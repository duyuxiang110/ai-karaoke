"""
ScoreEngine - K 歌打分引擎

打分维度:
  - 音准分 (50%): 用户音高与原唱基线的音分偏差
  - 节奏分 (30%): 用户咬字时间点与歌词时间戳的对齐程度
  - 气息分 (20%): 长音稳定性、气息控制

分数映射: 使用正态分布 CDF 将原始得分映射到 0-100，普通人得 70 分左右
"""
import numpy as np
from scipy.stats import norm


class ScoreEngine:

    def __init__(self):
        self.weights = {
            'pitch': 0.50,
            'rhythm': 0.30,
            'breath': 0.20,
        }
        # 正态分布参数: mu=0.6 表示平均原始准确率为 60%
        # 映射后: raw=0.6 → score=70, raw=0.9 → score=99, raw=0.3 → score=41
        self.norm_mu = 0.6
        self.norm_sigma = 0.15

    # ─── 核心打分入口 ───

    def calculate_final_score(self, user_pitches, baseline_pitches,
                              user_onsets=None, lyric_timestamps=None):
        # 完全没有可判定的演唱输入（全程静音/无浊音帧）时给 0 分。
        # 否则 raw=0 经 CDF 映射会落在 40 分底线上，
        # 「没唱」和「唱了但极差」就成了同一个分数
        if not self._has_voiced_input(user_pitches):
            return {
                'total': 0.0,
                'pitch': 0.0,
                'rhythm': 0.0,
                'breath': 0.0,
            }

        pitch_raw = self._calculate_pitch_score(user_pitches, baseline_pitches)
        breath_raw = self._calculate_breath_score(user_pitches)

        has_rhythm_data = user_onsets and lyric_timestamps
        if has_rhythm_data:
            rhythm_raw = self._calculate_rhythm_score(user_onsets, lyric_timestamps)
            weighted = (
                pitch_raw * self.weights['pitch']
                + rhythm_raw * self.weights['rhythm']
                + breath_raw * self.weights['breath']
            )
        else:
            # 无节奏数据时把节奏权重重新分配给音准和气息，避免 0 分拖低总分
            remaining = self.weights['pitch'] + self.weights['breath']
            weighted = (
                pitch_raw * self.weights['pitch'] / remaining
                + breath_raw * self.weights['breath'] / remaining
            )
            rhythm_raw = weighted

        return {
            'total': float(round(self._normalize_score(weighted), 1)),
            'pitch': float(round(self._normalize_score(pitch_raw), 1)),
            'rhythm': float(round(self._normalize_score(rhythm_raw), 1)),
            'breath': float(round(self._normalize_score(breath_raw), 1)),
        }

    # ─── 音准分 ───

    def _has_voiced_input(self, user_pitches):
        """用户音高里是否存在浊音帧；静音演唱时客户端不会收到任何有效帧"""
        if not user_pitches:
            return False
        return any((p.get('frequency') or 0) > 0 for p in user_pitches)

    def _calculate_pitch_score(self, user_pitches, baseline_pitches):
        """
        将用户音高与原唱基线在时间轴上对齐，计算每个帧的音分偏差。
        音分 (cents): 100 cents = 1 semitone, 1200 cents = 1 octave
        最近邻匹配基线浊音帧：不能对 frequency 跨静音线性插值，
        否则静音段会插出幻影中间频率冤枉短语边缘帧；时差 >50ms 不评。
        八度偏差给 60% 部分分（童声常比原唱高八度，全删会冤杀），不算满分。
        分级准确率: 偏差 0 → 1 分, ≥150 cents → 0 分, 线性过渡。
        返回 [0, 1] 的准确率。
        """
        if not user_pitches or not baseline_pitches:
            return 0.0

        user_times = np.array([p['time'] for p in user_pitches])
        user_freqs = np.array([p['frequency'] for p in user_pitches])
        base_times = np.array([p['time'] for p in baseline_pitches])
        base_freqs = np.array([p['frequency'] for p in baseline_pitches])

        user_voiced = (user_freqs > 0) & np.isfinite(user_freqs)
        base_voiced = (base_freqs > 0) & np.isfinite(base_freqs)
        if not user_voiced.any() or not base_voiced.any():
            return 0.0

        u_t = user_times[user_voiced]
        u_f = user_freqs[user_voiced]
        v_t = base_times[base_voiced]
        v_f = base_freqs[base_voiced]

        # 最近邻基线浊音帧（基线帧间隔 10ms，50ms 容差足够）
        idx = np.clip(np.searchsorted(v_t, u_t), 1, len(v_t) - 1)
        left, right = idx - 1, idx
        nearest = np.where(
            np.abs(v_t[left] - u_t) <= np.abs(v_t[right] - u_t), left, right
        )
        valid = np.abs(v_t[nearest] - u_t) < 0.05
        if not valid.any():
            return 0.0

        cents = 1200 * np.log2(u_f[valid] / v_f[nearest][valid])

        # 分级准确率: 本调全分; 八度偏差(|cents|≈1200)按 60% 部分分计
        acc_direct = np.clip(1.0 - np.abs(cents) / 150.0, 0.0, 1.0)
        acc_octave = 0.6 * np.clip(
            1.0 - np.abs(np.abs(cents) - 1200.0) / 150.0, 0.0, 1.0
        )
        acc = np.maximum(acc_direct, acc_octave)
        return float(np.mean(acc))

    # ─── 节奏分 ───

    def _calculate_rhythm_score(self, user_onsets, lyric_timestamps):
        """
        对比用户咬字时间点与原唱歌词时间戳。
        偏差 < 200ms 视为命中。
        返回 [0, 1] 的命中率。
        """
        if not user_onsets or not lyric_timestamps:
            return 0.0  # 无歌词数据时给中性分

        # 为每个歌词词找到最近的用户 onset
        hits = 0
        total = 0
        for lyric in lyric_timestamps:
            lyric_time = lyric.get('start', 0)
            min_diff = min(
                abs(uo - lyric_time) for uo in user_onsets
            ) if user_onsets else float('inf')
            if min_diff < 0.2:  # 200ms 容差
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    # ─── 气息分 ───

    def _calculate_breath_score(self, user_pitches):
        """
        分析长音稳定性: 在持续 voiced 段中，音高的标准差越小越好。
        同时检测气息断裂 (voiced → unvoiced 频繁切换)。
        返回 [0, 1] 的稳定性评分。
        """
        if not user_pitches:
            return 0.0

        freqs = np.array([p['frequency'] for p in user_pitches])
        voiced = freqs > 0

        if not voiced.any():
            return 0.0

        # 检测 voiced 段
        voiced_freqs = freqs[voiced]
        if len(voiced_freqs) < 2:
            return 0.0

        # 长音稳定性: 浊音段内逐帧音高跳变的 cents 中位数。
        # 不能用全程 std/mean——唱旋律必然跨音高，std 恒大会把稳定性压到 0，
        # 总分永远 40 多；中位数不受换音处的大跳影响，只反映颤音/抖动
        idx = np.flatnonzero(voiced)
        jumps = []
        run_start = 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or idx[i] != idx[i - 1] + 1:
                run = voiced_freqs[run_start:i]
                if len(run) >= 2:
                    cents = 1200 * np.log2(run[1:] / run[:-1])
                    jumps.extend(np.abs(cents).tolist())
                run_start = i
        if not jumps:
            return 0.0
        med_jump = float(np.median(jumps))
        # 中位跳变 < 20 cents 视为优秀, > 100 cents 视为差
        stability = max(0.0, min(1.0, 1.0 - med_jump / 100.0))

        # 气息断裂: 统计 voiced/unvoiced 切换次数，按演唱时长归一
        # （除以帧数会对采样率敏感：同样唱法 10Hz 和 100Hz 输入分数不同）
        transitions = np.diff(voiced.astype(int))
        breaks = np.sum(transitions != 0)
        times = np.array([p['time'] for p in user_pitches])
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        breaks_per_sec = breaks / duration if duration > 1e-6 else 1.0
        # <0.5 次/秒 视为好（正常短语换气），≥2 次/秒 视为断裂频繁
        continuity = max(0.0, min(1.0, 1.0 - breaks_per_sec / 2.0))

        return 0.6 * stability + 0.4 * continuity

    # ─── 正态分布映射 ───

    def _normalize_score(self, raw_score):
        """
        正态分布 CDF 映射:
          final = 40 + 60 * Φ((raw - mu) / sigma)

        raw = mu (0.6) → final = 70  (普通人水平)
        raw = 0.9     → final ≈ 99  (优秀)
        raw = 0.3     → final ≈ 41  (较差)
        raw = 0.0     → final ≈ 40  (底线)
        raw = 1.0     → final ≈ 100 (完美)
        """
        cdf = norm.cdf((raw_score - self.norm_mu) / self.norm_sigma)
        return 40 + 60 * cdf
