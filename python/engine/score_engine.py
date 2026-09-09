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
        偏差 < 50 cents 视为命中。
        返回 [0, 1] 的准确率。
        """
        if not user_pitches or not baseline_pitches:
            return 0.0

        # 时间对齐: 线性插值 baseline 到 user 的时间点
        user_times = np.array([p['time'] for p in user_pitches])
        user_freqs = np.array([p['frequency'] for p in user_pitches])
        base_times = np.array([p['time'] for p in baseline_pitches])
        base_freqs = np.array([p['frequency'] for p in baseline_pitches])

        # 过滤掉 unvoiced 帧 (frequency == 0 or None)
        voiced_mask = (user_freqs > 0) & np.isfinite(user_freqs)
        if not voiced_mask.any():
            return 0.0

        user_times = user_times[voiced_mask]
        user_freqs = user_freqs[voiced_mask]

        # 插值 baseline 到 user 时间点
        base_interp = np.interp(user_times, base_times, base_freqs,
                               left=0, right=0)
        base_voiced = base_interp > 0
        if not base_voiced.any():
            return 0.0

        # 计算音分偏差: cents = 1200 * log2(f1/f2)
        user_f = user_freqs[base_voiced]
        base_f = base_interp[base_voiced]
        cents = 1200 * np.log2(user_f / base_f)

        # 八度等价：将 cents 归约到 [-600, 600]，唱高/低一个八度仍算命中
        cents = ((cents + 600) % 1200) - 600

        # 命中率: |cents| < 50 视为命中 (quarter-tone 容差)
        hits = np.abs(cents) < 50
        return float(np.mean(hits)) if len(hits) > 0 else 0.0

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

        # 长音稳定性: 音高的变异系数 (CV = std/mean)
        mean_f = np.mean(voiced_freqs)
        std_f = np.std(voiced_freqs)
        cv = std_f / mean_f if mean_f > 0 else 1.0

        # CV < 0.05 视为优秀, CV > 0.2 视为差
        stability = max(0.0, min(1.0, 1.0 - cv / 0.2))

        # 气息断裂: 统计 voiced/unvoiced 切换次数
        transitions = np.diff(voiced.astype(int))
        breaks = np.sum(transitions != 0)
        break_rate = breaks / len(freqs) if len(freqs) > 0 else 1.0
        # break_rate < 0.05 视为好
        continuity = max(0.0, 1.0 - break_rate / 0.1)

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
