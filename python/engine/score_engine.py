"""
ScoreEngine - K 歌打分引擎

评分模型:
    Final = Quality × Completion

      Quality(已唱内容质量) = 0.50·音准 + 0.30·节奏 + 0.20·气息
      Completion(演唱完成度) = 用户唱到的「可演唱区间」 ÷ 歌曲全部「可演唱区间」

三个分项各自先映射成显示分再加权，所以总分 = 界面上那三个数的加权平均
× 完成度，用户拿计算器就能自己对账。节奏弃评时权重按比例摊回音准和气息。

旧版把这两件事混成了一件——只算「已唱部分的平均质量」，于是唱 1 分钟
和唱完整首只要质量相同分数就相同；再叠加 40 + 60·CDF 的硬底线，raw=0
也能拿 40 分，「没唱」「唱了一半」「唱得极差」全被压在同一档分数上。

「可演唱区间」不等于歌曲总时长：前奏 / 间奏 / 尾奏 / 纯音乐本来就不要求
用户唱，所以完成度的分母是原唱真正在唱的那些区间。

分数映射统一用 100·raw**0.85：单调、无硬底线，
raw 0.0→0 分，0.5→55 分，0.7→73 分，1.0→100 分。

节奏只能在句子级评：歌词时间轴是行级的，拿不到每个字的时刻，所以评的是
「每一句有没有及时唱」，不是「每个字踩得准不准」。评不了时 rhythm 返回
None（未参评）而不是塞一个别的维度的综合分进去冒充节奏分。

时间戳不推进（没按播放就开唱）时数据不可用，直接判 0 分并在 warning 里
说明原因，不给一个看起来像评分、实际毫无含义的数字。
"""
import numpy as np

# ─── 质量权重 ───
W_PITCH = 0.50
W_RHYTHM = 0.30
W_BREATH = 0.20

# 0~1 原始分 → 0~100 显示分的曲线指数。
# 取 <1 让中段更好读（raw 0.6 → 65 分，普通人水平），且不设任何硬底线
SCORE_CURVE = 0.85

# ─── 音准 ───
PITCH_MATCH_TOL = 0.05      # 与基线浊音帧的最大时间差；基线帧间隔 10ms
PITCH_ZERO_CENTS = 150.0    # 偏差达到该音分数即 0 分（100 cents = 半音）
OCTAVE_CENTS = 1200.0
OCTAVE_PARTIAL = 0.6        # 八度偏差的部分分：童声常高八度，全删会冤杀

# ─── 可演唱区间 ───
BASE_FRAME_PAD = 0.005      # 基线 DIO frame_period=10ms，取半帧
BASE_MERGE_GAP = 0.35       # 与 vocal_segments.MERGE_GAP 一致，换气不切成两段
BASE_MIN_SEGMENT = 0.30     # 与 vocal_segments.MIN_SEGMENT 一致，滤掉碎帧
DEFAULT_LINE_DUR = 3.0      # 歌词缺 end 且后面没有行时的兜底行长
MAX_LINE_DUR = 15.0         # 单行时长上限，防止异常歌词吞掉整段间奏

# ─── 用户演唱区间 ───
USER_FRAME_PAD = 0.06       # 前端 2048 样本 @44100Hz ≈ 46ms 一帧，取半帧稍大
USER_MERGE_GAP = 0.25       # 用户浊音帧间隔超过该值视为换气/断句

# ─── 节奏 ───
RHYTHM_EARLY_TOL = 0.25     # 允许比歌词起点早开口的时间
RHYTHM_LATE_MAX = 1.0       # 晚开口容差上限，长句不因此获得无限宽容
LINE_MIN_COVER = 0.40       # 连贯唱过一句时，该句内至少要唱满这个比例

# ─── 时间轴有效性 ───
MIN_FRAMES_FOR_TIMING = 8   # 少于此帧数谈不上「时间轴」，不判退化
DEGENERATE_SPAN = 0.20      # 这么多帧却挤在这么短的时间里 → 播放时钟没走
LOW_COMPLETION_HINT = 0.50  # 完成度低于此值时提示，解释总分为何被折算

# ─── 气息 ───
PHRASE_GAP = 0.25           # 用户帧间隔超过该值即一次断气
NOTE_SPLIT_CENTS = 100.0    # 音高台阶超过一个半音视为「换音」而非「抖动」
NOTE_CENTER_WINDOW = 0.5    # 估计当前音中位音高时回看的时间跨度
MIN_NOTE_DUR = 0.15         # 新音高要维持这么久才算真换音，否则是抖动
STABILITY_GOOD_CENTS = 15.0
STABILITY_BAD_CENTS = 80.0
BREAKS_GOOD_PER_SEC = 0.5   # 正常短语换气
BREAKS_BAD_PER_SEC = 2.0    # 断裂频繁
BREATH_STABILITY_WEIGHT = 0.6


class ScoreEngine:

    # ─── 核心打分入口 ───

    def calculate_final_score(self, user_pitches, baseline_pitches,
                              user_onsets=None, lyric_timestamps=None):
        user = self._voiced_points(user_pitches)

        # 全程静音 / 无浊音帧 → 0 分。没有 40 分保底，
        # 「没唱」就是 0，不会和「唱了但极差」撞在同一个分数上
        if not user:
            return self._zero_result(
                '未检测到有效演唱（全程静音或没有浊音帧）', lyric_timestamps)

        # 播放时钟没走时几百帧会全盖同一个时间戳，这种数据算出来的分数
        # （实测 0.9 分）没有任何含义，必须直接拦下并告诉用户为什么
        if self._timeline_degenerate(user):
            return self._zero_result(
                '演唱时间轴没有推进：请先播放伴奏再开始演唱，'
                '否则无法判定你唱到了歌曲的哪个位置',
                lyric_timestamps,
            )

        singable = self._singable_intervals(baseline_pitches, lyric_timestamps)

        pitch_raw = self._calculate_pitch_score(
            user, baseline_pitches, singable)
        breath_raw = self._calculate_breath_score(user)
        coverage = self._calculate_coverage(user, singable)

        phrases = self._user_phrases(user)
        onsets = user_onsets or self._extract_user_onsets(user)

        # 先各自映射成显示分，再加权。总分必须能由界面上那三个数直接
        # 算出来（加权平均 × 完成度），否则用户拿着 11.6 / 9.3 / 92.4 怎么按都
        # 按不出总分，只能当成算错了。把曲线加在加权之后虽然权重语义更
        # 纯粹，但分项相差悬殊时会凭空多出 5~6 分的 Jensen 间隙，无法对账
        pitch = self._map_score(pitch_raw)
        breath = self._map_score(breath_raw)

        if self._timed_lyrics(lyric_timestamps):
            rhythm = self._map_score(
                self._calculate_rhythm_score(onsets, phrases, lyric_timestamps))
            quality = pitch * W_PITCH + rhythm * W_RHYTHM + breath * W_BREATH
        else:
            # 歌词没有可用时间轴就无从评节奏：权重按比例摊回音准和气息，
            # 避免 0 分拖低总分；但节奏本身必须如实报「未参评」（None）。
            # 旧版把摊完的综合分塞进 rhythm，界面上那一栏明明写着「节奏」，
            # 展示的却是音准+气息 —— 用户根本无从察觉那不是节奏分
            rhythm = None
            remaining = W_PITCH + W_BREATH
            quality = (pitch * W_PITCH + breath * W_BREATH) / remaining

        total = quality * coverage ** SCORE_CURVE
        completion = round(100.0 * coverage, 1)

        return {
            'total': round(min(max(total, 0.0), 100.0), 1),
            'pitch': round(pitch, 1),
            'rhythm': None if rhythm is None else round(rhythm, 1),
            'breath': round(breath, 1),
            'completion': completion,
            # 总分被完成度折算时必须解释原因，否则「音准气息都还行、
            # 总分 1 分」看起来就像算错了
            'warning': None if coverage >= LOW_COMPLETION_HINT else (
                f'只唱到全曲 {completion:.0f}% 的可演唱部分，'
                f'总分已按完成度折算（总分 = 已唱内容质量 × 完成度）'
            ),
        }

    def _zero_result(self, warning, lyric_timestamps=None):
        """没有任何可评内容时的统一出口：全 0，并带上原因

        rhythm 仍要看歌词有没有时间轴：非同步歌词下节奏本来就评不了，
        这里顺手写 0.0 等于又造了一个假节奏分。
        """
        return {
            'total': 0.0,
            'pitch': 0.0,
            'rhythm': 0.0 if self._timed_lyrics(lyric_timestamps) else None,
            'breath': 0.0,
            'completion': 0.0,
            'warning': warning,
        }

    @staticmethod
    def _timeline_degenerate(user):
        """帧数不少却全挤在同一时刻 → 时间戳不可用

        前端约 46ms 出一帧，MIN_FRAMES_FOR_TIMING 帧正常至少横跨 0.3s。
        全部盖上同一个时间戳只可能是播放时钟没走：usePlaybackClock 在
        isPlaying=false 时直接返回冻结的 currentTime（没按播放就开唱）。
        """
        if len(user) < MIN_FRAMES_FOR_TIMING:
            return False
        return user[-1]['time'] - user[0]['time'] < DEGENERATE_SPAN

    # ─── 分数映射 ───

    def _map_score(self, raw_score):
        """
        0~1 原始分 → 0~100 显示分: 100 · raw**0.85

        单调、无硬底线。旧版 40 + 60·Φ((raw-0.6)/0.15) 规定 raw=0 也有
        40 分，raw 0.0/0.1/0.2/0.3 全落在 40~41，低质量结果被压成一片。
        """
        raw = min(max(float(raw_score), 0.0), 1.0)
        return 100.0 * raw ** SCORE_CURVE

    @staticmethod
    def _ramp(value, good, bad):
        """good 以内满分，bad 以外 0 分，中间线性过渡"""
        if bad <= good:
            return 0.0
        return float(min(max((bad - value) / (bad - good), 0.0), 1.0))

    # ─── 数据整形 ───

    @staticmethod
    def _voiced_points(pitches):
        """只留下有效浊音帧并按时间排序（前端只上传浊音帧，这里做防御性过滤）"""
        points = []
        for p in pitches or []:
            freq = p.get('frequency')
            time = p.get('time')
            if freq is None or time is None:
                continue
            freq = float(freq)
            time = float(time)
            if freq <= 0 or not np.isfinite(freq) or not np.isfinite(time):
                continue
            points.append({'time': time, 'frequency': freq})
        points.sort(key=lambda q: q['time'])
        return points

    @staticmethod
    def _intervals_from_times(times, pad, merge_gap, min_len):
        """把一串时间点还原成「这段时间在唱」的区间

        帧是采样点不是时长，直接拿首尾相减会漏掉帧间隔本身；
        先把每帧扩成 [t-pad, t+pad]，再把相邻的合并成区间。
        """
        if len(times) == 0:
            return []

        ordered = sorted(float(t) for t in times)
        intervals = []
        start = ordered[0]
        prev_end = ordered[0] + pad
        for t in ordered[1:]:
            if t - pad <= prev_end + merge_gap:
                prev_end = t + pad
            else:
                intervals.append((start - pad, prev_end))
                start = t
                prev_end = t + pad
        intervals.append((start - pad, prev_end))

        return [(s, e) for s, e in intervals if e - s >= min_len]

    @staticmethod
    def _merge_overlaps(intervals):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _intersect(a, b):
        """两组区间的交集；入参需各自有序且不重叠"""
        out = []
        i = j = 0
        while i < len(a) and j < len(b):
            start = max(a[i][0], b[j][0])
            end = min(a[i][1], b[j][1])
            if start < end:
                out.append((start, end))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return out

    @staticmethod
    def _inside_mask(times, intervals):
        mask = np.zeros(len(times), dtype=bool)
        for start, end in intervals:
            mask |= (times >= start) & (times <= end)
        return mask

    # ─── 可演唱区间 ───

    def _singable_intervals(self, baseline_pitches, lyric_timestamps):
        """
        用户「应该唱」的区间，也就是完成度的分母。

        以原唱基线的浊音区间为准：基线已被 mask_to_singing_segments
        限制在人声段内，段外 frequency=0，所以浊音帧的自然分组就是
        「可演唱区间」，前奏 / 间奏 / 尾奏 / 纯音乐自动被排除。

        有歌词时间轴时再与歌词求交——lrc 把每行的 end 设成下一行的
        start，单独用歌词会把整段间奏也算成「应该唱」。
        """
        baseline = self._intervals_from_times(
            [p['time'] for p in self._voiced_points(baseline_pitches)],
            pad=BASE_FRAME_PAD,
            merge_gap=BASE_MERGE_GAP,
            min_len=BASE_MIN_SEGMENT,
        )
        lyric = self._lyric_intervals(lyric_timestamps)

        if baseline and lyric:
            # 歌词整体错位时交集可能为空，退回基线区间，别把完成度打成 0
            return self._intersect(baseline, lyric) or baseline
        return baseline or lyric

    @staticmethod
    def _timed_lyrics(lyric_timestamps):
        """歌词里真正带时间轴的条目 → 有序的 [(start, end)]

        LRC / ID3 的非同步歌词 start 为 null，整批过滤掉；
        行尾缺失时用下一行起点兜底，最后一行给个默认行长。
        """
        lines = []
        for lyric in lyric_timestamps or []:
            start = lyric.get('start')
            if start is None:
                continue
            start = float(start)
            end = lyric.get('end')
            end = float(end) if end is not None and float(
                end) > start else None
            lines.append((start, end))

        if not lines:
            return []

        lines.sort(key=lambda x: x[0])
        timed = []
        for i, (start, end) in enumerate(lines):
            if end is None:
                end = lines[i + 1][0] if i + \
                    1 < len(lines) else start + DEFAULT_LINE_DUR
            timed.append((start, min(end, start + MAX_LINE_DUR)))
        return timed

    @staticmethod
    def _lyric_intervals(lyric_timestamps):
        return ScoreEngine._merge_overlaps(
            ScoreEngine._timed_lyrics(lyric_timestamps)
        )

    def _calculate_coverage(self, user, singable):
        """
        完成度 = 用户实际唱到的可演唱区间 ÷ 全部可演唱区间

        只算交集：在前奏里唱再多也不涨完成度，
        中途停下没唱的部分老老实实扣分。
        """
        total = sum(end - start for start, end in singable)
        if total <= 0:
            # 基线和歌词都为空、拿不到任何可演唱区间信息时不乘完成度。
            # 这种情况下音准也无从对齐，质量分本身已经说明问题
            return 1.0

        sung = self._user_phrases(user)
        covered = sum(end - start for start,
                      end in self._intersect(sung, singable))
        return min(max(covered / total, 0.0), 1.0)

    @staticmethod
    def _user_phrases(user):
        """用户「连续在唱」的段落区间

        前端只上传浊音帧，相邻帧间隔超过 USER_MERGE_GAP 就是换气/断句，
        据此还原出一个个演唱段落。完成度和节奏都要用它，
        不能各自按帧重新推一遍。
        """
        return ScoreEngine._intervals_from_times(
            [p['time'] for p in user],
            pad=USER_FRAME_PAD,
            merge_gap=USER_MERGE_GAP,
            min_len=0.0,
        )

    @staticmethod
    def _extract_user_onsets(user):
        """从连续浊音帧里提取真正的起音：每段连续演唱只算一次

        把每一帧都当起音的话，唱 10 秒会产生两百多个候选，
        one onset ↔ one lyric 形同虚设 —— 逐句匹配时任何一句都能
        顺手抓到一帧，「命中即消费」根本约束不了什么。
        """
        if not user:
            return []
        times = np.array([p['time'] for p in user], dtype=float)
        onsets = [float(times[0])]
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > USER_MERGE_GAP:
                onsets.append(float(times[i]))
        return onsets

    # ─── 音准分 ───

    def _calculate_pitch_score(self, user, baseline_pitches, singable):
        """
        用户音高与原唱基线的音分偏差，返回 [0, 1] 的准确率。

        只评价「可演唱区间」内的帧：在前奏 / 间奏里自己哼的不算演唱，
        既不该拿它抬分，也不该拿它扣分。

        最近邻匹配基线浊音帧：不能对 frequency 跨静音线性插值，
        否则静音段会插出幻影中间频率冤枉短语边缘帧；时差 >50ms 不评。

        分级准确率: 偏差 0 → 1 分，≥150 cents → 0 分，线性过渡；
        八度偏差(|cents|≈1200)按 60% 部分分计，不算满分。
        """
        baseline = self._voiced_points(baseline_pitches)
        if not user or not baseline:
            return 0.0

        u_t = np.array([p['time'] for p in user])
        u_f = np.array([p['frequency'] for p in user])
        if singable:
            inside = self._inside_mask(u_t, singable)
            u_t, u_f = u_t[inside], u_f[inside]
        if not len(u_t):
            return 0.0

        v_t = np.array([p['time'] for p in baseline])
        v_f = np.array([p['frequency'] for p in baseline])

        # 最近邻基线浊音帧（基线帧间隔 10ms，50ms 容差足够）
        idx = np.clip(np.searchsorted(v_t, u_t), 1, len(v_t) - 1)
        left, right = idx - 1, idx
        nearest = np.where(
            np.abs(v_t[left] - u_t) <= np.abs(v_t[right] - u_t), left, right
        )
        valid = np.abs(v_t[nearest] - u_t) < PITCH_MATCH_TOL
        if not valid.any():
            return 0.0

        cents = 1200 * np.log2(u_f[valid] / v_f[nearest][valid])

        acc_direct = np.clip(1.0 - np.abs(cents) / PITCH_ZERO_CENTS, 0.0, 1.0)
        acc_octave = OCTAVE_PARTIAL * np.clip(
            1.0 - np.abs(np.abs(cents) - OCTAVE_CENTS) / PITCH_ZERO_CENTS,
            0.0, 1.0,
        )
        return float(np.mean(np.maximum(acc_direct, acc_octave)))

    # ─── 节奏分 ───

    def _calculate_rhythm_score(self, onsets, phrases, lyric_timestamps):
        """
        逐句判定「这一句有没有及时唱」，返回 [0, 1] 的命中率。

        歌词时间轴是行级的（LRC 一行 / _distribute_over_vocals 一段），
        拿不到每个字的时刻，所以这里评的只能是句子级的进入时机，
        不是「每个字踩得准不准」。

        一句命中有两种方式：
          1. 有一个起音落在 [start-早开口容差, start+晚开口容差] 内 ——
             重新开口且开在点上。起音命中即被消费，一个起音只能算一句；
             旧实现让每句各自取全局最近的 onset，一个 1.05 的音能把
             1.00/1.10/1.20/1.30 四句全部命中。
          2. 没有新起音，但这句开始时用户正连着上文在唱，且在这句里
             唱满了 LINE_MIN_COVER —— 连贯唱法不该被当成漏唱。
             要求唱满比例是为了挡住「短促一声蹭到后面几句」。
        """
        lines = self._timed_lyrics(lyric_timestamps)
        if not lines:
            return 0.0
        onsets = sorted(float(o) for o in onsets or [])

        hits = 0
        cursor = 0
        for start, end in lines:
            window_start = start - RHYTHM_EARLY_TOL
            window_end = start + min(
                max(RHYTHM_EARLY_TOL, 0.5 * (end - start)), RHYTHM_LATE_MAX
            )
            # 比起唱窗口还早的起音不属于这一句，也不能留给后面的句子
            while cursor < len(onsets) and onsets[cursor] < window_start:
                cursor += 1
            if cursor < len(onsets) and onsets[cursor] <= window_end:
                hits += 1
                cursor += 1
            elif self._covers_line(phrases, start, end):
                hits += 1

        return hits / len(lines)

    @staticmethod
    def _covers_line(phrases, start, end):
        """是否有演唱段落跨过 start，且在 [start, end] 内唱满了 LINE_MIN_COVER"""
        if end <= start:
            return False
        for phrase_start, phrase_end in phrases:
            if phrase_end < start:
                continue
            if phrase_start > start:
                return False        # 后面的段落都在这一句之后才开始
            return (min(phrase_end, end) - start) >= LINE_MIN_COVER * (end - start)
        return False

    # ─── 气息分 ───

    def _calculate_breath_score(self, user):
        """
        气息 = 0.6·长音稳定性 + 0.4·气息连续性，返回 [0, 1]。

        稳定性不能用「相邻帧音分跳变」衡量：C4→D4→E4 是正常的旋律换音，
        算成抖动会让唱得越有旋律的人气息分越低。这里先把浊音帧按音高
        台阶切成一个个「音」，只统计音内部的偏离——超过一个半音、
        且新音高站得住的台阶才是换音，音内残差才是颤音 / 抖动 / 破音。

        音内偏离取均值而不是中位数：已经按音切开了，不存在旧版那种
        「换音处的大跳混进来」的问题，而抖动帧往往不到半数，
        取中位数会被大量的 0 偏离淹没、永远测不出抖。

        连续性看帧间隔：前端只上传浊音帧，静音不会成为帧，
        所以相邻帧时间差超过 PHRASE_GAP 就是一次断气。按演唱时长归一，
        不用帧数——同样的唱法在不同帧率下不该拿到不同分数。
        """
        if len(user) < 2:
            return 0.0

        times = np.array([p['time'] for p in user])
        freqs = np.array([p['frequency'] for p in user])

        deviations = []
        for run_t, run_f in self._phrase_runs(times, freqs):
            deviations.extend(self._within_note_deviations(run_t, run_f))

        continuity = self._continuity(times)
        if not deviations:
            # 每个音都只有一帧（极快的转音）时测不出音内稳定性，只看连续性
            return continuity

        jitter = float(np.mean(deviations))
        stability = self._ramp(
            jitter, STABILITY_GOOD_CENTS, STABILITY_BAD_CENTS)
        return (
            BREATH_STABILITY_WEIGHT * stability
            + (1 - BREATH_STABILITY_WEIGHT) * continuity
        )

    @staticmethod
    def _phrase_runs(times, freqs):
        """按时间间隔切短语：跨换气的两帧不该被当成一次音高跳变

        旧版按数组下标是否连续来切，但前端只上传浊音帧，
        下标永远连续 —— 整场演唱被当成一个短语，
        跨 20 秒间奏的音高跳变也被算进了稳定性。
        """
        runs = []
        start = 0
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > PHRASE_GAP:
                runs.append((times[start:i], freqs[start:i]))
                start = i
        runs.append((times[start:], freqs[start:]))
        return [(t, f) for t, f in runs if len(f) >= 2]

    @staticmethod
    def _within_note_deviations(times, freqs):
        """把一个短语切成一个个「音」，返回每帧相对所属音中位音高的偏离(cents)"""
        cents = 1200 * np.log2(freqs)
        n = len(cents)

        notes = []
        start = 0
        for i in range(1, n):
            # 当前音的中位音高：只回看 NOTE_CENTER_WINDOW，长音缓慢跑调也能跟上
            center_from = times[i] - NOTE_CENTER_WINDOW
            lo = max(start, int(np.searchsorted(times, center_from)))
            center = float(np.median(cents[lo:i]))
            if abs(cents[i] - center) <= NOTE_SPLIT_CENTS:
                continue

            # 候选换音还得「站得住」：往后 MIN_NOTE_DUR 的平均音高仍远离旧音
            # 才算真换音。抖动会弹回旧音高——±60 cents 的交替抖动峰峰值有
            # 120 cents，只看单帧差会把它误判成一路换音，抖动就永远测不出来。
            # 这里用均值不用中位数：抖动周期比窗口短时中位数会锁在某一侧极值上
            hold = int(np.searchsorted(times, times[i] + MIN_NOTE_DUR))
            hold = min(max(hold, i + 2), n)
            if abs(float(np.mean(cents[i:hold])) - center) <= NOTE_SPLIT_CENTS:
                continue

            notes.append(cents[start:i])
            start = i
        notes.append(cents[start:])

        deviations = []
        for note in notes:
            if len(note) < 2:
                continue
            center = float(np.median(note))
            deviations.extend(np.abs(note - center).tolist())
        return deviations

    @staticmethod
    def _continuity(times):
        gaps = np.diff(times)
        breaks = int(np.sum(gaps > PHRASE_GAP))
        duration = float(times[-1] - times[0])
        if duration <= 1e-6:
            return 1.0
        return ScoreEngine._ramp(
            breaks / duration, BREAKS_GOOD_PER_SEC, BREAKS_BAD_PER_SEC
        )
