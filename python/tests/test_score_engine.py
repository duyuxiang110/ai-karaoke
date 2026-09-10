"""
ScoreEngine 打分模型测试

歌曲结构（对应用户提出的场景）:
    0   ── 20s    前奏
    20  ── 60s    第一段      ← 可演唱
    60  ── 80s    间奏
    80  ── 140s   第二段      ← 可演唱
    140 ── 180s   尾奏

可演唱区间合计 100s，歌曲总长 180s，两者必须区分开。

运行:
    cd python && venv/bin/python tests/test_score_engine.py -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.score_engine import (  # noqa: E402
    ScoreEngine, SCORE_CURVE, W_BREATH, W_PITCH, W_RHYTHM,
    PITCH_ALIGN_GRID, PITCH_ZERO_CENTS,
)

SONG_DURATION = 180.0
SINGABLE = [(20.0, 60.0), (80.0, 140.0)]
SINGABLE_TOTAL = sum(e - s for s, e in SINGABLE)   # 100s

BASE_FRAME = 0.01      # 基线 DIO frame_period=10ms
USER_FRAME = 0.046     # 前端 2048 样本 @44100Hz ≈ 46ms 一帧
NOTE_DUR = 0.5         # 基线旋律每个音 0.5s
BASE_FREQ = 220.0
SCALE = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]


def melody_freq(t):
    """t 时刻原唱应该在的音高；不在可演唱区间内返回 0"""
    for start, end in SINGABLE:
        if start <= t < end:
            idx = int((t - start) / NOTE_DUR) % len(SCALE)
            return BASE_FREQ * SCALE[idx]
    return 0.0


def make_baseline():
    """整首歌的基线音高帧：演唱段外 frequency=0（mask_to_singing_segments 的效果）"""
    n = int(SONG_DURATION / BASE_FRAME)
    return [
        {'time': round(k * BASE_FRAME, 4), 'frequency': melody_freq(k * BASE_FRAME)}
        for k in range(n)
    ]


def make_user(intervals, ratio=1.0, wobble_cents=0.0, step=USER_FRAME):
    """按原唱旋律在给定区间内生成用户浊音帧

    ratio:      整体音高倍率（2.0 = 高八度，3.0 = 完全跑调）
    wobble_cents: 逐帧交替 ±该音分，模拟气息抖动
    """
    out = []
    k = 0
    for seg_start, seg_end in intervals:
        t = seg_start
        while t < seg_end:
            freq = melody_freq(t) * ratio
            if wobble_cents:
                freq *= 2 ** ((wobble_cents if k % 2 == 0 else -wobble_cents) / 1200.0)
            out.append({'time': round(t, 4), 'frequency': freq})
            t += step
            k += 1
    return out


def make_tone(intervals, freq=300.0, step=USER_FRAME):
    """在给定区间内生成恒定音高的浊音帧（不跟着原唱旋律走）"""
    out = []
    for seg_start, seg_end in intervals:
        t = seg_start
        while t < seg_end:
            out.append({'time': round(t, 4), 'frequency': freq})
            t += step
    return out


def make_user_drift(intervals, max_delay, step=USER_FRAME):
    """跟着原唱旋律唱，但速度跟不上：段首准点，段尾已经差了 max_delay 秒

    K 歌场景的常态。每个音都唱对了，只是比原唱晚（max_delay>0）
    或早（max_delay<0）一点落到时间轴上，偏差随时间线性累积。
    """
    out = []
    for seg_start, seg_end in intervals:
        span = max(seg_end - seg_start, 1e-9)
        t = seg_start
        while t < seg_end:
            delay = max_delay * (t - seg_start) / span
            # 取音高时刻限在本段内，别因为平移而伸到间奏里拿到 0
            src = min(max(t - delay, seg_start), seg_end - 1e-6)
            out.append({'time': round(t, 4), 'frequency': melody_freq(src)})
            t += step
    return out


def make_lyrics(line_dur=4.0):
    """可演唱区间内每 line_dur 秒一句歌词"""
    lines = []
    for seg_start, seg_end in SINGABLE:
        t = seg_start
        while t + 0.5 < seg_end:
            lines.append({
                'text': 'la',
                'start': round(t, 3),
                'end': round(min(t + line_dur, seg_end), 3),
            })
            t += line_dur
    return lines


def reconcile(result):
    """拿界面上显示的那三个分项把总分重算一遍

    用户要的就是这个：总分 = 分项加权平均 × 完成度，拿计算器一按就能验。
    """
    if result['rhythm'] is None:
        quality = (W_PITCH * result['pitch'] + W_BREATH * result['breath']) \
            / (W_PITCH + W_BREATH)
    else:
        quality = (W_PITCH * result['pitch']
                   + W_RHYTHM * result['rhythm']
                   + W_BREATH * result['breath'])
    if result['completion'] is None:
        return quality      # 完成度无法评估时不折算，总分就等于质量分
    return quality * (result['completion'] / 100.0) ** SCORE_CURVE


class ScoreEngineTest(unittest.TestCase):

    def setUp(self):
        self.engine = ScoreEngine()
        self.baseline = make_baseline()
        self.lyrics = make_lyrics()

    # ─── 全程没唱 ───

    def test_全程没唱_总分为0(self):
        silent = [{'time': round(k * USER_FRAME, 4), 'frequency': 0.0} for k in range(100)]
        for user in ([], silent):
            result = self.engine.calculate_final_score(
                user_pitches=user,
                baseline_pitches=self.baseline,
                user_onsets=None,
                lyric_timestamps=self.lyrics,
            )
            self.assertEqual(0.0, result['total'])
            self.assertEqual(0.0, result['pitch'])
            self.assertEqual(0.0, result['rhythm'])
            self.assertEqual(0.0, result['breath'])
            self.assertEqual(0.0, result['completion'])

    # ─── 分数映射：删掉 40 分保底 ───

    def test_原始分映射没有40分保底(self):
        # 旧版 40 + 60*CDF 把 raw=0 / 0.1 / 0.2 / 0.3 全压在 40~41 分
        self.assertEqual(0.0, self.engine._map_score(0.0))
        self.assertEqual(100.0, self.engine._map_score(1.0))
        self.assertLess(self.engine._map_score(0.1), 20.0)
        self.assertLess(self.engine._map_score(0.2), 30.0)
        self.assertLess(self.engine._map_score(0.3), 40.0)
        # 单调递增
        raws = [i / 20.0 for i in range(21)]
        mapped = [self.engine._map_score(r) for r in raws]
        self.assertEqual(mapped, sorted(mapped))

    def test_唱完整首但音准全错_不再有40分保底(self):
        # 音准全错(×3 ≈ +1902 cents，既不是本调也不是八度)，
        # 且没有歌词数据 → 质量只剩气息，旧版会落在 40 分保底上
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=3.0),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=None,
        )
        self.assertLess(result['pitch'], 5.0)
        self.assertLess(result['total'], 38.0)

    # ─── 完成度：按可演唱区间算，不按歌曲总时长算 ───

    def test_完成度按可演唱区间计算_前奏间奏尾奏不计入(self):
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        # 唱满 20~60 + 80~140 就是 100% 完成度，
        # 而不是 100/180 = 55.6%
        self.assertAlmostEqual(100.0, result['completion'], delta=3.0)
        self.assertGreater(result['total'], 85.0)

    def test_只在前奏间奏里出声_完成度为0(self):
        # 前奏/间奏本来就不要求用户唱，在那里唱不算完成度，也不该评音准
        result = self.engine.calculate_final_score(
            user_pitches=make_tone([(1.0, 19.0), (61.0, 79.0), (141.0, 179.0)]),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertEqual(0.0, result['completion'])
        self.assertEqual(0.0, result['total'])

    # ─── 中途停止：分数必须明显下降 ───

    def test_中途停止_总分明显下降(self):
        full = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        # 只唱第一段 40s，覆盖 40/100 = 40% 的可演唱区间
        partial = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE[:1]),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertAlmostEqual(40.0, partial['completion'], delta=3.0)
        self.assertGreater(full['total'], 85.0)
        self.assertLess(partial['total'], 50.0)
        self.assertGreater(full['total'] - partial['total'], 35.0)

    def test_唱完整首质量一般_落在60到78分(self):
        # 整体偏低约 90 cents：音准只拿 ~40% 准确率，节奏和气息仍完整
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=2 ** (90 / 1200.0)),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertAlmostEqual(100.0, result['completion'], delta=3.0)
        self.assertGreaterEqual(result['total'], 60.0)
        self.assertLessEqual(result['total'], 78.0)

    def test_八度偏差仍有部分分_不全删(self):
        # 童声常比原唱高八度，全删会冤杀，但也不能算满分
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=2.0),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertGreater(result['pitch'], 50.0)
        self.assertLess(result['pitch'], 80.0)

    def test_恒定大幅偏差不能被八度宽容救回来(self):
        # 采样率对不上时用户频率会被整体缩放：48k 的音频按 44.1k 解，
        # 每个音都低 146.9 音分（差一个半音还多）。这既不是本调也不是
        # 八度，分数必须比「只偏 90 音分」更低，不能反过来更高。
        #
        # 把八度折叠塞进 DTW 代价里就会出这种事：-146.9 的帧被拿去贴
        # 半秒前的邻音，解释成「八度差 35 音分」，实测能骗到 69.5 分
        detuned = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=2 ** (-146.9 / 1200.0)),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        mild = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=2 ** (90.0 / 1200.0)),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertLess(
            detuned['pitch'], 25.0,
            msg=f'整体差一个半音还有 {detuned["pitch"]} 分: {detuned}')
        self.assertLess(
            detuned['pitch'], mild['pitch'],
            msg=f'偏差更大反而分更高: {detuned["pitch"]} vs {mild["pitch"]}')

    def test_音准诊断_能一眼看出频率链路错位(self):
        # 「音准 40」既可能是唱得差，也可能是两侧 frequency 压根不是一个
        # 定义（采样率对不上、单位不一致）。诊断必须把这两种情况分开：
        # 对齐率说明有没有大面积丢帧，两侧中位频率的比值说明有没有整体缩放
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=2 ** (-146.9 / 1200.0)),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        diag = result['diagnostics']
        self.assertIsNotNone(diag, msg='没给出音准诊断')
        # 每个音都唱到了，不存在「帧被判无效而丢掉」
        self.assertEqual(diag['frames'], diag['aligned'])
        # 带符号中位偏差直接把这个常数暴露出来；只报绝对值就分不出
        # 「忽高忽低」和「整体偏一边」，后者才是链路错位的特征
        self.assertAlmostEqual(-146.9, diag['med_cents'], delta=5.0)
        # 用户频率整体被压到基线的 0.919 倍（= 44100/48000）
        self.assertAlmostEqual(
            44100 / 48000, diag['user_hz'] / diag['base_hz'], delta=0.01)

    def test_没唱或没基线时_诊断为空而不是编一个(self):
        # 零分出口也得带上这个键，否则后端取诊断时要处处判空
        for user in ([], make_tone([(1.0, 19.0)])):
            result = self.engine.calculate_final_score(
                user_pitches=user,
                baseline_pitches=self.baseline,
                user_onsets=None,
                lyric_timestamps=self.lyrics,
            )
            self.assertIn('diagnostics', result)
        empty = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=[],
            user_onsets=None,
            lyric_timestamps=None,
        )
        self.assertIsNone(empty['diagnostics'])

    # ─── 音准对齐：允许时间伸缩，不死卡绝对秒数 ───

    def test_唱得准但越唱越慢_音准不该被判跑调(self):
        # 用户每个音都唱对了，只是比原唱晚了最多 0.45s（差不到一个音）。
        # 死卡绝对秒数的最近邻匹配拿 t 时刻的原唱音去比用户 t 时刻的音，
        # 旋律早就走到下一个音了，于是「音准很好」被算成大片跑调帧
        for delay in (0.45, -0.45):
            result = self.engine.calculate_final_score(
                user_pitches=make_user_drift(SINGABLE, max_delay=delay),
                baseline_pitches=self.baseline,
                user_onsets=None,
                lyric_timestamps=self.lyrics,
            )
            self.assertGreaterEqual(
                result['pitch'], 85.0,
                msg=f'偏移 {delay}s 时音准被误判: {result}')

    def test_恒定长音不能被时间伸缩救回来(self):
        # DTW 会主动寻找最省力的对应关系，必须确认它不会把
        # 「从头到尾一个音」拉伸成「跟着旋律走」
        result = self.engine.calculate_final_score(
            user_pitches=make_tone(SINGABLE, freq=BASE_FREQ),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertLess(result['pitch'], 45.0, msg=f'长音蹭分: {result}')

    def test_偏移超出对齐带宽_不强行平移凑分(self):
        # 晚了 3 秒才开口，唱的已经是完全不同的段落。时间伸缩只容下
        # 小幅 tempo 差，不能把整段旋律平移过去凑分。
        #
        # 这条直接测 _dtw_cents 而不走端到端：夹具旋律是 7 个音循环、
        # 周期 3.5s，平移 3s 恰好等于「早半拍」，同一个音真的就落在
        # 1s 带宽里，DTW 那样配对是合法的，端到端根本测不出带宽。
        # 所以换成一路向上、绝不重复的轮廓：平移出带宽之后，
        # 再没有任何便宜的对应关系可捡。
        n = 200
        v_t = np.arange(n) * PITCH_ALIGN_GRID
        v_c = np.arange(n) * 40.0        # 每格升 40 音分，即 800 音分/秒
        u_c = v_c.copy()                 # 音一个不差

        cents = self.engine._dtw_cents(v_t + 3.0, u_c, v_t, v_c)
        self.assertGreater(len(cents), 0, msg='带宽内一个候选都没有，测不出东西')
        # 带宽只有 ±1s，最省力的配对也差了 1600 音分，远在 0 分线之外
        self.assertGreater(
            float(np.min(np.abs(cents))), PITCH_ZERO_CENTS,
            msg=f'3s 偏移被带宽外的基线救回: {np.min(np.abs(cents))}')

        # 同样的轮廓、偏移收进带宽内就必须能对上，
        # 否则上一条只是在证明「DTW 什么都不认」
        cents = self.engine._dtw_cents(v_t + 0.4, u_c, v_t, v_c)
        self.assertLess(float(np.median(np.abs(cents))), PITCH_ZERO_CENTS)

    def test_准点演唱_对齐改动不能让分数反而变低(self):
        # 完全跟着原唱唱时，弹性对齐的最优路径就是恒等映射，
        # 结果必须与严格最近邻一致，不能因为换算法而凭空掉分
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertGreaterEqual(result['pitch'], 95.0)

    # ─── 节奏：起音提取 + one onset ↔ one lyric ───

    def test_起音从连续浊音帧提取_每段只算一次(self):
        # 把每一帧都当起音的话，唱 10 秒会产生两百多个候选，
        # one onset ↔ one lyric 形同虚设 —— 任何一句都能顺手抓到一帧
        frames = make_tone([(10.0, 10.2), (12.5, 12.7), (15.2, 15.4)])
        self.assertGreater(len(frames), 10)
        onsets = self.engine._extract_user_onsets(frames)
        self.assertEqual([10.0, 12.5, 15.2], [round(o, 2) for o in onsets])

    def test_节奏一对一匹配_单个onset不能命中多句(self):
        # 旧实现：1.05 一个 onset 会让下面四句全部命中，节奏直接 100%
        lyrics = [
            {'text': '你', 'start': 1.00, 'end': 1.10},
            {'text': '好', 'start': 1.10, 'end': 1.20},
            {'text': '世', 'start': 1.20, 'end': 1.30},
            {'text': '界', 'start': 1.30, 'end': 1.40},
        ]
        self.assertAlmostEqual(
            0.25,
            self.engine._calculate_rhythm_score([1.05], [], lyrics),
            delta=1e-9,
        )
        # 四个 onset 各归各句才是满分
        self.assertAlmostEqual(
            1.0,
            self.engine._calculate_rhythm_score(
                [1.02, 1.12, 1.22, 1.32], [], lyrics),
            delta=1e-9,
        )

    def test_短促一声_不能靠演唱段落蹭到后面几句(self):
        # 极短的一声：起音只够命中第一句，
        # 后面几句既没有新起音、也没被唱够时长，必须算未命中
        lyrics = [
            {'text': '你', 'start': 1.00, 'end': 1.10},
            {'text': '好', 'start': 1.10, 'end': 1.20},
            {'text': '世', 'start': 1.20, 'end': 1.30},
            {'text': '界', 'start': 1.30, 'end': 1.40},
        ]
        frames = make_tone([(1.05, 1.06)])
        rate = self.engine._calculate_rhythm_score(
            self.engine._extract_user_onsets(frames),
            self.engine._user_phrases(frames),
            lyrics,
        )
        self.assertAlmostEqual(0.25, rate, delta=1e-9)

    def test_连贯唱过整句_没有新起音也算命中(self):
        # 歌词时间轴是行级的，只能判「这一句有没有及时唱」。
        # 从 6s 一路连贯唱到 14s 的人在 10s 这句上没有新起音，
        # 但他确实正在唱这句 —— 只认起音会把连贯唱法冤枉成节奏全错
        lyrics = [{'text': 'x', 'start': 10.0, 'end': 14.0}]
        frames = make_tone([(6.0, 14.2)])
        onsets = self.engine._extract_user_onsets(frames)
        self.assertEqual([6.0], [round(o, 2) for o in onsets])
        rate = self.engine._calculate_rhythm_score(
            onsets, self.engine._user_phrases(frames), lyrics
        )
        self.assertAlmostEqual(1.0, rate, delta=1e-9)

    def test_节奏_每句都要对上_漏唱的句子算未命中(self):
        onsets = [line['start'] + 0.05 for line in self.lyrics[:5]]
        rate = self.engine._calculate_rhythm_score(onsets, [], self.lyrics)
        self.assertAlmostEqual(5 / len(self.lyrics), rate, delta=1e-9)

    def test_无时间轴歌词_节奏不参评而不是伪造分数(self):
        # LRC / ID3 的非同步歌词 start 是 null，这种歌词评不了节奏。
        # 权重必须摊回音准和气息，且节奏要如实报「未参评」——
        # 旧版把「音准+气息」的综合分塞进 rhythm，
        # 界面上那一栏明明写着「节奏」，展示的却是别的东西
        unsynced = [{'text': 'la', 'start': None, 'end': None}
                    for _ in range(10)]
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=unsynced,
        )
        self.assertIsNone(result['rhythm'])
        self.assertGreater(result['total'], 85.0)

    def test_零分出口也不能伪造节奏分(self):
        # 全程静音 / 时间轴没推进都会走零分出口，那里顺手把 rhythm 写成 0.0
        # 又是一个假节奏分：歌词本来就没有时间轴，节奏根本无从评起
        unsynced = [{'text': 'la', 'start': None, 'end': None}]
        silent = [{'time': round(k * USER_FRAME, 4), 'frequency': 0.0}
                  for k in range(100)]
        frozen = [{'time': 30.0, 'frequency': melody_freq(30.0)}
                  for _ in range(300)]
        for user in (silent, frozen):
            result = self.engine.calculate_final_score(
                user_pitches=user,
                baseline_pitches=self.baseline,
                user_onsets=None,
                lyric_timestamps=unsynced,
            )
            self.assertEqual(0.0, result['total'])
            self.assertIsNone(result['rhythm'])

    # ─── 总分必须能由显示的分项直接算出来 ───

    def test_总分能由显示的分项直接算出来(self):
        # 旧口径先加权原始分、再过映射曲线，曲线加在加权之后，
        # 显示分和总分之间就凭空差出 2~6 分（分项相差越悬殊差越多），
        # 用户拿着 11.6 / 9.3 / 92.4 怎么按都按不出 0.9，只能当成算错了
        cases = [
            make_user(SINGABLE),                            # 唱满，三项都高
            make_user(SINGABLE, ratio=3.0),                 # 唱满但音准全错
            make_user(SINGABLE, ratio=2 ** (90 / 1200.0)),  # 整体偏低 90 音分
            make_user(SINGABLE[:1]),                        # 只唱第一段
            make_user([(30.0, 34.0)]),                      # 只唱 4 秒
        ]
        for user in cases:
            result = self.engine.calculate_final_score(
                user_pitches=user,
                baseline_pitches=self.baseline,
                user_onsets=None,
                lyric_timestamps=self.lyrics,
            )
            self.assertAlmostEqual(
                reconcile(result), result['total'], delta=1.0,
                msg=f'总分对不上账: {result}')

    def test_节奏未评时_总分按音准和气息对账(self):
        # 节奏弃评时权重摊回剩下两项，但摊完必须照样能用显示分算出总分
        unsynced = [{'text': 'la', 'start': None, 'end': None}
                    for _ in range(10)]
        for user in (make_user(SINGABLE), make_user(SINGABLE, ratio=3.0)):
            result = self.engine.calculate_final_score(
                user_pitches=user,
                baseline_pitches=self.baseline,
                user_onsets=None,
                lyric_timestamps=unsynced,
            )
            self.assertIsNone(result['rhythm'])
            self.assertAlmostEqual(
                reconcile(result), result['total'], delta=1.0,
                msg=f'总分对不上账: {result}')

    # ─── 完成度无法评估时不能假装唱完整了 ───

    def test_没有可演唱区间数据_完成度报无法评估(self):
        # 基线和歌词都为空 → 拿不到任何「该唱什么」的信息。
        # 旧行为是 coverage = 1.0，等于宣称「你唱完整了」，
        # 于是只剩气息的质量分照样乘 1 进了总分，
        # 注释里写「没有可评内容」、实际却给了满分完成度
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=[],
            user_onsets=None,
            lyric_timestamps=None,
        )
        self.assertIsNone(result['completion'])
        self.assertTrue(result['warning'])
        self.assertIn('完成度', result['warning'])
        self.assertAlmostEqual(reconcile(result), result['total'], delta=0.5)

    def test_基线全静音且无歌词_完成度同样无法评估(self):
        # 基线有帧但全未浊音（mask_to_singing_segments 把所有帧都掩了），
        # 一样得不出可演唱区间，不能归到 100%
        silent = [{'time': round(k * BASE_FRAME, 4), 'frequency': 0.0}
                  for k in range(int(SONG_DURATION / BASE_FRAME))]
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=silent,
            user_onsets=None,
            lyric_timestamps=None,
        )
        self.assertIsNone(result['completion'])
        self.assertIsNone(result['rhythm'])
        self.assertTrue(result['warning'])

    def test_完成度无法评估时_不能比唱完整首得分高(self):
        # 旧口径下「什么都没得比」反而拿到 coverage=1.0，
        # 可能比真唱完整首但略有瑕疵的人分还高 —— 逆向激励
        no_data = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=[],
            user_onsets=None,
            lyric_timestamps=None,
        )
        full = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE, ratio=2 ** (90 / 1200.0)),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertLess(no_data['total'], full['total'])

    # ─── 时间轴没推进（没按播放就开唱）───

    def test_时间戳不推进_判为无效演唱并给出提示(self):
        # 播放时钟冻结时几百帧全盖同一个时间戳：完成度≈0、
        # 节奏只命中一句、音准对着基线上一个点比。
        # 旧行为是默默给出 0.9 分，用户完全无法理解发生了什么
        frozen = [{'time': 30.0, 'frequency': melody_freq(30.0)}
                  for _ in range(300)]
        result = self.engine.calculate_final_score(
            user_pitches=frozen,
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertEqual(0.0, result['total'])
        self.assertTrue(result['warning'])
        self.assertIn('播放', result['warning'])

    def test_只唱一小段_提示里说明完成度(self):
        # 总分被完成度折算到接近 0 时必须解释原因，
        # 否则「音准/气息都还行、总分 1 分」看起来就像算错了
        result = self.engine.calculate_final_score(
            user_pitches=make_user([(30.0, 34.0)]),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertLess(result['total'], 20.0)
        self.assertTrue(result['warning'])
        self.assertIn('%', result['warning'])

    def test_正常唱完整首_没有警告(self):
        result = self.engine.calculate_final_score(
            user_pitches=make_user(SINGABLE),
            baseline_pitches=self.baseline,
            user_onsets=None,
            lyric_timestamps=self.lyrics,
        )
        self.assertIsNone(result['warning'])

    # ─── 气息：正常换音不算不稳 ───

    def test_气息_快速正常换音不算不稳(self):
        # 快速转音：前端 46ms 一帧，旋律换音比帧率还快时每一帧都是一个新音。
        # C4→D4→E4→… 是正常的旋律台阶，不是气息问题。
        # 旧实现把所有相邻帧 cents 跳变混在一起取中位数，
        # 这种输入下 100% 的跳变都是 +200 cents 级别，气息稳定性被判成 0
        run = [
            {
                'time': round(i * USER_FRAME, 4),
                'frequency': BASE_FREQ * SCALE[i % len(SCALE)],
            }
            for i in range(30)
        ]
        self.assertGreater(self.engine._calculate_breath_score(run), 0.9)

    def test_气息_音高抖动会被扣分(self):
        steady = make_user([(20.0, 26.0)])
        wobbly = make_user([(20.0, 26.0)], wobble_cents=60.0)
        steady_score = self.engine._calculate_breath_score(steady)
        wobbly_score = self.engine._calculate_breath_score(wobbly)
        self.assertGreater(steady_score, 0.9)
        self.assertLess(wobbly_score, steady_score - 0.2)

    def test_气息_频繁断气会被扣分(self):
        # 同样是 6s，一段连续唱完 vs 每 0.4s 断一次
        continuous = make_user([(20.0, 26.0)])
        broken = []
        t = 20.0
        while t < 26.0:
            broken.extend(make_user([(t, t + 0.12)]))
            t += 0.4
        self.assertGreater(self.engine._calculate_breath_score(continuous), 0.9)
        self.assertLess(self.engine._calculate_breath_score(broken), 0.7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
