"""
人声演唱段检测

demucs 分离出的人声轨在纯器乐段仍会残留乐器能量（鼓、吉他谐波等），
仅靠「有没有周期性」会把乐器谐波也当成音高。这里用 DIO 浊音占比
把「有声段」进一步区分为「演唱段」与「器乐残留段」，供歌词铺排和
基线音高掩码共用。
"""
import librosa
import numpy as np
import pyworld as pw

# 合并间隔小于该值的有声段，避免换气被切成两段；
# 再大就会把真实的短语边界也合并掉
MERGE_GAP = 0.35
MIN_SEGMENT = 0.3

RMS_FRAME = 1024
RMS_HOP = 256
# 有声阈值取绝对 dBFS：以噪声底（第 20 百分位）为基准上抬若干 dB。
# 用相对峰值的阈值会落进歌声区间，把短语切碎，
# 且录音电平整体偏低的歌曲会一段都检测不到
NOISE_FLOOR_PERCENTILE = 20
NOISE_MARGIN_DB = 9.0
# 有声段里浊音帧占比低于该值时视为器乐残留（前奏/间奏）而非演唱。
# 实测前奏段约 0.52 且 f0 中位偏高（乐器谐波），真唱段 0.74~0.89
VOICED_RATIO_MIN = 0.65


def vocal_segments(vocal_path):
    """检测人声轨的有声段落，返回 ([(start, end)], duration)"""
    y, sr = librosa.load(vocal_path, sr=22050, mono=True)
    return segments_from_audio(y, sr), len(y) / sr


def segments_from_audio(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=RMS_FRAME, hop_length=RMS_HOP)[0]
    if rms.size == 0:
        return []

    # 绝对 dBFS；数字静音会得到 -inf，算百分位前必须滤掉
    db = librosa.amplitude_to_db(rms, ref=1.0)
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        return []

    floor = float(np.percentile(finite, NOISE_FLOOR_PERCENTILE))
    threshold = floor + NOISE_MARGIN_DB

    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=RMS_HOP)
    segments = segments_above(db, times, threshold)

    if not segments:
        # 噪声底估计时素材异常，退到中位数电平再试一次
        threshold = float(np.percentile(finite, 50))
        segments = segments_above(db, times, threshold)

    return segments


def segments_above(db, times, threshold):
    """把响度包络中高于阈值的连续区间合并成有声段落"""
    raw = []
    start = None
    for i, active in enumerate(db > threshold):
        if active and start is None:
            start = times[i]
        elif not active and start is not None:
            raw.append((start, times[i]))
            start = None
    if start is not None:
        raw.append((start, float(times[-1])))

    segments = []
    for seg_start, seg_end in raw:
        if segments and seg_start - segments[-1][1] <= MERGE_GAP:
            segments[-1] = (segments[-1][0], seg_end)
        else:
            segments.append((seg_start, seg_end))

    return [(s, e) for s, e in segments if e - s >= MIN_SEGMENT]


def voiced_ratio(y, sr, segment):
    """一段音频里 DIO 判为浊音的帧占比，真唱高、纯器乐低"""
    start, end = segment
    x = y[int(start * sr):int(end * sr)].astype(np.float64)
    if x.size < int(sr * 0.3):
        return 0.0
    f0, time_axis = pw.dio(x, sr, f0_floor=80, f0_ceil=600, frame_period=10)
    f0 = pw.stonemask(x, f0, time_axis, sr)
    voiced = f0[f0 > 0]
    return len(voiced) / len(f0) if len(f0) else 0.0


def singing_segments(vocal_path):
    """
    在有声段落里剔除非演唱段（前奏/间奏的伴奏残留）

    若不过滤，歌词第一行会被铺到前奏上整体错位一行，
    基线音高也会把乐器谐波画成「原唱在唱」
    """
    y, sr = librosa.load(vocal_path, sr=22050, mono=True)
    segments = segments_from_audio(y, sr)

    kept = [s for s in segments if voiced_ratio(y, sr, s) >= VOICED_RATIO_MIN]
    if not kept:
        print('[VocalSegments] Voiced filter removed all segments, keeping raw')
        return segments

    dropped = len(segments) - len(kept)
    if dropped:
        print(f'[VocalSegments] Dropped {dropped} non-singing segment(s)')
    return kept
