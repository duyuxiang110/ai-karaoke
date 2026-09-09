"""
音高检测引擎
- extract_f0(): 从完整音频提取 F0 基线 (pyworld DIO + StoneMask)
- detect_realtime(): 实时处理 PCM 块，返回当前音高
- freq_to_note_cents(): 频率 → 音名 + 音分偏差
"""
import numpy as np
import pyworld as pw

from engine.vocal_segments import singing_segments

SAMPLE_RATE = 44100
F0_FLOOR = 80.0   # 最低检测频率 (E2)
F0_CEIL = 600.0   # 最高检测频率 (D5+)


def extract_f0(audio_path):
    """
    从完整音频文件提取 F0 基线
    返回: [{time, frequency, note, cents}]
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float64)

    # DIO: 快速 F0 估计
    _f0, t = pw.dio(y, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=10)
    # StoneMask: 精细化 F0
    f0 = pw.stonemask(y, _f0, t, sr)

    pitches = []
    for i, freq in enumerate(f0):
        note, cents = freq_to_note_cents(freq)
        pitches.append({
            'time': float(t[i]),
            'frequency': float(freq) if freq > 0 else 0.0,
            'note': note,
            'cents': cents,
        })

    voiced_count = sum(1 for p in pitches if p['frequency'] > 0)
    print(f'[PitchDetector] Extracted {len(pitches)} frames, {voiced_count} voiced')
    return pitches


def mask_to_singing_segments(pitches, vocal_path):
    """
    把基线音高限制在演唱段内

    DIO 只判周期性、不判是不是人声，demucs 人声轨里残留的鼓/吉他谐波
    也会被算成音高，导致歌手没唱的地方仍画出基线。演唱段外的帧置为未发声。
    """
    segments = singing_segments(vocal_path)
    if not segments:
        return pitches

    masked = 0
    for p in pitches:
        t = p['time']
        if not any(start <= t <= end for start, end in segments):
            p['frequency'] = 0.0
            p['note'] = None
            p['cents'] = None
            masked += 1

    print(f'[PitchDetector] Masked {masked} non-singing frames out of {len(pitches)}')
    return pitches


def detect_realtime(pcm_bytes, sample_rate=SAMPLE_RATE):
    """
    实时处理一块 PCM 数据，返回当前音高
    pcm_bytes: Int16 PCM 小端序二进制数据
    返回: {frequency, note, cents, rms}
    """
    # Int16 → float64
    int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio = int16.astype(np.float64) / 32768.0

    # RMS: 能量检测
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    # 静音帧跳过
    if rms < 0.015:
        return {'frequency': None, 'note': None, 'cents': None, 'rms': rms}

    # DIO + StoneMask
    try:
        _f0, t = pw.dio(
            audio, sample_rate,
            f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
            frame_period=10,
        )
        f0 = pw.stonemask(audio, _f0, t, sample_rate)
    except Exception:
        return {'frequency': None, 'note': None, 'cents': None, 'rms': rms}

    # 取 voiced 帧的中位数 F0
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return {'frequency': None, 'note': None, 'cents': None, 'rms': rms}

    freq = float(np.median(voiced))
    note, cents = freq_to_note_cents(freq)

    return {'frequency': freq, 'note': note, 'cents': cents, 'rms': rms}


def freq_to_note_cents(freq):
    """
    频率 → 音名 + 音分偏差
    A4 = 440Hz = MIDI 69
    """
    if freq is None or freq <= 0:
        return None, None

    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_rounded = round(midi)

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_rounded // 12 - 1
    note_name = note_names[midi_rounded % 12] + str(octave)

    cents = (midi - midi_rounded) * 100

    return note_name, float(cents)
