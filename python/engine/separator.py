"""
伴奏分离引擎
优先使用 Demucs (htdemucs) 高质量分离，未安装或失败时 fallback 到 librosa HPSS + 中置声道去除
"""
import hashlib
import os
import re
import sys
import numpy as np
import librosa
import soundfile as sf

try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# 打包态由 electron 注入 KARAOKE_OUTPUT_DIR 指向可写的用户数据目录，
# 避免往只读的 app Resources 里写分离产物；dev 不设置时保持仓库内默认目录
OUTPUT_DIR = os.environ.get('KARAOKE_OUTPUT_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'output'
)

_UNSAFE_RE = re.compile(r'[^\w\u4e00-\u9fff.-]+')


def _output_paths(file_path, output_dir):
    """
    每首歌用独立的输出文件名

    固定名 vocal.wav 会被下一首歌覆盖，而服务端按输入路径缓存分离结果，
    切回旧歌时会命中缓存却读到新歌的音频，音高基线和歌词时间轴都会错
    """
    abs_path = os.path.abspath(file_path)
    try:
        stat = os.stat(abs_path)
        identity = f'{abs_path}|{stat.st_size}|{int(stat.st_mtime)}'
    except OSError:
        identity = abs_path

    digest = hashlib.md5(identity.encode('utf-8')).hexdigest()[:12]
    stem = _UNSAFE_RE.sub('_', os.path.splitext(os.path.basename(abs_path))[0])[:40]
    base = f'{stem}_{digest}'
    return (
        os.path.join(output_dir, base + '.vocals.wav'),
        os.path.join(output_dir, base + '.instrumental.wav'),
    )


def _write_wav(path, data, sr):
    """先写临时文件再原子替换，避免中断留下半截 wav 被当成缓存命中"""
    tmp = path + '.part'
    # 临时名不是 .wav 结尾，必须显式指定格式，否则 soundfile 无法推断
    sf.write(tmp, data, sr, format='WAV')
    os.replace(tmp, path)


def separate_vocals(file_path, output_dir=None):
    """
    分离人声和伴奏
    返回: (vocal_path, instrumental_path, sample_rate, duration)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    vocal_path, instrumental_path = _output_paths(file_path, output_dir)

    if os.path.exists(vocal_path) and os.path.exists(instrumental_path):
        info = sf.info(vocal_path)
        print(f'[Separator] Reusing separation: {os.path.basename(vocal_path)}', flush=True)
        return vocal_path, instrumental_path, info.samplerate, info.frames / info.samplerate

    if DEMUCS_AVAILABLE:
        try:
            return _separate_with_demucs(file_path, vocal_path, instrumental_path)
        except Exception as e:
            print(f'[Separator] Demucs failed: {e}', file=sys.stderr, flush=True)
            print('[Separator] Falling back to HPSS...', flush=True)

    return _separate_with_hpss(file_path, vocal_path, instrumental_path)


def _separate_with_demucs(file_path, vocal_path, instrumental_path):
    """使用 Demucs htdemucs 模型分离"""
    print('[Separator] Loading Demucs htdemucs model...', flush=True)

    model = get_model('htdemucs')
    model.eval()

    # CPU 更稳定，MPS 在 Demucs 上有已知的挂起问题
    device = 'cpu'
    model = model.to(device)
    sr = model.samplerate
    print(f'[Separator] Model loaded. Sample rate: {sr}', flush=True)

    print(f'[Separator] Loading audio: {file_path}', flush=True)
    wav, sr_orig = torchaudio.load(file_path)
    if sr_orig != sr:
        wav = torchaudio.functional.resample(wav, sr_orig, sr)
    print(f'[Separator] Audio loaded. Shape: {wav.shape}, duration: {wav.shape[1]/sr:.1f}s', flush=True)

    # Demucs 期望 (batch, channels, samples)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.abs().max()
    wav = wav.unsqueeze(0).to(device)

    print('[Separator] Running Demucs separation (this may take a while)...', flush=True)
    with torch.no_grad():
        # apply_model 返回 (batch, num_sources, channels, length)
        # [0] 去掉 batch 维 → (num_sources=4, channels, length)
        sources = apply_model(model, wav, split=True, overlap=0.25)[0]

    sources = sources * ref.abs().max() + ref.mean()
    print('[Separator] Separation complete, extracting stems...', flush=True)

    # htdemucs stems: [drums, bass, other, vocals]
    stems = ['drums', 'bass', 'other', 'vocals']
    vocal_idx = stems.index('vocals')          # 3
    instrumental_idx = [i for i in range(len(stems)) if i != vocal_idx]  # [0, 1, 2]

    # sources shape: (4, channels, length)
    # 正确索引: sources[vocal_idx] → (channels, length)
    vocal = sources[vocal_idx].cpu().numpy()
    instrumental = sources[instrumental_idx].sum(0).cpu().numpy()

    # 转为 mono
    if vocal.ndim > 1:
        vocal = vocal.mean(0)
    if instrumental.ndim > 1:
        instrumental = instrumental.mean(0)

    _write_wav(vocal_path, vocal, sr)
    _write_wav(instrumental_path, instrumental, sr)

    duration = len(vocal) / sr
    print(f'[Separator] Demucs done. Duration: {duration:.1f}s', flush=True)
    return vocal_path, instrumental_path, sr, duration


def _separate_with_hpss(file_path, vocal_path, instrumental_path):
    """
    Fallback: librosa HPSS + 中置声道去除
    立体声歌曲中，人声通常在中央 (L≈R)，用 L-R 可去除人声得到伴奏
    """
    print('[Separator] Demucs not available, using HPSS + center removal...')

    y, sr = librosa.load(file_path, sr=44100, mono=False)

    if y.ndim == 1:
        # Mono: HPSS 分离谐波/打击
        harmonic, percussive = librosa.effects.hpss(y)
        vocal = harmonic
        instrumental = y - harmonic  # 原曲减去谐波部分
    else:
        # Stereo: 中置声道去除 (L-R) 得到无伴奏，HPSS 得到人声
        # 人声提取: HPSS 谐波分量
        y_mono = librosa.to_mono(y)
        harmonic, _ = librosa.effects.hpss(y_mono)
        vocal = harmonic

        # 伴奏: 原曲 - 人声（近似）
        # 更好的方法: 中置去除
        minus = (y[0] - y[1]) / 2
        instrumental = np.stack([minus, -minus], axis=0)
        # 转 mono
        instrumental = librosa.to_mono(instrumental)

    _write_wav(vocal_path, vocal, sr)
    _write_wav(instrumental_path, instrumental, sr)

    duration = len(vocal) / sr
    print(f'[Separator] HPSS done. Duration: {duration:.1f}s')
    return vocal_path, instrumental_path, sr, duration
