"""
伴奏分离引擎
统一使用 Demucs (htdemucs) 四轨分离：vocals 作为人声，drums+bass+other 合成伴奏。
不再提供 HPSS 等自动 fallback —— 环境缺失时直接报错，
保证所有机器对同一首歌得到完全一致的分离结果。
"""
import hashlib
import os
import re
import sys
import numpy as np
import soundfile as sf

# Demucs 是本项目唯一的分离引擎，直接硬依赖：缺失就让导入失败，绝不静默换算法
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

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
        # st_mtime_ns 精确到纳秒，避免同一秒内替换文件时错误复用旧分离产物
        identity = f'{abs_path}|{stat.st_size}|{stat.st_mtime_ns}'
    except OSError:
        identity = abs_path

    digest = hashlib.md5(identity.encode('utf-8')).hexdigest()[:12]
    stem = _UNSAFE_RE.sub('_', os.path.splitext(os.path.basename(abs_path))[0])[:40]
    base = f'{stem}_{digest}'
    return (
        os.path.join(output_dir, base + '.vocals.mp3'),
        os.path.join(output_dir, base + '.instrumental.mp3'),
    )


def _write_mp3(path, data, sr):
    """先写临时文件再原子替换，用 MP3 格式输出（兼容所有浏览器/平台）"""
    import lameenc
    # clip 到 [-1,1] 再转 int16
    data = np.clip(data, -1.0, 1.0)
    int16 = (data * 32767).astype(np.int16)

    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)

    mp3 = enc.encode(int16.tobytes())
    mp3 += enc.flush()

    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        f.write(mp3)
    os.replace(tmp, path)


def check_demucs():
    """
    启动时校验 Demucs 环境，并打印 torch/torchaudio 版本，
    便于排查「同一首歌在不同机器上分离结果不一致」的问题。
    可用返回 True，不可用返回 False（调用方据此抛出明确错误）。
    """
    try:
        print('[Separator] Demucs available', flush=True)
        print(f'[Separator] Torch: {torch.__version__}', flush=True)
        print(f'[Separator] Torchaudio: {torchaudio.__version__}', flush=True)
        return True
    except Exception as e:
        print(
            f'[Separator] Demucs unavailable: {e}', file=sys.stderr, flush=True)
        return False


def separate_vocals(file_path, output_dir=None):
    """
    分离人声和伴奏（仅 Demucs，无 fallback）
    返回: (vocal_path, instrumental_path, sample_rate, duration)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    vocal_path, instrumental_path = _output_paths(file_path, output_dir)

    if os.path.exists(vocal_path) and os.path.exists(instrumental_path):
        info = sf.info(vocal_path)
        print(
            f'[Separator] Reusing Demucs separation: {os.path.basename(vocal_path)}', flush=True)
        return vocal_path, instrumental_path, info.samplerate, info.frames / info.samplerate

    return _separate_with_demucs(file_path, vocal_path, instrumental_path)


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

    _write_mp3(vocal_path, vocal, sr)
    _write_mp3(instrumental_path, instrumental, sr)

    duration = len(vocal) / sr
    print(f'[Separator] Demucs done. Duration: {duration:.1f}s', flush=True)
    return vocal_path, instrumental_path, sr, duration


# 启动即校验 Demucs：缺失就直接抛明确错误，绝不静默退回 HPSS 之类的其它算法，
# 从根源消除跨机器分离结果不一致（伴奏里混进人声 / 人声里没有 vocals）的问题
if not check_demucs():
    raise RuntimeError(
        'Demucs is unavailable. '
        'Please install the required Demucs environment.'
    )
