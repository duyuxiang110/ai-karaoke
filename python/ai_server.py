"""
AI 智能 K 歌打分系统 - Python AI 后端
FastAPI 服务入口，提供伴奏分离、音高检测、歌词对齐、打分等 API
"""
import argparse
import asyncio
import os
import time
import numpy as np
import pyworld as pw
import uvicorn
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from engine.score_engine import ScoreEngine
from engine.separator import separate_vocals
from engine.pitch_detector import (
    extract_f0,
    freq_to_note_cents,
    mask_to_singing_segments,
)
from engine.lyric_aligner import align as align_lyrics

app = FastAPI(title="AI Karaoke Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

score_engine = ScoreEngine()

# 缓存已处理的歌曲，避免重复分离
_separation_cache: dict = {}


# ─── 数据模型 ───

class SeparateRequest(BaseModel):
    file_path: str
    output_dir: Optional[str] = None


class SeparateResponse(BaseModel):
    vocal_path: str
    instrumental_path: str
    sample_rate: int
    duration: float


class BaselineResponse(BaseModel):
    pitches: list
    lyrics: list
    duration: float
    lyric_source: dict


class ScoreRequest(BaseModel):
    user_pitches: list
    baseline_pitches: list
    user_onsets: Optional[list] = None
    lyric_timestamps: Optional[list] = None


class ScoreResponse(BaseModel):
    total: float
    pitch: float
    rhythm: float
    breath: float


# ─── REST 接口 ───

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/audio")
async def serve_audio(path: str):
    """通过 HTTP 提供音频文件，支持 Range 请求（音频元素需要才能获取 duration 和 seek）"""
    import mimetypes
    media_type, _ = mimetypes.guess_type(path)
    if not media_type:
        ext = os.path.splitext(path)[1].lower()
        media_type = 'audio/mpeg' if ext == '.mp3' else 'audio/wav'
    return FileResponse(path, media_type=media_type)


@app.post("/api/separate", response_model=SeparateResponse)
async def separate_vocals_api(req: SeparateRequest):
    """
    伴奏分离 - Demucs (优先) 或 HPSS (fallback)
    """
    cache_key = req.file_path
    if cache_key in _separation_cache:
        cached = _separation_cache[cache_key]
        return SeparateResponse(**cached)

    # 在线程池中运行（分离是 CPU 密集型操作，设 5 分钟超时）
    try:
        vocal_path, instrumental_path, sr, duration = await asyncio.wait_for(
            asyncio.to_thread(separate_vocals, req.file_path, req.output_dir),
            timeout=300,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail='分离超时（超过5分钟），请尝试更短的音频文件')

    result = {
        'vocal_path': vocal_path,
        'instrumental_path': instrumental_path,
        'sample_rate': sr,
        'duration': duration,
    }
    _separation_cache[cache_key] = result
    return SeparateResponse(**result)


@app.get("/api/baseline", response_model=BaselineResponse)
async def get_baseline(file_path: str, original_path: str = None, lrc_path: str = None):
    """
    获取原唱音高基线 + 歌词时间轴
    file_path: 分离后的人声 WAV 路径
    original_path: 原始 MP3 路径（用于读取内嵌歌词/ID3 标签）
    lrc_path: 用户手动上传的 .lrc 路径，优先级最高
    """
    # 并行提取音高和歌词
    pitches, (lyrics, lyric_source) = await asyncio.gather(
        asyncio.to_thread(extract_f0, file_path),
        asyncio.to_thread(align_lyrics, file_path, original_path, lrc_path),
    )

    # 去掉前奏/间奏里器乐残留造成的假音高，基线只保留人声演唱段
    pitches = await asyncio.to_thread(mask_to_singing_segments, pitches, file_path)

    duration = max(
        [p['time'] for p in pitches] if pitches else [0.0]
    )

    return BaselineResponse(
        pitches=pitches,
        lyrics=lyrics,
        duration=duration,
        lyric_source=lyric_source,
    )


@app.post("/api/score", response_model=ScoreResponse)
async def calculate_score(req: ScoreRequest):
    """最终打分"""
    result = score_engine.calculate_final_score(
        user_pitches=req.user_pitches,
        baseline_pitches=req.baseline_pitches,
        user_onsets=req.user_onsets,
        lyric_timestamps=req.lyric_timestamps,
    )
    return ScoreResponse(**result)


# ─── WebSocket 实时音高流 ───

def _voiced_in_window(window, sr):
    """对窗口跑 DIO + StoneMask，返回浊音帧频率数组（供线程池调用）"""
    _f0, t_axis = pw.dio(window, sr, f0_floor=80, f0_ceil=600, frame_period=20)
    f0 = pw.stonemask(window, _f0, t_axis, sr)
    return f0[f0 > 0]


@app.websocket("/ws/pitch")
async def pitch_stream(websocket: WebSocket):
    """
    实时音高检测 WebSocket
    前端发送: 二进制 PCM 帧 (16-bit, 44100Hz, mono, little-endian)
    后端返回: JSON {frequency, note, cents, rms, timestamp}
    """
    await websocket.accept()

    # DIO 需要 ~8192 样本 (186ms) 才能可靠检测音高
    # 服务端累积 PCM 块，达到阈值后统一处理
    MIN_SAMPLES = 8192
    audio_buffer = np.array([], dtype=np.float64)
    frame_count = 0
    samples_per_sec = 44100

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()

            # Int16 → float64，追加到缓冲区
            chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64) / 32768.0
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # RMS（当前块）
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0

            # 累积不足时，仅返回 RMS
            if len(audio_buffer) < MIN_SAMPLES:
                timestamp = frame_count * len(chunk) / samples_per_sec
                await websocket.send_json({
                    'type': 'pitch',
                    'frequency': None,
                    'note': None,
                    'cents': None,
                    'timestamp': round(timestamp, 3),
                    'rms': round(rms, 4),
                })
                continue

            # 取最近 MIN_SAMPLES 样本做 DIO
            window = audio_buffer[-MIN_SAMPLES:]

            # 静音帧跳过 DIO（门限放低，软声/气声也要出红线）
            if rms < 0.008:
                # 保留 75% 重叠，静音结束后下一块即可出音高
                audio_buffer = audio_buffer[-(MIN_SAMPLES - len(chunk)):]
                timestamp = frame_count * len(chunk) / samples_per_sec
                await websocket.send_json({
                    'type': 'pitch',
                    'frequency': None,
                    'note': None,
                    'cents': None,
                    'timestamp': round(timestamp, 3),
                    'rms': round(rms, 4),
                })
                frame_count += 1
                continue

            # DIO + StoneMask 放线程里跑：同步执行会阻塞收包循环，
            # 音高消息到达变成突发，前端红线一跳一跳
            try:
                voiced = await asyncio.to_thread(
                    _voiced_in_window, window, samples_per_sec
                )
            except Exception:
                voiced = np.array([])

            timestamp = frame_count * len(chunk) / samples_per_sec

            if len(voiced) > 0:
                freq = float(np.median(voiced))
                note, cents = freq_to_note_cents(freq)
            else:
                freq = None
                note = None
                cents = None

            await websocket.send_json({
                'type': 'pitch',
                'frequency': freq,
                'note': note,
                'cents': round(cents, 1) if cents is not None else None,
                'timestamp': round(timestamp, 3),
                'rms': round(rms, 4),
            })

            # 保留 75% 重叠：下一块进来即凑满 MIN_SAMPLES，pitch 每块一条
            audio_buffer = audio_buffer[-(MIN_SAMPLES - len(chunk)):]
            frame_count += 1

    except WebSocketDisconnect:
        print('[WS] Client disconnected from pitch stream')
    except Exception as e:
        print(f'[WS] Error: {e}')


# ─── 启动 ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Karaoke Python Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print("Starting AI Karaoke server on {}:{}".format(args.host, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
