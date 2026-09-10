"""
AI 智能 K 歌打分系统 - Python AI 后端
FastAPI 服务入口，提供伴奏分离、音高检测、歌词对齐、打分等 API
"""
import argparse
import asyncio
import json
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

# 前端没上报采样率时的兜底值。只是兜底，不是假设：真实值必须
# 由前端从 AudioContext 读出来发过来（见 /ws/pitch）
DEFAULT_SAMPLE_RATE = 44100
# 采样率的合理区间：超出就当没收到，宁可沿用旧值也不要拿它去跑 DIO
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000


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
    # 歌词没有可用时间轴时为 null，表示节奏未参评。
    # 旧版把「音准+气息」的综合分塞进这一栏，界面上却写着「节奏」
    rhythm: Optional[float] = None
    breath: float
    # 演唱完成度（0~100）：总分 = 已唱内容质量 × 完成度。
    # 界面不展示，但少了它无法解释总分到底为何偏低。
    # 原唱基线为空时为 null —— 那是「无从评估」，不能拿 100 假装唱完整了
    completion: Optional[float] = None
    # 时间轴无效、或总分被完成度折算时的解释文案
    warning: Optional[str] = None


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


def _log_score_input(req, result, diagnostics=None):
    """把打分输入摘要写进日志：分数有争议时，光看输出反推不出输入形态

    实测过一次「音准 11.6 / 节奏 9.3 / 气息 92.4 / 总分 0.9」，
    四个数字本身分不清是算法错还是输入退化（时间戳没推进、只唱了两秒）。
    帧数与时间跨度一摆出来就能直接定位。

    音准诊断同理：「音准 40」既可能是唱得差，也可能是两侧 frequency
    压根不是一个定义（采样率对不上、单位不一致），光看分数永远分不出来。
    """
    try:
        times = [float(p['time']) for p in req.user_pitches
                 if isinstance(p, dict) and p.get('time') is not None]
        span = max(times) - min(times) if times else 0.0
    except (TypeError, ValueError):
        span = -1.0
    print(
        f'[Score] user_frames={len(req.user_pitches)} span={span:.2f}s '
        f'baseline_frames={len(req.baseline_pitches)} '
        f'lyrics={len(req.lyric_timestamps or [])} '
        f'onsets={len(req.user_onsets or [])} -> {result}',
        flush=True,
    )
    if diagnostics:
        print(f'[Score] 音准诊断 {_pitch_diag_text(diagnostics)}', flush=True)


def _pitch_diag_text(d):
    """把音准诊断拼成一行，并直接给出怎么读的结论

    比值是这里最关键的数：两侧走的是同一条旋律，所以它该是 1
    （或八度时的 2 / 0.5）。不是的话就说明频率被整体缩放过，
    那是链路问题而不是唱功问题 —— 44100/48000 = 0.9188，正好 -146.9 音分
    """
    base_hz = d.get('base_hz') or 0.0
    ratio = (d.get('user_hz') or 0.0) / base_hz if base_hz else 0.0
    line = (
        f'aligned={d.get("aligned")}/{d.get("frames")} '
        f'med_cents={d.get("med_cents")} '
        f'user_hz={d.get("user_hz")} base_hz={base_hz} ratio={ratio:.4f}'
    )
    for label, want in (('同调', 1.0), ('整体高八度', 2.0), ('整体低八度', 0.5)):
        if abs(ratio - want) < 0.02:
            return f'{line} [{label}]'
    return f'{line} [两侧频率不是同一个定义，先查采样率，别先怀疑评分公式]'


@app.post("/api/score", response_model=ScoreResponse)
async def calculate_score(req: ScoreRequest):
    """最终打分"""
    result = score_engine.calculate_final_score(
        user_pitches=req.user_pitches,
        baseline_pitches=req.baseline_pitches,
        user_onsets=req.user_onsets,
        lyric_timestamps=req.lyric_timestamps,
    )
    # 诊断只写日志、不进响应：它是解释分数的材料，不是分数本身
    diagnostics = result.pop('diagnostics', None)
    _log_score_input(req, result, diagnostics)
    return ScoreResponse(**result)


# ─── WebSocket 实时音高流 ───

def _voiced_in_window(window, sr):
    """对窗口跑 DIO + StoneMask，返回浊音帧频率数组（供线程池调用）

    frame_period 必须 ≤10：186ms 窗 + frame_period=20 时 DIO 边界效应
    让 10 帧里只有 1 帧浊音，实测整条 WebSocket 一帧音高都回不去
    """
    _f0, t_axis = pw.dio(window, sr, f0_floor=80, f0_ceil=600, frame_period=10)
    f0 = pw.stonemask(window, _f0, t_axis, sr)
    return f0[f0 > 0]


def _apply_config_frame(payload, current):
    """读前端的配置帧，返回生效的采样率

    任何不对劲都沿用旧值并打日志，不抛异常：采样率只影响音高数值，
    没必要为了一条格式错的文本帧把整条音高流断开。
    """
    try:
        msg = json.loads(payload)
    except (ValueError, TypeError):
        print(f'[WS] 配置帧不是合法 JSON，采样率仍按 {current}: {payload[:120]!r}')
        return current
    if not isinstance(msg, dict) or msg.get('type') != 'config':
        return current

    sr = msg.get('sampleRate')
    # bool 是 int 的子类，true 会变成 1Hz，先排掉
    if isinstance(sr, bool) or not isinstance(sr, (int, float)):
        print(f'[WS] 配置帧里没有可用采样率，仍按 {current}: {msg}')
        return current
    sr = int(sr)
    if not MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE:
        print(f'[WS] 采样率 {sr} 超出合理区间，仍按 {current} 处理')
        return current
    if sr != current:
        print(f'[WS] 前端上报采样率 {sr}（原按 {current} 处理）')
    return sr


@app.websocket("/ws/pitch")
async def pitch_stream(websocket: WebSocket):
    """
    实时音高检测 WebSocket
    前端发送: 连上后先发一条 JSON 文本帧 {type:'config', sampleRate:int}，
              之后是二进制 PCM 帧 (16-bit, mono, little-endian)
    后端返回: JSON {frequency, note, cents, rms, timestamp}

    配置帧故意不回 ack：前端靠「发一块 PCM 换一条回包」的 FIFO 对账采集
    时刻，多回一条消息就会把整条时间轴错位。
    """
    await websocket.accept()

    # DIO 需要 ~8192 样本 (186ms @44100Hz) 才能可靠检测音高
    # 服务端累积 PCM 块，达到阈值后统一处理
    MIN_SAMPLES = 8192
    audio_buffer = np.array([], dtype=np.float64)
    frame_count = 0
    # 采样率必须由前端上报：AudioContext({sampleRate:44100}) 只是「提示」，
    # 浏览器给不了就静默用自己的（常见 48000）。DIO 拿错的采样率去解，
    # 所有频率会被整体缩放（48k 当 44.1k 解 = ×0.919 = 低 146.9 音分，
    # 差一个半音还多），音准分直接废掉，而且从分数上完全看不出是链路问题
    samples_per_sec = DEFAULT_SAMPLE_RATE

    try:
        while True:
            message = await websocket.receive()
            if message['type'] == 'websocket.disconnect':
                break

            # 文本帧 = 配置；二进制帧 = PCM。用 receive_bytes() 的话文本帧
            # 会直接报错断连，采样率就永远只能靠猜
            text = message.get('text')
            if text is not None:
                samples_per_sec = _apply_config_frame(text, samples_per_sec)
                continue

            pcm_bytes = message.get('bytes')
            if not pcm_bytes:
                continue

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

            # 静音帧跳过 DIO：用整窗 RMS 而非当前块
            # 当前块(2048样本=46ms)刚出声时 RMS 就过门限，
            # 但 8192 窗里大部分还是静音，DIO 会在噪声上产生假音高
            # 门限不能太高：轻唱/气声的整窗 RMS 偏低，太高会把真唱当静音
            window_rms = float(np.sqrt(np.mean(window ** 2))) if len(window) > 0 else 0.0
            if window_rms < 0.0015:
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

            # 整窗 RMS 门已挡住静音窗；噪声窗实测 0/19 浊音帧不会出假音高，
            # 不再加占比门——frame_period=20 时代 10 帧仅 1 帧浊音，占比门会把真唱全滤掉
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
