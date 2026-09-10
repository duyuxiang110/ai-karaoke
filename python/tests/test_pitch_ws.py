"""
实时音高 WebSocket 的采样率协商测试

跑法:
    cd python && venv/bin/python -m unittest tests.test_pitch_ws -v

不走 starlette 的 TestClient：两个运行时都没装 httpx，为一个测试往生产
依赖里塞东西不划算。改用一个最小的假 WebSocket 直接驱动真实的
pitch_stream 协程 —— 收包分支、DIO、回包全都是生产代码。

盯的是一件事：AudioContext({sampleRate: 44100}) 按规范只是「提示」，
浏览器给不了就静默用 48000。后端要是照 44100 去解，所有频率会被整体
缩放到 0.919 倍（-146.9 音分，差一个半音还多），音准分直接废掉，
而且从分数上完全看不出是链路问题。
"""
import asyncio
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_server import (  # noqa: E402
    DEFAULT_SAMPLE_RATE, _apply_config_frame, pitch_stream,
)

TONE_HZ = 440.0
TRUE_SR = 48000      # 浏览器实际给的采样率
BLOCK = 2048         # pcm-processor.js 的块长
WINDOW = 8192        # 后端跑 DIO 的窗长


def tone_blocks(freq=TONE_HZ, sr=TRUE_SR, samples=WINDOW, amplitude=0.3):
    """把一段正弦切成前端会发的那些 PCM 块"""
    t = np.arange(samples) / sr
    wave = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return [wave[i:i + BLOCK].tobytes() for i in range(0, samples, BLOCK)]


def text_frame(payload):
    return {'type': 'websocket.receive', 'text': json.dumps(payload)}


def binary_frame(data):
    return {'type': 'websocket.receive', 'bytes': data}


class FakeWebSocket:
    """只实现 pitch_stream 用到的那三个方法"""

    def __init__(self, messages):
        self._queue = list(messages)
        self.sent = []

    async def accept(self):
        pass

    async def receive(self):
        if not self._queue:
            return {'type': 'websocket.disconnect'}
        return self._queue.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)


def run_ws(messages):
    ws = FakeWebSocket(messages)
    asyncio.run(pitch_stream(ws))
    return ws.sent


class SampleRateNegotiationTest(unittest.TestCase):

    def test_上报真实采样率后_440的音就该报440(self):
        blocks = tone_blocks()
        sent = run_ws([text_frame({'type': 'config', 'sampleRate': TRUE_SR})]
                      + [binary_frame(b) for b in blocks])

        # 配置帧不能换来回包：前端靠「一块 PCM 换一条回包」的 FIFO 对账
        # 采集时刻，多一条就把整条时间轴错位了
        self.assertEqual(len(blocks), len(sent), msg='配置帧破坏了 1:1 对账')

        freq = sent[-1]['frequency']
        self.assertIsNotNone(freq, msg=f'整窗都没检出音高: {sent[-1]}')
        self.assertAlmostEqual(
            TONE_HZ, freq, delta=5.0,
            msg=f'48k 的音频没按 48k 解: {freq}')

    def test_不上报采样率_频率就会整体偏低(self):
        # 这条钉住的不是「期望行为」，而是不修采样率的代价：
        # 后端猜 44100 去解 48k 的 PCM，440Hz 被报成 404Hz。
        # 少了它，将来谁把配置帧删了都不会有测试报警
        blocks = tone_blocks()
        sent = run_ws([binary_frame(b) for b in blocks])

        freq = sent[-1]['frequency']
        self.assertIsNotNone(freq)
        self.assertAlmostEqual(
            TONE_HZ * DEFAULT_SAMPLE_RATE / TRUE_SR, freq, delta=5.0)
        # 1200*log2(404.25/440) = -146.9 音分，比「0 分线」150 只差一点：
        # 一个采样率错位就能把唱得完全准的人判成音准几乎全错
        self.assertLess(freq, TONE_HZ - 20.0)

    def test_配置帧不对劲时沿用旧值而不是断流(self):
        # 采样率只影响音高数值，不值得为一条格式错的文本帧把整条音高流断开
        broken = [
            'not json',
            json.dumps({'type': 'other', 'sampleRate': TRUE_SR}),
            json.dumps({'type': 'config'}),
            # bool 是 int 的子类，不排掉的话 true 会变成 1Hz
            json.dumps({'type': 'config', 'sampleRate': True}),
            json.dumps({'type': 'config', 'sampleRate': '48000'}),
            json.dumps({'type': 'config', 'sampleRate': 0}),
            json.dumps({'type': 'config', 'sampleRate': 10 ** 7}),
        ]
        for payload in broken:
            with self.subTest(payload=payload):
                self.assertEqual(
                    DEFAULT_SAMPLE_RATE,
                    _apply_config_frame(payload, DEFAULT_SAMPLE_RATE))

    def test_合法配置帧生效(self):
        config = {'type': 'config', 'sampleRate': TRUE_SR}
        self.assertEqual(TRUE_SR, _apply_config_frame(json.dumps(config),
                                                     DEFAULT_SAMPLE_RATE))
        # 浮点也认，取整
        self.assertEqual(TRUE_SR, _apply_config_frame(
            json.dumps({'type': 'config', 'sampleRate': float(TRUE_SR)}),
            DEFAULT_SAMPLE_RATE))
        # 与当前值相同时原样返回
        self.assertEqual(DEFAULT_SAMPLE_RATE, _apply_config_frame(
            json.dumps({'type': 'config', 'sampleRate': DEFAULT_SAMPLE_RATE}),
            DEFAULT_SAMPLE_RATE))


if __name__ == '__main__':
    unittest.main()
