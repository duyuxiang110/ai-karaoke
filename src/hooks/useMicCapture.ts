import { useState, useRef, useCallback } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'

// 希望拿到的采样率。只是个希望：AudioContext 构造参数里的 sampleRate
// 按规范就是「提示」，浏览器给不了会静默用自己的（常见 48000）
const REQUESTED_SAMPLE_RATE = 44100

interface UseMicCaptureReturn {
  /** 启动采集，返回 AudioContext 的**真实**采样率（后端跑 DIO 必须用它） */
  start: (onPCM: (buffer: ArrayBuffer) => void) => Promise<number>
  stop: () => void
  isRecording: boolean
  error: string | null
}

export function useMicCapture(): UseMicCaptureReturn {
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const start = useCallback(async (onPCM: (buffer: ArrayBuffer) => void) => {
    try {
      setError(null)

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: REQUESTED_SAMPLE_RATE,
        },
      })
      streamRef.current = stream

      // 一律以 ctx.sampleRate 为准，不假定请求生效了。采样率一旦对不上，
      // 后端拿 44100 去解 48k 的 PCM，所有频率被整体压到 0.919 倍
      // （-146.9 音分，差一个半音还多），音准分直接废掉 —— 而且从分数上
      // 完全看不出来，只会以为唱得差
      let ctx: AudioContext
      try {
        ctx = new AudioContext({ sampleRate: REQUESTED_SAMPLE_RATE })
      } catch {
        // 不支持指定采样率（部分 Safari / 旧内核），那就用系统给的
        ctx = new AudioContext()
      }
      audioCtxRef.current = ctx
      if (ctx.sampleRate !== REQUESTED_SAMPLE_RATE) {
        console.warn(
          `[useMicCapture] AudioContext 实际采样率 ${ctx.sampleRate}，` +
          `非请求的 ${REQUESTED_SAMPLE_RATE}；按实际值上报后端`
        )
      }
      if (ctx.state === 'suspended') {
        await ctx.resume()
      }

      await ctx.audioWorklet.addModule('./worklets/pcm-processor.js')

      const source = ctx.createMediaStreamSource(stream)
      const workletNode = new AudioWorkletNode(ctx, 'pcm-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
      })

      workletNode.port.onmessage = (e: MessageEvent) => {
        onPCM(e.data as ArrayBuffer)
      }

      source.connect(workletNode)
      workletNodeRef.current = workletNode

      useKaraokeStore.getState().setRecording(true)
      setIsRecording(true)
      return ctx.sampleRate
    } catch (err: any) {
      setError(err.message || '麦克风启动失败')
      console.error('[useMicCapture] Error:', err)
      // 没起成也得给个数：调用方拿它去建 WebSocket，不能返回 undefined
      return REQUESTED_SAMPLE_RATE
    }
  }, [])

  const stop = useCallback(() => {
    workletNodeRef.current?.disconnect()
    audioCtxRef.current?.close()
    streamRef.current?.getTracks().forEach((t) => t.stop())

    workletNodeRef.current = null
    audioCtxRef.current = null
    streamRef.current = null

    useKaraokeStore.getState().setRecording(false)
    setIsRecording(false)
  }, [])

  return { start, stop, isRecording, error }
}
