import { useState, useRef, useCallback } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'

interface UseMicCaptureReturn {
  start: (onPCM: (buffer: ArrayBuffer) => void) => Promise<void>
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
          sampleRate: 44100,
        },
      })
      streamRef.current = stream

      const ctx = new AudioContext({ sampleRate: 44100 })
      audioCtxRef.current = ctx
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
    } catch (err: any) {
      setError(err.message || '麦克风启动失败')
      console.error('[useMicCapture] Error:', err)
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
