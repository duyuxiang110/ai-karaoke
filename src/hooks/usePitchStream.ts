import { useState, useRef, useCallback, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { usePlaybackClock } from '@/hooks/usePlaybackClock'
import type { PitchResult } from '@/types'

// 后端收包循环里三个分支（缓冲不足 / 静音窗 / 正常）都各自 send_json 一次，
// 每个 PCM 块必定换来一条回包，所以「发送」与「回包」严格 1:1。
// 正常管道深度只有 4~8 帧（8192 样本窗 + DIO 计算 + 网络），
// 队列涨到这个量说明有请求没等到回包，只能丢掉最旧的重新对齐
const MAX_PENDING = 64

// 后端对第 N 块跑的是「最近 WINDOW_SAMPLES 个样本」的窗（处理完只丢当前块、
// 保留 75% 重叠），即窗尾正好是第 N 块末尾。报回的频率是整个窗浊音帧的
// 中位数，代表的是窗中心而不是窗尾，所以采集时刻还要再往前推半个窗。
// 不补的话用户音高会系统性地晚半个窗对到基线上，基线音符几百毫秒一换，
// 足以把音符边界附近的帧全部判成跑调。
//
// 8192 是「样本数」，换算成秒必须用真实采样率：实际是 48000 而按 44100
// 折算，这里会多推 7.6ms —— 比采样率本身造成的 -146.9 音分小得多，
// 但同样是白送的误差
const WINDOW_SAMPLES = 8192
const FALLBACK_SAMPLE_RATE = 44100

interface UsePitchStreamReturn {
  /** 建连。sampleRate 为 AudioContext 的真实采样率，缺省时沿用上一次的值 */
  connect: (sampleRate?: number) => void
  disconnect: () => void
  sendPCM: (buffer: ArrayBuffer) => void
  isConnected: boolean
}

export function usePitchStream(baseUrl: string): UsePitchStreamReturn {
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const getTime = usePlaybackClock()

  const getTimeRef = useRef(getTime)
  getTimeRef.current = getTime

  // 采样率由 useMicCapture 从 AudioContext 读出来传进 connect()，
  // 再随配置帧交给后端；重连时沿用上一次的，不用再问麦克风要
  const sampleRateRef = useRef(FALLBACK_SAMPLE_RATE)
  const windowHalfSecRef = useRef(WINDOW_SAMPLES / FALLBACK_SAMPLE_RATE / 2)

  // 采集时刻队列：在 sendPCM 里记下这一帧声音对应的播放位置，
  // 回包时按 FIFO 取回。等到回包才取时钟的话，时间戳会系统性滞后
  // 整条管道延迟（186ms 缓冲 + DIO 计算 + 往返），基线音符 0.5s 一换，
  // 0.2~0.3s 的偏移足以把最近邻匹配推到下一个音上，音准直接崩掉
  const pendingTimesRef = useRef<number[]>([])

  const shouldReconnectRef = useRef(false)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptsRef = useRef(0)

  const connect = useCallback((sampleRate?: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return

    if (sampleRate && sampleRate > 0) {
      sampleRateRef.current = sampleRate
      windowHalfSecRef.current = WINDOW_SAMPLES / sampleRate / 2
    }

    shouldReconnectRef.current = true
    // 新连接上没有历史回包要对账，旧队列留着只会整体错位
    pendingTimesRef.current = []

    const wsUrl = baseUrl.replace('http', 'ws') + '/ws/pitch'
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0
      setIsConnected(true)
      // 发 PCM 之前先把真实采样率告诉后端：后端要用它跑 DIO，猜错的话
      // 所有频率会被整体缩放（48k 当 44.1k 解 = 低 146.9 音分）。
      // 后端不会 ack 这一条：它靠「一块 PCM 换一条回包」对账采集时刻，
      // 多一条回包就会把整条时间轴错位
      ws.send(JSON.stringify({
        type: 'config',
        sampleRate: sampleRateRef.current,
      }))
    }

    ws.onclose = () => {
      setIsConnected(false)
      if (shouldReconnectRef.current) {
        const delay = Math.min(500 * Math.pow(2, reconnectAttemptsRef.current), 4000)
        reconnectAttemptsRef.current++
        if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null
          if (shouldReconnectRef.current) connect()
        }, delay)
      }
    }

    ws.onerror = () => {
      setIsConnected(false)
    }

    ws.onmessage = (e: MessageEvent) => {
      // 每条回包（含静音帧）都消费一个采集时刻，FIFO 才不会错位
      const capturedAt = pendingTimesRef.current.shift()
      try {
        const data: PitchResult = JSON.parse(e.data)
        const store = useKaraokeStore.getState()

        store.setLivePitch(data)

        if (data.frequency != null && data.frequency > 0 && !store.isSeeking) {
          store.addUserPitch({
            time: capturedAt ?? getTimeRef.current(),
            frequency: data.frequency,
          })
        }
      } catch (err) {
        console.error('[usePitchStream] Parse error:', err)
      }
    }

    wsRef.current = ws
  }, [baseUrl])

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false
    reconnectAttemptsRef.current = 0
    pendingTimesRef.current = []

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    wsRef.current?.close()
    wsRef.current = null
    setIsConnected(false)
  }, [])

  const sendPCM = useCallback((buffer: ArrayBuffer) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return
    const pending = pendingTimesRef.current
    if (pending.length >= MAX_PENDING) {
      pending.splice(0, pending.length - MAX_PENDING + 1)
    }
    // 记下的是这一窗声音真正发生的时刻，不是回包时刻
    pending.push(Math.max(0, getTimeRef.current() - windowHalfSecRef.current))
    wsRef.current.send(buffer)
  }, [])

  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false
      pendingTimesRef.current = []
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [])

  return { connect, disconnect, sendPCM, isConnected }
}
