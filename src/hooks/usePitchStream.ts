import { useState, useRef, useCallback, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { usePlaybackClock } from '@/hooks/usePlaybackClock'
import type { PitchResult } from '@/types'

// 后端收包循环里三个分支（缓冲不足 / 静音窗 / 正常）都各自 send_json 一次，
// 每个 PCM 块必定换来一条回包，所以「发送」与「回包」严格 1:1。
// 正常管道深度只有 4~8 帧（8192 样本窗 + DIO 计算 + 网络），
// 队列涨到这个量说明有请求没等到回包，只能丢掉最旧的重新对齐
const MAX_PENDING = 64

// 后端对第 N 块跑的是「最近 8192 样本」窗（处理完只丢当前块、保留 6144 样本重叠），
// 即窗尾正好是第 N 块末尾、窗长 185.7ms。报回的频率是整个窗浊音帧的中位数，
// 代表的是窗中心而不是窗尾，所以采集时刻还要再往前推半个窗。
// 不补的话用户音高会系统性地晚 93ms 对到基线上，基线音符几百毫秒一换，
// 足以把音符边界附近的帧全部判成跑调
const WINDOW_HALF_SEC = 8192 / 44100 / 2

interface UsePitchStreamReturn {
  connect: () => void
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

  // 采集时刻队列：在 sendPCM 里记下这一帧声音对应的播放位置，
  // 回包时按 FIFO 取回。等到回包才取时钟的话，时间戳会系统性滞后
  // 整条管道延迟（186ms 缓冲 + DIO 计算 + 往返），基线音符 0.5s 一换，
  // 0.2~0.3s 的偏移足以把最近邻匹配推到下一个音上，音准直接崩掉
  const pendingTimesRef = useRef<number[]>([])

  const shouldReconnectRef = useRef(false)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptsRef = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return

    shouldReconnectRef.current = true
    // 新连接上没有历史回包要对账，旧队列留着只会整体错位
    pendingTimesRef.current = []

    const wsUrl = baseUrl.replace('http', 'ws') + '/ws/pitch'
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0
      setIsConnected(true)
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
    pending.push(Math.max(0, getTimeRef.current() - WINDOW_HALF_SEC))
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
