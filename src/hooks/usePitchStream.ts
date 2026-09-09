import { useState, useRef, useCallback, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { usePlaybackClock } from '@/hooks/usePlaybackClock'
import type { PitchResult } from '@/types'

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

  const shouldReconnectRef = useRef(false)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptsRef = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return

    shouldReconnectRef.current = true

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
      try {
        const data: PitchResult = JSON.parse(e.data)
        const store = useKaraokeStore.getState()

        store.setLivePitch(data)

        if (data.frequency != null && data.frequency > 0 && !store.isSeeking) {
          store.addUserPitch({
            time: getTimeRef.current(),
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

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    wsRef.current?.close()
    wsRef.current = null
    setIsConnected(false)
  }, [])

  const sendPCM = useCallback((buffer: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(buffer)
    }
  }, [])

  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [])

  return { connect, disconnect, sendPCM, isConnected }
}
