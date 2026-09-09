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

  // onmessage 的闭包在 connect 时就固定了，用 ref 才能读到最新的时钟
  const getTimeRef = useRef(getTime)
  getTimeRef.current = getTime

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const wsUrl = baseUrl.replace('http', 'ws') + '/ws/pitch'
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => setIsConnected(true)
    ws.onclose = () => setIsConnected(false)
    ws.onerror = () => setIsConnected(false)

    ws.onmessage = (e: MessageEvent) => {
      try {
        const data: PitchResult = JSON.parse(e.data)
        const store = useKaraokeStore.getState()

        store.setLivePitch(data)

        if (data.frequency != null) {
          // 用歌曲位置而不是服务端回传的 timestamp：基线音高在歌曲时间轴上，
          // 两者必须同轴，否则音准打分的插值整体错位
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
      wsRef.current?.close()
    }
  }, [])

  return { connect, disconnect, sendPCM, isConnected }
}
