import { useEffect, useRef } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'

const BAR_COUNT = 24
const SAMPLE_MS = 60
const MIN_HEIGHT_PX = 4
const MAX_HEIGHT_PX = 44

// rms 是 [-1,1] 浮点音频的均方根，换算成 dB 再按听觉范围归一
const DB_FLOOR = -54
const DB_CEIL = -16

function rmsToLevel(rms: number): number {
  if (!(rms > 0)) return 0
  const db = 20 * Math.log10(rms)
  return Math.min(1, Math.max(0, (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)))
}

function levelColor(level: number): string {
  if (level >= 0.8) return 'var(--danger)'
  if (level >= 0.5) return 'var(--warn)'
  return 'var(--green)'
}

export function MicLevel() {
  const isRecording = useKaraokeStore((s) => s.isRecording)
  const barsRef = useRef<(HTMLSpanElement | null)[]>([])
  const historyRef = useRef<number[]>(new Array(BAR_COUNT).fill(0))

  useEffect(() => {
    const paint = () => {
      const history = historyRef.current
      for (let i = 0; i < BAR_COUNT; i++) {
        const el = barsRef.current[i]
        if (!el) continue
        const level = history[i]
        const active = level > 0.02
        el.style.height = `${MIN_HEIGHT_PX + level * (MAX_HEIGHT_PX - MIN_HEIGHT_PX)}px`
        el.style.background = active ? levelColor(level) : 'var(--text-muted)'
        el.style.opacity = active ? '1' : '0.35'
      }
    }

    if (!isRecording) {
      historyRef.current = new Array(BAR_COUNT).fill(0)
      paint()
      return
    }

    // 直接改 DOM 而不是 setState：服务端约 21Hz 回传一次 rms，
    // 走 React 状态会造成每秒二十多次整棵子树重渲染
    let raf = 0
    let lastSample = 0
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      if (now - lastSample < SAMPLE_MS) return
      lastSample = now
      const rms = useKaraokeStore.getState().livePitch?.rms ?? 0
      historyRef.current.push(rmsToLevel(rms))
      historyRef.current.shift()
      paint()
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [isRecording])

  return (
    <div
      className={`mic-level ${isRecording ? 'live' : ''}`}
      title={isRecording ? '麦克风音量' : '开始演唱后显示麦克风音量'}
    >
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <span
          key={i}
          className="mic-level-bar"
          ref={(el) => {
            barsRef.current[i] = el
          }}
        />
      ))}
    </div>
  )
}
