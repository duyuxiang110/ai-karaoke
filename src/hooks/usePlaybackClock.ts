import { useEffect, useRef, useCallback } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'

/**
 * 播放时钟
 *
 * audio 的 timeupdate 事件只有约 4Hz，直接拿它驱动画面会一顿一顿。
 * 这里以每次 store 更新为锚点，在两帧之间用 performance.now() 线性外推，
 * 返回一个取当前播放位置的函数（不触发重渲染，供 RAF 循环调用）。
 */
export function usePlaybackClock(): () => number {
  const currentTime = useKaraokeStore((s) => s.currentTime)
  const isPlaying = useKaraokeStore((s) => s.isPlaying)
  const duration = useKaraokeStore((s) => s.duration)

  const anchor = useRef({ base: currentTime, wall: performance.now() })

  useEffect(() => {
    anchor.current = { base: currentTime, wall: performance.now() }
  }, [currentTime, isPlaying])

  return useCallback(() => {
    const { base, wall } = anchor.current
    if (!isPlaying) return base
    const estimated = base + (performance.now() - wall) / 1000
    return duration > 0 ? Math.min(estimated, duration) : estimated
  }, [isPlaying, duration])
}
