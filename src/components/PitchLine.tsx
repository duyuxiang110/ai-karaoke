import { useRef, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { usePlaybackClock } from '@/hooks/usePlaybackClock'
import type { PitchPoint } from '@/types'

const PITCH_MIN = 80
const PITCH_MAX = 600
// 相邻音高点间隔超过该值视为中间没唱，红线断开；短于它的换气仍相连
const USER_BREAK_GAP = 0.4
const WINDOW_SEC = 16

function freqToY(freq: number, height: number): number {
  if (freq <= 0) return height
  const midi = 69 + 12 * Math.log2(freq / 440)
  const midiMin = 69 + 12 * Math.log2(PITCH_MIN / 440)
  const midiMax = 69 + 12 * Math.log2(PITCH_MAX / 440)
  const ratio = (midi - midiMin) / (midiMax - midiMin)
  return height - ratio * height
}

// 音高点按时间升序，二分定位窗口左边界，避免每帧遍历整条曲线
function lowerBound(points: PitchPoint[], time: number): number {
  let lo = 0
  let hi = points.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (points[mid].time < time) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function PitchLine() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const userPitches = useKaraokeStore((s) => s.userPitches)
  const baselinePitches = useKaraokeStore((s) => s.baselinePitches)
  const isRecording = useKaraokeStore((s) => s.isRecording)
  const getTime = usePlaybackClock()

  // 用 ref 存最新数据，RAF 循环直接读取，不依赖 React 重渲染
  const pitchesRef = useRef({ userPitches, baselinePitches, isRecording })
  pitchesRef.current = { userPitches, baselinePitches, isRecording }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf: number

    const drawCurve = (
      points: PitchPoint[],
      tStart: number,
      tEnd: number,
      timeToX: (t: number) => number,
      height: number,
      color: string,
      lineWidth: number,
      smooth = false,
      breakGap = 0
    ): { x: number; y: number; t: number } | null => {
      if (points.length === 0) return null
      ctx.beginPath()
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      let started = false
      let prevT = -1
      let tip: { x: number; y: number; t: number } | null = null
      const from = Math.max(0, lowerBound(points, tStart) - 2)
      for (let i = from; i < points.length; i++) {
        const p = points[i]
        if (p.time > tEnd) continue
        if (p.frequency > 0) {
          let freq = p.frequency
          if (smooth) {
            // 5 点中值滤波（i-2..i+2），抗八度误差和单点跳变
            const vals: number[] = [freq]
            for (let j = -2; j <= 2; j++) {
              if (j === 0) continue
              const f = points[i + j]?.frequency
              if (f && f > 0) vals.push(f)
            }
            vals.sort((a, b) => a - b)
            freq = vals[Math.floor(vals.length / 2)]
          }
          const x = timeToX(p.time)
          const y = freqToY(freq, height)
          if (p.time >= tStart) {
            if (!started || (breakGap > 0 && p.time - prevT > breakGap)) {
              ctx.moveTo(x, y)
              started = true
            } else {
              ctx.lineTo(x, y)
            }
            prevT = p.time
            tip = { x, y, t: p.time }
          }
        } else {
          started = false
        }
      }
      ctx.stroke()
      return tip
    }

    const draw = () => {
      raf = requestAnimationFrame(draw)

      const { userPitches: up, baselinePitches: bp } =
        pitchesRef.current
      const effectiveTime = getTime()

      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr
        canvas.height = h * dpr
        ctx.scale(dpr, dpr)
      }

      ctx.clearRect(0, 0, w, h)

      // 网格
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.05)'
      ctx.lineWidth = 1
      for (let i = 0; i <= 8; i++) {
        const y = (h * i) / 8
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      // 音符标签
      ctx.fillStyle = 'rgba(255, 0, 0, 0.51)'
      ctx.font = '10px monospace'
      ctx.textAlign = 'left'
      const noteLabels = ['C3', 'C4', 'C5']
      noteLabels.forEach((label, i) => {
        const freq = 130.81 * Math.pow(2, i)
        const y = freqToY(freq, h)
        ctx.fillText(label, 4, y - 2)
      })

      // 时间窗口：显示当前 ±8 秒
      const tStart = Math.max(0, effectiveTime - WINDOW_SEC / 2)
      const tEnd = tStart + WINDOW_SEC

      const timeToX = (t: number): number => {
        return ((t - tStart) / WINDOW_SEC) * w
      }

      // 基线音高（原唱）
      drawCurve(bp, tStart, tEnd, timeToX, h, 'rgba(99, 102, 241, 0.5)', 1.5)

      // 当前时间指示线
      const cursorX = timeToX(effectiveTime)

      // 用户实时音高：中值平滑 + 静音断线
      drawCurve(
        up, tStart, tEnd, timeToX, h, '#ec4899', 2, true, USER_BREAK_GAP
      )

      ctx.beginPath()
      ctx.moveTo(cursorX, 0)
      ctx.lineTo(cursorX, h)
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [getTime])

  return (
    <div className="pitch-line">
      <div className="pitch-line-header">
        <span className="legend-item">
          <span className="legend-dot baseline"></span> 原唱
        </span>
        <span className="legend-item">
          <span className="legend-dot user"></span> 我的
        </span>
      </div>
      <canvas ref={canvasRef} className="pitch-canvas" />
      {userPitches.length === 0 && (
        <div className="pitch-empty-overlay">
          <span>实时音高曲线</span>
        </div>
      )}
    </div>
  )
}
