import { useRef, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { usePlaybackClock } from '@/hooks/usePlaybackClock'
import type { PitchPoint } from '@/types'

const PITCH_MIN = 80
const PITCH_MAX = 600
// 麦克风超过 200ms 没检测到有效音高就断线，避免停唱后红线还拖着走
const USER_BREAK_GAP = 0.2
// 麦克风检测链路（AudioContext→analyser→pitch→JS）有缓冲延迟，
// 允许实时红线最多超前播放位置这么多，否则当前帧刚检测到的点可能画不出来
const REALTIME_DRAW_AHEAD = 0.05
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

// 实时曲线只能用「过去 + 当前」点做中值平滑，绝不能偷看 i+1 / i+2，
// 否则等于把未来的音高混进当前帧，红线会提前画出还没唱到的走势
function getSmoothFrequency(
  points: PitchPoint[],
  index: number,
  mode: 'none' | 'past'
): number {
  const current = points[index]?.frequency ?? 0
  if (current <= 0) return current
  if (mode === 'none') return current
  const values: number[] = []
  for (let j = -2; j <= 0; j++) {
    const f = points[index + j]?.frequency
    if (f && f > 0) values.push(f)
  }
  values.sort((a, b) => a - b)
  return values[Math.floor(values.length / 2)]
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
      smooth: 'none' | 'past' = 'none',
      breakGap = 0,
      maxTime = Infinity
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
        // 点已按时间升序：超过硬上限 maxTime 或窗口右边界 tEnd 都可以直接 break
        // 实时红线绝不能画到当前播放位置之后
        if (p.time > maxTime) break
        if (p.time > tEnd) break
        if (p.frequency > 0) {
          const freq = getSmoothFrequency(points, i, smooth)
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

      // 基线音高（原唱）：允许画到未来，作为跟唱参考；不平滑
      drawCurve(bp, tStart, tEnd, timeToX, h, 'rgba(99, 102, 241, 0.5)', 1.5, 'none')

      // 当前时间指示线
      const cursorX = timeToX(effectiveTime)

      // 用户实时音高：只用过去点平滑 + 静音断线，
      // 并硬限制最多画到「当前播放位置 + 小容差」，绝不提前画未来
      const maxUserTime = effectiveTime + REALTIME_DRAW_AHEAD
      drawCurve(
        up, tStart, tEnd, timeToX, h, '#ec4899', 2, 'past', USER_BREAK_GAP, maxUserTime
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
