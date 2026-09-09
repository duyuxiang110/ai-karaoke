import { useRef, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { scoreColor } from '@/utils/scoreColor'

const AXES = [
  { label: '音准', key: 'pitch' as const, angle: -Math.PI / 2 },
  { label: '节奏', key: 'rhythm' as const, angle: -Math.PI / 2 + (2 * Math.PI) / 3 },
  { label: '气息', key: 'breath' as const, angle: -Math.PI / 2 + (4 * Math.PI) / 3 },
]

export function ScoreRadar() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const score = useKaraokeStore((s) => s.score)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    const cx = w / 2
    const cy = h / 2
    const radius = Math.min(w, h) * 0.36

    ctx.clearRect(0, 0, w, h)

    // 网格圈
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath()
      const r = (radius * i) / 4
      AXES.forEach((axis, idx) => {
        const x = cx + Math.cos(axis.angle) * r
        const y = cy + Math.sin(axis.angle) * r
        if (idx === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.closePath()
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // 轴线
    AXES.forEach((axis) => {
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(
        cx + Math.cos(axis.angle) * radius,
        cy + Math.sin(axis.angle) * radius
      )
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.12)'
      ctx.stroke()
    })

    // 得分多边形，颜色跟随总分
    if (score) {
      const totalColor = scoreColor(score.total)
      ctx.beginPath()
      AXES.forEach((axis, idx) => {
        const value = score[axis.key] / 100
        const r = radius * value
        const x = cx + Math.cos(axis.angle) * r
        const y = cy + Math.sin(axis.angle) * r
        if (idx === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.closePath()
      ctx.globalAlpha = 0.16
      ctx.fillStyle = totalColor
      ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = totalColor
      ctx.lineWidth = 2.5
      ctx.stroke()
    }

    // 标签
    ctx.font = '13px -apple-system, sans-serif'
    ctx.fillStyle = '#64748b'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    AXES.forEach((axis) => {
      const labelR = radius + 22
      const x = cx + Math.cos(axis.angle) * labelR
      const y = cy + Math.sin(axis.angle) * labelR
      ctx.fillText(axis.label, x, y)

      if (score) {
        const value = score[axis.key]
        ctx.fillStyle = scoreColor(value)
        ctx.font = 'bold 18px -apple-system, sans-serif'
        ctx.fillText(value.toFixed(0), x, y + 17)
        ctx.fillStyle = '#64748b'
        ctx.font = '13px -apple-system, sans-serif'
      }
    })

    // 中心总分
    if (score) {
      ctx.fillStyle = scoreColor(score.total)
      ctx.font = 'bold 36px -apple-system, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(score.total.toFixed(0), cx, cy - 7)
      ctx.font = '11px -apple-system, sans-serif'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText('总分', cx, cy + 16)
    } else {
      ctx.fillStyle = '#94a3b8'
      ctx.font = '13px -apple-system, sans-serif'
      ctx.fillText('等待打分', cx, cy)
    }
  }, [score])

  return (
    <div className="score-radar">
      <h3>打分雷达图</h3>
      <canvas ref={canvasRef} className="radar-canvas" />
    </div>
  )
}
