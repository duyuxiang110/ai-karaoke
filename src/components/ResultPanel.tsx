import { useKaraokeStore } from '@/stores/karaokeStore'
import { scoreColor } from '@/utils/scoreColor'

export function ResultPanel() {
  const score = useKaraokeStore((s) => s.score)

  if (!score) {
    return (
      <div className="result-panel empty">
        <h3>结算</h3>
        <p className="empty-hint">演唱结束后显示最终成绩</p>
      </div>
    )
  }

  const grade =
    score.total >= 90 ? 'S' :
    score.total >= 80 ? 'A' :
    score.total >= 70 ? 'B' :
    score.total >= 60 ? 'C' : 'D'

  const totalColor = scoreColor(score.total)
  const rhythm = score.rhythm

  return (
    <div className="result-panel">
      <h3>演唱成绩</h3>
      <div className="result-grade" style={{ color: totalColor }}>
        {grade}
      </div>
      <div className="result-total">
        <span className="result-score" style={{ color: totalColor }}>
          {score.total.toFixed(1)}
        </span>
        <span className="result-unit">分</span>
      </div>
      {/* 总分被完成度折算、或时间轴无效时必须说明原因，
          否则「音准气息都还行、总分 1 分」看起来就像算错了 */}
      {score.warning && <p className="score-warning">{score.warning}</p>}
      <div className="result-breakdown">
        <div className="breakdown-item">
          <span className="breakdown-label">音准</span>
          <span className="breakdown-value" style={{ color: scoreColor(score.pitch) }}>
            {score.pitch.toFixed(1)}
          </span>
        </div>
        <div className="breakdown-item">
          <span className="breakdown-label">节奏</span>
          {rhythm == null ? (
            // 未参评不能显示成 0.0：那是「唱得极差」，和「没得评」是两件事
            <span
              className="breakdown-value na"
              title="这首歌的歌词没有同步时间轴，节奏无从评起；权重已按比例摊回音准和气息"
            >
              未评
            </span>
          ) : (
            <span className="breakdown-value" style={{ color: scoreColor(rhythm) }}>
              {rhythm.toFixed(1)}
            </span>
          )}
        </div>
        <div className="breakdown-item">
          <span className="breakdown-label">气息</span>
          <span className="breakdown-value" style={{ color: scoreColor(score.breath) }}>
            {score.breath.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  )
}
