// 分数配色，全站统一：低分红、中分黄、高分绿
export function scoreColor(value: number): string {
  if (value >= 75) return '#10b981'
  if (value >= 60) return '#f59e0b'
  return '#ef4444'
}
