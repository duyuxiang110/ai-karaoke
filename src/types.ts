export interface Song {
  id: string
  name: string
  path: string
  duration: number
  lrcPath?: string | null
}

export interface LyricLine {
  text: string
  start: number
  end: number
}

export interface LyricSource {
  type: 'id3' | 'lrc' | 'online' | 'none'
  label: string
  synced: boolean
  reason: string
}

export interface PitchPoint {
  time: number
  frequency: number
}

export interface PitchResult {
  type: string
  frequency: number | null
  note: string | null
  cents: number | null
  timestamp: number | null
  rms: number
}

export interface Score {
  total: number
  pitch: number
  /** 歌词没有可用时间轴时为 null，表示节奏未参评（不是 0 分） */
  rhythm: number | null
  breath: number
  /** 演唱完成度百分比，仅用于排查总分偏低的原因，界面不展示 */
  completion?: number
  /** 时间轴无效、或总分被完成度折算时的解释文案 */
  warning?: string | null
}
