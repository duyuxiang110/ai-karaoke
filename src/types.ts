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
  rhythm: number
  breath: number
}
