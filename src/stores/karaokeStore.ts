import { create } from 'zustand'
import type { Song, LyricLine, LyricSource, PitchPoint, PitchResult, Score } from '@/types'

interface KaraokeState {
  songs: Song[]
  currentSong: Song | null
  // 实际播放地址由 isVocalPlayback 在这两个之间派生
  instrumentalUrl: string | null
  vocalUrl: string | null
  isVocalPlayback: boolean

  isPlaying: boolean
  currentTime: number
  duration: number

  lyrics: LyricLine[]
  lyricSource: LyricSource | null
  currentLyricIndex: number

  isRecording: boolean
  isSeeking: boolean
  userPitches: PitchPoint[]
  baselinePitches: PitchPoint[]
  livePitch: PitchResult | null

  score: Score | null

  pythonReady: boolean
  isProcessing: boolean
  processMessage: string

  addSong: (song: Song) => void
  removeSong: (id: string) => void
  selectSong: (song: Song) => void
  setSongLrc: (id: string, lrcPath: string | null) => void
  setAudioSources: (instrumental: string | null, vocal: string | null) => void
  setVocalPlayback: (on: boolean) => void

  setPlaying: (playing: boolean) => void
  setCurrentTime: (t: number) => void
  setDuration: (d: number) => void
  setSeeking: (seeking: boolean) => void

  setLyrics: (lyrics: LyricLine[]) => void
  setLyricSource: (source: LyricSource | null) => void
  setCurrentLyricIndex: (i: number) => void

  setRecording: (recording: boolean) => void
  addUserPitch: (p: PitchPoint) => void
  clearUserPitches: () => void
  setBaselinePitches: (p: PitchPoint[]) => void
  setLivePitch: (p: PitchResult | null) => void

  setScore: (s: Score | null) => void

  setPythonReady: (ready: boolean) => void
  setProcessing: (processing: boolean, msg?: string) => void
}

export const useKaraokeStore = create<KaraokeState>((set) => ({
  songs: [],
  currentSong: null,
  instrumentalUrl: null,
  vocalUrl: null,
  isVocalPlayback: false,

  isPlaying: false,
  currentTime: 0,
  duration: 0,

  lyrics: [],
  lyricSource: null,
  currentLyricIndex: -1,

  isRecording: false,
  isSeeking: false,
  userPitches: [],
  baselinePitches: [],
  livePitch: null,

  score: null,

  pythonReady: false,
  isProcessing: false,
  processMessage: '',

  addSong: (song) => set((s) => ({ songs: [...s.songs, song] })),
  removeSong: (id) => set((s) => ({ songs: s.songs.filter((x) => x.id !== id) })),
  selectSong: (song) => set({
    currentSong: song,
    instrumentalUrl: null,
    vocalUrl: null,
    currentTime: 0,
    isPlaying: false,
    isSeeking: false,
    userPitches: [],
    score: null,
    lyrics: [],
    lyricSource: null,
    currentLyricIndex: -1,
  }),
  setSongLrc: (id, lrcPath) => set((s) => ({
    songs: s.songs.map((x) => (x.id === id ? { ...x, lrcPath } : x)),
    currentSong:
      s.currentSong && s.currentSong.id === id
        ? { ...s.currentSong, lrcPath }
        : s.currentSong,
  })),
  setAudioSources: (instrumental, vocal) => set({
    instrumentalUrl: instrumental,
    vocalUrl: vocal,
  }),
  setVocalPlayback: (on) => set({ isVocalPlayback: on }),

  setPlaying: (playing) => set({ isPlaying: playing }),
  setCurrentTime: (t) => set({ currentTime: t }),
  setDuration: (d) => set({ duration: d }),
  setSeeking: (seeking) => set({ isSeeking: seeking }),

  setLyrics: (lyrics) => set({ lyrics }),
  setLyricSource: (source) => set({ lyricSource: source }),
  setCurrentLyricIndex: (i) => set({ currentLyricIndex: i }),

  setRecording: (recording) => set({ isRecording: recording }),
  addUserPitch: (p) => set((s) => ({ userPitches: s.userPitches.concat(p) })),
  clearUserPitches: () => set({ userPitches: [] }),
  setBaselinePitches: (p) => set({ baselinePitches: p }),
  setLivePitch: (p) => set({ livePitch: p }),

  setScore: (s) => set({ score: s }),

  setPythonReady: (ready) => set({ pythonReady: ready }),
  setProcessing: (processing, msg = '') => set({ isProcessing: processing, processMessage: msg }),
}))
