import { useRef, useEffect } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { usePlaybackClock } from '@/hooks/usePlaybackClock'
import type { LyricLine } from '@/types'

const SOURCE_PREFIX: Record<string, string> = {
  id3: 'MP3 内嵌歌词',
  lrc: '本地 LRC',
  online: '网易云音乐',
}

// 落在间奏等空隙时返回 -1，调用方据此保持上一行高亮
function lineIndexAt(lyrics: LyricLine[], t: number): number {
  return lyrics.findIndex((line) => t >= line.start && t < line.end)
}

export function LyricsScroll() {
  const lyrics = useKaraokeStore((s) => s.lyrics)
  const lyricSource = useKaraokeStore((s) => s.lyricSource)
  const currentLyricIndex = useKaraokeStore((s) => s.currentLyricIndex)
  const isPlaying = useKaraokeStore((s) => s.isPlaying)
  const currentTime = useKaraokeStore((s) => s.currentTime)
  const getTime = usePlaybackClock()
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (lyrics.length === 0) return

    const applyAt = (t: number) => {
      const idx = lineIndexAt(lyrics, t)
      if (idx >= 0 && idx !== useKaraokeStore.getState().currentLyricIndex) {
        useKaraokeStore.getState().setCurrentLyricIndex(idx)
      }
    }

    // 暂停/拖动进度条时 timeupdate 是唯一时间来源，直接用 store 的值算一次
    if (!isPlaying) {
      applyAt(currentTime)
      return
    }

    let raf = requestAnimationFrame(function tick() {
      applyAt(getTime())
      raf = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(raf)
  }, [lyrics, isPlaying, currentTime, getTime])

  useEffect(() => {
    if (currentLyricIndex < 0 || !containerRef.current) return
    const el = containerRef.current.querySelector(
      `[data-lyric-idx="${currentLyricIndex}"]`
    ) as HTMLElement | null
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [currentLyricIndex])

  if (lyrics.length === 0) {
    return (
      <div className="lyrics-scroll empty">
        <p className="empty-hint">
          {lyricSource?.reason || '导入歌曲后，歌词将在此滚动显示'}
        </p>
        {!lyricSource?.reason && (
          <p className="empty-sub">（依次尝试 MP3 内嵌歌词、同名 .lrc、在线检索）</p>
        )}
      </div>
    )
  }

  const prefix = lyricSource ? SOURCE_PREFIX[lyricSource.type] : ''

  return (
    <div className="lyrics-pane">
      {prefix && (
        <div className="lyrics-source">
          {/* <span className="lyrics-source-tag">{prefix}</span> */}
          {lyricSource?.label && (
            <span className="lyrics-source-label">{lyricSource.label}</span>
          )}
          {/* {lyricSource && !lyricSource.synced && (
            <span className="lyrics-source-note">时间轴按人声段落推算</span>
          )} */}
        </div>
      )}
      <div className="lyrics-scroll" ref={containerRef}>
        {lyrics.map((line, idx) => (
          <p
            key={idx}
            data-lyric-idx={idx}
            className={`lyric-line ${idx === currentLyricIndex ? 'active' : ''}`}
          >
            {line.text}
          </p>
        ))}
      </div>
    </div>
  )
}
