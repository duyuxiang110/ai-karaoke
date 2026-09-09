import { useRef, useEffect, useCallback } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'

export function Player() {
  const instRef = useRef<HTMLAudioElement | null>(null)
  const vocalRef = useRef<HTMLAudioElement | null>(null)
  const isSeekingRef = useRef(false)
  const {
    currentSong,
    instrumentalUrl,
    vocalUrl,
    isVocalPlayback,
    isPlaying,
    currentTime,
    duration,
    setPlaying,
    setCurrentTime,
    setDuration,
    setVocalPlayback,
  } = useKaraokeStore()

  // 伴奏/原唱各用一个独立 audio 元素，切换时不换 src：
  // 同一元素在 mp3 与 wav 之间换源会重新初始化解码器，
  // 部分 macOS（实测 12.5）上切回后会无声
  const activeRef = useRef<HTMLAudioElement | null>(null)
  activeRef.current = isVocalPlayback ? vocalRef.current : instRef.current

  const activeEl = () => (isVocalPlayback ? vocalRef.current : instRef.current)

  // 事件只认当前激活的元素，避免暂停的那路干扰进度
  useEffect(() => {
    const els = [instRef.current, vocalRef.current].filter(
      (x): x is HTMLAudioElement => !!x
    )

    const onTimeUpdate = (e: Event) => {
      if (e.target !== activeRef.current) return
      if (isSeekingRef.current) return
      const audio = e.target as HTMLAudioElement
      if (audio.duration && audio.currentTime >= audio.duration - 0.1) {
        audio.pause()
        setPlaying(false)
        setCurrentTime(audio.duration)
      } else {
        setCurrentTime(audio.currentTime)
      }
    }
    const onLoadedMetadata = (e: Event) => {
      if (e.target !== activeRef.current) return
      const audio = e.target as HTMLAudioElement
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration)
      }
    }
    const onEnded = (e: Event) => {
      if (e.target !== activeRef.current) return
      setPlaying(false)
      setCurrentTime(0)
    }

    els.forEach((el) => {
      el.addEventListener('timeupdate', onTimeUpdate)
      el.addEventListener('loadedmetadata', onLoadedMetadata)
      el.addEventListener('ended', onEnded)
    })
    return () => {
      els.forEach((el) => {
        el.removeEventListener('timeupdate', onTimeUpdate)
        el.removeEventListener('loadedmetadata', onLoadedMetadata)
        el.removeEventListener('ended', onEnded)
      })
    }
  }, [instrumentalUrl, vocalUrl, setCurrentTime, setDuration, setPlaying])

  // 切换原唱/伴奏：把进度搬到另一路再播，不重建解码器
  useEffect(() => {
    const to = activeEl()
    if (!to) return
    const from = isVocalPlayback ? instRef.current : vocalRef.current
    const t = from && from.readyState > 0 ? from.currentTime : currentTime
    const wasPlaying = isPlaying
    from?.pause()
    try {
      to.currentTime = t
    } catch {
      // 元数据未就绪时等 canplay 再定位
      to.addEventListener(
        'canplay',
        () => {
          to.currentTime = t
          if (wasPlaying) to.play().catch(() => {})
        },
        { once: true }
      )
      return
    }
    setCurrentTime(t)
    if (wasPlaying) to.play().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVocalPlayback])

  useEffect(() => {
    const audio = activeEl()
    if (!audio) return
    if (isPlaying) {
      audio.play().catch(() => {})
    } else {
      audio.pause()
    }
  }, [isPlaying, isVocalPlayback])

  const togglePlay = () => setPlaying(!isPlaying)

  const handleSeekStart = useCallback(() => {
    isSeekingRef.current = true
    useKaraokeStore.getState().setSeeking(true)
  }, [])

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = isVocalPlayback ? vocalRef.current : instRef.current
    if (!audio) return
    const t = parseFloat(e.target.value)
    audio.currentTime = t
    setCurrentTime(t)
  }, [setCurrentTime, isVocalPlayback])

  const handleSeekEnd = useCallback(() => {
    isSeekingRef.current = false
    useKaraokeStore.getState().setSeeking(false)
  }, [])

  const formatTime = (s: number) => {
    if (!s || isNaN(s)) return '00:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="player-bar">
      {instrumentalUrl && (
        <audio ref={instRef} src={instrumentalUrl} preload="metadata" />
      )}
      {vocalUrl && <audio ref={vocalRef} src={vocalUrl} preload="metadata" />}
      <button
        className="btn-play"
        onClick={togglePlay}
        disabled={!currentSong}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>
      <span className="time-display">{formatTime(currentTime)}</span>
      <input
        type="range"
        className="progress-bar"
        min={0}
        max={duration || 0}
        value={currentTime}
        onChange={handleSeek}
        onInput={handleSeek}
        onMouseDown={handleSeekStart}
        onMouseUp={handleSeekEnd}
        onTouchStart={handleSeekStart}
        onTouchEnd={handleSeekEnd}
        disabled={!currentSong}
      />
      <span className="time-display">{formatTime(duration)}</span>
      <button
        className={`mode-toggle ${isVocalPlayback ? 'on' : ''}`}
        onClick={() => setVocalPlayback(!isVocalPlayback)}
        disabled={!instrumentalUrl || !vocalUrl}
        title={isVocalPlayback ? '正在播放原唱，点击切回伴奏' : '正在播放伴奏，点击切到原唱'}
      >
        <span className="mode-toggle-track">
          <span className="mode-toggle-knob"></span>
        </span>
        <span className="mode-toggle-label">{isVocalPlayback ? '原唱' : '伴奏'}</span>
      </button>
    </div>
  )
}
