import { useRef, useEffect, useCallback } from 'react'
import { useKaraokeStore } from '@/stores/karaokeStore'

export function Player() {
  const audioRef = useRef<HTMLAudioElement | null>(null)
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

  // 播放源由原唱开关派生，默认伴奏
  const audioUrl = isVocalPlayback ? vocalUrl : instrumentalUrl

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !audioUrl) return

    const onTimeUpdate = () => {
      if (isSeekingRef.current) return
      // Safety: 如果 ended 事件未触发，检测到播放到末尾时手动停止
      if (audio.duration && audio.currentTime >= audio.duration - 0.1) {
        audio.pause()
        setPlaying(false)
        setCurrentTime(audio.duration)
      } else {
        setCurrentTime(audio.currentTime)
      }
    }
    const onLoadedMetadata = () => {
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration)
      }
    }
    const onDurationChange = () => {
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration)
      }
    }
    const onEnded = () => {
      setPlaying(false)
      setCurrentTime(0)
    }

    audio.addEventListener('timeupdate', onTimeUpdate)
    audio.addEventListener('loadedmetadata', onLoadedMetadata)
    audio.addEventListener('durationchange', onDurationChange)
    audio.addEventListener('ended', onEnded)

    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate)
      audio.removeEventListener('loadedmetadata', onLoadedMetadata)
      audio.removeEventListener('durationchange', onDurationChange)
      audio.removeEventListener('ended', onEnded)
    }
  }, [audioUrl, setCurrentTime, setDuration, setPlaying])

  // 伴奏 / 原唱互换会替换 <audio> 的 src，元素随即被重置，
  // 必须等新源元数据就绪后把播放进度和播放状态接回去
  const prevUrlRef = useRef(audioUrl)
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (prevUrlRef.current === audioUrl) return
    prevUrlRef.current = audioUrl

    const { currentTime: resumeAt, isPlaying: wasPlaying } = useKaraokeStore.getState()
    if (!audioUrl || (resumeAt <= 0 && !wasPlaying)) return

    const restore = () => {
      audio.currentTime = resumeAt
      // 换源重置瞬间可能触发一次 timeupdate 把 store 写成 0，暂停时不会再有事件纠正
      setCurrentTime(resumeAt)
      if (wasPlaying) audio.play().catch(() => {})
    }
    audio.addEventListener('loadedmetadata', restore, { once: true })
    audio.load()
  }, [audioUrl, setCurrentTime])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying) {
      audio.play().catch(() => {})
    } else {
      audio.pause()
    }
  }, [isPlaying])

  const togglePlay = () => setPlaying(!isPlaying)

  const handleSeekStart = useCallback(() => {
    isSeekingRef.current = true
  }, [])

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current
    if (!audio) return
    const t = parseFloat(e.target.value)
    audio.currentTime = t
    setCurrentTime(t)
  }, [setCurrentTime])

  const handleSeekEnd = useCallback(() => {
    isSeekingRef.current = false
  }, [])

  const formatTime = (s: number) => {
    if (!s || isNaN(s)) return '00:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="player-bar">
      <audio ref={audioRef} src={audioUrl || undefined} />
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
        disabled={!audioUrl}
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
