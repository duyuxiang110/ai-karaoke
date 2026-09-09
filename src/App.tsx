import { useEffect, useState, useCallback } from 'react'
import { SongList } from '@/components/SongList'
import { Player } from '@/components/Player'
import { LyricsScroll } from '@/components/LyricsScroll'
import { MicLevel } from '@/components/MicLevel'
import { ScoreRadar } from '@/components/ScoreRadar'
import { PitchLine } from '@/components/PitchLine'
import { ResultPanel } from '@/components/ResultPanel'
import { useKaraokeStore } from '@/stores/karaokeStore'
import { useMicCapture } from '@/hooks/useMicCapture'
import { usePitchStream } from '@/hooks/usePitchStream'

declare global {
  interface Window {
    electronAPI: {
      pythonBaseUrl: string
      selectMP3File: () => Promise<string | null>
      selectLrcFile?: () => Promise<string | null>
      getPythonStatus: () => Promise<boolean>
    }
  }
}

const baseUrl = window.electronAPI?.pythonBaseUrl || 'http://127.0.0.1:8765'

function App() {
  const currentSong = useKaraokeStore((s) => s.currentSong)
  const setAudioSources = useKaraokeStore((s) => s.setAudioSources)
  const setDuration = useKaraokeStore((s) => s.setDuration)
  const setBaselinePitches = useKaraokeStore((s) => s.setBaselinePitches)
  const setLyrics = useKaraokeStore((s) => s.setLyrics)
  const setLyricSource = useKaraokeStore((s) => s.setLyricSource)
  const pythonReady = useKaraokeStore((s) => s.pythonReady)
  const setPythonReady = useKaraokeStore((s) => s.setPythonReady)
  const isRecording = useKaraokeStore((s) => s.isRecording)
  const isProcessing = useKaraokeStore((s) => s.isProcessing)
  const processMessage = useKaraokeStore((s) => s.processMessage)
  const setProcessing = useKaraokeStore((s) => s.setProcessing)
  const clearUserPitches = useKaraokeStore((s) => s.clearUserPitches)
  const setScore = useKaraokeStore((s) => s.setScore)

  const { start: startMic, stop: stopMic, error: micError } = useMicCapture()
  const { connect, disconnect, sendPCM } = usePitchStream(baseUrl)

  const [scoring, setScoring] = useState(false)

  // Check Python server status
  useEffect(() => {
    const check = async () => {
      const ok = await window.electronAPI?.getPythonStatus?.()
      setPythonReady(!!ok)
    }
    check()
    const timer = setInterval(check, 3000)
    return () => clearInterval(timer)
  }, [setPythonReady])

  // When song is selected, run separation + baseline extraction
  useEffect(() => {
    if (!currentSong) {
      setAudioSources(null, null)
      return
    }

    let cancelled = false

    const processSong = async () => {
      try {
        setProcessing(true, '正在加载 AI 模型并分离伴奏（首次可能需要下载模型，请耐心等待）...')
        clearUserPitches()
        setScore(null)

        // 1. 伴奏分离（Demucs 首次运行可能需要下载模型，设 5 分钟超时）
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 300000)
        let sep: any
        try {
          const sepRes = await fetch(`${baseUrl}/api/separate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: currentSong.path }),
            signal: controller.signal,
          })
          clearTimeout(timeoutId)
          if (!sepRes.ok) throw new Error('伴奏分离失败')
          sep = await sepRes.json()
        } catch (fetchErr: any) {
          clearTimeout(timeoutId)
          if (fetchErr.name === 'AbortError') throw new Error('分离超时（超过5分钟），请尝试更短的音频文件')
          throw fetchErr
        }

        if (cancelled) return

        // 2. 登记两个播放源：伴奏（分离出的 instrumental）与原唱（原始 mp3 完整混音）
        //    实际播哪个由底栏的原唱开关派生；都走 Python HTTP 端点以支持 Range 请求
        setAudioSources(
          `${baseUrl}/audio?path=${encodeURIComponent(sep.instrumental_path)}`,
          `${baseUrl}/audio?path=${encodeURIComponent(currentSong.path)}`
        )
        setDuration(sep.duration)

        // 3. 获取原唱音高基线 + 歌词（传入原始 MP3 路径用于读取内嵌歌词；
        //    若用户手动挂了 .lrc 则一并传入，优先级最高）
        setProcessing(true, '正在提取音高基线...')
        const lrcParam = currentSong.lrcPath
          ? `&lrc_path=${encodeURIComponent(currentSong.lrcPath)}`
          : ''
        const baseRes = await fetch(
          `${baseUrl}/api/baseline?file_path=${encodeURIComponent(sep.vocal_path)}&original_path=${encodeURIComponent(currentSong.path)}${lrcParam}`
        )
        if (!baseRes.ok) throw new Error('基线提取失败')
        const base = await baseRes.json()

        if (cancelled) return

        // 4. 更新 store
        const baselinePitches = base.pitches.map((p: any) => ({
          time: p.time,
          frequency: p.frequency,
        }))
        const lyrics = base.lyrics.map((l: any) => ({
          text: l.text,
          start: l.start,
          end: l.end,
        }))

        setBaselinePitches(baselinePitches)
        setLyrics(lyrics)
        setLyricSource(base.lyric_source ?? null)
        setProcessing(false, '')
      } catch (err) {
        console.error('Processing error:', err)
        setProcessing(false, '')
        setLyricSource({
          type: 'none',
          label: '',
          synced: false,
          reason: '歌曲处理失败，未能获取歌词',
        })
        // Fallback: 分离失败，两个音源都退化成原始 mp3
        const fallbackUrl = `${baseUrl}/audio?path=${encodeURIComponent(currentSong.path)}`
        setAudioSources(fallbackUrl, fallbackUrl)
      }
    }

    processSong()

    return () => { cancelled = true }
  }, [currentSong, baseUrl, setAudioSources, setBaselinePitches, setLyrics, setLyricSource, setDuration, setProcessing, clearUserPitches, setScore])

  const handleStartSinging = useCallback(async () => {
    clearUserPitches()
    setScore(null)
    connect()
    await startMic(sendPCM)
  }, [clearUserPitches, setScore, connect, startMic, sendPCM])

  const handleStopSinging = useCallback(async () => {
    stopMic()
    disconnect()

    setScoring(true)
    const { userPitches, baselinePitches } = useKaraokeStore.getState()
    try {
      const res = await fetch(`${baseUrl}/api/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_pitches: userPitches,
          baseline_pitches: baselinePitches,
        }),
      })
      const data = await res.json()
      setScore(data)
    } catch (err) {
      console.error('Failed to get score:', err)
    } finally {
      setScoring(false)
    }
  }, [stopMic, disconnect, baseUrl, setScore])

  return (
    <div className="app">
      <header className="app-header">
        <div className='app-header-left'>
          <h1>AI智能音乐打分系统</h1>
          <span>淳安县实验小学 姜小菡 著</span>
        </div>
        <div className="header-status">
          <span className={`status-dot ${pythonReady ? 'ok' : 'err'}`}></span>
          <span>AI {pythonReady ? '已连接' : '未连接'}</span>
          {isRecording && (
            <span className="rec-indicator">
              <span className="rec-dot"></span> 录音中
            </span>
          )}
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar-left">
          <SongList />
        </aside>

        <main className="main-content">
          {isProcessing ? (
            <div className="processing-overlay">
              <div className="processing-spinner"></div>
              <p className="processing-text">{processMessage}</p>
            </div>
          ) : (
            <>
              <LyricsScroll />
              <PitchLine />
            </>
          )}
        </main>

        <aside className="sidebar-right">
          <ScoreRadar />
          <ResultPanel />
        </aside>
      </div>

      <footer className="app-footer">
        <Player />
        <div className="mic-controls">
          <MicLevel />
          <button
            className={`btn-sing ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? handleStopSinging : handleStartSinging}
            disabled={!currentSong || scoring || isProcessing}
          >
            {scoring ? '⏳ 打分中...' : isRecording ? '⏹ 结束演唱' : '🎤 开始演唱'}
          </button>
          {micError && <span className="mic-error">{micError}</span>}
        </div>
      </footer>
    </div>
  )
}

export default App
