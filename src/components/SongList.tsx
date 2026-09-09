import { useKaraokeStore } from '@/stores/karaokeStore'
import type { Song } from '@/types'

let songIdCounter = 0

export function SongList() {
  const { songs, currentSong, addSong, removeSong, selectSong, setSongLrc } =
    useKaraokeStore()

  const handleImport = async () => {
    const filePath = await window.electronAPI?.selectMP3File?.()
    if (!filePath) return

    const name = filePath.split('/').pop()?.replace(/\.mp3$/i, '') || 'Unknown'
    const song: Song = {
      id: `song-${++songIdCounter}`,
      name,
      path: filePath,
      duration: 0,
    }
    addSong(song)
    selectSong(song)
  }

  const handleSelect = (song: Song) => {
    selectSong(song)
  }

  const handleImportLrc = async (e: React.MouseEvent, song: Song) => {
    e.stopPropagation()
    const lrcPath = await window.electronAPI?.selectLrcFile?.()
    if (!lrcPath) return
    setSongLrc(song.id, lrcPath)
  }

  const handleRemove = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    removeSong(id)
  }

  return (
    <div className="song-list">
      <div className="song-list-header">
        <h2>歌曲列表</h2>
        <button className="btn-import" onClick={handleImport}>
          + 导入 MP3
        </button>
      </div>
      <div className="song-list-items">
        {songs.length === 0 ? (
          <p className="empty-hint">点击上方按钮导入歌曲</p>
        ) : (
          songs.map((song) => (
            <div
              key={song.id}
              className={`song-item ${currentSong?.id === song.id ? 'active' : ''}`}
              onClick={() => handleSelect(song)}
            >
              <span className="song-name">{song.name}</span>
              <button
                className={`song-lrc ${song.lrcPath ? 'has-lrc' : ''}`}
                title={
                  song.lrcPath
                    ? `已挂歌词：${song.lrcPath.split('/').pop()}，点击更换`
                    : '曲库没有歌词时，手动上传 .lrc 歌词文件'
                }
                onClick={(e) => handleImportLrc(e, song)}
              >
                词
              </button>
              <button
                className="song-remove"
                onClick={(e) => handleRemove(e, song.id)}
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
