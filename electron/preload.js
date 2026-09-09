const { contextBridge, ipcRenderer } = require('electron')

const PYTHON_PORT = 8765

contextBridge.exposeInMainWorld('electronAPI', {
  pythonBaseUrl: `http://127.0.0.1:${PYTHON_PORT}`,
  selectMP3File: () => ipcRenderer.invoke('dialog:select-mp3'),
  selectLrcFile: () => ipcRenderer.invoke('dialog:select-lrc'),
  getPythonStatus: () => ipcRenderer.invoke('python:status'),
})
