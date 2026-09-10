const { app, BrowserWindow, ipcMain, dialog, protocol, net } = require('electron')
const { spawn, execFileSync } = require('child_process')
const path = require('path')
const http = require('http')
const fs = require('fs')
const os = require('os')

const PYTHON_PORT = 8765
const PYTHON_DIR = path.join(__dirname, '..', 'python')
const AI_SERVER = path.join(PYTHON_DIR, 'ai_server.py')
const VENV_PYTHON = path.join(PYTHON_DIR, 'venv', 'bin', 'python')

protocol.registerSchemesAsPrivileged([{
  scheme: 'karaoke',
  privileges: {
    standard: true,
    secure: true,
    supportFetchAPI: true,
    stream: true,
  },
}])

let pythonProcess = null
let mainWindow = null

function getPythonExecutable() {
  if (fs.existsSync(VENV_PYTHON)) {
    return VENV_PYTHON
  }
  return 'python3'
}

// dev 与打包态的 Python 启动参数不同：
// 打包态没有 venv，用 extraResources 带进去的独立运行时 + site-packages
function resolvePythonEnv() {
  if (app.isPackaged) {
    const resources = process.resourcesPath
    const userData = app.getPath('userData')
    return {
      exe: path.join(resources, 'python-runtime', 'python', 'bin', 'python3'),
      script: path.join(resources, 'python-src', 'ai_server.py'),
      cwd: path.join(resources, 'python-src'),
      env: {
        PYTHONPATH: path.join(resources, 'python-site'),
        // 内置 demucs 权重，首跑无需联网下载
        TORCH_HOME: path.join(resources, 'models'),
        PYTHONNOUSERSITE: '1',
        // Resources 只读，字节码与 numba JIT 缓存改写到用户数据目录
        PYTHONPYCACHEPREFIX: path.join(userData, 'pycache'),
        NUMBA_CACHE_DIR: path.join(userData, 'numba-cache'),
        KARAOKE_OUTPUT_DIR: path.join(userData, 'output'),
      },
    }
  }
  return {
    exe: getPythonExecutable(),
    script: AI_SERVER,
    cwd: PYTHON_DIR,
    env: {},
  }
}

let pythonSpawnCount = 0
let pythonLogStream = null
let appQuitting = false
let pythonLastExit = null
let pythonLastExe = null

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function listPortListeners() {
  try {
    const out = execFileSync(
      'lsof',
      ['-ti', `tcp:${PYTHON_PORT}`, '-sTCP:LISTEN'],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
    )
    return out
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
  } catch (err) {
    // 没有监听者时 lsof 退出码非 0；lsof 不存在也走这里，
    // 两种情况都当成「端口空闲」，行为跟没加这段逻辑时一致
    return []
  }
}

function processCommand(pid) {
  try {
    return execFileSync('ps', ['-p', String(pid), '-o', 'command='], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch (err) {
    return ''
  }
}

// 上一次运行遗留的 ai_server 会一直占着端口：新进程 bind 失败退出（exit 1），
// 而 waitForPythonServer 只探 /health —— 孤儿进程回个 200 就被判成「已就绪」，
// 界面从此一直跟旧代码对话，改了打分逻辑界面上一点变化都没有。
// 所以 spawn 之前先回收端口，且只回收命令行里确认是我们自己 ai_server.py 的
// 进程；别的应用正好用这个端口时不碰它，只把情况写进日志。
async function reclaimPythonPort() {
  const listeners = listPortListeners()
  if (listeners.length === 0) return true

  for (const pid of listeners) {
    const command = processCommand(pid)
    if (!command.includes('ai_server.py')) {
      console.warn(
        `[Electron] 端口 ${PYTHON_PORT} 被其它进程占用，未处理: pid=${pid} ${command}`
      )
      continue
    }
    console.log(`[Electron] 回收遗留的 AI 服务进程: pid=${pid}`)
    try {
      process.kill(Number(pid), 'SIGTERM')
    } catch (err) {
      console.error(`[Electron] 结束遗留进程失败 pid=${pid}: ${err.message}`)
    }
  }

  // SIGTERM 之后 uvicorn 还要收尾，等端口真正空出来再 spawn，
  // 否则新进程照样 bind 失败，白白烧掉一次重试机会
  for (let i = 0; i < 20 && listPortListeners().length > 0; i++) {
    await sleep(250)
  }
  return listPortListeners().length === 0
}

function schedulePythonRestart() {
  setTimeout(async () => {
    await reclaimPythonPort()
    startPythonServer()
  }, 2000)
}

function startPythonServer() {
  const { exe, script, cwd, env } = resolvePythonEnv()
  pythonSpawnCount++
  pythonLastExe = exe

  const logDir = path.join(app.getPath('userData'), 'logs')
  fs.mkdirSync(logDir, { recursive: true })
  const logPath = path.join(logDir, 'python.log')
  pythonLogStream = fs.createWriteStream(logPath, {
    flags: pythonSpawnCount === 1 ? 'w' : 'a',
  })
  pythonLogStream.write(
    `\n=== spawn #${pythonSpawnCount} ${new Date().toISOString()} exe=${exe} cwd=${cwd} ===\n`
  )

  console.log(`[Electron] Starting Python: ${exe} ${script} (log: ${logPath})`)

  pythonProcess = spawn(exe, [script, '--port', String(PYTHON_PORT)], {
    cwd,
    env: { ...process.env, ...env },
    stdio: ['pipe', 'pipe', 'pipe'],
  })

  const tee = (stream, level) => {
    stream.on('data', (data) => {
      const text = data.toString().trim()
      if (level === 'err') console.error(`[Python:ERR] ${text}`)
      else console.log(`[Python] ${text}`)
      pythonLogStream.write(data)
    })
  }
  tee(pythonProcess.stdout, 'out')
  tee(pythonProcess.stderr, 'err')

  pythonProcess.on('exit', (code) => {
    console.log(`[Electron] Python process exited with code ${code}`)
    pythonLastExit = code
    if (pythonLogStream) pythonLogStream.write(`=== exited code=${code} ===\n`)
    pythonProcess = null
    // 启动早期崩溃（端口被抢、瞬时失败等）自动重试，避免界面一直「AI 未连接」
    if (!appQuitting && code !== 0 && pythonSpawnCount < 3) {
      schedulePythonRestart()
    }
  })

  pythonProcess.on('error', (err) => {
    console.error(`[Electron] Failed to start Python: ${err.message}`)
    if (pythonLogStream) pythonLogStream.write(`=== spawn error: ${err.message} ===\n`)
    pythonProcess = null
    if (!appQuitting && pythonSpawnCount < 3) {
      schedulePythonRestart()
    }
  })
}

function waitForPythonServer(maxRetries = 30) {
  return new Promise((resolve, reject) => {
    let retries = 0

    const checkHealth = () => {
      const req = http.get(
        `http://127.0.0.1:${PYTHON_PORT}/health`,
        (res) => {
          if (res.statusCode === 200) {
            resolve()
          } else {
            retry()
          }
          res.resume()
        }
      )
      req.on('error', () => retry())
      req.setTimeout(1000, () => {
        req.destroy()
        retry()
      })
    }

    const retry = () => {
      retries++
      if (retries >= maxRetries) {
        reject(new Error('Python server failed to start within 15 seconds'))
        return
      }
      setTimeout(checkHealth, 500)
    }

    checkHealth()
  })
}

function killPythonServer() {
  if (pythonProcess) {
    console.log('[Electron] Killing Python process...')
    pythonProcess.kill('SIGTERM')
    pythonProcess = null
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 640,
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const isDev = !app.isPackaged

  if (isDev) {
    mainWindow.loadURL('http://localhost:5180')
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

ipcMain.handle('dialog:select-mp3', async () => {
  if (!mainWindow) return null
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择 MP3 文件',
    filters: [{ name: 'MP3 文件', extensions: ['mp3'] }],
    properties: ['openFile'],
  })
  if (result.canceled) return null
  return result.filePaths[0]
})

ipcMain.handle('dialog:select-lrc', async () => {
  if (!mainWindow) return null
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择 LRC 歌词文件',
    filters: [{ name: 'LRC 歌词', extensions: ['lrc'] }],
    properties: ['openFile'],
  })
  if (result.canceled) return null
  return result.filePaths[0]
})

ipcMain.handle('python:base-url', () => {
  return `http://127.0.0.1:${PYTHON_PORT}`
})

ipcMain.handle('python:status', async () => {
  return new Promise((resolve) => {
    const req = http.get(
      `http://127.0.0.1:${PYTHON_PORT}/health`,
      (res) => {
        resolve(res.statusCode === 200)
        res.resume()
      }
    )
    req.on('error', () => resolve(false))
    req.setTimeout(2000, () => {
      req.destroy()
      resolve(false)
    })
  })
})

// 「AI 未连接」时供界面直接展示原因，避免让用户自己翻日志
ipcMain.handle('python:diagnostics', async () => {
  const logPath = path.join(app.getPath('userData'), 'logs', 'python.log')
  let tail = ''
  try {
    tail = fs.readFileSync(logPath, 'utf8').split('\n').slice(-40).join('\n')
  } catch (err) {
    tail = `(读取日志失败: ${err.message})`
  }
  return {
    arch: process.arch,
    platform: process.platform,
    osRelease: os.release(),
    exe: pythonLastExe,
    spawnCount: pythonSpawnCount,
    lastExit: pythonLastExit,
    logPath,
    tail,
  }
})

// 系统代理（clash/v2ray 等）会把对 127.0.0.1 的请求也送进代理导致失败，
// 表现为界面一直「AI 未连接」；loopback 必须直连
app.commandLine.appendSwitch('proxy-bypass-list', '<-loopback>')

app.whenReady().then(async () => {
  protocol.handle('karaoke', (request) => {
    const url = new URL(request.url)
    let filePath = decodeURIComponent(url.pathname)
    // pathname 以 / 开头；如果原始路径也是绝对路径（/ 开头），
    // decodeURIComponent 后会得到 //xxx，需要去掉多余的 /
    if (filePath.startsWith('//')) {
      filePath = filePath.slice(1)
    }
    return net.fetch('file://' + filePath)
  })

  await reclaimPythonPort()
  startPythonServer()

  try {
    await waitForPythonServer()
    console.log('[Electron] Python server is ready')
  } catch (err) {
    console.error(`[Electron] ${err.message}`)
  }

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    appQuitting = true
    killPythonServer()
    app.quit()
  }
})

app.on('before-quit', () => {
  appQuitting = true
  killPythonServer()
})
