'use strict'

const {
  app,
  BrowserWindow,
  session,
  shell,
  ipcMain
} = require('electron')
const path = require('path')
const fs = require('fs')

const { initIpc } = require('./ipc')
const { initTray, destroyTray } = require('./tray')
const { initUpdater } = require('./updater')
const { nodeManager } = require('./node_mgr')

// ─── Constants ────────────────────────────────────────────────────────────────
const IS_DEV = !app.isPackaged
const WIN_STATE_FILE = path.join(app.getPath('userData'), 'window-state.json')
const ICON_PATH = path.join(__dirname, '..', 'build', 'icons', 'icon.png')
const MIN_WIDTH = 900
const MIN_HEIGHT = 640
const DEFAULT_WIDTH = 1100
const DEFAULT_HEIGHT = 750
const APP_VERSION = '0.3.2'

// ─── Single instance lock ─────────────────────────────────────────────────────
const gotTheLock = app.requestSingleInstanceLock()
if (!gotTheLock) {
  app.quit()
  process.exit(0)
}

// ─── Globals ──────────────────────────────────────────────────────────────────
let mainWindow = null

// ─── Window state persistence ─────────────────────────────────────────────────
function loadWindowState () {
  try {
    if (fs.existsSync(WIN_STATE_FILE)) {
      const raw = fs.readFileSync(WIN_STATE_FILE, 'utf8')
      const state = JSON.parse(raw)
      // Validate reasonable values
      if (
        typeof state.x === 'number' &&
        typeof state.y === 'number' &&
        typeof state.width === 'number' &&
        typeof state.height === 'number' &&
        state.width >= MIN_WIDTH &&
        state.height >= MIN_HEIGHT
      ) {
        return state
      }
    }
  } catch (_) {
    // Ignore corrupt state
  }
  return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT, x: undefined, y: undefined, maximized: false }
}

function saveWindowState (win) {
  try {
    const bounds = win.getBounds()
    const state = {
      x: bounds.x,
      y: bounds.y,
      width: bounds.width,
      height: bounds.height,
      maximized: win.isMaximized()
    }
    fs.writeFileSync(WIN_STATE_FILE, JSON.stringify(state, null, 2), 'utf8')
  } catch (_) {
    // Best-effort — don't crash on save failure
  }
}

// ─── CSP header ───────────────────────────────────────────────────────────────
function installCsp () {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          IS_DEV
            ? "default-src 'self' 'unsafe-inline' 'unsafe-eval' http://localhost:5173 ws://localhost:5173; connect-src 'self' http://127.0.0.1:8545 https://rpc.positronic-ai.network http://localhost:5173 ws://localhost:5173;"
            : "default-src 'self' 'unsafe-inline' 'unsafe-eval'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; connect-src 'self' http://127.0.0.1:8545 https://rpc.positronic-ai.network; img-src 'self' data:; style-src 'self' 'unsafe-inline';"
        ]
      }
    })
  })
}

// ─── Create main window ───────────────────────────────────────────────────────
async function createWindow () {
  const winState = loadWindowState()

  const windowOpts = {
    width: winState.width,
    height: winState.height,
    minWidth: MIN_WIDTH,
    minHeight: MIN_HEIGHT,
    title: `Positronic Node v${APP_VERSION}`,
    backgroundColor: '#0f172a',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webSecurity: false,
      spellcheck: false,
      devTools: true
    }
  }

  // Restore position only if it was explicitly saved
  if (typeof winState.x === 'number' && typeof winState.y === 'number') {
    windowOpts.x = winState.x
    windowOpts.y = winState.y
  }

  // Set icon if file exists
  if (fs.existsSync(ICON_PATH)) {
    windowOpts.icon = ICON_PATH
  }

  mainWindow = new BrowserWindow(windowOpts)

  if (winState.maximized) {
    mainWindow.maximize()
  }

  // ── Diagnostic logging (always active) ──
  const logFile = path.join(app.getPath('userData'), 'error.log')
  mainWindow.webContents.on('did-fail-load', (e, code, desc) => {
    fs.appendFileSync(logFile, `LOAD FAIL: ${code} ${desc}\n`)
  })
  mainWindow.webContents.on('render-process-gone', (e, details) => {
    fs.appendFileSync(logFile, `RENDERER GONE: ${JSON.stringify(details)}\n`)
  })

  // ── Load content ──
  if (IS_DEV) {
    await mainWindow.loadURL('http://localhost:5173')
    mainWindow.webContents.openDevTools()
  } else {
    const rendererPath = path.join(__dirname, '..', 'renderer', 'dist', 'index.html')
    await mainWindow.loadFile(rendererPath)
    // loadFile resolves after page + deferred scripts finish loading
    // Show now so the window is always visible (ready-to-show is a fallback)
    if (!mainWindow.isDestroyed()) {
      mainWindow.show()
      mainWindow.focus()
    }
  }

  // ready-to-show is the primary show trigger in dev mode;
  // in production the explicit show() above handles it
  mainWindow.once('ready-to-show', () => {
    if (!mainWindow.isVisible()) {
      mainWindow.show()
      mainWindow.focus()
    }
  })

  // ── Block external navigation ──
  mainWindow.webContents.on('will-navigate', (event, url) => {
    const parsedUrl = new URL(url)
    const isLocalhost = parsedUrl.hostname === 'localhost' || parsedUrl.hostname === '127.0.0.1'
    const isFileProtocol = parsedUrl.protocol === 'file:'
    if (!IS_DEV && !isLocalhost && !isFileProtocol) {
      event.preventDefault()
    }
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    // Open all new windows externally
    shell.openExternal(url)
    return { action: 'deny' }
  })

  // ── Hide to tray on close ──
  mainWindow.on('close', (event) => {
    if (!app.isQuitting) {
      event.preventDefault()
      mainWindow.hide()
    }
  })

  // ── Persist window state ──
  const persistState = () => saveWindowState(mainWindow)
  mainWindow.on('resize', persistState)
  mainWindow.on('move', persistState)

  // ── Wire node state forwarding ──
  // nodeManager emits events; forward to renderer
  nodeManager.on('state-change', ({ state, error }) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('node-state', { state, error })
    }
  })

  nodeManager.on('log-line', (line) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log-line', line)
    }
  })

  return mainWindow
}

// ─── App lifecycle ────────────────────────────────────────────────────────────
app.on('ready', async () => {
  installCsp()

  const win = await createWindow()

  initIpc(win)
  initTray(win, app)
  initUpdater(win)
})

app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    if (!mainWindow.isVisible()) mainWindow.show()
    mainWindow.focus()
  }
})

app.on('window-all-closed', () => {
  // On macOS it is common for applications to stay open until explicitly quit
  if (process.platform !== 'darwin') {
    app.isQuitting = true
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS re-create window if dock icon is clicked and no windows open
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  } else if (mainWindow) {
    mainWindow.show()
    mainWindow.focus()
  }
})

app.on('before-quit', () => {
  app.isQuitting = true
  destroyTray()
  // Give the node manager a chance to stop gracefully
  nodeManager.stop().catch(() => {})
})
