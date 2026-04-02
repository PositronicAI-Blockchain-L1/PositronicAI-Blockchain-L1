'use strict'

/**
 * Auto-updater module for Positronic Node.
 *
 * Uses electron-updater with a generic HTTP publish provider.
 * The update feed URL is configured via electron-builder.yml publish.url.
 *
 * Exports:
 *   initUpdater(mainWindow)  — called once from main/index.js after window ready
 *   checkForUpdates()        — called from ipc.js; returns {available, version, url}
 */

const { autoUpdater } = require('electron-updater')
const { app } = require('electron')

// ─── Module state ─────────────────────────────────────────────────────────────
let _mainWindow = null
let _latestInfo = null   // UpdateInfo from autoUpdater once an update is found
let _available = false

// ─── autoUpdater configuration ────────────────────────────────────────────────
// Do not automatically download — ask the user first via renderer UI
autoUpdater.autoDownload = false
// Do not silently install on quit — let the user trigger it
autoUpdater.autoInstallOnAppQuit = false
// Suppress the built-in logger noise in production; errors still surface via events
autoUpdater.logger = null

// ─── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Send an event to the renderer safely (window may have been closed).
 * @param {string} channel
 * @param {any}    payload
 */
function sendToRenderer (channel, payload) {
  if (_mainWindow && !_mainWindow.isDestroyed()) {
    try {
      _mainWindow.webContents.send(channel, payload)
    } catch (_) {
      // webContents may be gone
    }
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Attach all event listeners and schedule the initial update check.
 * Must be called once after the BrowserWindow is ready.
 *
 * @param {Electron.BrowserWindow} mainWindow
 */
function initUpdater (mainWindow) {
  _mainWindow = mainWindow

  // ── update-available ────────────────────────────────────────────────────────
  autoUpdater.on('update-available', (info) => {
    _available = true
    _latestInfo = info

    const payload = {
      version: info.version || null,
      releaseNotes: info.releaseNotes || null,
      releaseDate: info.releaseDate || null
    }

    sendToRenderer('update-available', payload)
  })

  // ── update-not-available ────────────────────────────────────────────────────
  autoUpdater.on('update-not-available', (_info) => {
    _available = false
    _latestInfo = null
  })

  // ── download-progress ───────────────────────────────────────────────────────
  autoUpdater.on('download-progress', (progress) => {
    // progress.percent is a float 0–100
    const percent = typeof progress.percent === 'number'
      ? Math.round(progress.percent)
      : 0

    sendToRenderer('update-progress', percent)
  })

  // ── update-downloaded ───────────────────────────────────────────────────────
  autoUpdater.on('update-downloaded', (info) => {
    const payload = {
      version: info.version || null,
      releaseNotes: info.releaseNotes || null,
      releaseDate: info.releaseDate || null
    }

    sendToRenderer('update-ready', payload)

    // Also emit on ipcMain so other main-process modules can react if needed
    try {
      const { ipcMain } = require('electron')
      ipcMain.emit('update-ready', null, payload)
    } catch (_) {}
  })

  // ── error ────────────────────────────────────────────────────────────────────
  autoUpdater.on('error', (err) => {
    // Never crash the app on update errors
    sendToRenderer('update-error', {
      message: err && err.message ? err.message : String(err)
    })
  })

  // ── Schedule first check 3 seconds after window is ready ────────────────────
  if (!app.isPackaged) {
    // Dev mode: skip actual network check — only log
    console.log('[updater] Dev mode — skipping auto-update check')
    return
  }

  setTimeout(() => {
    autoUpdater.checkForUpdates().catch((err) => {
      // Swallow — already handled by the error event above, but .catch() keeps
      // the rejected promise from causing an unhandledRejection in some envs.
      void err
    })
  }, 3000)
}

/**
 * Imperatively check for updates and return a structured result.
 * Called from ipc.js in response to the renderer's `checkUpdate` IPC request.
 *
 * @returns {Promise<{available: boolean, version: string|null, url: string|null, error?: string}>}
 */
async function checkForUpdates () {
  // Dev mode guard
  if (!app.isPackaged) {
    return { available: false, version: null, url: null }
  }

  try {
    const result = await autoUpdater.checkForUpdates()

    // result may be null if the check was throttled or no update is available
    if (!result || !result.updateInfo) {
      return { available: false, version: null, url: null }
    }

    const info = result.updateInfo
    const currentVersion = app.getVersion()
    const remoteVersion = info.version || null

    // Treat as "available" only when remote version differs from current
    const isAvailable = Boolean(
      remoteVersion &&
      remoteVersion !== currentVersion
    )

    // Build a direct download URL from the publish config if possible
    let downloadUrl = null
    try {
      const publishConf = autoUpdater.getFeedURL()
      if (publishConf && remoteVersion) {
        // Generic provider: files live at <url>/<artifactName>
        // We expose a human-friendly URL to the download page instead
        downloadUrl = `https://positronic-ai.network/download/`
      }
    } catch (_) {
      downloadUrl = 'https://positronic-ai.network/download/'
    }

    return {
      available: isAvailable,
      version: remoteVersion,
      url: isAvailable ? downloadUrl : null
    }
  } catch (err) {
    // Never throw out of this function — ipc.js has its own try/catch but
    // we want a clean {available: false} rather than an ugly stack trace
    return {
      available: false,
      version: null,
      url: null,
      error: err && err.message ? err.message : String(err)
    }
  }
}

/**
 * Download the available update. Progress is reported via 'update-progress' events.
 * @returns {Promise<void>}
 */
async function downloadUpdate () {
  if (!app.isPackaged) return
  await autoUpdater.downloadUpdate()
}

/**
 * Quit the application and install the downloaded update.
 * The app will restart automatically after install.
 */
function installUpdate () {
  // isSilent=false (show installer), isForceRunAfter=true (restart app after install)
  autoUpdater.quitAndInstall(false, true)
}

module.exports = { initUpdater, checkForUpdates, downloadUpdate, installUpdate }
