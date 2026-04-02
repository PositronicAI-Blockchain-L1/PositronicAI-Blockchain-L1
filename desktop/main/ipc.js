'use strict'

const { ipcMain, app, shell } = require('electron')
const path = require('path')
const fs = require('fs')
const { localRpc, remoteRpc } = require('./rpc')
const { nodeManager } = require('./node_mgr')
const { encrypt, decrypt } = require('./crypto')
const { checkForUpdates, downloadUpdate, installUpdate } = require('./updater')

// Allowed URL schemes for openExternal
const ALLOWED_SCHEMES = ['https:']

// Validate a URL is safe to open externally
function isAllowedExternalUrl (url) {
  try {
    const parsed = new URL(url)
    return ALLOWED_SCHEMES.includes(parsed.protocol)
  } catch (_) {
    return false
  }
}

/**
 * Register all ipcMain handlers.
 * Called once from main/index.js after the window is created.
 * @param {Electron.BrowserWindow} mainWindow
 */
function initIpc (mainWindow) { // eslint-disable-line no-unused-vars

  // ── RPC: local node ──────────────────────────────────────────────────────
  ipcMain.handle('rpc', async (_event, method, params = []) => {
    try {
      return await localRpc(method, params)
    } catch (err) {
      return null
    }
  })

  // ── RPC: remote fallback ─────────────────────────────────────────────────
  ipcMain.handle('rpcRemote', async (_event, method, params = []) => {
    try {
      return await remoteRpc(method, params)
    } catch (err) {
      return null
    }
  })

  // ── Node lifecycle ───────────────────────────────────────────────────────
  ipcMain.handle('nodeStart', async (_event, validatorMode = false, settings = {}) => {
    try {
      if (settings && typeof settings === 'object') {
        nodeManager.setSettings(settings)
      }
      await nodeManager.start(validatorMode)
      return { success: true }
    } catch (err) {
      return { success: false, error: err.message || String(err) }
    }
  })

  ipcMain.handle('nodeStop', async (_event) => {
    try {
      await nodeManager.stop()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.message || String(err) }
    }
  })

  ipcMain.handle('nodeStatus', (_event) => {
    try {
      return nodeManager.getStatus()
    } catch (err) {
      return { state: 'ERROR', error: err.message || String(err), pid: null, uptimeSeconds: 0 }
    }
  })

  // ── Wallet crypto ────────────────────────────────────────────────────────
  ipcMain.handle('walletEncrypt', async (_event, plaintext, password) => {
    try {
      if (typeof plaintext !== 'string' || typeof password !== 'string') {
        return { encrypted: null, error: 'Invalid arguments' }
      }
      const encrypted = await encrypt(plaintext, password)
      return { encrypted }
    } catch (err) {
      return { encrypted: null, error: err.message || String(err) }
    }
  })

  ipcMain.handle('walletDecrypt', async (_event, encrypted, password) => {
    try {
      if (typeof encrypted !== 'string' || typeof password !== 'string') {
        return { data: null, error: 'Invalid arguments' }
      }
      const data = await decrypt(encrypted, password)
      return { data }
    } catch (err) {
      return { data: null, error: err.message || String(err) }
    }
  })

  // ── App metadata ─────────────────────────────────────────────────────────
  ipcMain.handle('getVersion', () => {
    try {
      const pkg = require('../package.json')
      return pkg.version || '0.3.1'
    } catch (_) {
      return '0.3.1'
    }
  })

  ipcMain.handle('getDataDir', () => {
    return app.getPath('userData')
  })

  ipcMain.handle('checkUpdate', async () => {
    try {
      return await checkForUpdates()
    } catch (err) {
      return { available: false, error: err.message || String(err) }
    }
  })

  ipcMain.handle('downloadUpdate', async () => {
    try {
      await downloadUpdate()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.message || String(err) }
    }
  })

  ipcMain.handle('installUpdate', () => {
    try {
      installUpdate()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.message || String(err) }
    }
  })

  // ── Renderer error logging ───────────────────────────────────────────────
  ipcMain.handle('logError', (_event, msg) => {
    try {
      const logFile = path.join(app.getPath('userData'), 'renderer-error.log')
      fs.appendFileSync(logFile, `[${new Date().toISOString()}] ${msg}\n---\n`)
    } catch (_) {}
  })

  ipcMain.handle('openExternal', async (_event, url) => {
    if (typeof url !== 'string' || !isAllowedExternalUrl(url)) {
      throw new Error(`URL not allowed: ${url}`)
    }
    await shell.openExternal(url)
  })
}

module.exports = { initIpc }
