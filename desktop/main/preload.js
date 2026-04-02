'use strict'

const { contextBridge, ipcRenderer } = require('electron')

// ─── Helper: typed invoke wrapper ─────────────────────────────────────────────
function invoke (channel, ...args) {
  return ipcRenderer.invoke(channel, ...args)
}

// ─── API surface exposed to renderer ─────────────────────────────────────────
contextBridge.exposeInMainWorld('api', {

  // ── JSON-RPC ──
  /**
   * Call a method on the local node (127.0.0.1:8545).
   * @param {string} method  RPC method name
   * @param {Array}  params  RPC parameters
   * @returns {Promise<any>}  result field from JSON-RPC response
   */
  rpc (method, params = []) {
    return invoke('rpc', method, params)
  },

  /**
   * Call a method on the remote fallback RPC.
   * @param {string} method
   * @param {Array}  params
   * @returns {Promise<any>}
   */
  rpcRemote (method, params = []) {
    return invoke('rpcRemote', method, params)
  },

  // ── Node lifecycle ──
  /**
   * Start the local Python node.
   * @param {boolean} validatorMode  true to start in validator mode
   * @returns {Promise<{success: boolean, error?: string}>}
   */
  nodeStart (validatorMode = false, settings = {}) {
    return invoke('nodeStart', validatorMode, settings)
  },

  /**
   * Stop the local Python node.
   * @returns {Promise<{success: boolean}>}
   */
  nodeStop () {
    return invoke('nodeStop')
  },

  /**
   * Query current node status.
   * @returns {Promise<{state: string, error: string|null, pid: number|null, uptime: number}>}
   */
  nodeStatus () {
    return invoke('nodeStatus')
  },

  // ── Wallet crypto ──
  /**
   * Encrypt a plaintext string (e.g. wallet JSON) with a password.
   * @param {string} plaintext
   * @param {string} password
   * @returns {Promise<{encrypted: string}>}
   */
  walletEncrypt (plaintext, password) {
    return invoke('walletEncrypt', plaintext, password)
  },

  /**
   * Decrypt an encrypted string with a password.
   * @param {string} encrypted
   * @param {string} password
   * @returns {Promise<{data: string, error?: string}>}
   */
  walletDecrypt (encrypted, password) {
    return invoke('walletDecrypt', encrypted, password)
  },

  // ── App metadata ──
  /**
   * Get the current application version.
   * @returns {Promise<string>}
   */
  getVersion () {
    return invoke('getVersion')
  },

  /**
   * Get the user data directory path.
   * @returns {Promise<string>}
   */
  getDataDir () {
    return invoke('getDataDir')
  },

  /**
   * Check for available updates.
   * @returns {Promise<{available: boolean, version?: string, url?: string}>}
   */
  checkUpdate () {
    return invoke('checkUpdate')
  },

  /**
   * Download the available update. Progress events are emitted via onUpdateProgress.
   * @returns {Promise<{success: boolean, error?: string}>}
   */
  downloadUpdate () {
    return invoke('downloadUpdate')
  },

  /**
   * Install the downloaded update and restart the app.
   * @returns {Promise<{success: boolean, error?: string}>}
   */
  installUpdate () {
    return invoke('installUpdate')
  },

  /**
   * Open a URL in the system default browser.
   * Only https:// URLs are allowed.
   * @param {string} url
   * @returns {Promise<void>}
   */
  openExternal (url) {
    return invoke('openExternal', url)
  },

  // ── Event subscriptions ──
  /**
   * Subscribe to streaming log lines from the Python node process.
   * @param {Function} callback  called with (event, line: string)
   * @returns {Function}  call to unsubscribe
   */
  onLog (callback) {
    const handler = (event, line) => callback(line)
    ipcRenderer.on('log-line', handler)
    return () => ipcRenderer.removeListener('log-line', handler)
  },

  /**
   * Unsubscribe a previously registered log callback.
   * @param {Function} callback
   */
  offLog (callback) {
    ipcRenderer.removeListener('log-line', callback)
  },

  /**
   * Subscribe to node state change events.
   * @param {Function} callback  called with (event, {state, error})
   * @returns {Function}  call to unsubscribe
   */
  onNodeState (callback) {
    const handler = (event, payload) => callback(payload)
    ipcRenderer.on('node-state', handler)
    return () => ipcRenderer.removeListener('node-state', handler)
  },

  /**
   * Unsubscribe a previously registered node state callback.
   * @param {Function} callback
   */
  offNodeState (callback) {
    ipcRenderer.removeListener('node-state', callback)
  },

  /**
   * Subscribe to update-available events.
   * @param {Function} callback  called with {version, releaseNotes, releaseDate}
   * @returns {Function} unsubscribe
   */
  onUpdateAvailable (callback) {
    const handler = (_event, payload) => callback(payload)
    ipcRenderer.on('update-available', handler)
    return () => ipcRenderer.removeListener('update-available', handler)
  },

  /**
   * Subscribe to download progress events.
   * @param {Function} callback  called with percent (0-100)
   * @returns {Function} unsubscribe
   */
  onUpdateProgress (callback) {
    const handler = (_event, percent) => callback(percent)
    ipcRenderer.on('update-progress', handler)
    return () => ipcRenderer.removeListener('update-progress', handler)
  },

  /**
   * Subscribe to update-ready (download complete) events.
   * @param {Function} callback  called with {version, releaseNotes, releaseDate}
   * @returns {Function} unsubscribe
   */
  onUpdateReady (callback) {
    const handler = (_event, payload) => callback(payload)
    ipcRenderer.on('update-ready', handler)
    return () => ipcRenderer.removeListener('update-ready', handler)
  },

  /**
   * Subscribe to update error events.
   * @param {Function} callback  called with {message}
   * @returns {Function} unsubscribe
   */
  onUpdateError (callback) {
    const handler = (_event, payload) => callback(payload)
    ipcRenderer.on('update-error', handler)
    return () => ipcRenderer.removeListener('update-error', handler)
  },

  /**
   * Log a renderer-side error to renderer-error.log in userData.
   * @param {string} msg
   */
  logError (msg) {
    return invoke('logError', String(msg))
  }
})
