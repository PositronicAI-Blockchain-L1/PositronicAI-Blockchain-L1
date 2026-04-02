'use strict'

const { spawn } = require('child_process')
const { EventEmitter } = require('events')
const path = require('path')
const { app } = require('electron')
const { localRpc, resetLocalCircuit } = require('./rpc')

// ─── States ───────────────────────────────────────────────────────────────────
const STATE = {
  STOPPED: 'STOPPED',
  STARTING: 'STARTING',
  RUNNING: 'RUNNING',
  SYNCING: 'SYNCING',
  ERROR: 'ERROR',
  STOPPING: 'STOPPING'
}

// ─── Config ───────────────────────────────────────────────────────────────────
const READY_POLL_INTERVAL_MS = 1500     // poll interval while waiting for RPC
const READY_POLL_TIMEOUT_MS = 90_000   // max wait for node to become ready
const STOP_SIGKILL_DELAY_MS = 5000     // wait before SIGKILL after SIGTERM
const MAX_RESTARTS = 3                 // max auto-restarts within window
const RESTART_WINDOW_MS = 5 * 60_000  // 5 minutes

// ─── Python candidate executables ────────────────────────────────────────────
const PYTHON_CANDIDATES = ['python', 'python3', 'py']

// ─── NodeManager ──────────────────────────────────────────────────────────────
class NodeManager extends EventEmitter {
  constructor () {
    super()

    this._state = STATE.STOPPED
    this._error = null
    this._process = null
    this._startTime = null         // Date when process was spawned
    this._pythonExe = null         // cached working python exe

    this._readyTimer = null        // setInterval handle for ready polling
    this._readyDeadline = null     // Date when ready-check times out
    this._killTimer = null         // setTimeout handle for SIGKILL

    this._restartCount = 0
    this._restartWindowStart = null // Date of first restart in current window

    this._validatorMode = false    // remember for auto-restart
    this._settings = {}            // extra settings from renderer (maxPeers, logLevel, etc.)
  }

  // ── State helpers ──────────────────────────────────────────────────────────
  _setState (state, error = null) {
    this._state = state
    this._error = error
    this.emit('state-change', { state, error })
  }

  _log (line) {
    this.emit('log-line', line)
  }

  // ── Find a working Python executable ──────────────────────────────────────
  async _findPython () {
    if (this._pythonExe) return this._pythonExe

    for (const exe of PYTHON_CANDIDATES) {
      try {
        await new Promise((resolve, reject) => {
          const proc = spawn(exe, ['--version'], {
            stdio: 'ignore',
            timeout: 3000
          })
          proc.on('close', (code) => (code === 0 ? resolve() : reject()))
          proc.on('error', reject)
        })
        this._pythonExe = exe
        return exe
      } catch (_) {
        // try next
      }
    }

    throw new Error(
      `Python not found. Install Python 3.8+ and ensure it is in PATH. Tried: ${PYTHON_CANDIDATES.join(', ')}`
    )
  }

  // ── Start ──────────────────────────────────────────────────────────────────
  /**
   * Update extra settings that will be passed as env vars on next start.
   * @param {Object} settings  e.g. { maxPeers, mempoolMax, logLevel, aiEnabled }
   */
  setSettings (settings) {
    this._settings = { ...this._settings, ...settings }
  }

  /**
   * Spawn the Python node process.
   * @param {boolean} validatorMode
   */
  async start (validatorMode = false) {
    if (this._state === STATE.RUNNING || this._state === STATE.STARTING) {
      this._log(`[node_mgr] Node already ${this._state} — ignoring start request`)
      return
    }
    if (this._state === STATE.STOPPING) {
      throw new Error('Node is currently stopping — try again shortly')
    }

    this._validatorMode = validatorMode
    this._setState(STATE.STARTING)
    this._log(`[node_mgr] Starting node (validator=${validatorMode})…`)

    let spawnExe, spawnArgs, spawnCwd

    if (app.isPackaged) {
      // Packaged app: use the bundled headless node executable (PyInstaller)
      const exeName = process.platform === 'win32' ? 'PositronicNode.exe' : 'PositronicNode'
      const exeDir  = path.join(process.resourcesPath, 'node_entry.dist')
      spawnExe  = path.join(exeDir, exeName)
      spawnArgs = [validatorMode ? '--validator' : '--no-validator']
      spawnCwd  = exeDir
    } else {
      // Development: spawn Python module from project root
      let pythonExe
      try {
        pythonExe = await this._findPython()
      } catch (err) {
        this._setState(STATE.ERROR, err.message)
        throw err
      }
      spawnExe  = pythonExe
      spawnArgs = ['-m', 'positronic.node_entry', validatorMode ? '--validator' : '--no-validator']
      spawnCwd  = path.join(__dirname, '..', '..')
    }

    this._log(`[node_mgr] Spawning: ${spawnExe} ${spawnArgs.join(' ')}`)

    // Build environment with settings from renderer
    const spawnEnv = { ...process.env }
    const s = this._settings || {}
    if (s.maxPeers != null && s.maxPeers > 0) spawnEnv.POSITRONIC_MAX_PEERS = String(s.maxPeers)
    if (s.logLevel)                            spawnEnv.POSITRONIC_LOG_LEVEL = s.logLevel
    if (s.aiEnabled != null)                   spawnEnv.POSITRONIC_AI_ENABLED = s.aiEnabled ? '1' : '0'

    const proc = spawn(spawnExe, spawnArgs, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: spawnEnv,
      detached: false,
      cwd: spawnCwd,
    })

    this._process = proc
    this._startTime = new Date()

    // ── stdout — primary log stream ──
    proc.stdout.setEncoding('utf8')
    let stdoutBuf = ''
    proc.stdout.on('data', (chunk) => {
      stdoutBuf += chunk
      let newline
      while ((newline = stdoutBuf.indexOf('\n')) !== -1) {
        const line = stdoutBuf.slice(0, newline).trimEnd()
        stdoutBuf = stdoutBuf.slice(newline + 1)
        if (line) this._log(`[node] ${line}`)
      }
    })

    // ── stderr — silently consumed (same content as stdout) ──
    proc.stderr.setEncoding('utf8')
    proc.stderr.on('data', () => {})

    // ── Process exit ──
    proc.on('close', (code, signal) => {
      this._log(`[node_mgr] Process exited — code=${code} signal=${signal}`)
      this._clearReadyTimer()
      this._clearKillTimer()

      const wasRunning = this._state === STATE.RUNNING || this._state === STATE.SYNCING
      const wasStopping = this._state === STATE.STOPPING

      this._process = null

      if (wasStopping) {
        this._setState(STATE.STOPPED)
        return
      }

      // Unexpected exit — attempt auto-restart
      if (wasRunning || this._state === STATE.STARTING) {
        this._handleCrash(code, signal)
      }
    })

    proc.on('error', (err) => {
      this._log(`[node_mgr] Process error: ${err.message}`)
      this._clearReadyTimer()
      this._process = null
      this._setState(STATE.ERROR, err.message)
    })

    // Reset circuit-breaker so we can immediately poll
    resetLocalCircuit()

    // Start polling for readiness
    this._startReadyPolling()
  }

  // ── Ready polling ──────────────────────────────────────────────────────────
  _startReadyPolling () {
    this._clearReadyTimer()
    this._readyDeadline = Date.now() + READY_POLL_TIMEOUT_MS

    this._readyTimer = setInterval(async () => {
      if (this._state !== STATE.STARTING) {
        this._clearReadyTimer()
        return
      }

      // Timed out
      if (Date.now() > this._readyDeadline) {
        this._clearReadyTimer()
        this._log('[node_mgr] Timed out waiting for node RPC — check node logs')
        this._setState(STATE.ERROR, 'Node did not become ready within 90s')
        return
      }

      try {
        const result = await localRpc('eth_chainId', [])
        if (result !== null && result !== undefined) {
          this._clearReadyTimer()
          this._log(`[node_mgr] Node ready — chainId=${result}`)
          // Check if node is syncing
          try {
            const info = await localRpc('positronic_nodeInfo', [])
            if (info && info.syncing) {
              this._setState(STATE.SYNCING)
              this._startSyncPolling()
              return
            }
          } catch (_) {}
          this._setState(STATE.RUNNING)
        }
        // else: no response yet, keep polling
      } catch (_) {
        // keep polling
      }
    }, READY_POLL_INTERVAL_MS)
  }

  _startSyncPolling () {
    this._clearReadyTimer()
    this._readyTimer = setInterval(async () => {
      if (this._state !== STATE.SYNCING) {
        this._clearReadyTimer()
        return
      }
      try {
        const info = await localRpc('positronic_nodeInfo', [])
        if (info && !info.syncing) {
          this._clearReadyTimer()
          this._log('[node_mgr] Sync complete — node fully operational')
          this._setState(STATE.RUNNING)
        }
      } catch (_) {}
    }, 5000)
  }

  _clearReadyTimer () {
    if (this._readyTimer) {
      clearInterval(this._readyTimer)
      this._readyTimer = null
    }
  }

  _clearKillTimer () {
    if (this._killTimer) {
      clearTimeout(this._killTimer)
      this._killTimer = null
    }
  }

  // ── Crash handler / auto-restart ───────────────────────────────────────────
  _handleCrash (code, signal) {
    const now = Date.now()

    // Reset restart window after 5 minutes
    if (!this._restartWindowStart || now - this._restartWindowStart > RESTART_WINDOW_MS) {
      this._restartCount = 0
      this._restartWindowStart = now
    }

    if (this._restartCount >= MAX_RESTARTS) {
      const msg = `Node crashed ${MAX_RESTARTS} times within ${RESTART_WINDOW_MS / 60_000}m — giving up`
      this._log(`[node_mgr] ${msg}`)
      this._setState(STATE.ERROR, msg)
      return
    }

    this._restartCount++
    const msg = `Node crashed (code=${code}, signal=${signal}) — auto-restart ${this._restartCount}/${MAX_RESTARTS}`
    this._log(`[node_mgr] ${msg}`)
    this._setState(STATE.STOPPED)

    // Brief pause before restarting to avoid spin-loops
    setTimeout(() => {
      if (this._state === STATE.STOPPED) {
        this.start(this._validatorMode).catch((err) => {
          this._log(`[node_mgr] Restart failed: ${err.message}`)
          this._setState(STATE.ERROR, err.message)
        })
      }
    }, 1500)
  }

  // ── Stop ───────────────────────────────────────────────────────────────────
  /**
   * Gracefully stop the node process.
   * Sends SIGTERM, then SIGKILL after 5 seconds if still alive.
   * @returns {Promise<void>} resolves when process has exited
   */
  stop () {
    return new Promise((resolve) => {
      if (!this._process || this._state === STATE.STOPPED) {
        this._setState(STATE.STOPPED)
        resolve()
        return
      }

      this._clearReadyTimer()
      this._setState(STATE.STOPPING)
      this._log('[node_mgr] Sending SIGTERM…')

      const proc = this._process

      const onExit = () => {
        this._clearKillTimer()
        this._process = null
        this._setState(STATE.STOPPED)
        resolve()
      }

      proc.once('close', onExit)

      // Send SIGTERM (or taskkill on Windows since SIGTERM isn't supported)
      try {
        if (process.platform === 'win32') {
          spawn('taskkill', ['/pid', String(proc.pid), '/t', '/f'], {
            stdio: 'ignore',
            detached: true
          }).unref()
        } else {
          proc.kill('SIGTERM')
        }
      } catch (err) {
        this._log(`[node_mgr] SIGTERM error: ${err.message}`)
      }

      // SIGKILL fallback
      this._killTimer = setTimeout(() => {
        if (this._process) {
          this._log('[node_mgr] Process did not exit — sending SIGKILL')
          try {
            if (process.platform === 'win32') {
              spawn('taskkill', ['/pid', String(proc.pid), '/t', '/f'], {
                stdio: 'ignore',
                detached: true
              }).unref()
            } else {
              proc.kill('SIGKILL')
            }
          } catch (_) {}
        }
      }, STOP_SIGKILL_DELAY_MS)
    })
  }

  // ── Restart ────────────────────────────────────────────────────────────────
  async restart (validatorMode) {
    if (validatorMode === undefined) validatorMode = this._validatorMode
    await this.stop()
    await this.start(validatorMode)
  }

  // ── Status ─────────────────────────────────────────────────────────────────
  /**
   * @returns {{state: string, error: string|null, pid: number|null, uptimeSeconds: number}}
   */
  getStatus () {
    const uptimeSeconds = this._startTime
      ? Math.floor((Date.now() - this._startTime.getTime()) / 1000)
      : 0

    return {
      state: this._state,
      error: this._error,
      pid: this._process ? this._process.pid : null,
      uptimeSeconds
    }
  }
}

// ─── Singleton ────────────────────────────────────────────────────────────────
const nodeManager = new NodeManager()

module.exports = { nodeManager, STATE }
