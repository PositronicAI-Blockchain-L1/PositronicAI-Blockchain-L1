import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

// ── Dev mock: stub window.api when running in plain browser (not Electron) ──
if (!window.api) {
  const noop = () => Promise.resolve(null)
  const unsub = () => () => {}
  window.api = {
    rpc:          noop,
    rpcRemote:    noop,
    nodeStart:    noop,
    nodeStop:     noop,
    nodeStatus:   () => Promise.resolve({ state: 'stopped' }),
    walletEncrypt: noop,
    walletDecrypt: noop,
    getVersion:   () => Promise.resolve('0.3.1-dev'),
    getDataDir:   () => Promise.resolve('~/.positronic'),
    checkUpdate:  noop,
    downloadUpdate: noop,
    installUpdate:  noop,
    openExternal: noop,
    onLog:        unsub,
    onNodeState:  unsub,
    onUpdateAvailable: unsub,
    onUpdateProgress:  unsub,
    onUpdateReady:     unsub,
    onUpdateError:     unsub,
    logError:     noop,
  }
}

// ── Inject spin keyframe for update spinner ──
const style = document.createElement('style')
style.textContent = '@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }'
document.head.appendChild(style)

// ── Capture uncaught errors and write to error.log via IPC ──
window.addEventListener('error', (event) => {
  const msg = `Uncaught error: ${event.message} @ ${event.filename}:${event.lineno}:${event.colno}\n${event?.error?.stack || ''}`
  window.api.logError?.(msg).catch(() => {})
})
window.addEventListener('unhandledrejection', (event) => {
  const msg = `Unhandled rejection: ${event.reason?.message || String(event.reason)}\n${event.reason?.stack || ''}`
  window.api.logError?.(msg).catch(() => {})
})

// ── Error Boundary: shows error text instead of blank screen ──
class ErrorBoundary extends React.Component {
  constructor (props) {
    super(props)
    this.state = { error: null, stack: null }
  }

  static getDerivedStateFromError (error) {
    return { error: error?.message || String(error), stack: error?.stack || '' }
  }

  componentDidCatch (error, info) {
    const msg = `React render error: ${error?.message}\nStack: ${error?.stack}\nComponent: ${info?.componentStack}`
    window.api.logError?.(msg).catch(() => {})
  }

  render () {
    if (this.state.error) {
      return (
        <div style={{
          background:  '#0f0000',
          color:       '#ff6666',
          padding:     '24px',
          fontFamily:  'monospace',
          fontSize:    '13px',
          height:      '100vh',
          overflow:    'auto',
          whiteSpace:  'pre-wrap',
          wordBreak:   'break-word',
        }}>
          {'React Error — see %APPDATA%\\Positronic Node\\renderer-error.log\n\n'}
          {this.state.error}
          {'\n\n'}
          {this.state.stack}
        </div>
      )
    }
    return this.props.children
  }
}

const root = createRoot(document.getElementById('root'))
root.render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
)
