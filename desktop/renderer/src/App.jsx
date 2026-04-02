import React, {
  useState, useEffect, useRef, useCallback,
} from 'react'
import {
  LayoutDashboard, Wallet, Shield, Network as NetworkIcon,
  Layers, ScrollText, Settings, Zap, X,
  RefreshCw, Square, Play, AlertTriangle, Download, RotateCw, Check,
} from 'lucide-react'
import { COLORS, FONTS, SHADOWS } from './theme.js'
import {
  nodeStart, nodeStop, rpc, rpcRemote, checkUpdate, onLog, onNodeState,
  downloadUpdate, installUpdate, onUpdateAvailable, onUpdateProgress, onUpdateReady, onUpdateError,
} from './api.js'

import Dashboard  from './tabs/Dashboard.jsx'
import WalletTab  from './tabs/Wallet.jsx'
import Validator  from './tabs/Validator.jsx'
import Network    from './tabs/Network.jsx'
import Ecosystem  from './tabs/Ecosystem.jsx'
import Logs       from './tabs/Logs.jsx'
import SettingsTab from './tabs/Settings.jsx'

// ─── Constants ────────────────────────────────────────────────────────────────
const MAX_LOGS   = 500
const TABS = [
  { id: 'dashboard', label: 'Dashboard', Icon: LayoutDashboard },
  { id: 'wallet',    label: 'Wallet',    Icon: Wallet           },
  { id: 'validator', label: 'Validator', Icon: Shield           },
  { id: 'network',   label: 'Network',   Icon: NetworkIcon      },
  { id: 'ecosystem', label: 'Ecosystem', Icon: Layers           },
  { id: 'logs',      label: 'Logs',      Icon: ScrollText       },
  { id: 'settings',  label: 'Settings',  Icon: Settings         },
]

function statusColor(state) {
  switch (state) {
    case 'running':  return COLORS.success
    case 'syncing':  return COLORS.blue
    case 'starting': return COLORS.warning
    case 'error':    return COLORS.danger
    default:         return COLORS.textMuted
  }
}
function statusLabel(state) {
  switch (state) {
    case 'running':  return 'RUNNING'
    case 'syncing':  return 'SYNCING'
    case 'starting': return 'STARTING'
    case 'error':    return 'ERROR'
    default:         return 'STOPPED'
  }
}

// ─── Sidebar Tab Item ─────────────────────────────────────────────────────────
function SidebarItem({ id, label, Icon, active, onClick, badge }) {
  const [hovered, setHovered] = useState(false)
  const isActive = active === id

  return (
    <button
      onClick={() => onClick(id)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display:        'flex',
        alignItems:     'center',
        gap:            '10px',
        width:          '100%',
        padding:        '10px 14px',
        background:     isActive
          ? `linear-gradient(90deg, ${COLORS.accent}18, transparent)`
          : hovered ? `${COLORS.bgElevated}` : 'transparent',
        border:         'none',
        borderLeft:     isActive ? `3px solid ${COLORS.accent}` : '3px solid transparent',
        borderRadius:   '0 8px 8px 0',
        cursor:         'pointer',
        color:          isActive ? COLORS.accent : hovered ? COLORS.text : COLORS.textDim,
        fontSize:       '13px',
        fontWeight:     isActive ? 600 : 400,
        fontFamily:     FONTS.heading,
        textAlign:      'left',
        transition:     'all 0.15s ease',
        marginBottom:   '2px',
        position:       'relative',
      }}
    >
      <Icon
        size={16}
        strokeWidth={isActive ? 2 : 1.5}
        style={{ flexShrink: 0 }}
      />
      {label}
      {badge != null && badge > 0 && (
        <span style={{
          marginLeft:   'auto',
          background:   COLORS.danger,
          color:        '#fff',
          borderRadius: '10px',
          fontSize:     '10px',
          fontWeight:   700,
          padding:      '1px 6px',
          minWidth:     '18px',
          textAlign:    'center',
        }}>
          {badge > 99 ? '99+' : badge}
        </span>
      )}
    </button>
  )
}

// ─── Status Banner ────────────────────────────────────────────────────────────
function StatusBanner({ nodeState }) {
  const color = statusColor(nodeState)
  const label = statusLabel(nodeState)

  if (nodeState === 'running' || nodeState === 'syncing') return null // hide when operational

  return (
    <div style={{
      padding:        '6px 20px',
      background:     `${color}18`,
      borderBottom:   `1px solid ${color}40`,
      display:        'flex',
      alignItems:     'center',
      gap:            '8px',
      fontSize:       '12px',
      fontWeight:     500,
      color,
      flexShrink:     0,
    }}>
      <span style={{
        width:        '7px',
        height:       '7px',
        borderRadius: '50%',
        background:   color,
        flexShrink:   0,
        boxShadow:    nodeState === 'starting' ? `0 0 6px ${color}` : 'none',
        animation:    nodeState === 'starting' ? 'pulse-dot 1.2s infinite' : 'none',
      }} />
      Node is {label}
      {nodeState === 'stopped' && (
        <span style={{ color: COLORS.textMuted, marginLeft: '4px' }}>
          — Click "Start Node" to begin
        </span>
      )}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab,    setActiveTab]    = useState('dashboard')
  const [nodeState,    setNodeState]    = useState('stopped')
  const [version,      setVersion]      = useState('')
  const [logs,         setLogs]         = useState([])
  const [errorBadge,   setErrorBadge]   = useState(0)
  const [emergencyBanner, setEmergencyBanner] = useState(null)
  const [updateBanner, setUpdateBanner] = useState(null)
  const [updatePhase, setUpdatePhase]   = useState('idle')   // idle | downloading | ready | error
  const [updateProgress, setUpdateProgress] = useState(0)
  const [nodeActionBusy, setNodeActionBusy] = useState(false)
  const [dashData,     setDashData]     = useState(null)
  const [netData,      setNetData]      = useState(null)
  const [validatorData,setValidatorData]= useState(null)
  const [ecoData,      setEcoData]      = useState(null)
  const [miningHistory,setMiningHistory]= useState([])
  const [walletAddress,setWalletAddress]= useState('')
  const [walletKey,    setWalletKey]    = useState('')

  const prevHeightRef  = useRef(0)
  const ecoTimerRef    = useRef(null)
  const pollTimerRef   = useRef(null)
  const mountedRef     = useRef(true)

  // ── init: version + update check ──
  useEffect(() => {
    mountedRef.current = true

    async function init() {
      try {
        const v = await window.api.getVersion()
        if (mountedRef.current) setVersion(v)
      } catch {}

      try {
        const upd = await checkUpdate()
        if (mountedRef.current && upd?.available) {
          setUpdateBanner({ version: upd.version, url: upd.url })
        }
      } catch {}
    }
    init()

    return () => { mountedRef.current = false }
  }, [])

  // ── auto-update event listeners ──
  useEffect(() => {
    const unsubs = [
      onUpdateAvailable(info => {
        if (mountedRef.current) {
          setUpdateBanner({ version: info.version })
          setUpdatePhase('idle')
        }
      }),
      onUpdateProgress(percent => {
        if (mountedRef.current) setUpdateProgress(percent)
      }),
      onUpdateReady(info => {
        if (mountedRef.current) {
          setUpdatePhase('ready')
          setUpdateProgress(100)
        }
      }),
      onUpdateError(err => {
        if (mountedRef.current) setUpdatePhase('error')
      }),
    ]
    return () => unsubs.forEach(fn => fn && fn())
  }, [])

  // ── node state listener ──
  useEffect(() => {
    const unsub = onNodeState(payload => {
      if (mountedRef.current) {
        const s = (payload?.state ?? payload ?? '').toLowerCase()
        setNodeState(s)
      }
    })
    return unsub
  }, [])

  // ── log listener ──
  useEffect(() => {
    const unsub = onLog(entry => {
      if (!mountedRef.current) return
      setLogs(prev => {
        const next = [...prev, entry]
        return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next
      })
      if (entry?.level === 'error' || entry?.level === 'ERROR') {
        setErrorBadge(n => n + 1)
      }
    })
    return unsub
  }, [])

  // ── node status polling (3s) ──
  useEffect(() => {
    let timer

    async function pollStatus() {
      try {
        const s = await window.api.nodeStatus()
        if (mountedRef.current && s?.state) setNodeState(s.state.toLowerCase())
      } catch {}
    }
    pollStatus()
    timer = setInterval(pollStatus, 3000)
    return () => clearInterval(timer)
  }, [])

  // ── data polling (5s) ──
  const fetchDashboard = useCallback(async () => {
    try {
      // Use local RPC when node is running to get accurate local sync progress
      let info = null
      if (nodeState === 'running' || nodeState === 'syncing') {
        try {
          info = await rpc('positronic_nodeInfo', [])
        } catch {}
      }
      // Fall back to remote RPC if local unavailable
      if (!info) {
        info = await rpcRemote('positronic_nodeInfo', [])
      }
      if (!mountedRef.current) return
      setDashData(info)
      setNetData(info)

      // update mining history: count blocks since last poll
      const newHeight = info?.height ?? 0
      const delta = newHeight - prevHeightRef.current
      if (prevHeightRef.current > 0 && delta > 0) {
        setMiningHistory(prev => {
          const next = [...prev, delta]
          return next.length > 60 ? next.slice(next.length - 60) : next
        })
      }
      prevHeightRef.current = newHeight

      // emergency check
      try {
        const em = await rpcRemote('positronic_emergencyStatus', [])
        if (em?.state >= 2 && mountedRef.current) {
          setEmergencyBanner({ state: em.state, message: em.message })
        } else if (mountedRef.current) {
          setEmergencyBanner(null)
        }
      } catch {}
    } catch {}
  }, [nodeState])

  const fetchValidatorData = useCallback(async () => {
    if (!walletAddress) return
    try {
      const si = await rpcRemote('positronic_getStakingInfo', [walletAddress])
      if (mountedRef.current) setValidatorData(si)
    } catch {}
  }, [walletAddress])

  useEffect(() => {
    fetchDashboard()
    pollTimerRef.current = setInterval(fetchDashboard, 5000)
    return () => clearInterval(pollTimerRef.current)
  }, [fetchDashboard])

  useEffect(() => {
    if (walletAddress) fetchValidatorData()
    const t = setInterval(fetchValidatorData, 5000)
    return () => clearInterval(t)
  }, [fetchValidatorData])

  // ── ecosystem polling (30s) ──
  const fetchEcosystem = useCallback(async () => {
    const safe = async (method, params = []) => {
      try { return await rpcRemote(method, params) } catch { return null }
    }
    const [neural, consensus, did, gov, trust, bridge, depin, rwa, agents, mkt, zkml] =
      await Promise.all([
        safe('positronic_getNeuralStatus'),
        safe('positronic_getConsensusInfo'),
        safe('positronic_getDIDStats'),
        safe('positronic_getGovernanceStats'),
        safe('positronic_getTrustStats'),
        safe('positronic_getBridgeStats'),
        safe('positronic_getDePINStats'),
        safe('positronic_getRWAStats'),
        safe('positronic_getAgentStats'),
        safe('positronic_mktGetStats'),
        safe('positronic_getZKMLStats'),
      ])
    if (mountedRef.current) {
      setEcoData({ neural, consensus, did, gov, trust, bridge, depin, rwa, agents, mkt, zkml })
    }
  }, [])

  useEffect(() => {
    fetchEcosystem()
    ecoTimerRef.current = setInterval(fetchEcosystem, 30000)
    return () => clearInterval(ecoTimerRef.current)
  }, [fetchEcosystem])

  // ── node control ──
  async function handleNodeToggle() {
    setNodeActionBusy(true)
    try {
      if (nodeState === 'running' || nodeState === 'syncing') {
        await nodeStop()
      } else {
        const settings = (() => {
          try { return JSON.parse(localStorage.getItem('positronic_settings') ?? '{}') } catch { return {} }
        })()
        await nodeStart(settings.validatorMode ?? false, {
          maxPeers:  settings.maxPeers,
          logLevel:  settings.logLevel,
          aiEnabled: settings.aiEnabled,
        })
      }
    } finally {
      setNodeActionBusy(false)
    }
  }

  // ── clear error badge when navigating to Logs ──
  function handleTabChange(id) {
    setActiveTab(id)
    if (id === 'logs') setErrorBadge(0)
  }

  // ─── Layout ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      display:   'flex',
      width:     '100vw',
      height:    '100vh',
      background: COLORS.bg,
      overflow:  'hidden',
      fontFamily: FONTS.body,
    }}>
      <GlobalStyles />

      {/* ── Sidebar ── */}
      <aside style={{
        width:       '200px',
        flexShrink:  0,
        background:  COLORS.bgDark,
        borderRight: `1px solid ${COLORS.border}`,
        display:     'flex',
        flexDirection:'column',
        overflow:    'hidden',
      }}>
        {/* Logo */}
        <div style={{
          padding:      '18px 14px 14px',
          borderBottom: `1px solid ${COLORS.separator}`,
          display:      'flex',
          alignItems:   'center',
          gap:          '8px',
          flexShrink:   0,
        }}>
          <Zap size={20} color={COLORS.accent} strokeWidth={2} />
          <div>
            <div style={{
              color:       COLORS.text,
              fontSize:    '13px',
              fontWeight:  700,
              fontFamily:  FONTS.heading,
              letterSpacing:'0.04em',
            }}>
              POSITRONIC
            </div>
            <div style={{ color: COLORS.textMuted, fontSize: '10px', letterSpacing: '0.06em' }}>
              NODE v{version || '…'}
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{
          flex:       1,
          overflowY:  'auto',
          padding:    '10px 0',
        }}>
          {TABS.map(({ id, label, Icon }) => (
            <SidebarItem
              key={id}
              id={id}
              label={label}
              Icon={Icon}
              active={activeTab}
              onClick={handleTabChange}
              badge={id === 'logs' ? errorBadge : undefined}
            />
          ))}
        </nav>

        {/* Node toggle at bottom */}
        <div style={{
          padding:    '12px',
          borderTop:  `1px solid ${COLORS.separator}`,
          flexShrink: 0,
        }}>
          <button
            onClick={handleNodeToggle}
            disabled={nodeActionBusy || nodeState === 'starting'}
            style={{
              width:          '100%',
              display:        'flex',
              alignItems:     'center',
              justifyContent: 'center',
              gap:            '6px',
              padding:        '8px',
              borderRadius:   '8px',
              border:         'none',
              cursor:         nodeActionBusy || nodeState === 'starting' ? 'not-allowed' : 'pointer',
              opacity:        nodeActionBusy || nodeState === 'starting' ? 0.6 : 1,
              background:     (nodeState === 'running' || nodeState === 'syncing')
                ? `${COLORS.danger}22`
                : `linear-gradient(90deg, ${COLORS.blue}, ${COLORS.accent})`,
              color:          (nodeState === 'running' || nodeState === 'syncing') ? COLORS.danger : '#000',
              fontSize:       '12px',
              fontWeight:     700,
              fontFamily:     FONTS.heading,
              transition:     'all 0.15s',
            }}
          >
            {(nodeState === 'running' || nodeState === 'syncing')
              ? <><Square size={13} strokeWidth={2} /> Stop Node</>
              : <><Play  size={13} strokeWidth={2} /> Start Node</>
            }
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div style={{
        flex:          1,
        display:       'flex',
        flexDirection: 'column',
        overflow:      'hidden',
        minWidth:      0,
      }}>
        {/* Header */}
        <header style={{
          background:   COLORS.bgDark,
          borderBottom: `1px solid ${COLORS.border}`,
          padding:      '0 20px',
          height:       '48px',
          display:      'flex',
          alignItems:   'center',
          justifyContent:'space-between',
          flexShrink:   0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{
              color:       COLORS.text,
              fontSize:    '14px',
              fontWeight:  600,
              fontFamily:  FONTS.heading,
              letterSpacing:'0.06em',
            }}>
              {TABS.find(t => t.id === activeTab)?.label ?? ''}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Network status */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{
                width:        '8px',
                height:       '8px',
                borderRadius: '50%',
                background:   statusColor(nodeState),
                flexShrink:   0,
                boxShadow:    nodeState === 'running' ? `0 0 8px ${COLORS.success}` : 'none',
                animation:    nodeState === 'starting' ? 'pulse-dot 1.2s infinite' : 'none',
              }} />
              <span style={{
                color:        statusColor(nodeState),
                fontSize:     '11px',
                fontWeight:   600,
                letterSpacing:'0.06em',
                fontFamily:   FONTS.heading,
              }}>
                {statusLabel(nodeState)}
              </span>
            </div>

            <span style={{
              color:     COLORS.textMuted,
              fontSize:  '11px',
              background: COLORS.bgCard,
              padding:   '3px 8px',
              borderRadius:'4px',
              border:    `1px solid ${COLORS.border}`,
              fontFamily: FONTS.mono,
            }}>
              Chain 420420
            </span>
          </div>
        </header>

        {/* Banners */}
        {emergencyBanner && (
          <div style={{
            padding:      '8px 20px',
            background:   `${emergencyBanner.state >= 3 ? COLORS.danger : COLORS.warning}22`,
            borderBottom: `1px solid ${emergencyBanner.state >= 3 ? COLORS.danger : COLORS.warning}60`,
            display:      'flex',
            alignItems:   'center',
            gap:          '8px',
            flexShrink:   0,
          }}>
            <AlertTriangle size={15} color={emergencyBanner.state >= 3 ? COLORS.danger : COLORS.warning} strokeWidth={2} />
            <span style={{
              color:     emergencyBanner.state >= 3 ? COLORS.danger : COLORS.warning,
              fontSize:  '12px',
              fontWeight:600,
            }}>
              EMERGENCY STATE {emergencyBanner.state}
            </span>
            <span style={{ color: COLORS.textDim, fontSize: '12px' }}>
              {emergencyBanner.message ?? 'Network is in emergency mode.'}
            </span>
          </div>
        )}

        {updateBanner && (
          <div style={{
            padding:      '8px 20px',
            background:   updatePhase === 'ready' ? `${COLORS.success}18` : `${COLORS.blue}18`,
            borderBottom: `1px solid ${updatePhase === 'ready' ? COLORS.success : COLORS.blue}40`,
            display:      'flex',
            alignItems:   'center',
            gap:          '10px',
            flexShrink:   0,
          }}>
            {updatePhase === 'ready'
              ? <Check size={14} color={COLORS.success} strokeWidth={2.5} />
              : updatePhase === 'downloading'
                ? <RotateCw size={14} color={COLORS.blue} strokeWidth={2} style={{ animation: 'spin 1s linear infinite' }} />
                : <Download size={14} color={COLORS.blue} strokeWidth={2} />
            }
            <span style={{
              color: updatePhase === 'ready' ? COLORS.success : COLORS.blue,
              fontSize: '12px', fontWeight: 600,
            }}>
              {updatePhase === 'ready'
                ? `v${updateBanner.version} ready to install`
                : updatePhase === 'downloading'
                  ? `Downloading v${updateBanner.version}... ${updateProgress}%`
                  : `Update available: v${updateBanner.version}`
              }
            </span>

            {/* Progress bar during download */}
            {updatePhase === 'downloading' && (
              <div style={{
                flex: 1, maxWidth: '200px', height: '4px',
                background: `${COLORS.blue}30`, borderRadius: '2px', overflow: 'hidden',
              }}>
                <div style={{
                  width: `${updateProgress}%`, height: '100%',
                  background: COLORS.blue, borderRadius: '2px',
                  transition: 'width 0.3s ease',
                }} />
              </div>
            )}

            {/* Action buttons */}
            {updatePhase === 'idle' && (
              <button
                onClick={async () => {
                  setUpdatePhase('downloading')
                  setUpdateProgress(0)
                  try { await downloadUpdate() }
                  catch { setUpdatePhase('error') }
                }}
                style={{
                  padding: '3px 12px', fontSize: '11px', fontWeight: 600,
                  background: COLORS.blue, color: '#fff', border: 'none',
                  borderRadius: '4px', cursor: 'pointer',
                }}
              >
                Download
              </button>
            )}
            {updatePhase === 'ready' && (
              <button
                onClick={() => installUpdate()}
                style={{
                  padding: '3px 12px', fontSize: '11px', fontWeight: 600,
                  background: COLORS.success, color: '#fff', border: 'none',
                  borderRadius: '4px', cursor: 'pointer',
                }}
              >
                Install & Restart
              </button>
            )}
            {updatePhase === 'error' && (
              <span style={{ color: COLORS.danger, fontSize: '11px' }}>
                Download failed —
                <button
                  onClick={() => setUpdatePhase('idle')}
                  style={{
                    background: 'none', border: 'none', color: COLORS.blue,
                    cursor: 'pointer', fontSize: '11px', textDecoration: 'underline',
                    padding: '0 4px',
                  }}
                >retry</button>
              </span>
            )}

            <button
              onClick={() => setUpdateBanner(null)}
              style={{
                marginLeft: 'auto', background: 'none', border: 'none',
                cursor: 'pointer', color: COLORS.textMuted,
                display: 'flex', alignItems: 'center',
              }}
            >
              <X size={14} />
            </button>
          </div>
        )}

        <StatusBanner nodeState={nodeState} />

        {/* Tab content */}
        <main style={{
          flex:      1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding:   '20px',
          minWidth:  0,
        }}>
          {activeTab === 'dashboard' && (
            <Dashboard
              data={dashData}
              nodeState={nodeState}
              miningHistory={miningHistory}
              validatorData={validatorData}
            />
          )}
          {activeTab === 'wallet' && (
            <WalletTab
              onAddressChange={setWalletAddress}
              onKeyChange={setWalletKey}
            />
          )}
          {activeTab === 'validator' && (
            <Validator
              validatorData={validatorData}
              address={walletAddress}
              secretKey={walletKey}
            />
          )}
          {activeTab === 'network' && (
            <Network netData={netData} />
          )}
          {activeTab === 'ecosystem' && (
            <Ecosystem ecoData={ecoData} />
          )}
          {activeTab === 'logs' && (
            <Logs logs={logs} />
          )}
          {activeTab === 'settings' && (
            <SettingsTab nodeState={nodeState} />
          )}
        </main>
      </div>
    </div>
  )
}

// ── inject global animation keyframes once ────────────────────────────────────
function GlobalStyles() {
  useEffect(() => {
    if (document.getElementById('global-anim-style')) return
    const s = document.createElement('style')
    s.id = 'global-anim-style'
    s.textContent = `
      @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.35; }
      }
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    `
    document.head.appendChild(s)
  }, [])
  return null
}
