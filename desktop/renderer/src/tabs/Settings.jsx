import React, { useState, useEffect } from 'react'
import {
  Settings, Shield, Cpu, RefreshCw,
  Save, RotateCcw, ExternalLink, Info,
  Monitor, Moon, Zap, Play, Square, FolderOpen,
  Copy, CheckCircle,
} from 'lucide-react'
import { COLORS, FONTS } from '../theme.js'
import Card    from '../components/Card.jsx'
import Button  from '../components/Button.jsx'
import InfoRow from '../components/InfoRow.jsx'
import Badge   from '../components/Badge.jsx'

const STORAGE_KEY = 'positronic_settings'

const DEFAULTS = {
  validatorMode:  false,
  maxPeers:       0,
  mempoolMax:     5000,
  aiEnabled:      true,
  logLevel:       'info',
  autoStart:      false,
  theme:          'dark',
}

function loadSettings() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') }
  } catch {
    return { ...DEFAULTS }
  }
}

function Toggle({ value, onChange, label, description, color }) {
  return (
    <div style={{
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'space-between',
      padding:        '10px 0',
      borderBottom:   `1px solid ${COLORS.separator}`,
    }}>
      <div>
        <div style={{ color: COLORS.text, fontSize: '13px', fontWeight: 500, marginBottom: '2px' }}>
          {label}
        </div>
        {description && (
          <div style={{ color: COLORS.textMuted, fontSize: '11px' }}>{description}</div>
        )}
      </div>
      <button
        onClick={() => onChange(!value)}
        style={{
          width:         '42px',
          height:        '22px',
          borderRadius:  '11px',
          border:        'none',
          cursor:        'pointer',
          background:    value ? (color ?? COLORS.accent) : COLORS.separator,
          position:      'relative',
          flexShrink:    0,
          transition:    'background 0.2s',
          boxShadow:     value ? `0 0 8px ${color ?? COLORS.accent}60` : 'none',
        }}
      >
        <span style={{
          position:   'absolute',
          top:        '3px',
          left:       value ? '22px' : '3px',
          width:      '16px',
          height:     '16px',
          borderRadius:'50%',
          background: '#fff',
          transition: 'left 0.2s',
        }} />
      </button>
    </div>
  )
}

function NumberField({ label, description, value, onChange, min, max, step = 1 }) {
  return (
    <div style={{
      display:      'flex',
      alignItems:   'center',
      justifyContent:'space-between',
      padding:      '10px 0',
      borderBottom: `1px solid ${COLORS.separator}`,
      gap:          '12px',
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ color: COLORS.text, fontSize: '13px', fontWeight: 500, marginBottom: '2px' }}>{label}</div>
        {description && <div style={{ color: COLORS.textMuted, fontSize: '11px' }}>{description}</div>}
      </div>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          width:        '90px',
          background:   COLORS.bgDark,
          border:       `1px solid ${COLORS.border}`,
          borderRadius: '6px',
          padding:      '5px 10px',
          color:        COLORS.text,
          fontSize:     '13px',
          fontFamily:   FONTS.mono,
          outline:      'none',
          textAlign:    'right',
        }}
      />
    </div>
  )
}

function TextField({ label, description, value, onChange, placeholder }) {
  return (
    <div style={{
      padding:      '10px 0',
      borderBottom: `1px solid ${COLORS.separator}`,
    }}>
      <div style={{ color: COLORS.text, fontSize: '13px', fontWeight: 500, marginBottom: '2px' }}>{label}</div>
      {description && <div style={{ color: COLORS.textMuted, fontSize: '11px', marginBottom: '6px' }}>{description}</div>}
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width:        '100%',
          boxSizing:    'border-box',
          background:   COLORS.bgDark,
          border:       `1px solid ${COLORS.border}`,
          borderRadius: '6px',
          padding:      '7px 10px',
          color:        COLORS.text,
          fontSize:     '13px',
          fontFamily:   FONTS.mono,
          outline:      'none',
        }}
      />
    </div>
  )
}

function SelectField({ label, description, value, onChange, options }) {
  return (
    <div style={{
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'space-between',
      padding:        '10px 0',
      borderBottom:   `1px solid ${COLORS.separator}`,
      gap:            '12px',
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ color: COLORS.text, fontSize: '13px', fontWeight: 500, marginBottom: '2px' }}>{label}</div>
        {description && <div style={{ color: COLORS.textMuted, fontSize: '11px' }}>{description}</div>}
      </div>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          background:   COLORS.bgDark,
          border:       `1px solid ${COLORS.border}`,
          borderRadius: '6px',
          padding:      '5px 10px',
          color:        COLORS.text,
          fontSize:     '13px',
          fontFamily:   FONTS.body,
          outline:      'none',
          cursor:       'pointer',
          minWidth:     '110px',
        }}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}

function CopyPathButton({ text }) {
  const [copied, setCopied] = useState(false)
  async function handle() {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1600) }
    catch {}
  }
  return (
    <button
      onClick={handle}
      title="Copy path"
      style={{
        background:'none', border:'none', cursor:'pointer',
        padding:'2px 6px', color: copied ? COLORS.success : COLORS.textMuted,
        display:'inline-flex', alignItems:'center', gap:'4px',
        fontSize:'11px', transition:'color 0.2s',
      }}
    >
      {copied ? <CheckCircle size={12} strokeWidth={1.5} /> : <Copy size={12} strokeWidth={1.5} />}
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}

function StatusDot({ state }) {
  const color = state === 'running' ? COLORS.success
              : state === 'starting' ? COLORS.warning
              : state === 'error'   ? COLORS.danger
              : COLORS.textMuted
  const label = state === 'running' ? 'Running'
              : state === 'starting' ? 'Starting…'
              : state === 'error'   ? 'Error'
              : 'Stopped'
  const pulse = state === 'running' || state === 'starting'
  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
      <span style={{
        display:'inline-block', width:'8px', height:'8px', borderRadius:'50%',
        background:color, flexShrink:0,
        boxShadow: pulse ? `0 0 6px ${color}` : 'none',
        animation: pulse ? 'pulse-dot 1.4s infinite' : 'none',
      }} />
      <span style={{ color, fontSize:'13px', fontWeight:600, fontFamily:FONTS.heading }}>{label}</span>
    </div>
  )
}

export default function SettingsTab({ nodeState: propNodeState }) {
  const nodeState = propNodeState ?? 'stopped'
  const [settings,  setSettings]  = useState(loadSettings)
  const [saved,     setSaved]     = useState(false)
  const [version,   setVersion]   = useState('')
  const [dataDir,   setDataDir]   = useState('')
  const [nodeBusy,  setNodeBusy]  = useState(false)
  const [nodeMsg,   setNodeMsg]   = useState('')

  useEffect(() => {
    window.api?.getVersion?.().then(v => v && setVersion(v)).catch(() => {})
    window.api?.getDataDir?.().then(d => d && setDataDir(d)).catch(() => {})
  }, [])

  function set(key, val) {
    setSettings(prev => ({ ...prev, [key]: val }))
    setSaved(false)
  }

  function handleSave() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch {}
  }

  function handleReset() {
    setSettings({ ...DEFAULTS })
    setSaved(false)
  }

  async function handleStart() {
    setNodeBusy(true); setNodeMsg('')
    try {
      // Pass all settings to the node process via environment variables
      const nodeSettings = {
        maxPeers:  settings.maxPeers,
        logLevel:  settings.logLevel,
        aiEnabled: settings.aiEnabled,
      }
      await window.api.nodeStart(settings.validatorMode, nodeSettings)
      setNodeMsg('Node started with saved settings.')
    } catch (e) { setNodeMsg(`Start failed: ${e?.message ?? String(e)}`) }
    setNodeBusy(false)
    setTimeout(() => setNodeMsg(''), 3000)
  }

  async function handleStop() {
    setNodeBusy(true); setNodeMsg('')
    try {
      await window.api.nodeStop()
      setNodeMsg('Node stop command sent.')
    } catch (e) { setNodeMsg(`Stop failed: ${e?.message ?? String(e)}`) }
    setNodeBusy(false)
    setTimeout(() => setNodeMsg(''), 3000)
  }

  function openDocs() {
    window.api?.openExternal?.('https://positronic.ai/docs/')
  }
  function openWebsite() {
    window.api?.openExternal?.('https://positronic.ai/')
  }
  function openWhitepaper() {
    window.api?.openExternal?.('https://positronic.ai/whitepaper.html')
  }

  const isRunning  = nodeState === 'running' || nodeState === 'syncing'
  const isStarting = nodeState === 'starting'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

      {/* ── App Info ── */}
      <Card title="App Information" icon={Info} accentColor={COLORS.accent}>
        <InfoRow label="Version"        value={version ? `v${version}` : '--'} />
        <InfoRow label="Chain ID"       value="420420" />
        <InfoRow label="Network"        value="Positronic Testnet" />
        <InfoRow label="Node Status"    value={<StatusDot state={nodeState} />} />
        {dataDir && (
          <div style={{ padding:'8px 0', borderBottom:`1px solid ${COLORS.separator}` }}>
            <div style={{ color:COLORS.textMuted, fontSize:'11px', marginBottom:'4px' }}>DATA DIRECTORY</div>
            <div style={{
              display:'flex', alignItems:'center', gap:'6px',
              background:COLORS.bgDark, border:`1px solid ${COLORS.separator}`,
              borderRadius:'6px', padding:'6px 10px',
            }}>
              <FolderOpen size={12} color={COLORS.textMuted} strokeWidth={1.5} style={{flexShrink:0}} />
              <span style={{ fontFamily:FONTS.mono, fontSize:'11px', color:COLORS.textDim, flex:1, wordBreak:'break-all' }}>
                {dataDir}
              </span>
              <CopyPathButton text={dataDir} />
            </div>
          </div>
        )}
      </Card>

      {/* ── Node Control ── */}
      <Card title="Node Control" icon={Play} accentColor={COLORS.blue}>
        <div style={{ display:'flex', gap:'10px', flexWrap:'wrap', alignItems:'center', marginBottom:'10px' }}>
          <Button
            variant="primary"
            icon={<Play size={13} strokeWidth={2} />}
            onClick={handleStart}
            loading={nodeBusy && !isRunning}
            disabled={isRunning || isStarting || nodeBusy}
          >
            Start Node
          </Button>
          <Button
            variant="danger"
            icon={<Square size={13} strokeWidth={2} />}
            onClick={handleStop}
            loading={nodeBusy && isRunning}
            disabled={!isRunning || nodeBusy}
          >
            Stop Node
          </Button>
          <StatusDot state={nodeState} />
        </div>
        {nodeMsg && (
          <div style={{ color:COLORS.textDim, fontSize:'12px', marginBottom:'6px' }}>{nodeMsg}</div>
        )}
        <div style={{ color:COLORS.textMuted, fontSize:'11px' }}>
          Validator mode and port settings apply on next node start.
        </div>
      </Card>

      {/* ── Node settings ── */}
      <Card title="Node Configuration" icon={Zap} accentColor={COLORS.accent}>
        <Toggle
          label="Validator Mode"
          description="Run as a validator. Requires 32+ ASF staked."
          value={settings.validatorMode}
          onChange={v => set('validatorMode', v)}
          color={COLORS.success}
        />
        <Toggle
          label="AI Consensus"
          description="Enable AI-validated block scoring."
          value={settings.aiEnabled}
          onChange={v => set('aiEnabled', v)}
          color={COLORS.purple}
        />
        <Toggle
          label="Auto-Start Node"
          description="Start the node automatically when the app launches."
          value={settings.autoStart}
          onChange={v => set('autoStart', v)}
        />
        <NumberField
          label="Max Peers"
          description="Maximum peer connections (0 = unlimited, matching server default)."
          value={settings.maxPeers}
          onChange={v => set('maxPeers', v)}
          min={0} max={1000}
        />
        <NumberField
          label="Mempool Max"
          description="Maximum pending transactions in mempool."
          value={settings.mempoolMax}
          onChange={v => set('mempoolMax', v)}
          min={100} max={50000} step={100}
        />
        <SelectField
          label="Log Level"
          description="Minimum log level to display."
          value={settings.logLevel}
          onChange={v => set('logLevel', v)}
          options={[
            { value: 'debug', label: 'Debug' },
            { value: 'info',  label: 'Info'  },
            { value: 'warn',  label: 'Warn'  },
            { value: 'error', label: 'Error' },
          ]}
        />
      </Card>

      {/* ── Save / Reset ── */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        <Button
          variant="primary"
          icon={<Save size={14} strokeWidth={2} />}
          onClick={handleSave}
        >
          Save Settings
        </Button>
        <Button
          variant="ghost"
          icon={<RotateCcw size={14} strokeWidth={2} />}
          onClick={handleReset}
        >
          Reset to Defaults
        </Button>
        {saved && (
          <span style={{ color: COLORS.success, fontSize: '12px', fontWeight: 600 }}>
            ✓ Settings saved
          </span>
        )}
      </div>

      {/* ── App info ── */}
      <Card title="About Positronic" icon={Info} accentColor={COLORS.purple}>
        <InfoRow label="App Version"   value={version ? `v${version}` : '—'} color={COLORS.accent} />
        <InfoRow label="Chain ID"      value="420420" />
        <InfoRow label="Network"       value="Positronic Testnet" color={COLORS.accent} />
        <InfoRow label="RPC Endpoint"  value="http://127.0.0.1:8545" />
        <InfoRow label="Consensus"     value="AI-PoS (Proof of Stake)" />
        <div style={{
          color:COLORS.textDim, fontSize:'12px', lineHeight:'1.7',
          margin:'12px 0', padding:'10px 0', borderTop:`1px solid ${COLORS.separator}`,
        }}>
          Positronic is a Layer-1 blockchain with neural AI-validated consensus,
          post-quantum cryptography, decentralized identity, on-chain governance,
          and a rich ecosystem of DePIN, RWA, AI agents, and ZKML.
        </div>
        <div style={{ display:'flex', gap:'10px', flexWrap:'wrap' }}>
          <Button variant="ghost" size="sm" icon={<ExternalLink size={12} strokeWidth={1.5} />} onClick={openWebsite}>
            Website
          </Button>
          <Button variant="ghost" size="sm" icon={<ExternalLink size={12} strokeWidth={1.5} />} onClick={openWhitepaper}>
            Whitepaper
          </Button>
          <Button variant="ghost" size="sm" icon={<ExternalLink size={12} strokeWidth={1.5} />} onClick={openDocs}>
            Docs
          </Button>
        </div>
      </Card>

    </div>
  )
}
