import { useState, useEffect, useRef, useCallback } from 'react'

// ─── Core IPC wrappers ────────────────────────────────────────────────────────

export async function rpc(method, params = []) {
  return window.api.rpc(method, params)
}

export async function rpcRemote(method, params = []) {
  return window.api.rpcRemote(method, params)
}

export async function nodeStart(validatorMode = false, settings = {}) {
  return window.api.nodeStart(validatorMode, settings)
}

export async function nodeStop() {
  return window.api.nodeStop()
}

export async function nodeStatus() {
  return window.api.nodeStatus()
}

export async function walletEncrypt(plaintext, password) {
  return window.api.walletEncrypt(plaintext, password)
}

export async function walletDecrypt(encrypted, password) {
  return window.api.walletDecrypt(encrypted, password)
}

export async function getVersion() {
  return window.api.getVersion()
}

export async function checkUpdate() {
  return window.api.checkUpdate()
}

export function openExternal(url) {
  return window.api.openExternal(url)
}

export function onLog(cb) {
  return window.api.onLog(cb)
}

export function onNodeState(cb) {
  return window.api.onNodeState(cb)
}

// ── Auto-update API ──

export async function downloadUpdate() {
  return window.api.downloadUpdate()
}

export async function installUpdate() {
  return window.api.installUpdate()
}

export function onUpdateAvailable(cb) {
  return window.api.onUpdateAvailable(cb)
}

export function onUpdateProgress(cb) {
  return window.api.onUpdateProgress(cb)
}

export function onUpdateReady(cb) {
  return window.api.onUpdateReady(cb)
}

export function onUpdateError(cb) {
  return window.api.onUpdateError(cb)
}

// ─── High-level API object ────────────────────────────────────────────────────

/** Parse a hex string or integer value into a floating-point ASF amount */
function toASF(hexOrInt) {
  if (hexOrInt == null) return 0
  try {
    let big
    if (typeof hexOrInt === 'bigint') {
      big = hexOrInt
    } else if (typeof hexOrInt === 'string' && hexOrInt.startsWith('0x')) {
      big = BigInt(hexOrInt)
    } else if (typeof hexOrInt === 'string' && /^\d+$/.test(hexOrInt)) {
      big = BigInt(hexOrInt)
    } else if (typeof hexOrInt === 'number') {
      big = BigInt(Math.floor(hexOrInt))
    } else {
      return 0
    }
    return Number(big) / 1e18
  } catch {
    return 0
  }
}

export const api = {
  /** Fetch core dashboard data via remote RPC */
  async dashboard() {
    const info = await window.api.rpcRemote('positronic_nodeInfo', [])
    if (!info) return null
    const consensus = info.consensus || {}
    const network   = info.network   || {}
    const ai        = info.ai        || {}
    const state     = info.state     || {}
    return {
      online:           (network.peer_count || info.peers || 0) > 0,
      height:           info.height || 0,
      peers:            network.peer_count || info.peers || 0,
      maxPeers:         network.max_peers ?? 25,
      synced:           info.synced || false,
      chainId:          info.chain_id || 420420,
      networkType:      info.network_type || 'testnet',
      isValidator:      consensus.is_validator || false,
      totalStaked:      toASF(consensus.total_staked || 0),
      activeValidators: consensus.active_validators || 0,
      epoch:            consensus.epoch || consensus.current_epoch || 0,
      mempoolSize:      network.mempool_size || info.mempool_size || 0,
      accounts:         state.account_count || 0,
      totalTxs:         state.tx_count || 0,
      aiEnabled:        ai.ai_enabled || false,
      aiAccuracy:       ai.avg_score != null ? (ai.avg_score * 100).toFixed(1) + '%' : '--',
      aiSamples:        ai.training_samples || 0,
    }
  },

  async stakingInfo(address) {
    if (!address) return null
    const r = await window.api.rpcRemote('positronic_getStakingInfo', [address])
    if (!r) return null
    return {
      staked:      toASF(r.staked || 0),
      rewards:     toASF(r.rewards || 0),
      isValidator: r.is_validator || false,
      available:   toASF(r.available || 0),
    }
  },

  async walletBalance(address) {
    const bal   = await window.api.rpcRemote('eth_getBalance', [address, 'latest'])
    const nonce = await window.api.rpcRemote('eth_getTransactionCount', [address, 'latest'])
    return {
      balance: toASF(bal || '0x0'),
      txCount: parseInt(nonce || '0x0', 16),
    }
  },

  async txHistory(address, limit = 20) {
    const r = await window.api.rpcRemote('positronic_getAddressTransactions', [address, limit])
    return Array.isArray(r) ? r : []
  },

  async networkInfo() {
    const info   = await window.api.rpcRemote('positronic_nodeInfo', [])
    const health = await window.api.rpcRemote('positronic_getNetworkHealth', [])
    if (!info) return null
    const net   = info.network || {}
    const peers = health?.peer_list || []
    return {
      peers:       net.peer_count || 0,
      maxPeers:    net.max_peers || 25,
      synced:      info.synced || false,
      height:      info.height || 0,
      targetHeight:health?.target_height || 0,
      mempoolSize: net.mempool_size || 0,
      networkType: info.network_type || 'testnet',
      p2pPort:     9000,
      rpcPort:     8545,
      peerList:    peers,
      bannedPeers: health?.banned_peers || 0,
    }
  },

  async ecosystemStats() {
    const safe = async (method, params = []) => {
      try { return await window.api.rpcRemote(method, params) } catch { return null }
    }
    const [neural, consensus, did, gov, bridge, depin, rwa, agents, mkt, pq, trust, token] =
      await Promise.all([
        safe('positronic_getNeuralStatus'),
        safe('positronic_getConsensusInfo'),
        safe('positronic_getDIDStats'),
        safe('positronic_getGovernanceStats'),
        safe('positronic_getBridgeStats'),
        safe('positronic_getDePINStats'),
        safe('positronic_getRWAStats'),
        safe('positronic_getAgentStats'),
        safe('positronic_mktGetStats'),
        safe('positronic_getPQStats'),
        safe('positronic_getTrustStats'),
        safe('positronic_getTokenRegistryStats'),
      ])
    return { neural, consensus, did, gov, bridge, depin, rwa, agents, mkt, pq, trust, token }
  },

  async nodeStatus() {
    return window.api.nodeStatus()
  },

  // ── Transaction methods ───────────────────────────────────────────────────

  async stake(address, amount, secretKey) {
    return window.api.rpcRemote('positronic_stake', [{
      from: address,
      amount: String(amount),
      secret_key: secretKey,
    }])
  },

  async unstake(address, secretKey, amount = 0) {
    return window.api.rpcRemote('positronic_unstake', [{
      from: address,
      amount: String(amount),
      secret_key: secretKey,
    }])
  },

  async claimRewards(address, secretKey) {
    return window.api.rpcRemote('positronic_claimStakingRewards', [{
      from: address,
      secret_key: secretKey,
    }])
  },

  async sendTransfer(fromAddress, toAddress, amount, secretKey) {
    return window.api.rpcRemote('positronic_transfer', [
      fromAddress,
      toAddress,
      String(amount),
    ])
  },
}

// ─── Polling hook ─────────────────────────────────────────────────────────────

/**
 * usePolling(fetchFn, intervalMs)
 * Calls fetchFn immediately, then every intervalMs ms.
 * Returns { data, loading, error, refresh }
 */
export function usePolling(fetchFn, intervalMs = 5000) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const timerRef   = useRef(null)
  const mountedRef = useRef(true)
  const fetchFnRef = useRef(fetchFn)

  useEffect(() => { fetchFnRef.current = fetchFn }, [fetchFn])

  const run = useCallback(async () => {
    try {
      const result = await fetchFnRef.current()
      if (mountedRef.current) {
        setData(result)
        setError(null)
        setLoading(false)
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err?.message ?? String(err))
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    run()
    timerRef.current = setInterval(run, intervalMs)
    return () => {
      mountedRef.current = false
      clearInterval(timerRef.current)
    }
  }, [run, intervalMs])

  return { data, loading, error, refresh: run }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Convert hex balance string (from eth_getBalance) to ASF float */
export function hexBalanceToAsf(hexStr) {
  if (!hexStr) return 0
  try {
    const wei = BigInt(hexStr)
    return Number(wei) / 1e18
  } catch {
    return 0
  }
}

/** Format ASF with N decimal places */
export function formatAsf(value, decimals = 4) {
  if (value == null || isNaN(value)) return '0.0000'
  return Number(value).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

/** Truncate address/hash: 0x1234...abcd */
export function truncateHex(str, start = 6, end = 4) {
  if (!str || str.length <= start + end + 3) return str ?? ''
  return `${str.slice(0, start)}...${str.slice(-end)}`
}

/** Copy text to clipboard */
export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
