"""
Positronic - Block Explorer Web Application
A self-contained web application using Python's built-in http.server.
Connects to the Positronic blockchain via JSON-RPC on localhost:8545.
Serves on port 8080 with a Positronic-themed UI.

Usage:
    python -m positronic.explorer.app
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from positronic.constants import CHAIN_ID, COIN_SYMBOL, DECIMALS, BASE_UNIT, DEFAULT_RPC_PORT

logger = logging.getLogger("positronic.explorer")


# ============================================================
# Configuration (from environment or defaults)
# ============================================================

RPC_URL = os.environ.get("POSITRONIC_RPC_URL", f"http://localhost:{DEFAULT_RPC_PORT}")
EXPLORER_HOST = os.environ.get("POSITRONIC_EXPLORER_HOST", "0.0.0.0")
EXPLORER_PORT = int(os.environ.get("POSITRONIC_EXPLORER_PORT", "8080"))


# ============================================================
# RPC Client
# ============================================================

class RPCClient:
    """Simple JSON-RPC 2.0 client using urllib."""

    def __init__(self, url: str = RPC_URL):
        self.url = url
        self._id = 0

    def call(self, method: str, params: list = None):
        """Make a JSON-RPC call. Returns the result or None on error."""
        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": self._id,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if "error" in body and body["error"] is not None:
                    return None
                return body.get("result")
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            return None

    # -- Convenience wrappers --

    def get_block_number(self):
        result = self.call("eth_blockNumber")
        return int(result, 16) if result else None

    def get_block(self, height: int):
        return self.call("eth_getBlockByNumber", [hex(height), True])

    def get_block_by_hash(self, block_hash: str):
        return self.call("eth_getBlockByHash", [block_hash, True])

    def get_transaction(self, tx_hash: str):
        return self.call("eth_getTransactionByHash", [tx_hash])

    def get_balance(self, address: str):
        result = self.call("eth_getBalance", [address])
        return int(result, 16) if result else 0

    def get_nonce(self, address: str):
        result = self.call("eth_getTransactionCount", [address])
        return int(result, 16) if result else 0

    def get_node_info(self):
        return self.call("positronic_nodeInfo")

    def get_network_health(self):
        return self.call("positronic_getNetworkHealth")

    def get_ai_stats(self):
        return self.call("positronic_getAIStats")

    def get_ai_score(self, tx_hash: str):
        return self.call("positronic_getAIScore", [tx_hash])

    def get_wallet_info(self, address: str):
        return self.call("positronic_getWalletInfo", [address])

    def get_wallet_stats(self):
        return self.call("positronic_getWalletStats")

    def get_ai_rank(self, address: str):
        return self.call("positronic_getAIRank", [address])

    def get_ai_rank_stats(self):
        return self.call("positronic_getAIRankStats")

    def get_node_rank(self, address: str):
        return self.call("positronic_getNodeRank", [address])

    def get_node_leaderboard(self, limit: int = 50):
        return self.call("positronic_getNodeLeaderboard", [str(limit)])

    def get_node_stats(self):
        return self.call("positronic_getNodeStats")

    def get_immune_status(self):
        return self.call("positronic_getImmuneStatus")

    def get_recent_threats(self, limit: int = 50):
        return self.call("positronic_getRecentThreats", [str(limit)])

    def get_governance_stats(self):
        return self.call("positronic_getGovernanceStats")

    def get_pending_proposals(self):
        return self.call("positronic_getPendingProposals")

    def get_proposal(self, proposal_id: str):
        return self.call("positronic_getProposal", [proposal_id])

    def get_game_stats(self):
        return self.call("positronic_getGameStats")

    def get_game_leaderboard(self, limit: int = 50):
        return self.call("positronic_getGameLeaderboard", [str(limit)])

    def get_player_profile(self, address: str):
        return self.call("positronic_getPlayerProfile", [address])

    def get_quarantine_pool(self):
        return self.call("positronic_getQuarantinePool")


# ============================================================
# Utility Functions
# ============================================================

def format_positronic(value):
    """Format a base-unit integer value as ASF with decimals."""
    if value is None:
        return "0 ASF"
    if isinstance(value, str):
        try:
            value = int(value, 16) if value.startswith("0x") else int(value)
        except (ValueError, TypeError):
            return "0 ASF"
    if value == 0:
        return "0 ASF"
    whole = value // BASE_UNIT
    frac = value % BASE_UNIT
    if frac == 0:
        return f"{whole:,} ASF"
    frac_str = f"{frac:018d}".rstrip("0")
    return f"{whole:,}.{frac_str} ASF"


def format_number(n):
    """Format a number with commas."""
    if n is None:
        return "0"
    if isinstance(n, str):
        try:
            n = int(n, 16) if n.startswith("0x") else int(n)
        except (ValueError, TypeError):
            return n
    return f"{n:,}"


def short_hash(h):
    """Shorten a hex hash for display."""
    if not h:
        return "N/A"
    s = str(h)
    if len(s) > 20:
        return s[:10] + "..." + s[-8:]
    return s


def timestamp_str(ts):
    """Convert timestamp to readable string."""
    if not ts:
        return "N/A"
    try:
        ts = float(ts)
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))
    except (ValueError, TypeError, OSError):
        return "N/A"


def time_ago(ts):
    """Convert timestamp to 'X ago' string."""
    if not ts:
        return "N/A"
    try:
        diff = time.time() - float(ts)
        if diff < 0:
            return "just now"
        if diff < 60:
            return f"{int(diff)}s ago"
        if diff < 3600:
            return f"{int(diff / 60)}m ago"
        if diff < 86400:
            return f"{int(diff / 3600)}h ago"
        return f"{int(diff / 86400)}d ago"
    except (ValueError, TypeError):
        return "N/A"


def safe_get(d, *keys, default="N/A"):
    """Safely navigate nested dicts."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def html_escape(s):
    """Basic HTML escaping."""
    if not isinstance(s, str):
        s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


TX_TYPE_NAMES = {
    0: "Transfer",
    1: "Contract Create",
    2: "Contract Call",
    3: "Stake",
    4: "Unstake",
    6: "Block Reward",
    7: "AI Treasury",
}

TX_STATUS_NAMES = {
    0: "Pending",
    1: "Accepted",
    2: "Quarantined",
    3: "Rejected",
    4: "Confirmed",
    5: "Failed",
}

TX_STATUS_COLORS = {
    0: "#FBD000",  # yellow
    1: "#43B047",  # green
    2: "#FF8C00",  # orange
    3: "#E52521",  # red
    4: "#049CD8",  # blue
    5: "#E52521",  # red
}

ALERT_COLORS = {
    "GREEN": "#43B047",
    "YELLOW": "#FBD000",
    "ORANGE": "#FF8C00",
    "RED": "#E52521",
    "BLACK": "#1a1a2e",
}

NODE_RANK_EMOJI = {
    1: "Probe",
    2: "Sentinel",
    3: "Circuit",
    4: "Relay",
    5: "Cortex",
    6: "Nexus",
    7: "Positronic",
    8: "Star Positronic",
}


# ============================================================
# HTML Templates
# ============================================================

CSS = """
:root {
    --positronic-red: #E52521;
    --positronic-blue: #049CD8;
    --positronic-yellow: #FBD000;
    --positronic-green: #43B047;
    --bg-dark: #1a1a2e;
    --bg-card: #16213e;
    --bg-card-hover: #1b2a4a;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0a0;
    --text-bright: #ffffff;
    --border-color: #2a3a5c;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: var(--bg-dark);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

a { color: var(--positronic-blue); text-decoration: none; }
a:hover { text-decoration: underline; color: #4db8e8; }

/* Navigation */
.navbar {
    background: linear-gradient(135deg, var(--positronic-red) 0%, #c41e1a 100%);
    padding: 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    position: sticky;
    top: 0;
    z-index: 1000;
}
.navbar-inner {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    flex-wrap: wrap;
}
.navbar-brand {
    font-size: 1.4rem;
    font-weight: bold;
    color: var(--positronic-yellow);
    padding: 12px 0;
    text-shadow: 2px 2px 0 rgba(0,0,0,0.3);
    letter-spacing: 1px;
}
.navbar-brand:hover { text-decoration: none; color: #fff; }
.nav-links {
    display: flex;
    gap: 0;
    flex-wrap: wrap;
}
.nav-links a {
    color: #fff;
    padding: 14px 16px;
    font-size: 0.9rem;
    transition: background 0.2s;
    border-bottom: 3px solid transparent;
}
.nav-links a:hover, .nav-links a.active {
    background: rgba(0,0,0,0.2);
    text-decoration: none;
    border-bottom: 3px solid var(--positronic-yellow);
}

/* Search bar */
.search-bar {
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 20px;
}
.search-bar form {
    display: flex;
    gap: 8px;
}
.search-bar input {
    flex: 1;
    padding: 12px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 0.95rem;
    outline: none;
    transition: border 0.2s;
}
.search-bar input:focus {
    border-color: var(--positronic-blue);
}
.search-bar input::placeholder { color: var(--text-secondary); }
.search-bar button {
    padding: 12px 24px;
    background: var(--positronic-blue);
    color: #fff;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    transition: background 0.2s;
}
.search-bar button:hover { background: #037ab5; }

/* Layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
.page-title {
    font-size: 1.6rem;
    font-weight: bold;
    color: var(--text-bright);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--positronic-red);
}

/* Cards */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    transition: transform 0.15s, border-color 0.15s;
}
.card:hover {
    transform: translateY(-2px);
    border-color: var(--positronic-blue);
}
.card-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.card-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: var(--text-bright);
}
.card-value.red { color: var(--positronic-red); }
.card-value.blue { color: var(--positronic-blue); }
.card-value.yellow { color: var(--positronic-yellow); }
.card-value.green { color: var(--positronic-green); }

/* Tables */
.table-wrapper {
    overflow-x: auto;
    margin-bottom: 24px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}
table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
}
thead th {
    background: rgba(0,0,0,0.3);
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 1px;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
    white-space: nowrap;
}
tbody td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border-color);
    font-size: 0.9rem;
    white-space: nowrap;
}
tbody tr:hover { background: var(--bg-card-hover); }
tbody tr:last-child td { border-bottom: none; }

/* Detail view */
.detail-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    margin-bottom: 24px;
    overflow: hidden;
}
.detail-section h2 {
    padding: 16px 20px;
    font-size: 1.1rem;
    background: rgba(0,0,0,0.2);
    border-bottom: 1px solid var(--border-color);
    color: var(--positronic-yellow);
}
.detail-row {
    display: flex;
    border-bottom: 1px solid var(--border-color);
    padding: 0;
}
.detail-row:last-child { border-bottom: none; }
.detail-key {
    flex: 0 0 200px;
    padding: 12px 20px;
    font-weight: 600;
    color: var(--text-secondary);
    background: rgba(0,0,0,0.1);
    font-size: 0.85rem;
}
.detail-val {
    flex: 1;
    padding: 12px 20px;
    word-break: break-all;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.85rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
    text-transform: uppercase;
}
.badge-green { background: rgba(67,176,71,0.2); color: var(--positronic-green); }
.badge-red { background: rgba(229,37,33,0.2); color: var(--positronic-red); }
.badge-blue { background: rgba(4,156,216,0.2); color: var(--positronic-blue); }
.badge-yellow { background: rgba(251,208,0,0.2); color: var(--positronic-yellow); }
.badge-orange { background: rgba(255,140,0,0.2); color: #FF8C00; }

/* AI Score bar */
.score-bar {
    display: inline-block;
    width: 120px;
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
    vertical-align: middle;
    margin-left: 8px;
}
.score-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

/* Alert banner */
.alert-banner {
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-weight: bold;
    text-align: center;
}

/* Error page */
.error-box {
    text-align: center;
    padding: 60px 20px;
}
.error-box h1 {
    font-size: 4rem;
    color: var(--positronic-red);
    margin-bottom: 10px;
}
.error-box p {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 30px 20px;
    color: var(--text-secondary);
    font-size: 0.8rem;
    border-top: 1px solid var(--border-color);
    margin-top: 40px;
}

/* Responsive */
@media (max-width: 768px) {
    .navbar-inner { flex-direction: column; }
    .nav-links { width: 100%; justify-content: center; }
    .nav-links a { padding: 10px 12px; font-size: 0.8rem; }
    .card-grid { grid-template-columns: 1fr 1fr; }
    .detail-row { flex-direction: column; }
    .detail-key { flex: none; }
    .search-bar form { flex-direction: column; }
    .search-bar button { width: 100%; }
}
@media (max-width: 480px) {
    .card-grid { grid-template-columns: 1fr; }
    .container { padding: 10px; }
}
"""


def base_template(title, content, active=""):
    """Wrap content in the base HTML layout."""
    def nav_class(name):
        return ' class="active"' if name == active else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_escape(title)} - Positronic Explorer</title>
<style>{CSS}</style>
</head>
<body>

<nav class="navbar">
<div class="navbar-inner">
    <a href="/" class="navbar-brand">Positronic Explorer</a>
    <div class="nav-links">
        <a href="/"{nav_class("home")}>Home</a>
        <a href="/network"{nav_class("network")}>Network</a>
        <a href="/game"{nav_class("game")}>Game</a>
        <a href="/governance"{nav_class("governance")}>Governance</a>
    </div>
</div>
</nav>

<div class="search-bar">
<form method="GET" action="/search">
    <input type="text" name="q" placeholder="Search by block height, tx hash, or address (0x...)" />
    <button type="submit">Search</button>
</form>
</div>

<div class="container">
{content}
</div>

<div class="footer">
    Positronic Block Explorer | Chain ID: {CHAIN_ID} | Powered by AI-Validated Blockchain (PoNC)
</div>

</body>
</html>"""


# ============================================================
# Page Renderers
# ============================================================

def render_home(rpc: RPCClient):
    """Home page: chain stats, latest blocks, network health."""
    # Fetch data
    node_info = rpc.get_node_info() or {}
    height = safe_get(node_info, "height", default=None)
    if height is None:
        height = rpc.get_block_number()

    ai_stats = safe_get(node_info, "ai", default={})
    quarantine = safe_get(node_info, "quarantine", default={})
    state = safe_get(node_info, "state", default={})
    immune = safe_get(node_info, "immune_system", default={})
    game = safe_get(node_info, "game", default={})
    governance = safe_get(node_info, "governance", default={})
    node_ranking = safe_get(node_info, "node_ranking", default={})

    chain_height = height if height is not None else "N/A"
    head_hash = safe_get(node_info, "head_hash", default="N/A")
    total_accounts = safe_get(state, "total_accounts", default=0)
    alert_level = safe_get(immune, "alert_level", default="GREEN")
    alert_color = ALERT_COLORS.get(alert_level, "#43B047")

    # Stats cards
    cards_html = f"""
    <div class="card-grid">
        <div class="card">
            <div class="card-label">Chain Height</div>
            <div class="card-value blue">{format_number(chain_height)}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Accounts</div>
            <div class="card-value green">{format_number(total_accounts)}</div>
        </div>
        <div class="card">
            <div class="card-label">AI Model Version</div>
            <div class="card-value yellow">{safe_get(ai_stats, "model_version", default="1")}</div>
        </div>
        <div class="card">
            <div class="card-label">Network Alert</div>
            <div class="card-value" style="color:{alert_color}">{html_escape(str(alert_level))}</div>
        </div>
        <div class="card">
            <div class="card-label">Quarantined TXs</div>
            <div class="card-value red">{safe_get(quarantine, "total_quarantined", default=0)}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Nodes</div>
            <div class="card-value blue">{safe_get(node_ranking, "total_nodes", default=0)}</div>
        </div>
        <div class="card">
            <div class="card-label">Game Players</div>
            <div class="card-value green">{safe_get(game, "total_players", default=0)}</div>
        </div>
        <div class="card">
            <div class="card-label">Token Proposals</div>
            <div class="card-value yellow">{safe_get(governance, "total_proposals", default=0)}</div>
        </div>
    </div>
    """

    # Latest blocks table
    blocks_rows = ""
    if isinstance(chain_height, int) and chain_height >= 0:
        start = max(0, chain_height - 9)
        for h in range(chain_height, start - 1, -1):
            block = rpc.get_block(h)
            if block:
                header = block.get("header", block)
                bh = safe_get(header, "height", default=h)
                bhash = safe_get(header, "hash", default="")
                ts = safe_get(header, "timestamp", default=0)
                gas_used = safe_get(header, "gas_used", default=0)
                txs = block.get("transactions", [])
                tx_count = len(txs) if isinstance(txs, list) else 0
                validator = safe_get(header, "validator_pubkey", default="")
                blocks_rows += f"""<tr>
                    <td><a href="/block/{bh}">{bh}</a></td>
                    <td>{time_ago(ts)}</td>
                    <td>{tx_count}</td>
                    <td>{format_number(gas_used)}</td>
                    <td><a href="/block/{bh}">{short_hash(bhash)}</a></td>
                    <td>{short_hash(validator)}</td>
                </tr>"""

    blocks_html = f"""
    <h2 class="page-title">Latest Blocks</h2>
    <div class="table-wrapper">
    <table>
        <thead>
            <tr><th>Height</th><th>Age</th><th>TXs</th><th>Gas Used</th><th>Hash</th><th>Validator</th></tr>
        </thead>
        <tbody>
            {blocks_rows if blocks_rows else '<tr><td colspan="6" style="text-align:center;color:var(--text-secondary)">No blocks found. Is the RPC server running on port 8545?</td></tr>'}
        </tbody>
    </table>
    </div>
    """

    # Head hash
    head_html = ""
    if head_hash and head_hash != "N/A":
        head_html = f"""
        <div class="detail-section">
            <h2>Chain Head</h2>
            <div class="detail-row">
                <div class="detail-key">Head Hash</div>
                <div class="detail-val">{html_escape(str(head_hash))}</div>
            </div>
            <div class="detail-row">
                <div class="detail-key">Chain ID</div>
                <div class="detail-val">{CHAIN_ID}</div>
            </div>
        </div>
        """

    return base_template(
        "Home",
        cards_html + blocks_html + head_html,
        active="home",
    )


def render_block(rpc: RPCClient, height: int):
    """Block detail page."""
    block = rpc.get_block(height)
    if not block:
        return render_error("Block Not Found", f"Block #{height} was not found on the chain.")

    header = block.get("header", block)
    txs = block.get("transactions", [])
    if not isinstance(txs, list):
        txs = []

    bhash = safe_get(header, "hash", default="N/A")
    prev_hash = safe_get(header, "previous_hash", default="N/A")
    ts = safe_get(header, "timestamp", default=0)
    validator = safe_get(header, "validator_pubkey", default="N/A")
    gas_limit = safe_get(header, "gas_limit", default=0)
    gas_used = safe_get(header, "gas_used", default=0)
    state_root = safe_get(header, "state_root", default="N/A")
    tx_root = safe_get(header, "transactions_root", default="N/A")
    ai_score_root = safe_get(header, "ai_score_root", default="N/A")
    ai_model = safe_get(header, "ai_model_version", default=1)
    slot = safe_get(header, "slot", default=0)
    epoch = safe_get(header, "epoch", default=0)
    chain_id = safe_get(header, "chain_id", default=CHAIN_ID)

    # Gas usage bar
    gas_pct = 0
    try:
        gas_pct = (int(gas_used) / int(gas_limit) * 100) if int(gas_limit) > 0 else 0
    except (ValueError, TypeError, ZeroDivisionError):
        gas_pct = 0
    gas_color = "#43B047" if gas_pct < 50 else "#FBD000" if gas_pct < 80 else "#E52521"

    # Navigation arrows
    prev_link = f'<a href="/block/{height - 1}">&larr; Block {height - 1}</a>' if height > 0 else ""
    next_link = f'<a href="/block/{height + 1}">Block {height + 1} &rarr;</a>'

    detail_html = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <h1 class="page-title" style="margin-bottom:0;">Block #{height}</h1>
        <div style="display:flex;gap:16px;">{prev_link} {next_link}</div>
    </div>
    <div class="detail-section">
        <h2>Block Header</h2>
        <div class="detail-row"><div class="detail-key">Block Hash</div><div class="detail-val">{html_escape(str(bhash))}</div></div>
        <div class="detail-row"><div class="detail-key">Previous Hash</div><div class="detail-val">{html_escape(str(prev_hash))}</div></div>
        <div class="detail-row"><div class="detail-key">Timestamp</div><div class="detail-val">{timestamp_str(ts)} ({time_ago(ts)})</div></div>
        <div class="detail-row"><div class="detail-key">Validator</div><div class="detail-val">{html_escape(str(validator))}</div></div>
        <div class="detail-row">
            <div class="detail-key">Gas Used / Limit</div>
            <div class="detail-val">
                {format_number(gas_used)} / {format_number(gas_limit)} ({gas_pct:.1f}%)
                <div class="score-bar" style="width:200px;margin-left:12px;">
                    <div class="score-fill" style="width:{gas_pct}%;background:{gas_color};"></div>
                </div>
            </div>
        </div>
        <div class="detail-row"><div class="detail-key">Slot / Epoch</div><div class="detail-val">{slot} / {epoch}</div></div>
        <div class="detail-row"><div class="detail-key">AI Model Version</div><div class="detail-val">{ai_model}</div></div>
        <div class="detail-row"><div class="detail-key">State Root</div><div class="detail-val">{html_escape(str(state_root))}</div></div>
        <div class="detail-row"><div class="detail-key">TX Root</div><div class="detail-val">{html_escape(str(tx_root))}</div></div>
        <div class="detail-row"><div class="detail-key">AI Score Root</div><div class="detail-val">{html_escape(str(ai_score_root))}</div></div>
        <div class="detail-row"><div class="detail-key">Chain ID</div><div class="detail-val">{chain_id}</div></div>
    </div>
    """

    # Transactions table
    tx_rows = ""
    for tx in txs:
        tx_hash = safe_get(tx, "tx_hash", default="N/A")
        tx_type = TX_TYPE_NAMES.get(safe_get(tx, "tx_type", default=0), "Unknown")
        sender = safe_get(tx, "sender", default="N/A")
        recipient = safe_get(tx, "recipient", default="N/A")
        value = safe_get(tx, "value", default=0)
        ai_score = safe_get(tx, "ai_score", default=0)
        status = safe_get(tx, "status", default=0)
        status_name = TX_STATUS_NAMES.get(status, "Unknown")
        status_color = TX_STATUS_COLORS.get(status, "#a0a0a0")

        try:
            score_pct = float(ai_score) * 100
        except (ValueError, TypeError):
            score_pct = 0
        score_color = "#43B047" if score_pct < 50 else "#FBD000" if score_pct < 85 else "#E52521"

        tx_rows += f"""<tr>
            <td><a href="/tx/{html_escape(str(tx_hash))}">{short_hash(tx_hash)}</a></td>
            <td>{tx_type}</td>
            <td><a href="/address/{html_escape(str(sender))}">{short_hash(sender)}</a></td>
            <td><a href="/address/{html_escape(str(recipient))}">{short_hash(recipient)}</a></td>
            <td>{format_positronic(value)}</td>
            <td>
                {ai_score}
                <div class="score-bar"><div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div></div>
            </td>
            <td><span class="badge" style="background:rgba(0,0,0,0.2);color:{status_color}">{status_name}</span></td>
        </tr>"""

    tx_html = f"""
    <div class="detail-section">
        <h2>Transactions ({len(txs)})</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>Hash</th><th>Type</th><th>From</th><th>To</th><th>Value</th><th>AI Score</th><th>Status</th></tr></thead>
        <tbody>
            {tx_rows if tx_rows else '<tr><td colspan="7" style="text-align:center;color:var(--text-secondary)">No transactions in this block</td></tr>'}
        </tbody>
    </table>
    </div>
    """

    return base_template(f"Block #{height}", detail_html + tx_html, active="home")


def render_transaction(rpc: RPCClient, tx_hash: str):
    """Transaction detail page."""
    tx = rpc.get_transaction(tx_hash)
    if not tx:
        return render_error("Transaction Not Found", f"Transaction {tx_hash} was not found.")

    tx_type = TX_TYPE_NAMES.get(safe_get(tx, "tx_type", default=0), "Unknown")
    sender = safe_get(tx, "sender", default="N/A")
    recipient = safe_get(tx, "recipient", default="N/A")
    value = safe_get(tx, "value", default=0)
    nonce = safe_get(tx, "nonce", default=0)
    gas_price = safe_get(tx, "gas_price", default=0)
    gas_limit = safe_get(tx, "gas_limit", default=0)
    data_hex = safe_get(tx, "data", default="")
    signature = safe_get(tx, "signature", default="N/A")
    timestamp = safe_get(tx, "timestamp", default=0)
    chain_id = safe_get(tx, "chain_id", default=CHAIN_ID)
    ai_score = safe_get(tx, "ai_score", default=0)
    ai_model = safe_get(tx, "ai_model_version", default=0)
    status = safe_get(tx, "status", default=0)
    status_name = TX_STATUS_NAMES.get(status, "Unknown")
    status_color = TX_STATUS_COLORS.get(status, "#a0a0a0")

    try:
        score_pct = float(ai_score) * 100
    except (ValueError, TypeError):
        score_pct = 0
    if score_pct < 50:
        score_badge = "badge-green"
        score_label = "Low Risk"
    elif score_pct < 85:
        score_badge = "badge-yellow"
        score_label = "Medium Risk"
    else:
        score_badge = "badge-red"
        score_label = "High Risk"
    score_color = "#43B047" if score_pct < 50 else "#FBD000" if score_pct < 85 else "#E52521"

    has_data = data_hex and data_hex != "" and data_hex != "0x"
    data_display = html_escape(str(data_hex))
    if len(data_display) > 200:
        data_display = data_display[:200] + "..."

    detail_html = f"""
    <h1 class="page-title">Transaction Details</h1>
    <div class="detail-section">
        <h2>Overview</h2>
        <div class="detail-row"><div class="detail-key">TX Hash</div><div class="detail-val">{html_escape(str(tx_hash))}</div></div>
        <div class="detail-row"><div class="detail-key">Type</div><div class="detail-val">{tx_type}</div></div>
        <div class="detail-row">
            <div class="detail-key">Status</div>
            <div class="detail-val"><span class="badge" style="background:rgba(0,0,0,0.2);color:{status_color}">{status_name}</span></div>
        </div>
        <div class="detail-row"><div class="detail-key">Timestamp</div><div class="detail-val">{timestamp_str(timestamp)} ({time_ago(timestamp)})</div></div>
        <div class="detail-row"><div class="detail-key">Chain ID</div><div class="detail-val">{chain_id}</div></div>
    </div>

    <div class="detail-section">
        <h2>Transfer Details</h2>
        <div class="detail-row"><div class="detail-key">From</div><div class="detail-val"><a href="/address/{html_escape(str(sender))}">{html_escape(str(sender))}</a></div></div>
        <div class="detail-row"><div class="detail-key">To</div><div class="detail-val"><a href="/address/{html_escape(str(recipient))}">{html_escape(str(recipient))}</a></div></div>
        <div class="detail-row"><div class="detail-key">Value</div><div class="detail-val" style="color:var(--positronic-green);font-weight:bold;">{format_positronic(value)}</div></div>
        <div class="detail-row"><div class="detail-key">Nonce</div><div class="detail-val">{nonce}</div></div>
        <div class="detail-row"><div class="detail-key">Gas Price</div><div class="detail-val">{format_number(gas_price)} Pixel</div></div>
        <div class="detail-row"><div class="detail-key">Gas Limit</div><div class="detail-val">{format_number(gas_limit)}</div></div>
        <div class="detail-row"><div class="detail-key">TX Fee</div><div class="detail-val">{format_positronic(int(gas_price) * int(gas_limit) if isinstance(gas_price, int) and isinstance(gas_limit, int) else 0)}</div></div>
    </div>

    <div class="detail-section">
        <h2>AI Validation</h2>
        <div class="detail-row">
            <div class="detail-key">AI Risk Score</div>
            <div class="detail-val">
                <span style="font-size:1.3rem;font-weight:bold;color:{score_color}">{ai_score}</span>
                <div class="score-bar" style="width:200px;margin-left:12px;">
                    <div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div>
                </div>
                <span class="{score_badge} badge" style="margin-left:12px;">{score_label}</span>
            </div>
        </div>
        <div class="detail-row"><div class="detail-key">AI Model Version</div><div class="detail-val">{ai_model}</div></div>
    </div>
    """

    if has_data:
        detail_html += f"""
    <div class="detail-section">
        <h2>Input Data</h2>
        <div class="detail-row"><div class="detail-key">Data</div><div class="detail-val" style="font-size:0.8rem;">{data_display}</div></div>
        <div class="detail-row"><div class="detail-key">Data Size</div><div class="detail-val">{len(data_hex) // 2} bytes</div></div>
    </div>
        """

    detail_html += f"""
    <div class="detail-section">
        <h2>Signature</h2>
        <div class="detail-row"><div class="detail-key">Signature</div><div class="detail-val" style="font-size:0.8rem;">{html_escape(str(signature))}</div></div>
    </div>
    """

    return base_template("Transaction Details", detail_html, active="home")


def render_address(rpc: RPCClient, address: str):
    """Address detail page: balance, wallet info, node rank, AI rank."""
    balance = rpc.get_balance(address)
    nonce = rpc.get_nonce(address)
    wallet_info = rpc.get_wallet_info(address)
    node_rank_info = rpc.get_node_rank(address)
    ai_rank_info = rpc.get_ai_rank(address)

    # Overview
    detail_html = f"""
    <h1 class="page-title">Address Details</h1>
    <div class="detail-section">
        <h2>Account Overview</h2>
        <div class="detail-row"><div class="detail-key">Address</div><div class="detail-val">{html_escape(address)}</div></div>
        <div class="detail-row"><div class="detail-key">Balance</div><div class="detail-val" style="color:var(--positronic-green);font-weight:bold;font-size:1.2rem;">{format_positronic(balance)}</div></div>
        <div class="detail-row"><div class="detail-key">Nonce (TX Count)</div><div class="detail-val">{format_number(nonce)}</div></div>
    </div>
    """

    # Wallet info
    if wallet_info:
        wallet_status = safe_get(wallet_info, "status", default="Unknown")
        verified = safe_get(wallet_info, "verified", default=False)
        trust_score = safe_get(wallet_info, "trust_score", default=0)
        total_volume = safe_get(wallet_info, "total_volume", default=0)
        tx_count = safe_get(wallet_info, "transaction_count", default=0)
        registered_at = safe_get(wallet_info, "registered_at", default=0)

        verified_badge = '<span class="badge badge-green">Verified</span>' if verified else '<span class="badge badge-yellow">Unverified</span>'
        status_badge_class = "badge-green" if wallet_status in ("active", "ACTIVE") else "badge-yellow" if wallet_status in ("registered", "REGISTERED") else "badge-red"

        detail_html += f"""
    <div class="detail-section">
        <h2>Wallet Registry</h2>
        <div class="detail-row"><div class="detail-key">Status</div><div class="detail-val"><span class="badge {status_badge_class}">{html_escape(str(wallet_status))}</span></div></div>
        <div class="detail-row"><div class="detail-key">Verified</div><div class="detail-val">{verified_badge}</div></div>
        <div class="detail-row"><div class="detail-key">Trust Score</div><div class="detail-val">{trust_score}</div></div>
        <div class="detail-row"><div class="detail-key">Total Volume</div><div class="detail-val">{format_positronic(total_volume)}</div></div>
        <div class="detail-row"><div class="detail-key">Transaction Count</div><div class="detail-val">{format_number(tx_count)}</div></div>
        <div class="detail-row"><div class="detail-key">Registered</div><div class="detail-val">{timestamp_str(registered_at)}</div></div>
    </div>
        """

    # Node rank
    if node_rank_info:
        rank_num = safe_get(node_rank_info, "rank", default=1)
        rank_name = safe_get(node_rank_info, "rank_name", default="Probe")
        uptime = safe_get(node_rank_info, "uptime_percentage", default=0)
        blocks_validated = safe_get(node_rank_info, "blocks_validated", default=0)
        days_active = safe_get(node_rank_info, "days_active", default=0)
        reward_mult = safe_get(node_rank_info, "reward_multiplier", default=1.0)
        penalties = safe_get(node_rank_info, "penalties", default=0)

        positronic_rank_label = NODE_RANK_EMOJI.get(rank_num, "Unknown")

        detail_html += f"""
    <div class="detail-section">
        <h2>Node Rank</h2>
        <div class="detail-row"><div class="detail-key">Rank</div><div class="detail-val" style="font-size:1.2rem;font-weight:bold;color:var(--positronic-yellow);">{html_escape(str(rank_name))} (Level {rank_num} - {positronic_rank_label})</div></div>
        <div class="detail-row"><div class="detail-key">Uptime</div><div class="detail-val">{uptime}%</div></div>
        <div class="detail-row"><div class="detail-key">Blocks Validated</div><div class="detail-val">{format_number(blocks_validated)}</div></div>
        <div class="detail-row"><div class="detail-key">Days Active</div><div class="detail-val">{days_active}</div></div>
        <div class="detail-row"><div class="detail-key">Reward Multiplier</div><div class="detail-val">{reward_mult}x</div></div>
        <div class="detail-row"><div class="detail-key">Penalties</div><div class="detail-val">{penalties}</div></div>
    </div>
        """

    # AI rank
    if ai_rank_info:
        ai_rank_name = safe_get(ai_rank_info, "rank_name", default="E1_PRIVATE")
        ai_rank_fa = safe_get(ai_rank_info, "rank_name_fa", default="")
        ai_accuracy = safe_get(ai_rank_info, "accuracy", default=0)
        ai_total_scored = safe_get(ai_rank_info, "total_scored", default=0)
        ai_uptime_hours = safe_get(ai_rank_info, "uptime_hours", default=0)
        ai_reward_mult = safe_get(ai_rank_info, "reward_multiplier", default=1.0)
        ai_promotions = safe_get(ai_rank_info, "promotions", default=0)
        ai_demotions = safe_get(ai_rank_info, "demotions", default=0)

        detail_html += f"""
    <div class="detail-section">
        <h2>AI Validator Rank</h2>
        <div class="detail-row"><div class="detail-key">Rank</div><div class="detail-val" style="font-size:1.2rem;font-weight:bold;color:var(--positronic-blue);">{html_escape(str(ai_rank_name))} {html_escape(str(ai_rank_fa))}</div></div>
        <div class="detail-row"><div class="detail-key">Accuracy</div><div class="detail-val">{ai_accuracy}</div></div>
        <div class="detail-row"><div class="detail-key">TXs Scored</div><div class="detail-val">{format_number(ai_total_scored)}</div></div>
        <div class="detail-row"><div class="detail-key">Uptime</div><div class="detail-val">{ai_uptime_hours} hours</div></div>
        <div class="detail-row"><div class="detail-key">Reward Multiplier</div><div class="detail-val">{ai_reward_mult}x</div></div>
        <div class="detail-row"><div class="detail-key">Promotions / Demotions</div><div class="detail-val">{ai_promotions} / {ai_demotions}</div></div>
    </div>
        """

    return base_template("Address Details", detail_html, active="home")


def render_network(rpc: RPCClient):
    """Network page: immune status, node leaderboard, AI rank stats."""
    immune = rpc.get_immune_status() or {}
    node_stats = rpc.get_node_stats() or {}
    ai_rank_stats = rpc.get_ai_rank_stats() or {}
    recent_threats = rpc.get_recent_threats(20) or []
    node_leaderboard = rpc.get_node_leaderboard(25) or []

    alert_level = safe_get(immune, "alert_level", default="GREEN")
    alert_color = ALERT_COLORS.get(alert_level, "#43B047")
    total_threats = safe_get(immune, "total_threats", default=0)
    blocked_addrs = safe_get(immune, "blocked_addresses", default=0)
    threat_types = safe_get(immune, "threat_types", default={})

    total_nodes = safe_get(node_stats, "total_nodes", default=0)
    rank_dist = safe_get(node_stats, "rank_distribution", default={})

    total_ai_validators = safe_get(ai_rank_stats, "total_validators", default=0)
    ai_rank_dist = safe_get(ai_rank_stats, "rank_distribution", default={})

    # Alert banner
    alert_html = f"""
    <div class="alert-banner" style="background:{alert_color};color:#fff;">
        Network Alert Level: {html_escape(str(alert_level))} | {total_threats} total threats detected | {blocked_addrs} addresses blocked
    </div>
    """

    # Stats cards
    cards_html = f"""
    <div class="card-grid">
        <div class="card">
            <div class="card-label">Alert Level</div>
            <div class="card-value" style="color:{alert_color}">{html_escape(str(alert_level))}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Threats</div>
            <div class="card-value red">{total_threats}</div>
        </div>
        <div class="card">
            <div class="card-label">Blocked Addresses</div>
            <div class="card-value red">{blocked_addrs}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Nodes</div>
            <div class="card-value blue">{total_nodes}</div>
        </div>
        <div class="card">
            <div class="card-label">AI Validators</div>
            <div class="card-value yellow">{total_ai_validators}</div>
        </div>
    </div>
    """

    # Node rank distribution
    rank_dist_rows = ""
    for rank_name, count in rank_dist.items() if isinstance(rank_dist, dict) else []:
        rank_display = rank_name.replace("_", " ").title()
        rank_dist_rows += f"<tr><td>{html_escape(rank_display)}</td><td>{count}</td></tr>"

    node_dist_html = f"""
    <div class="detail-section">
        <h2>Node Rank Distribution</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>Rank</th><th>Count</th></tr></thead>
        <tbody>{rank_dist_rows if rank_dist_rows else '<tr><td colspan="2" style="text-align:center;color:var(--text-secondary)">No node data available</td></tr>'}</tbody>
    </table>
    </div>
    """

    # AI rank distribution
    ai_dist_rows = ""
    for rank_name, count in ai_rank_dist.items() if isinstance(ai_rank_dist, dict) else []:
        ai_dist_rows += f"<tr><td>{html_escape(str(rank_name))}</td><td>{count}</td></tr>"

    ai_dist_html = f"""
    <div class="detail-section">
        <h2>AI Validator Rank Distribution</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>Rank</th><th>Count</th></tr></thead>
        <tbody>{ai_dist_rows if ai_dist_rows else '<tr><td colspan="2" style="text-align:center;color:var(--text-secondary)">No AI validator data available</td></tr>'}</tbody>
    </table>
    </div>
    """

    # Node leaderboard
    leaderboard_rows = ""
    for i, node in enumerate(node_leaderboard):
        address = safe_get(node, "address", default="N/A")
        rank_name = safe_get(node, "rank_name", default="Probe")
        uptime = safe_get(node, "uptime_percentage", default=0)
        blocks_validated = safe_get(node, "blocks_validated", default=0)
        days_active = safe_get(node, "days_active", default=0)
        reward_mult = safe_get(node, "reward_multiplier", default=1.0)

        leaderboard_rows += f"""<tr>
            <td>{i + 1}</td>
            <td><a href="/address/{html_escape(str(address))}">{short_hash(address)}</a></td>
            <td style="font-weight:bold;color:var(--positronic-yellow);">{html_escape(str(rank_name))}</td>
            <td>{uptime}%</td>
            <td>{format_number(blocks_validated)}</td>
            <td>{days_active}</td>
            <td>{reward_mult}x</td>
        </tr>"""

    leaderboard_html = f"""
    <div class="detail-section">
        <h2>Node Leaderboard</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>#</th><th>Address</th><th>Rank</th><th>Uptime</th><th>Blocks</th><th>Days</th><th>Multiplier</th></tr></thead>
        <tbody>{leaderboard_rows if leaderboard_rows else '<tr><td colspan="7" style="text-align:center;color:var(--text-secondary)">No nodes registered yet</td></tr>'}</tbody>
    </table>
    </div>
    """

    # Recent threats
    threat_rows = ""
    for threat in recent_threats:
        evt_type = safe_get(threat, "event_type", default="Unknown")
        severity = safe_get(threat, "severity", default=0)
        source = safe_get(threat, "source_address", default="N/A")
        ts = safe_get(threat, "timestamp", default=0)
        desc = safe_get(threat, "description", default="")
        bh = safe_get(threat, "block_height", default=0)
        resolved = safe_get(threat, "resolved", default=False)

        try:
            sev_pct = float(severity) * 100
        except (ValueError, TypeError):
            sev_pct = 0
        sev_color = "#43B047" if sev_pct < 50 else "#FBD000" if sev_pct < 80 else "#E52521"
        resolved_badge = '<span class="badge badge-green">Resolved</span>' if resolved else '<span class="badge badge-red">Active</span>'

        threat_rows += f"""<tr>
            <td>{html_escape(str(evt_type))}</td>
            <td><span style="color:{sev_color};font-weight:bold;">{severity}</span></td>
            <td><a href="/address/{html_escape(str(source))}">{short_hash(source)}</a></td>
            <td>{time_ago(ts)}</td>
            <td>{bh}</td>
            <td>{resolved_badge}</td>
        </tr>"""

    threats_html = f"""
    <div class="detail-section">
        <h2>Recent Threats</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>Type</th><th>Severity</th><th>Source</th><th>Time</th><th>Block</th><th>Status</th></tr></thead>
        <tbody>{threat_rows if threat_rows else '<tr><td colspan="6" style="text-align:center;color:var(--text-secondary)">No threats detected - network is healthy</td></tr>'}</tbody>
    </table>
    </div>
    """

    content = f"""
    <h1 class="page-title">Network Health & Node Rankings</h1>
    {alert_html}
    {cards_html}
    {leaderboard_html}
    {node_dist_html}
    {ai_dist_html}
    {threats_html}
    """

    return base_template("Network", content, active="network")


def render_game(rpc: RPCClient):
    """Game page: player leaderboard, game stats."""
    game_stats = rpc.get_game_stats() or {}
    leaderboard = rpc.get_game_leaderboard(25) or []

    total_players = safe_get(game_stats, "total_players", default=0)
    total_games = safe_get(game_stats, "total_games_played", default=0)
    total_distributed = safe_get(game_stats, "total_distributed", default=0)
    pending_rewards = safe_get(game_stats, "pending_rewards", default=0)

    cards_html = f"""
    <div class="card-grid">
        <div class="card">
            <div class="card-label">Total Players</div>
            <div class="card-value green">{format_number(total_players)}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Games Played</div>
            <div class="card-value blue">{format_number(total_games)}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Rewards Distributed</div>
            <div class="card-value yellow">{format_positronic(total_distributed)}</div>
        </div>
        <div class="card">
            <div class="card-label">Pending Rewards</div>
            <div class="card-value red">{format_positronic(pending_rewards)}</div>
        </div>
    </div>
    """

    # Player leaderboard
    lb_rows = ""
    for i, player in enumerate(leaderboard):
        address = safe_get(player, "address", default="N/A")
        total_score = safe_get(player, "total_score", default=0)
        total_games_p = safe_get(player, "total_games", default=0)
        total_earned = safe_get(player, "total_rewards_earned", default=0)
        level = safe_get(player, "level", default=1)
        experience = safe_get(player, "experience", default=0)
        achievements = safe_get(player, "achievements", default=[])
        consecutive = safe_get(player, "consecutive_days", default=0)

        ach_count = len(achievements) if isinstance(achievements, list) else 0

        lb_rows += f"""<tr>
            <td style="font-weight:bold;color:var(--positronic-yellow);">{i + 1}</td>
            <td><a href="/address/{html_escape(str(address))}">{short_hash(address)}</a></td>
            <td>{format_number(total_score)}</td>
            <td>{format_number(total_games_p)}</td>
            <td>{format_positronic(total_earned)}</td>
            <td>Lv.{level}</td>
            <td>{ach_count}</td>
            <td>{consecutive}d</td>
        </tr>"""

    leaderboard_html = f"""
    <div class="detail-section">
        <h2>Player Leaderboard</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>#</th><th>Player</th><th>Score</th><th>Games</th><th>Earned</th><th>Level</th><th>Achievements</th><th>Streak</th></tr></thead>
        <tbody>{lb_rows if lb_rows else '<tr><td colspan="8" style="text-align:center;color:var(--text-secondary)">No players registered yet. Start playing to earn ASF!</td></tr>'}</tbody>
    </table>
    </div>
    """

    # Achievement reference
    achievement_html = """
    <div class="detail-section">
        <h2>Achievements Reference</h2>
        <div class="detail-row"><div class="detail-key">First Jump</div><div class="detail-val">Play your first game - Reward: 1 ASF</div></div>
        <div class="detail-row"><div class="detail-key">Coin Collector</div><div class="detail-val">Collect 100 coins in a game - Reward: 5 ASF</div></div>
        <div class="detail-row"><div class="detail-key">Boss Beater</div><div class="detail-val">Defeat 10 enemies - Reward: 10 ASF</div></div>
        <div class="detail-row"><div class="detail-key">Speed Runner</div><div class="detail-val">Complete a level under 60 seconds - Reward: 15 ASF</div></div>
        <div class="detail-row"><div class="detail-key">Perfect Run</div><div class="detail-val">Complete a level with no damage - Reward: 20 ASF</div></div>
        <div class="detail-row"><div class="detail-key">World Champion</div><div class="detail-val">Complete all worlds - Reward: 100 ASF</div></div>
        <div class="detail-row"><div class="detail-key">Daily Player</div><div class="detail-val">Play 7 consecutive days - Reward: 3 ASF</div></div>
        <div class="detail-row"><div class="detail-key">Marathon</div><div class="detail-val">Play 30 consecutive days - Reward: 50 ASF</div></div>
    </div>
    """

    content = f"""
    <h1 class="page-title">Play-to-Earn Game</h1>
    {cards_html}
    {leaderboard_html}
    {achievement_html}
    """

    return base_template("Game", content, active="game")


def render_governance(rpc: RPCClient):
    """Governance page: token proposals."""
    gov_stats = rpc.get_governance_stats() or {}
    pending = rpc.get_pending_proposals() or []

    total_proposals = safe_get(gov_stats, "total_proposals", default=0)
    total_deployed = safe_get(gov_stats, "total_deployed", default=0)
    council_members = safe_get(gov_stats, "council_members", default=0)
    pending_count = safe_get(gov_stats, "pending", default=0)

    cards_html = f"""
    <div class="card-grid">
        <div class="card">
            <div class="card-label">Total Proposals</div>
            <div class="card-value blue">{format_number(total_proposals)}</div>
        </div>
        <div class="card">
            <div class="card-label">Tokens Deployed</div>
            <div class="card-value green">{format_number(total_deployed)}</div>
        </div>
        <div class="card">
            <div class="card-label">Council Members</div>
            <div class="card-value yellow">{format_number(council_members)}</div>
        </div>
        <div class="card">
            <div class="card-label">Pending Proposals</div>
            <div class="card-value red">{format_number(pending_count)}</div>
        </div>
    </div>
    """

    # Pending proposals table
    proposal_rows = ""
    for p in pending:
        pid = safe_get(p, "proposal_id", default="N/A")
        proposer = safe_get(p, "proposer", default="N/A")
        status = safe_get(p, "status", default="SUBMITTED")
        token_name = safe_get(p, "token_name", default="N/A")
        token_symbol = safe_get(p, "token_symbol", default="N/A")
        token_supply = safe_get(p, "token_supply", default=0)
        ai_risk = safe_get(p, "ai_risk_score", default=0)
        votes_for = safe_get(p, "votes_for", default=0)
        votes_against = safe_get(p, "votes_against", default=0)
        created_at = safe_get(p, "created_at", default=0)

        status_color = {
            "SUBMITTED": "#FBD000",
            "AI_REVIEWING": "#049CD8",
            "AI_APPROVED": "#43B047",
            "AI_REJECTED": "#E52521",
            "COUNCIL_VOTING": "#049CD8",
            "APPROVED": "#43B047",
            "REJECTED": "#E52521",
            "DEPLOYED": "#43B047",
        }.get(status, "#a0a0a0")

        proposal_rows += f"""<tr>
            <td>{html_escape(str(pid))}</td>
            <td style="font-weight:bold;">{html_escape(str(token_name))}</td>
            <td>{html_escape(str(token_symbol))}</td>
            <td>{format_number(token_supply)}</td>
            <td><a href="/address/{html_escape(str(proposer))}">{short_hash(proposer)}</a></td>
            <td><span class="badge" style="background:rgba(0,0,0,0.2);color:{status_color}">{html_escape(str(status))}</span></td>
            <td>{ai_risk}</td>
            <td style="color:var(--positronic-green);">{votes_for}</td>
            <td style="color:var(--positronic-red);">{votes_against}</td>
            <td>{time_ago(created_at)}</td>
        </tr>"""

    proposals_html = f"""
    <div class="detail-section">
        <h2>Active Proposals</h2>
    </div>
    <div class="table-wrapper">
    <table>
        <thead><tr><th>ID</th><th>Token Name</th><th>Symbol</th><th>Supply</th><th>Proposer</th><th>Status</th><th>AI Risk</th><th>For</th><th>Against</th><th>Created</th></tr></thead>
        <tbody>{proposal_rows if proposal_rows else '<tr><td colspan="10" style="text-align:center;color:var(--text-secondary)">No active proposals</td></tr>'}</tbody>
    </table>
    </div>
    """

    # Governance process
    process_html = """
    <div class="detail-section">
        <h2>Governance Process</h2>
        <div class="detail-row"><div class="detail-key">Step 1: Submit</div><div class="detail-val">Anyone can submit a token creation proposal with name, symbol, and supply.</div></div>
        <div class="detail-row"><div class="detail-key">Step 2: AI Review</div><div class="detail-val">The AI system analyzes the proposal for risk (scam detection, rug pull analysis). Max risk score: 0.7</div></div>
        <div class="detail-row"><div class="detail-key">Step 3: Council Vote</div><div class="detail-val">Governance council votes. Requires minimum 3 votes and 60% approval rate.</div></div>
        <div class="detail-row"><div class="detail-key">Step 4: Deploy</div><div class="detail-val">Approved tokens are deployed as smart contracts on the Positronic network.</div></div>
    </div>
    """

    content = f"""
    <h1 class="page-title">Token Governance</h1>
    {cards_html}
    {proposals_html}
    {process_html}
    """

    return base_template("Governance", content, active="governance")


def render_search_result(rpc: RPCClient, query: str):
    """Handle search: detect if query is block number, tx hash, or address."""
    q = query.strip()

    if not q:
        return render_error("Empty Search", "Please enter a block number, transaction hash, or address.")

    # Block number
    if q.isdigit():
        height = int(q)
        block = rpc.get_block(height)
        if block:
            return render_block(rpc, height)
        return render_error("Not Found", f"Block #{height} was not found.")

    # Hex value - could be tx hash, block hash, or address
    clean = q.removeprefix("0x")

    # TX hash (64 bytes = 128 hex chars for SHA-512)
    if len(clean) == 128:
        tx = rpc.get_transaction(q)
        if tx:
            return render_transaction(rpc, q)
        block = rpc.get_block_by_hash(q)
        if block:
            header = block.get("header", block)
            h = safe_get(header, "height", default=0)
            return render_block(rpc, h)
        return render_error("Not Found", f"No transaction or block found for hash: {q}")

    # Address (20 bytes = 40 hex chars)
    if len(clean) == 40:
        return render_address(rpc, q)

    # Try as address anyway if it has 0x prefix
    if q.startswith("0x"):
        return render_address(rpc, q)

    # Try as block number in hex
    try:
        height = int(q, 16)
        block = rpc.get_block(height)
        if block:
            return render_block(rpc, height)
    except ValueError:
        pass

    return render_error("Not Found", f"Could not find results for: {html_escape(q)}")


def render_error(title: str, message: str):
    """Render an error page."""
    content = f"""
    <div class="error-box">
        <h1>404</h1>
        <h2 style="color:var(--positronic-yellow);margin-bottom:16px;">{html_escape(title)}</h2>
        <p>{message}</p>
        <p style="margin-top:20px;"><a href="/">Return to Home</a></p>
    </div>
    """
    return base_template(title, content)


def render_not_found():
    """Render a generic 404 page."""
    return render_error("Page Not Found", "The page you are looking for does not exist.")


# ============================================================
# HTTP Request Handler
# ============================================================

class ExplorerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Positronic block explorer."""

    rpc = RPCClient()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        try:
            if path == "/":
                html = render_home(self.rpc)
            elif path == "/network":
                html = render_network(self.rpc)
            elif path == "/game":
                html = render_game(self.rpc)
            elif path == "/governance":
                html = render_governance(self.rpc)
            elif path == "/search":
                query = params.get("q", [""])[0]
                html = render_search_result(self.rpc, query)
            elif path.startswith("/block/"):
                try:
                    height = int(path.split("/block/")[1])
                    html = render_block(self.rpc, height)
                except (ValueError, IndexError):
                    html = render_error("Invalid Block", "Please provide a valid block number.")
            elif path.startswith("/tx/"):
                tx_hash = path.split("/tx/")[1]
                html = render_transaction(self.rpc, tx_hash)
            elif path.startswith("/address/"):
                address = path.split("/address/")[1]
                html = render_address(self.rpc, address)
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                html = render_not_found()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            encoded = html.encode("utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        except Exception as e:
            logger.warning("explorer_request_failed: %s", e)
            error_html = render_error("Internal Error", f"An error occurred: {html_escape(str(e))}")
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            encoded = error_html.encode("utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[Explorer] {self.client_address[0]} - {format % args}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Start the Positronic Block Explorer."""
    print("=" * 60)
    print("  Positronic Block Explorer")
    print("=" * 60)
    print(f"  RPC endpoint:  {RPC_URL}")
    print(f"  Explorer URL:  http://localhost:{EXPLORER_PORT}")
    print(f"  Chain ID:      {CHAIN_ID}")
    print("=" * 60)
    print()

    # Test RPC connection
    rpc = RPCClient()
    height = rpc.get_block_number()
    if height is not None:
        print(f"[Explorer] Connected to RPC. Chain height: {height}")
    else:
        print("[Explorer] WARNING: Cannot connect to RPC server at " + RPC_URL)
        print("[Explorer]          The explorer will show empty data until the node is started.")
        print()

    server = HTTPServer((EXPLORER_HOST, EXPLORER_PORT), ExplorerHandler)
    print(f"[Explorer] Server started on http://{EXPLORER_HOST}:{EXPLORER_PORT}")
    print("[Explorer] Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Explorer] Shutting down...")
        server.server_close()
        print("[Explorer] Server stopped.")


if __name__ == "__main__":
    main()
