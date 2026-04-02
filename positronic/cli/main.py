"""
Positronic - CLI Entry Point
Full-featured command-line wallet and node management tool.
Supports: wallet, node, chain, ai, game, governance commands.
"""

import argparse
import asyncio
import getpass
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime

from positronic.crypto.keys import KeyPair
from positronic.crypto.address import address_from_hex, address_to_hex
from positronic.chain.blockchain import Blockchain
from positronic.wallet.wallet import Wallet
from positronic.wallet.keystore import KeyStore
from positronic.network.node import Node
from positronic.utils.config import NodeConfig
from positronic.utils.encoding import format_positronic, format_denomination
from positronic.constants import (
    BASE_UNIT,
    CHAIN_ID,
    COIN_NAME,
    COIN_SYMBOL,
    DECIMALS,
    DEFAULT_P2P_PORT,
    DEFAULT_RPC_PORT,
    TOTAL_SUPPLY,
    BLOCK_TIME,
    MAX_VALIDATORS,
    INITIAL_BLOCK_REWARD,
    HALVING_INTERVAL,
    DENOMINATIONS,
    HD_WALLET_COIN_TYPE,
)


# ============================================================================
#  Positronic ASCII banner
# ============================================================================

BANNER = r"""
   ____           _ _                   _
  |  _ \ ___  ___(_) |_ _ __ ___  _ __ (_) ___
  | |_) / _ \/ __| | __| '__/ _ \| '_ \| |/ __|
  |  __/ (_) \__ \ | |_| | | (_) | | | | | (__
  |_|   \___/|___/_|\__|_|  \___/|_| |_|_|\___|

  The World's First AI-Validated Blockchain
  Proof of Neural Consensus (PoNC) | Chain ID: {chain_id}
""".format(chain_id=CHAIN_ID)

SHORT_BANNER = (
    "Positronic - AI-Validated Blockchain | "
    "Chain ID: {chain_id} | PoNC Consensus"
).format(chain_id=CHAIN_ID)

from positronic import __version__ as VERSION


# ============================================================================
#  Utility / formatting helpers
# ============================================================================

def _separator(char="-", width=60):
    """Print a horizontal separator line."""
    print(char * width)


def _header(title):
    """Print a formatted section header."""
    _separator("=")
    print(f"  {title}")
    _separator("=")


def _kv(key, value, key_width=26):
    """Print a key-value pair with aligned formatting."""
    print(f"  {key:<{key_width}} {value}")


def _table(headers, rows, col_widths=None):
    """Print a simple ASCII table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(min(max_w + 2, 50))

    # Header
    header_line = ""
    for i, h in enumerate(headers):
        header_line += str(h).ljust(col_widths[i])
    print(f"  {header_line}")
    print("  " + "-" * sum(col_widths))

    # Rows
    for row in rows:
        line = ""
        for i, cell in enumerate(row):
            if i < len(col_widths):
                cell_str = str(cell)
                if len(cell_str) > col_widths[i] - 2:
                    cell_str = cell_str[: col_widths[i] - 5] + "..."
                line += cell_str.ljust(col_widths[i])
        print(f"  {line}")


def _get_data_dir(args):
    """Extract data directory from args, with default."""
    return getattr(args, "data_dir", "./data") or "./data"


def _get_db_path(args):
    """Build the database path from args."""
    data_dir = _get_data_dir(args)
    return os.path.join(data_dir, "positronic.db")


def _get_keystore_dir(args):
    """Build the keystore directory path from args."""
    data_dir = _get_data_dir(args)
    return os.path.join(data_dir, "keystore")


def _load_chain(args):
    """Load blockchain from disk, printing an error if it doesn't exist."""
    db_path = _get_db_path(args)
    if not os.path.exists(db_path):
        print(f"[ERROR] No blockchain data found at {db_path}")
        print("        Run 'positronic node start --founder' to create the genesis chain.")
        sys.exit(1)
    chain = Blockchain(db_path=db_path)
    chain._load_chain()
    return chain


def _prompt_password(confirm=False):
    """Prompt for a password securely."""
    password = getpass.getpass("Password: ")
    if confirm:
        password2 = getpass.getpass("Confirm password: ")
        if password != password2:
            print("[ERROR] Passwords do not match.")
            sys.exit(1)
    return password


# ============================================================================
#  WALLET commands
# ============================================================================

def cmd_wallet_create(args):
    """Create a new wallet account."""
    _header("Create New Wallet")
    keystore_dir = _get_keystore_dir(args)
    password = _prompt_password(confirm=True)

    w = Wallet(keystore_dir=keystore_dir)
    kp = w.create_account(password)

    print()
    _kv("Address:", kp.address_hex)
    _kv("Public Key:", kp.public_key_bytes.hex()[:40] + "...")
    _kv("Keystore:", keystore_dir)
    print()
    print("  Your key has been encrypted and saved.")
    print("  [!] REMEMBER YOUR PASSWORD - it cannot be recovered.")
    _separator()


def cmd_wallet_balance(args):
    """Check balance for an address."""
    _header(f"Wallet Balance")
    chain = _load_chain(args)

    address_hex = args.address
    addr = address_from_hex(address_hex)
    balance = chain.get_balance(addr)
    nonce = chain.get_nonce(addr)

    denom = getattr(args, "denomination", "posi") or "posi"

    print()
    _kv("Address:", address_hex)
    _kv("Balance:", format_denomination(balance, denom))
    _kv("Balance (ASF):", format_positronic(balance))
    _kv("Balance (raw):", f"{balance} Pixel")
    _kv("Nonce:", str(nonce))

    # Show in all denominations
    if getattr(args, "all_denominations", False):
        print()
        print("  All Denominations:")
        _separator("-", 40)
        for name in DENOMINATIONS:
            _kv(f"    {name.upper()}:", format_denomination(balance, name))

    _separator()


def cmd_wallet_send(args):
    """Send ASF coins to an address."""
    _header("Send ASF")
    chain = _load_chain(args)
    keystore_dir = _get_keystore_dir(args)
    w = Wallet(keystore_dir=keystore_dir)

    from_address = args.sender
    to_address = args.recipient
    amount = args.amount
    gas_price = getattr(args, "gas_price", 1) or 1
    rpc_url = getattr(args, "rpc_url", None)

    # Validate address format
    clean_to = to_address.removeprefix("0x")
    if len(clean_to) != 40 or not all(c in "0123456789abcdefABCDEF" for c in clean_to):
        print(f"[ERROR] Invalid recipient address: {to_address}")
        print("  Address must be a 40-character hex string (with optional 0x prefix)")
        sys.exit(1)

    password = _prompt_password()

    # Load sender key
    try:
        kp = w.load_account(from_address, password)
    except Exception as e:
        print(f"[ERROR] Could not load wallet: {e}")
        sys.exit(1)

    nonce = chain.get_nonce(kp.address)
    value = int(amount * BASE_UNIT)
    recipient = address_from_hex(to_address)

    # Check effective balance (excludes staked funds)
    acc = chain.state.get_account(kp.address)
    available = acc.effective_balance
    if available < value:
        print(f"[ERROR] Insufficient balance.")
        _kv("  Total balance:", format_positronic(acc.balance))
        _kv("  Staked:", format_positronic(acc.staked_amount))
        _kv("  Available:", format_positronic(available))
        _kv("  Required:", format_positronic(value))
        sys.exit(1)

    # Build and sign transaction
    tx = w.build_transfer(recipient, value, nonce, gas_price=gas_price, keypair=kp)

    print()
    _kv("TX Hash:", tx.tx_hash_hex)
    _kv("From:", kp.address_hex)
    _kv("To:", to_address)
    _kv("Amount:", format_positronic(value))
    _kv("Gas Price:", str(gas_price))
    _kv("Nonce:", str(nonce))
    print()
    print("  Transaction created and signed successfully.")

    # Broadcast to running node via RPC
    target_url = rpc_url or f"http://127.0.0.1:{DEFAULT_RPC_PORT}"
    try:
        import urllib.request
        import urllib.error
        rpc_payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "eth_sendRawTransaction",
            "params": [tx.to_dict()],
            "id": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=rpc_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if "error" in result:
                print(f"  [WARN] Node rejected TX: {result['error']}")
                print("  Transaction saved locally but not broadcast.")
            else:
                print(f"  Transaction broadcast to node at {target_url}")
                print(f"  Awaiting inclusion in a block...")
    except (urllib.error.URLError, OSError):
        print(f"  [INFO] Could not reach node at {target_url}")
        print("  Transaction saved locally. Submit manually when a node is running.")
    _separator()


def cmd_wallet_list(args):
    """List all wallet accounts."""
    _header("Wallet Accounts")
    keystore_dir = _get_keystore_dir(args)
    keys = KeyStore.list_keys(keystore_dir)

    if not keys:
        print()
        print("  No wallets found.")
        print(f"  Keystore directory: {keystore_dir}")
        print("  Run 'positronic wallet create' to create a new wallet.")
        _separator()
        return

    rows = []
    for i, key in enumerate(keys, 1):
        rows.append([str(i), key["address"], key.get("file", "")])

    print()
    _table(["#", "Address", "File"], rows)

    # If blockchain is available, show balances
    db_path = _get_db_path(args)
    if os.path.exists(db_path):
        try:
            chain = Blockchain(db_path=db_path)
            chain._load_chain()
            print()
            print("  Balances:")
            _separator("-", 40)
            for key in keys:
                addr = address_from_hex(key["address"])
                balance = chain.get_balance(addr)
                _kv(f"    {key['address'][:18]}...:", format_positronic(balance))
        except Exception as e:
            logging.getLogger(__name__).debug("Could not load balances: %s", e)

    _separator()


# ============================================================================
#  NODE commands
# ============================================================================

def cmd_node_start(args):
    """Start a Positronic node."""
    print(BANNER)
    _header("Starting Positronic Node")

    # Load config from file if provided, otherwise use defaults
    config_path = getattr(args, "config", None)
    if config_path:
        if os.path.exists(config_path):
            try:
                config = NodeConfig.load(config_path)
                print(f"  Loaded config from: {config_path}")
            except Exception as e:
                logging.warning("config_parse_failed: %s", e)
                print(f"  [WARN] Failed to parse config file: {e}")
                print(f"         Using default configuration.")
                config = NodeConfig()
        else:
            print(f"  [WARN] Config file not found: {config_path}")
            print(f"         Using default configuration.")
            config = NodeConfig()
    else:
        config = NodeConfig()

    port = getattr(args, "port", DEFAULT_P2P_PORT) or DEFAULT_P2P_PORT
    rpc_port = getattr(args, "rpc_port", DEFAULT_RPC_PORT) or DEFAULT_RPC_PORT
    data_dir = _get_data_dir(args)
    rpc_host = getattr(args, "rpc_host", "0.0.0.0") or "0.0.0.0"
    founder = getattr(args, "founder", False)
    validator = getattr(args, "validator", False)
    bootstrap = getattr(args, "bootstrap", []) or []
    tls_enabled = getattr(args, "tls", False)
    network_type = getattr(args, "network_type", "mainnet") or "mainnet"

    _kv("P2P Port:", str(port))
    _kv("RPC Port:", str(rpc_port))
    _kv("RPC Host:", rpc_host)
    _kv("Data Dir:", data_dir)
    _kv("Validator:", "Yes" if validator else "No")
    _kv("Founder Mode:", "Yes" if founder else "No")
    _kv("TLS:", "Enabled" if tls_enabled else "Disabled")
    _kv("Network:", network_type)
    if bootstrap:
        _kv("Bootstrap Nodes:", ", ".join(bootstrap))
    _separator()

    # Override config with CLI arguments
    config.network.p2p_port = port
    config.network.rpc_port = rpc_port
    config.network.rpc_host = rpc_host
    config.storage.data_dir = data_dir
    config.network.network_type = network_type
    config.validator.enabled = validator
    config._founder_mode = founder  # Use genesis keypair for mining rewards
    if bootstrap:
        config.network.bootstrap_nodes = list(bootstrap)

    async def run():
        node_instance = Node(config)
        await node_instance.start(founder_mode=founder)

        # RPC server is now started automatically by Node.start()

        if founder:
            print()
            _kv("Founder Address:", node_instance.keypair.address_hex)
            _kv("Genesis Balance:", format_positronic(
                node_instance.blockchain.get_balance(node_instance.keypair.address)
            ))
            _separator()

        print()
        print("  Node is running. Press Ctrl+C to stop.")
        _separator()

        # Graceful shutdown flag for SIGTERM (docker stop)
        shutdown_event = asyncio.Event()

        def _sigterm_handler(signum, frame):
            print("\n  Received SIGTERM signal.")
            shutdown_event.set()

        # Register SIGTERM handler (POSIX-only; harmless no-op on Windows)
        try:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except (OSError, ValueError):
            pass  # Windows or non-main thread

        try:
            # Wait until Ctrl+C or SIGTERM
            while not shutdown_event.is_set():
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print()
            print("  Shutting down gracefully...")
            await node_instance.stop()
            print("  Node stopped.")

    asyncio.run(run())


def cmd_node_status(args):
    """Show node and chain status."""
    _header("Positronic Node Status")
    chain = _load_chain(args)
    stats = chain.get_stats()

    print()
    _kv("Chain ID:", str(stats["chain_id"]))
    _kv("Block Height:", str(stats["height"]))
    head = stats.get("head_hash", "N/A")
    _kv("Head Hash:", (head[:40] + "...") if head and len(head) > 40 else str(head))
    print()

    # State
    state = stats.get("state", {})
    _kv("Accounts:", str(state.get("account_count", 0)))
    _kv("Total Balance:", format_positronic(state.get("total_balance", 0)))
    print()

    # AI
    ai = stats.get("ai", {})
    _kv("AI Enabled:", str(ai.get("ai_enabled", False)))
    _kv("AI Model Version:", str(ai.get("model_version", "N/A")))
    _kv("Total Scored:", str(ai.get("total_scored", 0)))
    _kv("Accepted:", str(ai.get("accepted", 0)))
    _kv("Quarantined:", str(ai.get("quarantined", 0)))
    _kv("Rejected:", str(ai.get("rejected", 0)))
    _kv("Kill Switch:", str(ai.get("kill_switch_triggered", False)))
    print()

    # Quarantine
    quar = stats.get("quarantine", {})
    _kv("Quarantine Pool Size:", str(quar.get("pool_size", 0)))
    print()

    # Node ranking
    nr = stats.get("node_ranking", {})
    _kv("Total Nodes:", str(nr.get("total_nodes", 0)))
    print()

    # Game
    game = stats.get("game", {})
    _kv("Total Players:", str(game.get("total_players", 0)))
    _kv("Total Games Played:", str(game.get("total_games_played", 0)))
    print()

    # Governance
    gov = stats.get("governance", {})
    _kv("Total Proposals:", str(gov.get("total_proposals", 0)))
    _kv("Deployed Tokens:", str(gov.get("total_deployed", 0)))
    _kv("Pending Proposals:", str(gov.get("pending", 0)))

    _separator()


def cmd_node_health(args):
    """Show comprehensive network health status."""
    _header("Positronic Network Health")
    chain = _load_chain(args)
    health = chain.get_network_health()

    # Immune System
    immune = health.get("immune_status", {})
    print()
    print("  --- Neural Immune System ---")
    alert = immune.get("alert_level", "GREEN")
    alert_val = immune.get("alert_level_value", 0)
    alert_display = f"{alert} (Level {alert_val})"
    _kv("Alert Level:", alert_display)
    _kv("Total Threats:", str(immune.get("total_threats", 0)))
    _kv("Blocked Addresses:", str(immune.get("blocked_addresses", 0)))
    threat_types = immune.get("threat_types", {})
    if threat_types:
        _kv("Threat Types:", "")
        for ttype, count in threat_types.items():
            _kv(f"    {ttype}:", str(count))
    print()

    # Node Ranking
    nr = health.get("node_ranking", {})
    print("  --- Node Ranking ---")
    _kv("Total Nodes:", str(nr.get("total_nodes", 0)))
    rank_dist = nr.get("rank_distribution", {})
    if rank_dist:
        _kv("Rank Distribution:", "")
        for rank, count in rank_dist.items():
            display_rank = rank.replace("_", " ").title()
            _kv(f"    {display_rank}:", str(count))
    print()

    # AI Ranks
    ai_ranks = health.get("ai_ranks", {})
    print("  --- AI Validator Ranks ---")
    _kv("Total AI Validators:", str(ai_ranks.get("total_validators", 0)))
    ai_dist = ai_ranks.get("rank_distribution", {})
    if ai_dist:
        _kv("Rank Distribution:", "")
        for rank, count in ai_dist.items():
            display_rank = rank.replace("_", " ").title()
            _kv(f"    {display_rank}:", str(count))
    print()

    # Wallet Registry
    wr = health.get("wallet_registry", {})
    print("  --- Wallet Registry ---")
    _kv("Total Wallets:", str(wr.get("total_wallets", 0)))
    _kv("Verified:", str(wr.get("verified", 0)))
    _kv("Blacklisted:", str(wr.get("blacklisted", 0)))
    print()

    # Game Engine
    ge = health.get("game_engine", {})
    print("  --- Game Engine ---")
    _kv("Total Players:", str(ge.get("total_players", 0)))
    _kv("Games Played:", str(ge.get("total_games_played", 0)))
    _kv("Total Distributed:", format_positronic(ge.get("total_distributed", 0)))
    _kv("Pending Rewards:", format_positronic(ge.get("pending_rewards", 0)))
    print()

    # Governance
    gov = health.get("governance", {})
    print("  --- Token Governance ---")
    _kv("Total Proposals:", str(gov.get("total_proposals", 0)))
    _kv("Deployed Tokens:", str(gov.get("total_deployed", 0)))
    _kv("Council Members:", str(gov.get("council_members", 0)))
    _kv("Pending:", str(gov.get("pending", 0)))
    print()

    # Forensics
    forensics = health.get("forensics", {})
    print("  --- Forensic Reporter ---")
    _kv("Total Reports:", str(forensics.get("total_reports", 0)))
    _kv("Total Flagged:", str(forensics.get("total_flagged", 0)))

    _separator()


def cmd_node_backup(args):
    """Create a backup of the node's database."""
    import shutil
    _header("Node Backup")
    db_path = _get_db_path(args)
    if not os.path.exists(db_path):
        print("  [ERROR] No database found to back up.")
        _separator()
        return

    backup_dir = getattr(args, "output", None) or os.path.join(_get_data_dir(args), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"positronic_backup_{timestamp}.db")

    try:
        shutil.copy2(db_path, backup_file)
        size_mb = os.path.getsize(backup_file) / (1024 * 1024)
        print()
        _kv("Source:", db_path)
        _kv("Backup:", backup_file)
        _kv("Size:", f"{size_mb:.2f} MB")
        print()
        print("  Backup created successfully.")
    except Exception as e:
        print(f"  [ERROR] Backup failed: {e}")
    _separator()


def cmd_node_restore(args):
    """Restore a node's database from a backup."""
    import shutil
    _header("Node Restore")
    backup_path = args.backup_file
    if not os.path.exists(backup_path):
        print(f"  [ERROR] Backup file not found: {backup_path}")
        _separator()
        return

    db_path = _get_db_path(args)
    if os.path.exists(db_path):
        print(f"  [WARN] Existing database will be overwritten: {db_path}")
        confirm = input("  Continue? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Restore cancelled.")
            _separator()
            return

    try:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        shutil.copy2(backup_path, db_path)
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print()
        _kv("Restored from:", backup_path)
        _kv("Database:", db_path)
        _kv("Size:", f"{size_mb:.2f} MB")
        print()
        print("  Database restored successfully.")
    except Exception as e:
        print(f"  [ERROR] Restore failed: {e}")
    _separator()


# ============================================================================
#  CHAIN commands
# ============================================================================

def cmd_chain_info(args):
    """Show blockchain information as JSON."""
    _header("Blockchain Info")
    chain = _load_chain(args)
    stats = chain.get_stats()
    print()
    print(json.dumps(stats, indent=2, default=str))
    _separator()


def cmd_chain_block(args):
    """Show details of a specific block."""
    _header(f"Block #{args.height}")
    chain = _load_chain(args)
    block = chain.get_block(args.height)
    if block:
        print()
        print(json.dumps(block.to_dict(), indent=2, default=str))
    else:
        print(f"\n  Block {args.height} not found.")
    _separator()


# ============================================================================
#  GAME commands
# ============================================================================

def cmd_game_stats(args):
    """Show Play-to-Earn game statistics."""
    _header("Positronic Game Stats")
    chain = _load_chain(args)
    stats = chain.game_engine.get_stats()

    print()
    _kv("Total Players:", str(stats.get("total_players", 0)))
    _kv("Total Games Played:", str(stats.get("total_games_played", 0)))
    _kv("Total Distributed:", format_positronic(stats.get("total_distributed", 0)))
    _kv("Pending Rewards:", format_positronic(stats.get("pending_rewards", 0)))
    print()

    # Show achievement rewards reference
    print("  --- Achievement Rewards ---")
    from positronic.game.play_to_earn import GameAchievement, ACHIEVEMENT_REWARDS
    for ach in GameAchievement:
        reward = ACHIEVEMENT_REWARDS.get(ach, 0)
        _kv(f"    {ach.name}:", format_positronic(reward))

    _separator()


def cmd_game_leaderboard(args):
    """Show game leaderboard."""
    _header("Positronic Game Leaderboard")
    chain = _load_chain(args)
    limit = getattr(args, "limit", 20) or 20
    leaderboard = chain.game_engine.get_leaderboard(limit=limit)

    if not leaderboard:
        print("\n  No players yet. Start playing to earn ASF!")
        _separator()
        return

    print()
    rows = []
    for i, p in enumerate(leaderboard, 1):
        addr = p.get("address", "")
        addr_short = f"0x{addr[:12]}..." if len(addr) > 12 else addr
        rows.append([
            str(i),
            addr_short,
            str(p.get("level", 1)),
            str(p.get("total_score", 0)),
            str(p.get("total_games", 0)),
            format_positronic(p.get("total_rewards_earned", 0)),
        ])

    _table(
        ["#", "Player", "Level", "Score", "Games", "Earned"],
        rows,
        col_widths=[4, 18, 7, 10, 7, 18],
    )
    _separator()


# ============================================================================
#  GOVERNANCE commands
# ============================================================================

def cmd_governance_proposals(args):
    """List pending governance proposals."""
    _header("Token Governance Proposals")
    chain = _load_chain(args)
    proposals = chain.token_governance.get_pending_proposals()
    gov_stats = chain.token_governance.get_stats()

    print()
    _kv("Total Proposals:", str(gov_stats.get("total_proposals", 0)))
    _kv("Deployed Tokens:", str(gov_stats.get("total_deployed", 0)))
    _kv("Council Members:", str(gov_stats.get("council_members", 0)))
    _kv("Pending:", str(gov_stats.get("pending", 0)))
    print()

    if not proposals:
        print("  No pending proposals.")
        _separator()
        return

    print("  --- Pending Proposals ---")
    _separator("-", 40)
    for p in proposals:
        d = p.to_dict()
        print()
        _kv("Proposal ID:", d.get("proposal_id", ""))
        _kv("Status:", d.get("status", ""))
        _kv("Token Name:", d.get("token_name", ""))
        _kv("Token Symbol:", d.get("token_symbol", ""))
        _kv("Supply:", str(d.get("token_supply", 0)))
        _kv("Decimals:", str(d.get("token_decimals", 18)))
        _kv("AI Risk Score:", f"{d.get('ai_risk_score', 0):.4f}")
        _kv("Votes For:", str(d.get("votes_for", 0)))
        _kv("Votes Against:", str(d.get("votes_against", 0)))
        proposer = d.get("proposer", "")
        _kv("Proposer:", f"0x{proposer[:16]}..." if len(proposer) > 16 else proposer)
        _separator("-", 40)

    _separator()


# ============================================================================
#  AI commands
# ============================================================================

def cmd_ai_rank(args):
    """Show AI rank for a specific address."""
    _header("AI Validator Rank")
    chain = _load_chain(args)

    address_hex = args.address
    addr = address_from_hex(address_hex)
    profile = chain.ai_rank_manager.get_profile(addr)

    if not profile:
        print(f"\n  No AI validator profile found for {address_hex}")
        print("  This address may not be registered as an AI validator.")
        _separator()
        return

    info = profile.to_dict()
    print()
    _kv("Address:", f"0x{info.get('address', '')}")
    _kv("Rank:", f"{info.get('rank_name', 'N/A')} (Level {info.get('rank', 0)})")
    _kv("Rank (Farsi):", info.get("rank_name_fa", ""))
    _kv("Total Scored:", str(info.get("total_scored", 0)))
    _kv("Accuracy:", f"{info.get('accuracy', 0):.4f}")
    _kv("Uptime (hours):", f"{info.get('uptime_hours', 0):.1f}")
    _kv("Days Active:", str(info.get("days_active", 0)))
    _kv("Reward Multiplier:", f"{info.get('reward_multiplier', 1.0):.1f}x")
    _kv("Promotions:", str(info.get("promotions", 0)))
    _kv("Demotions:", str(info.get("demotions", 0)))
    _separator()


def cmd_ai_immune_status(args):
    """Show neural immune system status."""
    _header("Neural Immune System Status")
    chain = _load_chain(args)
    status = chain.immune_system.get_status()

    alert = status.get("alert_level", "GREEN")
    alert_val = status.get("alert_level_value", 0)

    # Alert level visual indicator
    alert_indicators = {
        "GREEN": "[OK]",
        "YELLOW": "[!]",
        "ORANGE": "[!!]",
        "RED": "[!!!]",
        "BLACK": "[CRITICAL]",
    }
    indicator = alert_indicators.get(alert, "")

    print()
    _kv("Alert Level:", f"{alert} {indicator} (Level {alert_val})")
    _kv("Total Threats:", str(status.get("total_threats", 0)))
    _kv("Blocked Addresses:", str(status.get("blocked_addresses", 0)))
    print()

    threat_types = status.get("threat_types", {})
    if threat_types:
        print("  --- Threat Breakdown ---")
        for ttype, count in threat_types.items():
            _kv(f"    {ttype}:", str(count))
        print()

    # Recent threats
    limit = getattr(args, "limit", 10) or 10
    recent = chain.immune_system.get_recent_threats(limit=limit)
    if recent:
        print(f"  --- Recent Threats (last {len(recent)}) ---")
        for t in recent:
            src = t.get("source_address", "")
            src_short = f"0x{src[:12]}..." if len(src) > 12 else src
            severity = t.get("severity", 0)
            desc = t.get("description", "")
            ts = t.get("timestamp", 0)
            time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"
            print(f"    [{time_str}] severity={severity:.2f}  src={src_short}")
            if desc:
                print(f"      {desc}")
        print()

    # Recent responses
    responses = chain.immune_system.get_recent_responses(limit=limit)
    if responses:
        print(f"  --- Recent Responses (last {len(responses)}) ---")
        for r in responses:
            action = r.get("action_type", "")
            target = r.get("target_address", "")
            target_short = f"0x{target[:12]}..." if len(target) > 12 else target
            al = r.get("alert_level", "")
            details = r.get("details", "")
            print(f"    [{al}] {action} -> {target_short}")
            if details:
                print(f"      {details}")

    _separator()


def cmd_ai_status(args):
    """Show AI validation gate status (legacy command)."""
    _header("AI Validation Gate Status")
    chain = _load_chain(args)
    stats = chain.ai_gate.get_stats()

    print()
    _kv("Model Version:", str(stats.get("model_version", "N/A")))
    _kv("AI Enabled:", str(stats.get("ai_enabled", False)))
    _kv("Kill Switch:", str(stats.get("kill_switch_triggered", False)))
    _kv("Total Scored:", str(stats.get("total_scored", 0)))
    _kv("Accepted:", str(stats.get("accepted", 0)))
    _kv("Quarantined:", str(stats.get("quarantined", 0)))
    _kv("Rejected:", str(stats.get("rejected", 0)))
    _kv("False Positive Rate:", f"{stats.get('false_positive_rate', 0):.4f}")
    _separator()


# ============================================================================
#  INFO / VERSION command
# ============================================================================

# ============================================================================
#  PHASE 17: HD WALLET commands
# ============================================================================

def cmd_wallet_hd_create(args):
    """Create a new HD wallet with mnemonic phrase."""
    _header("Create HD Wallet")

    word_count = getattr(args, "words", 24) or 24
    passphrase = getattr(args, "passphrase", "") or ""

    try:
        from positronic.wallet.hd_wallet import HDWallet
        wallet = HDWallet.create(word_count=word_count)

        print()
        print("  [!] WRITE DOWN YOUR MNEMONIC PHRASE AND KEEP IT SAFE!")
        print("  [!] Anyone with this phrase can access ALL your funds!")
        _separator("-", 60)
        print()
        print(f"  {wallet.mnemonic}")
        print()
        _separator("-", 60)

        # Derive the first address
        first = wallet.derive_address(account=0, index=0)
        print()
        _kv("Word Count:", str(word_count))
        _kv("Coin Type:", str(HD_WALLET_COIN_TYPE))
        _kv("First Address:", first.address_hex)
        _kv("Derivation Path:", f"m/44'/{HD_WALLET_COIN_TYPE}'/0'/0/0")
        print()
        print("  HD wallet created successfully.")
        print("  Use 'positronic wallet hd-derive' to derive more addresses.")
    except Exception as e:
        print(f"  [ERROR] Failed to create HD wallet: {e}")
    _separator()


def cmd_wallet_hd_restore(args):
    """Restore an HD wallet from mnemonic phrase."""
    _header("Restore HD Wallet")

    mnemonic = args.mnemonic
    passphrase = getattr(args, "passphrase", "") or ""

    try:
        from positronic.wallet.hd_wallet import HDWallet
        wallet = HDWallet.from_mnemonic(mnemonic, passphrase=passphrase)

        # Derive the first few addresses
        count = getattr(args, "count", 5) or 5
        addresses = wallet.get_all_addresses(account=0, count=count)

        print()
        _kv("Mnemonic Words:", str(len(mnemonic.split())))
        _kv("Coin Type:", str(HD_WALLET_COIN_TYPE))
        print()

        print("  --- Derived Addresses ---")
        _separator("-", 50)
        rows = []
        for addr in addresses:
            rows.append([
                addr["path"],
                addr["address"][:22] + "...",
            ])
        _table(["Path", "Address"], rows)

        print()
        print("  HD wallet restored successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to restore HD wallet: {e}")
    _separator()


def cmd_wallet_hd_derive(args):
    """Derive a new address from HD wallet mnemonic."""
    _header("Derive HD Address")

    mnemonic = args.mnemonic
    account = getattr(args, "account", 0) or 0
    index = getattr(args, "index", 0) or 0
    passphrase = getattr(args, "passphrase", "") or ""

    try:
        from positronic.wallet.hd_wallet import HDWallet
        wallet = HDWallet.from_mnemonic(mnemonic, passphrase=passphrase)
        kp = wallet.derive_address(account=account, index=index)

        path = f"m/44'/{HD_WALLET_COIN_TYPE}'/{account}'/0/{index}"

        print()
        _kv("Derivation Path:", path)
        _kv("Address:", kp.address_hex)
        _kv("Public Key:", kp.public_key_bytes.hex()[:40] + "...")
        print()
        print("  Address derived successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to derive address: {e}")
    _separator()


def cmd_wallet_history(args):
    """Show transaction history for an address."""
    _header("Transaction History")

    address_hex = args.address
    limit = getattr(args, "limit", 20) or 20

    try:
        from positronic.wallet.tx_history import TxHistoryTracker
        tracker = TxHistoryTracker()

        # Scan the blockchain for this address
        chain = _load_chain(args)
        tracker.watch_address(address_hex.removeprefix("0x"))

        # Scan recent blocks
        start = max(0, chain.height - 1000)
        for h in range(start, chain.height + 1):
            block = chain.get_block(h)
            if block:
                for tx in block.transactions:
                    tx_dict = tx.to_dict()
                    tracker.record_transaction(
                        tx_dict, h,
                        timestamp=getattr(tx, "timestamp", 0),
                        status="confirmed",
                    )

        history = tracker.get_history(
            address_hex.removeprefix("0x"), limit=limit,
        )

        if not history:
            print(f"\n  No transactions found for {address_hex}")
            print(f"  (Scanned blocks {start} to {chain.height})")
            _separator()
            return

        print()
        _kv("Address:", address_hex)
        _kv("Transactions Found:", str(len(history)))
        print()

        rows = []
        for entry in history:
            direction = entry.direction.upper() if entry.direction else "?"
            value_str = format_positronic(entry.value) if entry.value else "0"
            counterparty = entry.counterparty[:16] + "..." if entry.counterparty else "N/A"
            rows.append([
                direction,
                entry.tx_type or "transfer",
                value_str,
                counterparty,
                str(entry.block_height),
            ])

        _table(
            ["Dir", "Type", "Value", "Counterparty", "Block"],
            rows,
            col_widths=[6, 14, 18, 20, 8],
        )

    except Exception as e:
        print(f"  [ERROR] Failed to get history: {e}")
    _separator()


def cmd_wallet_address_book(args):
    """Manage named addresses."""
    _header("Address Book")

    data_dir = _get_data_dir(args)
    book_path = os.path.join(data_dir, "address_book.json")

    try:
        from positronic.wallet.address_book import AddressBook
        book = AddressBook(book_path)

        action = getattr(args, "action", "list") or "list"

        if action == "add":
            name = getattr(args, "name", None)
            address = getattr(args, "addr", None)
            label = getattr(args, "label", "") or ""
            if not name or not address:
                print("  [ERROR] --name and --addr are required for 'add'")
                _separator()
                return
            book.add(name, address, label=label)
            print(f"  Added: {name} -> {address}")

        elif action == "remove":
            name = getattr(args, "name", None)
            if not name:
                print("  [ERROR] --name is required for 'remove'")
                _separator()
                return
            if book.remove(name):
                print(f"  Removed: {name}")
            else:
                print(f"  Not found: {name}")

        elif action == "search":
            query = getattr(args, "query", None)
            if not query:
                print("  [ERROR] --query is required for 'search'")
                _separator()
                return
            results = book.search(query)
            if not results:
                print(f"  No matches for '{query}'")
            else:
                rows = []
                for r in results:
                    rows.append([r["name"], r["address"][:22] + "...", r.get("label", "")])
                _table(["Name", "Address", "Label"], rows)

        else:  # list
            entries = book.list_all()
            if not entries:
                print("  Address book is empty.")
                print("  Use 'positronic wallet address-book add --name NAME --addr ADDRESS'")
            else:
                rows = []
                for e in entries:
                    rows.append([e["name"], e["address"][:22] + "...", e.get("label", "")])
                print()
                _table(["Name", "Address", "Label"], rows)

    except Exception as e:
        print(f"  [ERROR] {e}")
    _separator()


# ============================================================================
#  PHASE 17: CHAIN GAS INFO command
# ============================================================================

def cmd_chain_gas_info(args):
    """Show gas oracle information."""
    _header("Gas Oracle Info")

    try:
        from positronic.chain.gas_oracle import GasOracle
        oracle = GasOracle()

        # Try to load chain and update oracle with recent blocks
        try:
            chain = _load_chain(args)
            # Simulate recent block utilization
            for h in range(max(0, chain.height - 10), chain.height + 1):
                block = chain.get_block(h)
                if block:
                    tx_count = len(block.transactions)
                    gas_used = tx_count * 21000  # Estimate
                    oracle.update_base_fee(
                        gas_used, 30_000_000,
                        block_height=h, tx_count=tx_count,
                    )
        except SystemExit:
            pass  # No chain data, use defaults

        suggestion = oracle.get_fee_suggestion()
        stats = oracle.get_stats()

        print()
        print("  --- Current Gas Pricing ---")
        _kv("Base Fee:", str(suggestion["base_fee"]))
        _kv("Priority Fee (Low):", str(suggestion["priority_fee_low"]))
        _kv("Priority Fee (Medium):", str(suggestion["priority_fee_medium"]))
        _kv("Priority Fee (High):", str(suggestion["priority_fee_high"]))
        print()

        print("  --- Oracle Stats ---")
        _kv("Fee Floor:", str(stats["fee_floor"]))
        _kv("Fee Ceiling:", str(stats["fee_ceiling"]))
        _kv("Avg Utilization:", f"{stats['avg_utilization']:.2%}")
        _kv("History Size:", str(stats["history_size"]))
        print()

        # Show fee history
        history = oracle.get_fee_history(5)
        if history:
            print("  --- Recent Fee History ---")
            rows = []
            for entry in history:
                rows.append([
                    str(entry["height"]),
                    str(entry["base_fee"]),
                    f"{entry['utilization']:.2%}",
                    str(entry["tx_count"]),
                ])
            _table(
                ["Block", "Base Fee", "Utilization", "TXs"],
                rows,
                col_widths=[10, 12, 14, 8],
            )

    except Exception as e:
        print(f"  [ERROR] {e}")
    _separator()


# ============================================================================
#  INFO / VERSION command
# ============================================================================

def cmd_info(args):
    """Show Positronic chain info and parameters."""
    print(BANNER)
    _header("Chain Parameters")

    print()
    _kv("Coin Name:", COIN_NAME)
    _kv("Symbol:", COIN_SYMBOL)
    _kv("Chain ID:", str(CHAIN_ID))
    _kv("Decimals:", str(DECIMALS))
    _kv("Total Supply:", format_positronic(TOTAL_SUPPLY))
    _kv("Block Time:", f"{BLOCK_TIME} seconds")
    _kv("Max Validators:", str(MAX_VALIDATORS))
    _kv("Initial Block Reward:", format_positronic(INITIAL_BLOCK_REWARD))
    _kv("Halving Interval:", f"{HALVING_INTERVAL:,} blocks")
    _kv("Default P2P Port:", str(DEFAULT_P2P_PORT))
    _kv("Default RPC Port:", str(DEFAULT_RPC_PORT))
    print()

    print("  --- Denominations ---")
    _separator("-", 40)
    for name, value in DENOMINATIONS.items():
        _kv(f"    1 {name.upper()}:", f"{value:,} Pixel")

    _separator()


# ============================================================================
#  Argument parser construction
# ============================================================================

def build_parser():
    """Build the argparse parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog="positronic",
        description=SHORT_BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  positronic wallet create                     Create a new wallet\n"
            "  positronic wallet balance 0xabc123...         Check balance\n"
            "  positronic wallet send 0xfrom 0xto 10.5       Send 10.5 ASF\n"
            "  positronic wallet list                        List all wallets\n"
            "  positronic wallet hd-create --words 24        Create HD wallet\n"
            "  positronic wallet hd-restore 'word1 word2...' Restore from mnemonic\n"
            "  positronic wallet hd-derive 'mnemonic' --index 3  Derive address\n"
            "  positronic wallet history 0xabc123...         TX history\n"
            "  positronic wallet address-book                Address book\n"
            "  positronic node start --founder               Start genesis node\n"
            "  positronic node status                        Show chain status\n"
            "  positronic node health                        Network health report\n"
            "  positronic chain gas-info                     Gas oracle pricing\n"
            "  positronic game stats                         Game statistics\n"
            "  positronic game leaderboard                   Player leaderboard\n"
            "  positronic governance proposals               Pending proposals\n"
            "  positronic ai rank 0xabc123...                AI validator rank\n"
            "  positronic ai immune-status                   Immune system status\n"
            "  positronic info                               Show chain parameters\n"
        ),
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"%(prog)s {VERSION} ({COIN_NAME})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ----------------------------------------------------------------
    #  wallet
    # ----------------------------------------------------------------
    wallet_parser = subparsers.add_parser("wallet", help="Wallet management")
    wallet_sub = wallet_parser.add_subparsers(dest="wallet_cmd", help="Wallet commands")

    # wallet create
    wc = wallet_sub.add_parser("create", help="Create a new wallet account")
    wc.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    wc.set_defaults(func=cmd_wallet_create)

    # wallet balance
    wb = wallet_sub.add_parser("balance", help="Check balance for an address")
    wb.add_argument("address", help="Wallet address (0x-prefixed hex)")
    wb.add_argument("--data-dir", default="./data", help="Data directory")
    wb.add_argument("--denomination", "-d", default="posi",
                    choices=list(DENOMINATIONS.keys()),
                    help="Display denomination (default: posi)")
    wb.add_argument("--all-denominations", "-a", action="store_true",
                    help="Show balance in all denominations")
    wb.set_defaults(func=cmd_wallet_balance)

    # wallet send
    ws = wallet_sub.add_parser("send", help="Send ASF coins")
    ws.add_argument("sender", help="Sender address (0x-prefixed hex)")
    ws.add_argument("recipient", help="Recipient address (0x-prefixed hex)")
    ws.add_argument("amount", type=float, help="Amount of ASF to send")
    ws.add_argument("--data-dir", default="./data", help="Data directory")
    ws.add_argument("--gas-price", type=int, default=1, help="Gas price (default: 1)")
    ws.add_argument("--rpc-url", default=None,
                    help=f"RPC endpoint URL (default: http://127.0.0.1:{DEFAULT_RPC_PORT})")
    ws.set_defaults(func=cmd_wallet_send)

    # wallet list
    wl = wallet_sub.add_parser("list", help="List all wallet accounts")
    wl.add_argument("--data-dir", default="./data", help="Data directory")
    wl.set_defaults(func=cmd_wallet_list)

    # wallet hd-create (Phase 17)
    whc = wallet_sub.add_parser("hd-create", help="Create HD wallet with mnemonic")
    whc.add_argument("--words", type=int, default=24, choices=[12, 15, 18, 21, 24],
                     help="Mnemonic word count (default: 24)")
    whc.add_argument("--passphrase", default="", help="Optional BIP-39 passphrase")
    whc.set_defaults(func=cmd_wallet_hd_create)

    # wallet hd-restore (Phase 17)
    whr = wallet_sub.add_parser("hd-restore", help="Restore HD wallet from mnemonic")
    whr.add_argument("mnemonic", help="Mnemonic phrase (quoted string)")
    whr.add_argument("--passphrase", default="", help="Optional BIP-39 passphrase")
    whr.add_argument("--count", type=int, default=5, help="Number of addresses to derive (default: 5)")
    whr.set_defaults(func=cmd_wallet_hd_restore)

    # wallet hd-derive (Phase 17)
    whd = wallet_sub.add_parser("hd-derive", help="Derive address from HD mnemonic")
    whd.add_argument("mnemonic", help="Mnemonic phrase (quoted string)")
    whd.add_argument("--account", type=int, default=0, help="Account index (default: 0)")
    whd.add_argument("--index", type=int, default=0, help="Address index (default: 0)")
    whd.add_argument("--passphrase", default="", help="Optional BIP-39 passphrase")
    whd.set_defaults(func=cmd_wallet_hd_derive)

    # wallet history (Phase 17)
    whi = wallet_sub.add_parser("history", help="Show TX history for an address")
    whi.add_argument("address", help="Wallet address (0x-prefixed hex)")
    whi.add_argument("--data-dir", default="./data", help="Data directory")
    whi.add_argument("--limit", type=int, default=20, help="Max entries (default: 20)")
    whi.set_defaults(func=cmd_wallet_history)

    # wallet address-book (Phase 17)
    wab = wallet_sub.add_parser("address-book", help="Manage named addresses")
    wab.add_argument("action", nargs="?", default="list",
                     choices=["list", "add", "remove", "search"],
                     help="Action (default: list)")
    wab.add_argument("--name", help="Contact name")
    wab.add_argument("--addr", help="Contact address")
    wab.add_argument("--label", default="", help="Optional label")
    wab.add_argument("--query", help="Search query")
    wab.add_argument("--data-dir", default="./data", help="Data directory")
    wab.set_defaults(func=cmd_wallet_address_book)

    # ----------------------------------------------------------------
    #  node
    # ----------------------------------------------------------------
    node_parser = subparsers.add_parser("node", help="Node management")
    node_sub = node_parser.add_subparsers(dest="node_cmd", help="Node commands")

    # node start
    ns = node_sub.add_parser("start", help="Start a Positronic node")
    ns.add_argument("--port", "--p2p-port", type=int, default=DEFAULT_P2P_PORT,
                    dest="port",
                    help=f"P2P port (default: {DEFAULT_P2P_PORT})")
    ns.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT,
                    help=f"RPC port (default: {DEFAULT_RPC_PORT})")
    ns.add_argument("--rpc-host", default="0.0.0.0",
                    help="RPC server bind address (default: 0.0.0.0)")
    ns.add_argument("--data-dir", default="./data", help="Data directory")
    ns.add_argument("--founder", action="store_true",
                    help="Start as founder (create genesis block)")
    ns.add_argument("--validator", action="store_true",
                    help="Enable block production")
    ns.add_argument("--bootstrap", nargs="*", default=[],
                    help="Bootstrap node URLs")
    ns.add_argument("--tls", action="store_true",
                    help="Enable TLS (wss://) for P2P connections")
    ns.add_argument("--network-type", default="mainnet",
                    choices=["mainnet", "testnet", "local"],
                    help="Network type (default: mainnet)")
    ns.add_argument("--config", default=None,
                    help="Path to YAML configuration file")
    ns.set_defaults(func=cmd_node_start)

    # node status
    nst = node_sub.add_parser("status", help="Show node and chain status")
    nst.add_argument("--data-dir", default="./data", help="Data directory")
    nst.set_defaults(func=cmd_node_status)

    # node health
    nh = node_sub.add_parser("health", help="Show network health report")
    nh.add_argument("--data-dir", default="./data", help="Data directory")
    nh.set_defaults(func=cmd_node_health)

    # node backup
    nb = node_sub.add_parser("backup", help="Create database backup")
    nb.add_argument("--data-dir", default="./data", help="Data directory")
    nb.add_argument("--output", default=None, help="Backup output directory")
    nb.set_defaults(func=cmd_node_backup)

    # node restore
    nrs = node_sub.add_parser("restore", help="Restore database from backup")
    nrs.add_argument("backup_file", help="Path to backup file")
    nrs.add_argument("--data-dir", default="./data", help="Data directory")
    nrs.set_defaults(func=cmd_node_restore)

    # ----------------------------------------------------------------
    #  chain
    # ----------------------------------------------------------------
    chain_parser = subparsers.add_parser("chain", help="Blockchain inspection")
    chain_sub = chain_parser.add_subparsers(dest="chain_cmd", help="Chain commands")

    # chain info
    ci = chain_sub.add_parser("info", help="Show blockchain information (JSON)")
    ci.add_argument("--data-dir", default="./data", help="Data directory")
    ci.set_defaults(func=cmd_chain_info)

    # chain block
    cb = chain_sub.add_parser("block", help="Show block details")
    cb.add_argument("height", type=int, help="Block height")
    cb.add_argument("--data-dir", default="./data", help="Data directory")
    cb.set_defaults(func=cmd_chain_block)

    # chain gas-info (Phase 17)
    cg = chain_sub.add_parser("gas-info", help="Show gas oracle pricing info")
    cg.add_argument("--data-dir", default="./data", help="Data directory")
    cg.set_defaults(func=cmd_chain_gas_info)

    # ----------------------------------------------------------------
    #  game
    # ----------------------------------------------------------------
    game_parser = subparsers.add_parser("game", help="Play-to-Earn game commands")
    game_sub = game_parser.add_subparsers(dest="game_cmd", help="Game commands")

    # game stats
    gs = game_sub.add_parser("stats", help="Show game statistics")
    gs.add_argument("--data-dir", default="./data", help="Data directory")
    gs.set_defaults(func=cmd_game_stats)

    # game leaderboard
    gl = game_sub.add_parser("leaderboard", help="Show player leaderboard")
    gl.add_argument("--data-dir", default="./data", help="Data directory")
    gl.add_argument("--limit", type=int, default=20, help="Number of entries (default: 20)")
    gl.set_defaults(func=cmd_game_leaderboard)

    # ----------------------------------------------------------------
    #  governance
    # ----------------------------------------------------------------
    gov_parser = subparsers.add_parser("governance", help="Token governance commands")
    gov_sub = gov_parser.add_subparsers(dest="gov_cmd", help="Governance commands")

    # governance proposals
    gp = gov_sub.add_parser("proposals", help="List pending governance proposals")
    gp.add_argument("--data-dir", default="./data", help="Data directory")
    gp.set_defaults(func=cmd_governance_proposals)

    # ----------------------------------------------------------------
    #  ai
    # ----------------------------------------------------------------
    ai_parser = subparsers.add_parser("ai", help="AI validation commands")
    ai_sub = ai_parser.add_subparsers(dest="ai_cmd", help="AI commands")

    # ai rank
    ar = ai_sub.add_parser("rank", help="Show AI validator rank for an address")
    ar.add_argument("address", help="Validator address (0x-prefixed hex)")
    ar.add_argument("--data-dir", default="./data", help="Data directory")
    ar.set_defaults(func=cmd_ai_rank)

    # ai immune-status
    ais = ai_sub.add_parser("immune-status", help="Show neural immune system status")
    ais.add_argument("--data-dir", default="./data", help="Data directory")
    ais.add_argument("--limit", type=int, default=10,
                     help="Number of recent events to show (default: 10)")
    ais.set_defaults(func=cmd_ai_immune_status)

    # ai status (legacy - validation gate)
    ast = ai_sub.add_parser("status", help="Show AI validation gate status")
    ast.add_argument("--data-dir", default="./data", help="Data directory")
    ast.set_defaults(func=cmd_ai_status)

    # ----------------------------------------------------------------
    #  info (top-level)
    # ----------------------------------------------------------------
    info_parser = subparsers.add_parser("info", help="Show chain parameters and info")
    info_parser.set_defaults(func=cmd_info)

    return parser


# ============================================================================
#  Entrypoint
# ============================================================================

class _IPMaskFormatter(logging.Formatter):
    """Masks IPv4 addresses in log output — keeps first two octets, hides last two."""
    _IP_RE = re.compile(r'\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b')

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return self._IP_RE.sub(r'\1.x.x', msg)


def main():
    """Main CLI entrypoint."""
    # Configure logging so all getLogger() calls across the project
    # actually produce output instead of being silently discarded.
    log_level = os.environ.get("POSITRONIC_LOG_LEVEL", "INFO").upper()
    fmt = _IPMaskFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()  # Remove any pre-existing handlers
    root.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(handler)

    parser = build_parser()
    args = parser.parse_args()

    # If no command given, show banner + help
    if not args.command:
        print(BANNER)
        parser.print_help()
        return

    # If a subcommand group is given but no specific action, show its help
    if not hasattr(args, "func"):
        # Find the right sub-parser to print help for
        subparser_map = {
            "wallet": "wallet_cmd",
            "node": "node_cmd",
            "chain": "chain_cmd",
            "game": "game_cmd",
            "governance": "gov_cmd",
            "ai": "ai_cmd",
        }
        subcmd_attr = subparser_map.get(args.command)
        if subcmd_attr and not getattr(args, subcmd_attr, None):
            print(SHORT_BANNER)
            print()
            parser.parse_args([args.command, "--help"])
        else:
            parser.print_help()
        return

    # Execute the command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
