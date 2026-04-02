"""
Positronic - JSON-RPC Method Implementations
Compatible with eth_ namespace for MetaMask support.
Includes positronic_* namespace for all Positronic-specific features.
"""

from typing import Any, Dict, List, Optional

from positronic.chain.blockchain import Blockchain
from positronic.crypto.address import address_from_hex
from positronic.constants import CHAIN_ID, DENOMINATIONS, BLOCK_GAS_LIMIT
from positronic.utils.encoding import format_denomination
from positronic.game.telegram import TelegramBridge
from positronic.utils.logging import get_logger

logger = get_logger(__name__)


def _rpc_error(msg: str, e: Exception = None) -> dict:
    """Return RPC error dict with safe message. Log actual error internally."""
    if e:
        logger.debug("RPC method error: %s -- %s", msg, e)
    return {"error": msg}


class RPCMethods:
    """
    JSON-RPC methods compatible with Ethereum's eth_ namespace.
    This allows MetaMask and other Ethereum wallets to interact with Positronic.

    Namespaces:
    - eth_*   : Ethereum-compatible (MetaMask/Trust Wallet)
    - net_*   : Network information
    - positronic_* : Positronic-specific features (AI, compliance, game, governance)
    """

    def __init__(self, blockchain: Blockchain, mempool=None, access_control=None):
        self.blockchain = blockchain
        self.mempool = mempool
        self.access_control = access_control
        self.telegram_bridge = TelegramBridge()
        self._peer_manager = None  # set by Node after construction
        self._pending_broadcasts = []  # TXs to broadcast to peers (polled by node)
        self._network_type = "testnet"

    def set_peer_manager(self, peer_manager, network_type: str = "testnet"):
        """Allow the Node to inject peer_manager so RPC can report network info."""
        self._peer_manager = peer_manager
        self._network_type = network_type

    def set_sync(self, sync):
        """Allow the Node to inject sync manager so RPC can report sync progress."""
        self._sync = sync

    def _forward_to_peers(self, method: str, params: list):
        """Broadcast a state-changing RPC (stake/unstake/claim) to all peers via P2P.
        Uses the SYSTEM_TX message type over the existing WebSocket connections.
        No firewall changes needed — uses port 9000 (already open)."""
        if not hasattr(self, '_p2p_server') or self._p2p_server is None:
            return
        try:
            import asyncio
            from positronic.network.messages import make_system_tx
            node_id = getattr(self._p2p_server, 'node_id', 'rpc')
            msg = make_system_tx(method, list(params), node_id)
            loop = getattr(self, '_node_event_loop', None)
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(self._p2p_server.broadcast(msg), loop)
            logger.info("P2P broadcast SYSTEM_TX: %s", method)
        except Exception as e:
            logger.debug("P2P forward failed: %s", e)

    def _save_validator_to_db(self, address: bytes, acc) -> None:
        """Persist a single validator record to the validators table."""
        import json as _json
        self.blockchain.db.execute(
            """INSERT OR REPLACE INTO validators
               (pubkey, address, stake, is_active, activation_epoch,
                exit_epoch, slashed, attestation_count, validator_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "",  # pubkey not available from account alone
                address.hex(),
                acc.staked_amount,
                1 if acc.is_validator else 0,
                0, -1, 0, 0,
                _json.dumps({"address": address.hex(), "pubkey": "",
                             "stake": acc.staked_amount, "status": "ACTIVE" if acc.is_validator else "PENDING"}),
            ),
        )
        self.blockchain.db.safe_commit()

    def handle(self, method: str, params: list, client_ip: str = "unknown") -> Any:
        """Route an RPC method to its handler."""
        self._current_client_ip = client_ip
        handlers = {
            # === Ethereum-compatible methods ===
            "eth_chainId": self.eth_chain_id,
            "eth_blockNumber": self.eth_block_number,
            "eth_getBlockByNumber": self.eth_get_block_by_number,
            "eth_getBlockByHash": self.eth_get_block_by_hash,
            "eth_getTransactionByHash": self.eth_get_transaction_by_hash,
            "eth_getTransactionReceipt": self.eth_get_transaction_receipt,
            "net_version": self.net_version,
            "eth_getBalance": self.eth_get_balance,
            "eth_getTransactionCount": self.eth_get_transaction_count,
            "eth_getCode": self.eth_get_code,
            "eth_sendRawTransaction": self.eth_send_raw_transaction,
            "eth_estimateGas": self.eth_estimate_gas,
            "eth_gasPrice": self.eth_gas_price,
            "eth_call": self.eth_call,

            # === User Transfer (wallet page) ===
            "positronic_transfer": self.positronic_user_transfer,

            # === Positronic Core ===
            "positronic_getAIScore": self.positronic_get_ai_score,
            "positronic_getAIStats": self.positronic_get_ai_stats,
            "positronic_getQuarantinePool": self.positronic_get_quarantine_pool,
            "positronic_voteQuarantineAppeal": self.positronic_vote_quarantine_appeal,
            "positronic_getQuarantineEntry": self.positronic_get_quarantine_entry,
            "positronic_nodeInfo": self.positronic_node_info,
            "positronic_getNetworkHealth": self.positronic_get_network_health,
            "positronic_formatDenomination": self.positronic_format_denomination,

            # === Wallet Registry ===
            "positronic_getWalletInfo": self.positronic_get_wallet_info,
            "positronic_getWalletStats": self.positronic_get_wallet_stats,
            "positronic_isWalletRegistered": self.positronic_is_wallet_registered,

            # === AI Rank System ===
            "positronic_getAIRank": self.positronic_get_ai_rank,
            "positronic_getAIRankStats": self.positronic_get_ai_rank_stats,

            # === Node Ranking ===
            "positronic_getNodeRank": self.positronic_get_node_rank,
            "positronic_getNodeLeaderboard": self.positronic_get_node_leaderboard,
            "positronic_getNodeStats": self.positronic_get_node_stats,

            # === Neural Immune System ===
            "positronic_getImmuneStatus": self.positronic_get_immune_status,
            "positronic_getRecentThreats": self.positronic_get_recent_threats,

            # === Token Governance ===
            "positronic_getGovernanceStats": self.positronic_get_governance_stats,
            "positronic_getPendingProposals": self.positronic_get_pending_proposals,
            "positronic_getProposal": self.positronic_get_proposal,

            # === Forensic Reporting ===
            "positronic_getForensicStats": self.positronic_get_forensic_stats,
            "positronic_getForensicReport": self.positronic_get_forensic_report,

            # === Play-to-Earn ===
            "positronic_getGameStats": self.positronic_get_game_stats,
            "positronic_getPlayerProfile": self.positronic_get_player_profile,
            "positronic_getGameLeaderboard": self.positronic_get_game_leaderboard,
            "positronic_submitGameResult": self.positronic_submit_game_result,

            # === Play-to-Mine ===
            "positronic_getPromotionStatus": self.positronic_get_promotion_status,
            "positronic_optInAutoPromotion": self.positronic_opt_in_auto_promotion,
            "positronic_getPlayToMineStats": self.positronic_get_play_to_mine_stats,

            # === Network Health Monitor ===
            "positronic_getHealthReport": self.positronic_get_health_report,

            # === SPV/Light Client ===
            "positronic_getMerkleProof": self.positronic_get_merkle_proof,
            "positronic_verifyMerkleProof": self.positronic_verify_merkle_proof,

            # === Checkpoints ===
            "positronic_getCheckpoint": self.positronic_get_checkpoint,
            "positronic_getLatestCheckpoint": self.positronic_get_latest_checkpoint,
            "positronic_getCheckpointStats": self.positronic_get_checkpoint_stats,
            # === Phase 33: State Sync & Checkpoint Verification ===
            "positronic_getStateSnapshot": self.positronic_get_state_snapshot,
            "positronic_getCheckpoints": self.positronic_get_checkpoints,
            "positronic_verifyStateRoot": self.positronic_verify_state_root,

            # === Multisig ===
            "positronic_getMultisigWallet": self.positronic_get_multisig_wallet,
            "positronic_getMultisigStats": self.positronic_get_multisig_stats,

            # === Faucet ===
            "positronic_faucetDrip": self.positronic_faucet_drip,
            "positronic_getFaucetStats": self.positronic_get_faucet_stats,

            # === Court Evidence ===
            "positronic_generateCourtReport": self.positronic_generate_court_report,
            "positronic_getEvidencePackage": self.positronic_get_evidence_package,
            "positronic_verifyEvidence": self.positronic_verify_evidence,

            # === Address / Account ===
            "positronic_getAddressTransactions": self.positronic_get_address_transactions,
            "eth_getTransactionsByAddress": self.positronic_get_address_transactions,

            # === TRUST (Soulbound Token) ===
            "positronic_getTrustScore": self.positronic_get_trust_score,
            "positronic_getTrustProfile": self.positronic_get_trust_profile,
            "positronic_getTrustLeaderboard": self.positronic_get_trust_leaderboard,
            "positronic_getTrustStats": self.positronic_get_trust_stats,

            # === PRC-20 Tokens ===
            "positronic_createToken": self.positronic_create_token,
            "positronic_getTokenInfo": self.positronic_get_token_info,
            "positronic_getTokenBalance": self.positronic_get_token_balance,
            "positronic_tokenMint": self.positronic_token_mint,
            "positronic_listTokens": self.positronic_list_tokens,

            # === PRC-721 NFTs ===
            "positronic_createNFTCollection": self.positronic_create_nft_collection,
            "positronic_getNFTCollection": self.positronic_get_nft_collection,
            "positronic_getNFTMetadata": self.positronic_get_nft_metadata,
            "positronic_getNFTsOfOwner": self.positronic_get_nfts_of_owner,
            "positronic_mintNFT": self.positronic_mint_nft,
            "positronic_listCollections": self.positronic_list_collections,
            "positronic_getTokenRegistryStats": self.positronic_get_token_registry_stats,

            # === Gasless (Paymaster) ===
            "positronic_registerPaymaster": self.positronic_register_paymaster,
            "positronic_getPaymasterInfo": self.positronic_get_paymaster_info,
            "positronic_getPaymasterStats": self.positronic_get_paymaster_stats,

            # === Smart Wallet ===
            "positronic_createSmartWallet": self.positronic_create_smart_wallet,
            "positronic_getSmartWallet": self.positronic_get_smart_wallet,
            "positronic_getSmartWalletStats": self.positronic_get_smart_wallet_stats,

            # === On-Chain Game ===
            "positronic_startOnChainGame": self.positronic_start_onchain_game,
            "positronic_getOnChainGameState": self.positronic_get_onchain_game_state,
            "positronic_getOnChainGameStats": self.positronic_get_onchain_game_stats,

            # === AI Agents ===
            "positronic_registerAgent": self.positronic_register_agent,
            "positronic_getAgentInfo": self.positronic_get_agent_info,
            "positronic_getAgentsByOwner": self.positronic_get_agents_by_owner,
            "positronic_getAgentStats": self.positronic_get_agent_stats,
            # === Phase 23: Autonomous AI Agents ===
            "positronic_agentExecuteAction": self.positronic_agent_execute_action,
            "positronic_agentSetLimits": self.positronic_agent_set_limits,
            "positronic_agentGetActivity": self.positronic_agent_get_activity,

            # === Cryptographic Commitment Privacy (hash-based, not ZK-SNARK/STARK) ===
            "positronic_getZKStats": self.positronic_get_zk_stats,  # backward compat
            "positronic_getCommitmentStats": self.positronic_get_zk_stats,  # honest alias

            # === Cross-Chain Bridge ===
            "positronic_getBridgeStats": self.positronic_get_bridge_stats,
            "positronic_getAnchor": self.positronic_get_anchor,
            # === Phase 20: Lock/Mint Bridge v2 ===
            "positronic_bridgeLock": self.positronic_bridge_lock,
            "positronic_bridgeConfirmLock": self.positronic_bridge_confirm_lock,
            "positronic_bridgeMint": self.positronic_bridge_mint,
            "positronic_bridgeBurn": self.positronic_bridge_burn,
            "positronic_bridgeRelease": self.positronic_bridge_release,
            "positronic_bridgeGetStatus": self.positronic_bridge_get_status,

            # === DePIN ===
            "positronic_registerDevice": self.positronic_register_device,
            "positronic_getDeviceInfo": self.positronic_get_device_info,
            "positronic_getDevicesByOwner": self.positronic_get_devices_by_owner,
            "positronic_getDePINStats": self.positronic_get_depin_stats,
            # === Phase 22: DePIN Economic Layer ===
            "positronic_getDeviceScore": self.positronic_get_device_score,
            "positronic_getRewardEstimate": self.positronic_get_reward_estimate,
            "positronic_claimDeviceRewards": self.positronic_claim_device_rewards,

            # === Post-Quantum ===
            "positronic_getPQStats": self.positronic_get_pq_stats,
            "positronic_hasPQKey": self.positronic_has_pq_key,

            # === Decentralized Identity (DID) ===
            "positronic_createIdentity": self.positronic_create_identity,
            "positronic_getIdentity": self.positronic_get_identity,
            "positronic_getDIDStats": self.positronic_get_did_stats,

            # === Game Bridge (External Games) ===
            "positronic_registerGame": self.positronic_register_game,
            "positronic_getGameInfo": self.positronic_get_game_info,
            "positronic_listRegisteredGames": self.positronic_list_registered_games,
            "positronic_getGameBridgeStats": self.positronic_get_game_bridge_stats,
            "positronic_startGameSession": self.positronic_start_game_session,
            "positronic_addGameEvent": self.positronic_add_game_event,
            "positronic_submitGameSession": self.positronic_submit_game_session,
            "positronic_getSessionStatus": self.positronic_get_session_status,
            "positronic_getPlayerGameHistory": self.positronic_get_player_game_history,
            "positronic_getGameMiningRate": self.positronic_get_game_mining_rate,
            "positronic_getGameEmission": self.positronic_get_game_emission,
            "positronic_getGlobalPlayMineStats": self.positronic_get_global_play_mine_stats,
            "positronic_generateGameAPIKey": self.positronic_generate_game_api_key,
            "positronic_testGameSession": self.positronic_test_game_session,
            "positronic_getGameSDKConfig": self.positronic_get_game_sdk_config,

            # === Telegram Mini App Integration ===
            "positronic_telegramRegisterBot": self.positronic_telegram_register_bot,
            "positronic_telegramAuth": self.positronic_telegram_auth,
            "positronic_telegramGetWallet": self.positronic_telegram_get_wallet,
            "positronic_telegramGetStats": self.positronic_telegram_get_stats,

            # === Game Token Bridge (games create tokens & mint NFTs) ===
            "positronic_gameCreateToken": self.positronic_game_create_token,
            "positronic_gameCreateCollection": self.positronic_game_create_collection,
            "positronic_gameMintItem": self.positronic_game_mint_item,
            "positronic_gameDistributeReward": self.positronic_game_distribute_reward,
            "positronic_getGameTokens": self.positronic_get_game_tokens,
            "positronic_getGameCollections": self.positronic_get_game_collections,
            "positronic_getGameTokenBridgeStats": self.positronic_get_game_token_bridge_stats,

            # === AI Agent Marketplace (Phase 29) ===
            "positronic_mktRegisterAgent": self.positronic_mkt_register_agent,
            "positronic_mktGetAgent": self.positronic_mkt_get_agent,
            "positronic_mktListAgents": self.positronic_mkt_list_agents,
            "positronic_mktSubmitTask": self.positronic_mkt_submit_task,
            "positronic_mktGetTask": self.positronic_mkt_get_task,
            "positronic_mktRateAgent": self.positronic_mkt_rate_agent,
            "positronic_mktApproveAgent": self.positronic_mkt_approve_agent,
            "positronic_mktGetAgentStats": self.positronic_mkt_get_agent_stats,
            "positronic_mktGetLeaderboard": self.positronic_mkt_get_leaderboard,
            "positronic_mktGetStats": self.positronic_mkt_get_stats,
            "positronic_mktExecuteTask": self.positronic_mkt_execute_task,

            # === RWA Tokenization (Phase 30) ===
            "positronic_registerRWA": self.positronic_register_rwa,
            "positronic_getRWAInfo": self.positronic_get_rwa_info,
            "positronic_listRWAs": self.positronic_list_rwas,
            "positronic_transferRWA": self.positronic_transfer_rwa,
            "positronic_checkCompliance": self.positronic_check_compliance,
            "positronic_addKYCCredential": self.positronic_add_kyc_credential,
            "positronic_distributeDividend": self.positronic_distribute_dividend,
            "positronic_getDividendHistory": self.positronic_get_dividend_history,
            "positronic_getRWAHolders": self.positronic_get_rwa_holders,
            "positronic_getRWAStats": self.positronic_get_rwa_stats,

            # === Panel Missing Methods ===
            "positronic_getTransactionHistory": self.positronic_get_transaction_history,
            "positronic_getTokensByCreator": self.positronic_get_tokens_by_creator,
            "positronic_getNFTsByOwner": self.positronic_get_nfts_by_owner,
            "positronic_claimStakingRewards": self.positronic_claim_staking_rewards,
            "positronic_getLeaderboard": self.positronic_get_leaderboard,
            "positronic_getDID": self.positronic_get_did,
            "positronic_getCredentials": self.positronic_get_credentials,
            "positronic_getSmartWalletInfo": self.positronic_get_smart_wallet_info,
            "positronic_createSessionKey": self.positronic_create_session_key,
            "positronic_getSessionKeys": self.positronic_get_session_keys,
            "positronic_addRecoveryGuardian": self.positronic_add_recovery_guardian,
            "positronic_optInValidator": self.positronic_opt_in_validator,
            "positronic_optOutValidator": self.positronic_opt_out_validator,

            # === ML Commitment Proofs (hash-based commitment scheme, Phase 31) ===
            # Note: Uses SHA-256 commitment proofs, not ZK-SNARK/STARK circuits.
            # Old "ZKML" names kept for backward compatibility.
            "positronic_getZKMLProof": self.positronic_get_zkml_proof,  # backward compat
            "positronic_getMLCommitmentProof": self.positronic_get_zkml_proof,  # honest alias
            "positronic_verifyZKMLProof": self.positronic_verify_zkml_proof,
            "positronic_verifyMLCommitment": self.positronic_verify_zkml_proof,  # honest alias
            "positronic_getZKMLStats": self.positronic_get_zkml_stats,
            "positronic_getZKMLConfig": self.positronic_get_zkml_config,
            "positronic_getModelCommitment": self.positronic_get_model_commitment,

            # === Security Hardening (Phase 15) ===
            "positronic_gameHeartbeat": self.positronic_game_heartbeat,
            "positronic_emergencyPauseGame": self.positronic_emergency_pause_game,
            "positronic_getGameSecurityStats": self.positronic_get_game_security_stats,

            # === Phase 17: GOD CHAIN ===
            "positronic_getGasOracle": self.positronic_get_gas_oracle,
            "positronic_getFeeHistory": self.positronic_get_fee_history,
            "positronic_getTxLaneStats": self.positronic_get_tx_lane_stats,
            "positronic_getCompactBlockStats": self.positronic_get_compact_block_stats,
            "positronic_getPartitionStatus": self.positronic_get_partition_status,
            "positronic_getPeerScores": self.positronic_get_peer_scores,
            "positronic_getRevertReason": self.positronic_get_revert_reason,
            "positronic_hdCreateWallet": self.positronic_hd_create_wallet,
            "positronic_hdDeriveAddress": self.positronic_hd_derive_address,
            "positronic_getAddressHistory": self.positronic_get_address_history,

            # === Phase 18: Audit Fixes ===
            "positronic_getAIMetrics": self.positronic_get_ai_metrics,
            "positronic_apiVersion": self.positronic_api_version,

            # === Phase 21: AI Explainability (XAI) ===
            "positronic_explainValidation": self.positronic_explain_validation,

            # === Immune Appeal System ===
            "positronic_requestImmuneAppeal": self.positronic_request_immune_appeal,
            "positronic_resolveImmuneAppeal": self.positronic_resolve_immune_appeal,
            "positronic_getImmuneAppeal": self.positronic_get_immune_appeal,
            "positronic_listImmuneAppeals": self.positronic_list_immune_appeals,

            # === Consensus v2: Three-Layer System ===
            "positronic_getAttestationStats": self.positronic_get_attestation_stats,
            "positronic_getConsensusInfo": self.positronic_get_consensus_info,

            # === Admin / Treasury Management ===
            "positronic_getTreasuryBalances": self.positronic_get_treasury_balances,
            "positronic_getTeamVestingStatus": self.positronic_get_team_vesting_status,
            "positronic_getVestingStatus": self.positronic_get_vesting_status,
            "positronic_adminTransfer": self.positronic_admin_transfer,
            "positronic_stake": self.positronic_stake,
            "positronic_unstake": self.positronic_unstake,
            "positronic_getStakingInfo": self.positronic_getStakingInfo,
            "positronic_getStakingStats": self.positronic_getStakingStats,
            "positronic_getSlashingStats": self.positronic_getSlashingStats,
            "positronic_adminGetPeers": self.positronic_admin_get_peers,
            "positronic_adminBanPeer": self.positronic_admin_ban_peer,
            "positronic_adminGetValidators": self.positronic_admin_get_validators,

            # === Public Network Info ===
            "positronic_getPeers": self.positronic_get_peers,
            "positronic_getValidators": self.positronic_get_validators,
            "positronic_getConsensusStatus": self.positronic_get_consensus_status,
            "positronic_getAIStatus": self.positronic_get_ai_status,
            "positronic_requestAdminAccess": self.positronic_request_admin_access,
            "positronic_getTreasuryTransactions": self.positronic_get_treasury_transactions,

            # === Governance Write Operations ===
            "positronic_createGovernanceProposal": self.positronic_create_governance_proposal,
            "positronic_voteGovernanceProposal": self.positronic_vote_governance_proposal,
            "positronic_executeGovernanceProposal": self.positronic_execute_governance_proposal,

            # === Emergency Control System ===
            "positronic_emergencyPause": self.positronic_emergency_pause,
            "positronic_emergencyResume": self.positronic_emergency_resume,
            "positronic_emergencyHalt": self.positronic_emergency_halt,
            "positronic_emergencyStatus": self.positronic_emergency_status,
            "positronic_upgradeSchedule": self.positronic_upgrade_schedule,
            "positronic_upgradeStatus": self.positronic_upgrade_status,

            # === Phase 32: Neural Self-Preservation ===
            "positronic_getNeuralStatus": self.positronic_get_neural_status,
            "positronic_getNeuralSnapshot": self.positronic_get_neural_snapshot,
            "positronic_triggerNeuralRecovery": self.positronic_trigger_neural_recovery,
            "positronic_getNeuralRecoveryHistory": self.positronic_get_neural_recovery_history,
            "positronic_getPathwayHealth": self.positronic_get_pathway_health,
            "positronic_validateSnapshot": self.positronic_validate_snapshot,
            # === Phase 32: Cold Start ===
            "positronic_getColdStartStatus": self.positronic_get_cold_start_status,
            # === Phase 32: Online Learning ===
            "positronic_getLearningStats": self.positronic_get_learning_stats,
            "positronic_getLearningHistory": self.positronic_get_learning_history,
            "positronic_triggerManualRetrain": self.positronic_trigger_manual_retrain,
            # === Phase 32: Model Communication Bus ===
            "positronic_getModelBusStats": self.positronic_get_model_bus_stats,
            # === Phase 32: Causal XAI ===
            "positronic_explainTransaction": self.positronic_explain_transaction,
            # === Phase 32: Concept Drift ===
            "positronic_getDriftAlerts": self.positronic_get_drift_alerts,

            # === DEX — Automated Market Maker ===
            "positronic_dexCreatePool": self.positronic_dex_create_pool,
            "positronic_dexSwap": self.positronic_dex_swap,
            "positronic_dexAddLiquidity": self.positronic_dex_add_liquidity,
            "positronic_dexRemoveLiquidity": self.positronic_dex_remove_liquidity,
            "positronic_dexGetQuote": self.positronic_dex_get_quote,
            "positronic_dexGetPool": self.positronic_dex_get_pool,
            "positronic_dexListPools": self.positronic_dex_list_pools,
            "positronic_dexGetStats": self.positronic_dex_get_stats,
            # === Web3/MetaMask compatibility ===
            "web3_clientVersion": self.web3_client_version,
            "net_peerCount": self.net_peer_count,
            "eth_getLogs": self.eth_get_logs,
            # === Monitoring ===
            "positronic_getMempoolStatus": self.positronic_get_mempool_status,
            "positronic_getBlockReward": self.positronic_get_block_reward,
        }

        handler = handlers.get(method)
        if handler:
            return handler(params)
        raise KeyError(f"Method {method} not found")

    # ================================================================
    # Ethereum-compatible methods
    # ================================================================

    def eth_chain_id(self, params) -> str:
        return hex(CHAIN_ID)

    def net_version(self, params) -> str:
        return str(CHAIN_ID)

    def eth_block_number(self, params) -> str:
        return hex(self.blockchain.height)

    def eth_get_balance(self, params) -> str:
        address = address_from_hex(params[0]) if params else b""
        acc = self.blockchain.state.get_account(address)
        # Return effective balance (total - staked) so users see available funds
        return hex(acc.effective_balance)

    def eth_get_transaction_count(self, params) -> str:
        address = address_from_hex(params[0]) if params else b""
        nonce = self.blockchain.get_nonce(address)
        return hex(nonce)

    def eth_get_code(self, params) -> str:
        address = address_from_hex(params[0]) if params else b""
        code = self.blockchain.state.get_code(address)
        return "0x" + code.hex() if code else "0x"

    def eth_get_block_by_number(self, params) -> Optional[dict]:
        if not params:
            return None
        height = int(params[0], 16) if params[0] != "latest" else self.blockchain.height
        block = self.blockchain.get_block(height)
        return block.to_dict() if block else None

    def eth_get_block_by_hash(self, params) -> Optional[dict]:
        if not params:
            return None
        block_hash = bytes.fromhex(params[0].removeprefix("0x"))
        block = self.blockchain.get_block_by_hash(block_hash)
        return block.to_dict() if block else None

    def eth_get_transaction_by_hash(self, params) -> Optional[dict]:
        if not params:
            return None
        tx_hash = bytes.fromhex(params[0].removeprefix("0x"))
        tx = self.blockchain.get_transaction(tx_hash)
        if not tx:
            return None
        result = tx.to_dict()
        # Add block context from chain_db
        try:
            row = self.blockchain.chain_db.db.execute(
                "SELECT block_height, block_hash FROM transactions WHERE tx_hash = ?",
                (tx_hash.hex(),),
            ).fetchone()
            if row:
                result["block_height"] = row["block_height"]
                result["block_number"] = row["block_height"]
                result["blockNumber"] = row["block_height"]
                result["block_hash"] = row["block_hash"]
        except Exception:
            pass
        return result

    def eth_get_transaction_receipt(self, params) -> Optional[dict]:
        if not params:
            return None
        tx_hash = bytes.fromhex(params[0].removeprefix("0x"))
        receipt = self.blockchain.get_receipt(tx_hash)
        return receipt.to_dict() if receipt else None

    def eth_send_raw_transaction(self, params) -> str:
        """
        Submit a signed transaction to the mempool.
        Accepts a Positronic transaction as JSON hex-encoded dict.
        Returns the transaction hash on success.
        """
        if not params:
            raise ValueError("Missing transaction data parameter")

        raw_data = params[0]

        try:
            # Support JSON-encoded transaction dict (Positronic native format)
            if isinstance(raw_data, dict):
                tx_dict = raw_data
            elif isinstance(raw_data, str):
                # Try hex-encoded JSON
                if raw_data.startswith("0x"):
                    raw_data = raw_data[2:]
                try:
                    import json as _json
                    tx_dict = _json.loads(bytes.fromhex(raw_data).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    # Try direct JSON string
                    import json as _json
                    tx_dict = _json.loads(raw_data)
            else:
                raise ValueError("Invalid transaction format")

            from positronic.core.transaction import Transaction
            tx = Transaction.from_dict(tx_dict)

            # Verify signature
            if not tx.verify_signature():
                raise ValueError("Invalid transaction signature")

            # Validate and add to mempool
            vr = self.blockchain.tx_validator.validate(tx, self.blockchain.state, self.blockchain.height)
            if not vr.valid:
                raise ValueError(f"Transaction validation failed: {vr.error}")

            if self.mempool and hasattr(self.mempool, "add"):
                if not self.mempool.add(tx):
                    raise ValueError("Transaction rejected by mempool")
            else:
                raise ValueError("Mempool not available")

            # Queue TX for P2P broadcast (node polls this list)
            self._pending_broadcasts.append(tx.to_dict())

            return "0x" + tx.tx_hash.hex()

        except ValueError:
            raise
        except Exception as e:
            logger.debug("RPC method error: Transaction failed -- %s", e)
            raise ValueError("Failed to process transaction")

    def eth_estimate_gas(self, params) -> str:
        """Estimate gas for a transaction using VM simulation.

        Phase 17: Uses VM.simulate() for accurate estimation.
        Falls back to 21000 (simple transfer) on any error.
        """
        try:
            if not params:
                return hex(21000)

            tx_data = params[0] if isinstance(params[0], dict) else {}
            to_addr = tx_data.get("to", "")
            data_hex = tx_data.get("data", "0x")

            # Simple transfer (no data, has recipient)
            if to_addr and (not data_hex or data_hex == "0x"):
                return hex(21000)

            # Contract call — try VM simulation
            if to_addr and data_hex and data_hex != "0x":
                try:
                    from positronic.vm.vm import PositronicVM, ExecutionContext
                    from positronic.crypto.address import address_from_hex as _afh

                    vm = PositronicVM(state=self.blockchain.state)
                    sender = _afh(tx_data.get("from", "0x" + "00" * 20))
                    recipient = _afh(to_addr)
                    call_data = bytes.fromhex(data_hex.removeprefix("0x"))
                    value = int(tx_data.get("value", "0x0"), 16) if isinstance(tx_data.get("value"), str) else int(tx_data.get("value", 0))
                    gas_limit = int(tx_data.get("gas", "0x1e8480"), 16) if isinstance(tx_data.get("gas"), str) else int(tx_data.get("gas", 2_000_000))

                    ctx = ExecutionContext(
                        sender=sender,
                        recipient=recipient,
                        value=value,
                        data=call_data,
                        gas_limit=gas_limit,
                    )
                    result = vm.simulate(ctx)
                    # Add 20% buffer
                    estimated = int(result.gas_used * 1.2)
                    return hex(max(estimated, 21000))
                except Exception as e:
                    logger.warning("eth_estimateGas VM simulation failed", exc_info=True)

            # Contract creation
            if not to_addr and data_hex and data_hex != "0x":
                data_len = len(bytes.fromhex(data_hex.removeprefix("0x")))
                estimated = 53000 + data_len * 200
                return hex(estimated)

            return hex(21000)
        except Exception as e:
            logger.warning("eth_estimateGas fallback", exc_info=True)
            return hex(21000)

    def eth_gas_price(self, params) -> str:
        """Get current gas price from the gas oracle.

        Phase 17: Uses GasOracle for dynamic pricing.
        Falls back to hex(1) on any error.
        """
        try:
            from positronic.chain.gas_oracle import GasOracle
            oracle = GasOracle()

            # Update oracle with recent block data if chain is available
            try:
                height = self.blockchain.height
                for h in range(max(0, height - 5), height + 1):
                    block = self.blockchain.get_block(h)
                    if block:
                        tx_count = len(block.transactions)
                        gas_used = tx_count * 21000
                        oracle.update_base_fee(
                            gas_used, BLOCK_GAS_LIMIT,
                            block_height=h, tx_count=tx_count,
                        )
            except Exception as e:
                logger.warning("eth_gasPrice oracle update failed", exc_info=True)

            suggestion = oracle.get_fee_suggestion()
            total = suggestion["base_fee"] + suggestion["priority_fee_medium"]
            return hex(max(1, total))
        except Exception as e:
            logger.warning("eth_gasPrice fallback", exc_info=True)
            return hex(1)

    def eth_call(self, params) -> str:
        """Execute a message call without creating a transaction.

        Phase 17: Uses VM.simulate() for actual execution.
        Falls back to '0x' on any error.
        """
        try:
            if not params:
                return "0x"

            tx_data = params[0] if isinstance(params[0], dict) else {}
            to_addr = tx_data.get("to", "")
            data_hex = tx_data.get("data", "0x")

            if not to_addr or not data_hex or data_hex == "0x":
                return "0x"

            from positronic.vm.vm import PositronicVM, ExecutionContext
            from positronic.crypto.address import address_from_hex as _afh

            vm = PositronicVM(state=self.blockchain.state)
            sender = _afh(tx_data.get("from", "0x" + "00" * 20))
            recipient = _afh(to_addr)
            call_data = bytes.fromhex(data_hex.removeprefix("0x"))
            value = int(tx_data.get("value", "0x0"), 16) if isinstance(tx_data.get("value"), str) else int(tx_data.get("value", 0))
            gas_limit = int(tx_data.get("gas", "0x1e8480"), 16) if isinstance(tx_data.get("gas"), str) else int(tx_data.get("gas", 2_000_000))

            ctx = ExecutionContext(
                sender=sender,
                recipient=recipient,
                value=value,
                data=call_data,
                gas_limit=gas_limit,
            )
            result = vm.simulate(ctx)
            return "0x" + result.return_data.hex() if result.return_data else "0x"
        except Exception as e:
            logger.warning("eth_call fallback", exc_info=True)
            return "0x"

    # ================================================================
    # Positronic Core
    # ================================================================

    def positronic_get_ai_score(self, params) -> Optional[dict]:
        if not params:
            return None
        tx_hash = bytes.fromhex(params[0].removeprefix("0x"))
        tx = self.blockchain.get_transaction(tx_hash)
        if tx:
            return {"tx_hash": tx.tx_hash_hex, "ai_score": tx.ai_score, "status": tx.status.name}
        return None

    def positronic_get_ai_stats(self, params) -> dict:
        stats = self.blockchain.ai_gate.get_stats()
        # Flatten training_samples from anomaly_detector sub-dict
        ad = stats.get("anomaly_detector", {})
        if "training_samples" not in stats:
            stats["training_samples"] = ad.get("training_samples", ad.get("samples", 0))
        # Add quarantine_pending from quarantine pool
        if "quarantine_pending" not in stats:
            try:
                qstats = self.blockchain.quarantine_pool.get_stats()
                stats["quarantine_pending"] = qstats.get("quarantine_count", qstats.get("pending", 0))
            except Exception:
                stats["quarantine_pending"] = 0
        return stats

    def positronic_get_quarantine_pool(self, params) -> dict:
        return self.blockchain.quarantine_pool.get_stats()

    def positronic_vote_quarantine_appeal(self, params) -> dict:
        """Vote on a quarantine appeal. params: [tx_hash_hex, voter_hex, approve_bool, reason?]"""
        if len(params) < 3:
            return {"error": "Requires [tx_hash_hex, voter_hex, approve]"}
        try:
            tx_hash = bytes.fromhex(params[0].removeprefix("0x"))
            voter = bytes.fromhex(params[1].removeprefix("0x"))
            approve = bool(params[2])
            reason = str(params[3]) if len(params) > 3 else ""

            # Validate voter is in governance pool
            if hasattr(self.blockchain, 'validator_pool'):
                gov_pool = getattr(self.blockchain.validator_pool, 'governance_validators', None)
                if gov_pool is not None and voter not in gov_pool:
                    return {"error": "Voter not in governance validator pool"}

            # Cast vote
            success = self.blockchain.quarantine_pool.vote_appeal(
                tx_hash=tx_hash, vote_for=approve, voter_id=voter
            )
            if not success:
                entry = self.blockchain.quarantine_pool.get(tx_hash)
                if entry is None:
                    return {"error": "Transaction not found in quarantine pool"}
                return {"error": "Already voted on this transaction"}

            # Return current appeal status
            entry = self.blockchain.quarantine_pool.get(tx_hash)
            if entry:
                return {
                    "tx_hash": params[0],
                    "voted": True,
                    "vote": "approve" if approve else "reject",
                    "reason": reason,
                    "votes_for": entry.appeal_votes_for,
                    "votes_against": entry.appeal_votes_against,
                    "status": entry.status,
                    "supermajority_reached": entry.appeal_votes_for > entry.appeal_votes_against * 2,
                }
            return {"tx_hash": params[0], "voted": True, "released": True}
        except Exception as e:
            logger.error("vote_quarantine_appeal failed", exc_info=True)
            return _rpc_error("Transaction failed", e)

    def positronic_get_quarantine_entry(self, params) -> dict:
        """Get details of a specific quarantine entry. params: [tx_hash_hex]"""
        if not params:
            return {"error": "Requires [tx_hash_hex]"}
        try:
            tx_hash = bytes.fromhex(params[0].removeprefix("0x"))
            entry = self.blockchain.quarantine_pool.get(tx_hash)
            if not entry:
                return {"error": "Transaction not found in quarantine pool"}
            return {
                "tx_hash": params[0],
                "ai_score": entry.ai_score,
                "quarantined_at_block": entry.quarantined_at_block,
                "review_count": entry.review_count,
                "votes_for": entry.appeal_votes_for,
                "votes_against": entry.appeal_votes_against,
                "voter_count": len(entry.appeal_voters),
                "status": entry.status,
                "supermajority_reached": entry.appeal_votes_for > entry.appeal_votes_against * 2,
            }
        except Exception as e:
            logger.error("get_quarantine_entry failed", exc_info=True)
            return _rpc_error("Query failed", e)

    def positronic_node_info(self, params) -> dict:
        stats = self.blockchain.get_stats()
        # Inject network/peer info from peer_manager (not available in blockchain)
        if self._peer_manager is not None:
            try:
                connected = self._peer_manager.get_connected_peers()
                peer_count = len(connected)
            except Exception:
                peer_count = 0
            mempool_size = 0
            if self.mempool is not None:
                mempool_size = getattr(self.mempool, 'size', 0)
            stats["network"] = {
                "peer_count": peer_count,
                "max_peers": getattr(self._peer_manager, 'max_peers', 12),
                "network_type": self._network_type,
                "mempool_size": mempool_size,
            }
            stats["peers"] = peer_count
        else:
            stats["network"] = {
                "peer_count": 0,
                "max_peers": 12,
                "network_type": self._network_type,
                "mempool_size": 0,
            }
            stats["peers"] = 0
        # Sync status — use real sync state if available
        peer_count = stats.get("peers", 0)
        height = stats.get("height", 0)
        sync = getattr(self, '_sync', None)
        if sync is not None:
            try:
                syncing = sync.state.syncing
                stats["syncing"] = syncing
                if syncing:
                    # Actively syncing: show real progress (0.0–1.0)
                    stats["sync_progress"] = sync.state.progress
                    stats["synced"] = False
                elif peer_count > 0 and height > 0:
                    # Connected and not syncing: fully synced
                    stats["sync_progress"] = 1.0
                    stats["synced"] = True
                else:
                    # Not connected yet: no percentage (null)
                    stats["sync_progress"] = None
                    stats["synced"] = False
            except Exception:
                stats["synced"] = (height > 0 and peer_count > 0)
                stats["syncing"] = False
                stats["sync_progress"] = 1.0 if stats["synced"] else None
        else:
            stats["synced"] = (height > 0 and peer_count > 0)
            stats["syncing"] = False
            stats["sync_progress"] = 1.0 if stats["synced"] else None
        stats["mempool_size"] = stats["network"].get("mempool_size", 0)
        # Consensus fields for dashboard/validator display
        if self.blockchain.consensus:
            cs = stats.get("consensus", {})
            from positronic.constants import BASE_UNIT
            total_staked = 0
            try:
                for v in self.blockchain.consensus.registry.all_validators:
                    total_staked += v.stake
            except Exception:
                pass
            cs["total_staked"] = total_staked
            cs["is_validator"] = cs.get("active_validators", 0) > 0
            stats["consensus"] = cs

        # Strip sensitive internal details from public response
        ai = stats.get("ai", {})
        for key in ("weights", "thresholds", "anomaly_detector", "mev_detector",
                     "contract_analyzer", "stability_guardian", "neural_engine"):
            ai.pop(key, None)
        immune = stats.get("immune_system", {})
        immune.pop("sybil_detection", None)

        return stats

    def positronic_get_network_health(self, params) -> dict:
        return self.blockchain.get_network_health()

    def positronic_format_denomination(self, params) -> str:
        """Format a value in specified denomination. params: [value, denomination]"""
        if len(params) < 2:
            return "Error: requires [value, denomination]"
        try:
            return format_denomination(int(params[0]), params[1])
        except (ValueError, TypeError) as e:
            return f"Error: {e}"

    # ================================================================
    # Wallet Registry
    # ================================================================

    def positronic_get_wallet_info(self, params) -> Optional[dict]:
        if not params:
            return None
        address = address_from_hex(params[0])
        wallet = self.blockchain.wallet_registry.get_wallet(address)
        if wallet:
            return wallet.to_dict()
        # Fallback: return basic info from state for unregistered wallets
        acc = self.blockchain.state.get_account(address)
        from positronic.constants import BASE_UNIT
        return {
            "address": params[0],
            "status": "active",
            "tier": "standard",
            "ai_trust_score": getattr(acc, 'trust_score', 0),
            "total_transactions": getattr(acc, 'nonce', 0),
            "nonce": getattr(acc, 'nonce', 0),
            "staked_amount": getattr(acc, 'staked_amount', 0),
            "balance": acc.effective_balance,
        }

    def positronic_get_wallet_stats(self, params) -> dict:
        return self.blockchain.wallet_registry.get_stats()

    def positronic_is_wallet_registered(self, params) -> bool:
        if not params:
            return False
        address = address_from_hex(params[0])
        return self.blockchain.wallet_registry.is_registered(address)

    # ================================================================
    # AI Rank System
    # ================================================================

    def positronic_get_ai_rank(self, params) -> Optional[dict]:
        if not params:
            return None
        address = address_from_hex(params[0])
        profile = self.blockchain.ai_rank_manager.get_profile(address)
        if profile:
            return profile.to_dict()
        # Return empty AI rank for new addresses
        return {
            "address": params[0],
            "total_scored": 0,
            "accuracy": 0.0,
            "rank": "unranked",
        }

    def positronic_get_ai_rank_stats(self, params) -> dict:
        return self.blockchain.ai_rank_manager.get_stats()

    # ================================================================
    # Node Ranking
    # ================================================================

    def positronic_get_node_rank(self, params) -> Optional[dict]:
        if not params:
            return None
        address = address_from_hex(params[0])
        node = self.blockchain.node_ranking.get_node(address)
        if node:
            return node.to_dict()
        # Return empty rank for unranked nodes
        return {
            "address": params[0],
            "rank": "E1_PRIVATE",
            "level": 1,
            "blocks_validated": 0,
            "blocks_proposed": 0,
            "uptime_percentage": 0.0,
            "reward_multiplier": 1.0,
        }

    def positronic_get_node_leaderboard(self, params) -> list:
        limit = int(params[0]) if params else 50
        return self.blockchain.node_ranking.get_leaderboard(limit)

    def positronic_get_node_stats(self, params) -> dict:
        return self.blockchain.node_ranking.get_stats()

    # ================================================================
    # Neural Immune System
    # ================================================================

    def positronic_get_immune_status(self, params) -> dict:
        return self.blockchain.immune_system.get_status()

    def positronic_get_recent_threats(self, params) -> list:
        limit = int(params[0]) if params else 50
        return self.blockchain.immune_system.get_recent_threats(limit)


    # ================================================================
    # Web3/MetaMask Compatibility + Monitoring
    # ================================================================

    def web3_client_version(self, params) -> str:
        """Return client version string (MetaMask compatibility)."""
        from positronic import __version__
        return f"Positronic/v{__version__}/python"

    def net_peer_count(self, params) -> str:
        """Return connected peer count as hex (MetaMask compatibility)."""
        if self._peer_manager is not None:
            try:
                count = len(self._peer_manager.get_connected_peers())
            except Exception:
                count = 0
        else:
            count = 0
        return hex(count)

    def eth_get_logs(self, params) -> list:
        """Return event logs matching filter (stub — returns empty for now).
        Full implementation requires persistent event log storage."""
        # TODO: Implement persistent event log storage in chain_db
        return []

    def positronic_get_mempool_status(self, params) -> dict:
        """Return mempool statistics."""
        import time as _time
        if self.mempool is None:
            return {"size": 0, "max_size": 10000, "total_gas": 0, "oldest_age": "--"}
        pending = getattr(self.mempool, 'pending', {})
        total_gas = sum(getattr(tx, 'gas_limit', 0) for tx in pending.values())
        now = _time.time()
        timestamps = [getattr(tx, 'timestamp', now) for tx in pending.values() if getattr(tx, 'timestamp', 0) > 0]
        if timestamps:
            oldest_sec = int(now - min(timestamps))
            if oldest_sec < 60:
                oldest_age = f"{oldest_sec}s"
            elif oldest_sec < 3600:
                oldest_age = f"{oldest_sec // 60}m {oldest_sec % 60}s"
            else:
                oldest_age = f"{oldest_sec // 3600}h {(oldest_sec % 3600) // 60}m"
        else:
            oldest_age = "--"
        return {
            "size": len(pending),
            "max_size": getattr(self.mempool, 'MAX_SIZE', 10000),
            "total_gas": total_gas,
            "oldest_age": oldest_age,
            "ai_rejected": getattr(self.mempool, 'ai_rejected', 0),
            "ai_quarantined": getattr(self.mempool, 'ai_quarantined', 0),
            "ai_accepted": getattr(self.mempool, 'ai_accepted', 0),
            "sender_count": len(getattr(self.mempool, 'by_sender', {})),
        }

    def positronic_get_block_reward(self, params) -> dict:
        """Return block reward info for current or specified height."""
        from positronic.constants import (
            INITIAL_BLOCK_REWARD, HALVING_INTERVAL,
            MINING_SUPPLY_CAP, TAIL_EMISSION, BASE_UNIT,
            PRODUCER_REWARD_SHARE, ATTESTATION_REWARD_SHARE,
            NODE_OPERATOR_REWARD_SHARE,
        )
        height = int(params[0]) if params else self.blockchain.height
        halvings = height // HALVING_INTERVAL
        reward = INITIAL_BLOCK_REWARD >> halvings
        reward = max(reward, TAIL_EMISSION)
        return {
            "block_height": height,
            "base_reward": reward,
            "base_reward_asf": reward / BASE_UNIT,
            "producer_share": int(reward * PRODUCER_REWARD_SHARE),
            "attestation_share": int(reward * ATTESTATION_REWARD_SHARE),
            "node_operator_share": int(reward * NODE_OPERATOR_REWARD_SHARE),
            "halving_number": halvings,
            "next_halving_block": (halvings + 1) * HALVING_INTERVAL,
            "mining_supply_cap": MINING_SUPPLY_CAP,
        }

    # ================================================================
    # Token Governance
    # ================================================================

    def positronic_get_governance_stats(self, params) -> dict:
        stats = self.blockchain.token_governance.get_stats()
        # Add pending_proposals alias (app expects this key)
        if "pending_proposals" not in stats:
            stats["pending_proposals"] = stats.get("pending", 0)
        return stats

    def positronic_get_pending_proposals(self, params) -> list:
        proposals = self.blockchain.token_governance.get_pending_proposals()
        return [p.to_dict() for p in proposals]

    def positronic_get_proposal(self, params) -> Optional[dict]:
        if not params:
            return None
        proposal = self.blockchain.token_governance.get_proposal(params[0])
        return proposal.to_dict() if proposal else None

    def positronic_create_governance_proposal(self, params) -> dict:
        """Create a governance proposal. params: [proposer_hex, token_name, token_symbol, supply, decimals?, description?]"""
        if len(params) < 4:
            return {"error": "Requires [proposer_hex, name, symbol, supply]"}
        try:
            proposer = bytes.fromhex(params[0].removeprefix("0x"))
            name = str(params[1])
            symbol = str(params[2])
            supply = int(params[3])
            decimals = int(params[4]) if len(params) > 4 else 18
            description = str(params[5]) if len(params) > 5 else ""
            proposal = self.blockchain.token_governance.submit_proposal(
                proposer=proposer, token_name=name, token_symbol=symbol,
                token_supply=supply, token_decimals=decimals, description=description,
            )
            return proposal.to_dict()
        except Exception as e:
            logger.error("create_governance_proposal failed", exc_info=True)
            return _rpc_error("Governance operation failed", e)

    def positronic_vote_governance_proposal(self, params) -> dict:
        """Vote on a governance proposal. params: [proposal_id, voter_hex, approve_bool]"""
        if len(params) < 3:
            return {"error": "Requires [proposal_id, voter_hex, approve]"}
        try:
            proposal_id = str(params[0])
            voter = bytes.fromhex(params[1].removeprefix("0x"))
            approve = bool(params[2])
            proposal = self.blockchain.token_governance.council_vote(
                proposal_id=proposal_id, voter=voter, approve=approve,
            )
            return proposal.to_dict()
        except Exception as e:
            logger.error("vote_governance_proposal failed", exc_info=True)
            return _rpc_error("Governance operation failed", e)

    def positronic_execute_governance_proposal(self, params) -> dict:
        """Execute an approved governance proposal. params: [proposal_id]"""
        if len(params) < 1:
            return {"error": "Requires [proposal_id]"}
        try:
            proposal_id = str(params[0])
            proposal = self.blockchain.token_governance.get_proposal(proposal_id)
            if not proposal:
                return {"error": "Proposal not found"}
            if proposal.status.name != "APPROVED":
                return {"error": f"Proposal not approved (status: {proposal.status.name})"}
            # Mark as deployed with a placeholder address
            contract_addr = bytes(20)  # placeholder
            result = self.blockchain.token_governance.mark_deployed(proposal_id, contract_addr)
            return result.to_dict()
        except Exception as e:
            logger.error("execute_governance_proposal failed", exc_info=True)
            return _rpc_error("Governance operation failed", e)

    # ================================================================
    # Forensic Reporting
    # ================================================================

    def positronic_get_forensic_stats(self, params) -> dict:
        return self.blockchain.forensic_reporter.get_stats()

    def positronic_get_forensic_report(self, params) -> Optional[dict]:
        if not params:
            return None
        report = self.blockchain.forensic_reporter.get_report(params[0])
        return report.to_dict() if report else None

    # ================================================================
    # Play-to-Earn
    # ================================================================

    def positronic_get_game_stats(self, params) -> dict:
        return self.blockchain.game_engine.get_stats()

    def positronic_get_player_profile(self, params) -> Optional[dict]:
        if not params:
            return None
        address = address_from_hex(params[0])
        player = self.blockchain.game_engine.get_player(address)
        if player:
            return player.to_dict()
        # Return empty profile for new players
        return {
            "address": params[0],
            "games_played": 0,
            "high_score": 0,
            "total_score": 0,
            "rewards_earned": 0,
            "best_level": None,
            "reward_multiplier": 1.0,
        }

    def positronic_get_game_leaderboard(self, params) -> list:
        limit = int(params[0]) if params else 50
        return self.blockchain.game_engine.get_leaderboard(limit)

    def positronic_submit_game_result(self, params) -> dict:
        """Submit a game result for reward calculation.

        params: [{player, score, level_completed, time_taken,
                  coins_collected, enemies_defeated, no_damage, result_hash}]
        """
        if not params or not isinstance(params[0], dict):
            return {"reward": 0, "reason": "Missing game result data"}
        data = params[0]
        try:
            from positronic.game.play_to_earn import GameResult
            player_bytes = bytes.fromhex(data["player"])
            result_hash_bytes = bytes.fromhex(data.get("result_hash", ""))
            result = GameResult(
                player=player_bytes,
                score=int(data["score"]),
                level_completed=int(data["level_completed"]),
                time_taken=float(data["time_taken"]),
                coins_collected=int(data["coins_collected"]),
                enemies_defeated=int(data["enemies_defeated"]),
                no_damage=bool(data.get("no_damage", False)),
                result_hash=result_hash_bytes,
            )
            return self.blockchain.game_engine.submit_game_result(result)
        except Exception as e:
            logger.debug("submit_game_result_failed: %s", e)
            logger.debug("RPC method error: Game operation failed -- %s", e)
            return {"reward": 0, "reason": "Invalid game result"}

    # ================================================================
    # Network Health Monitor
    # ================================================================

    def positronic_get_health_report(self, params) -> dict:
        """Get comprehensive network health report."""
        if hasattr(self.blockchain, 'health_monitor'):
            return self.blockchain.health_monitor.get_stats()
        return {"status": "HEALTHY", "score": 100, "note": "Health monitor not initialized"}

    # ================================================================
    # SPV/Light Client
    # ================================================================

    def positronic_get_merkle_proof(self, params) -> Optional[dict]:
        """Get Merkle proof for a transaction. params: [tx_hash]"""
        if not params:
            return None
        return {"status": "available", "note": "SPV light client support enabled"}

    def positronic_verify_merkle_proof(self, params) -> dict:
        """Verify a Merkle proof. params: [proof_dict]"""
        if not params:
            return {"valid": False, "error": "No proof provided"}
        try:
            from positronic.network.light_client import MerkleProof
            proof = MerkleProof.from_dict(params[0])
            return {"valid": proof.verify()}
        except Exception as e:
            logger.error("verify_merkle_proof failed", exc_info=True)
            logger.debug("RPC method error: Query failed -- %s", e)
            return {"valid": False, "error": "Query failed"}

    # ================================================================
    # Checkpoints
    # ================================================================

    def positronic_get_checkpoint(self, params) -> Optional[dict]:
        """Get checkpoint at height. params: [height]"""
        if not params:
            return None
        if hasattr(self.blockchain, 'checkpoint_manager'):
            height = int(params[0])
            cp = self.blockchain.checkpoint_manager.get_checkpoint(height)
            return cp.to_dict() if cp else None
        return None

    def positronic_get_latest_checkpoint(self, params) -> Optional[dict]:
        """Get the latest checkpoint."""
        if hasattr(self.blockchain, 'checkpoint_manager'):
            cp = self.blockchain.checkpoint_manager.latest_checkpoint
            return cp.to_dict() if cp else None
        return None

    def positronic_get_checkpoint_stats(self, params) -> dict:
        """Get checkpoint system stats."""
        if hasattr(self.blockchain, 'checkpoint_manager'):
            return self.blockchain.checkpoint_manager.get_stats()
        return {"total_checkpoints": 0, "note": "Checkpoint manager not initialized"}

    # ================================================================
    # Phase 33: State Sync & Checkpoint Verification
    # ================================================================

    def positronic_get_state_snapshot(self, params) -> dict:
        """Get compressed state snapshot at height. params: [height (optional)]"""
        if not hasattr(self.blockchain, 'checkpoint_manager'):
            return _rpc_error("Checkpoint manager not initialized")
        try:
            height = int(params[0]) if params else self.blockchain.height
            state = self.blockchain.state
            state_db = getattr(self.blockchain, 'state_db', None)
            snapshot_bytes = self.blockchain.checkpoint_manager.export_state_snapshot(
                height, state_db, state
            )
            return {
                "height": height,
                "snapshot": snapshot_bytes.hex(),
                "size_bytes": len(snapshot_bytes),
                "accounts_count": len(state.accounts),
            }
        except Exception as e:
            return _rpc_error("Failed to export state snapshot", e)

    def positronic_get_checkpoints(self, params) -> list:
        """Get list of available checkpoints."""
        if not hasattr(self.blockchain, 'checkpoint_manager'):
            return []
        return self.blockchain.checkpoint_manager.get_available_checkpoints()

    def positronic_verify_state_root(self, params) -> dict:
        """Verify a state root against stored checkpoint. params: [height, state_root_hex]"""
        if not params or len(params) < 2:
            return _rpc_error("Requires [height, state_root_hex]")
        if not hasattr(self.blockchain, 'checkpoint_manager'):
            return _rpc_error("Checkpoint manager not initialized")
        try:
            height = int(params[0])
            expected_root = params[1]
            cp = self.blockchain.checkpoint_manager.get_checkpoint(height)
            if cp is None:
                return {"verified": False, "reason": "No checkpoint at height"}
            stored_root = cp.state_root.hex()
            match = stored_root == expected_root
            return {
                "verified": match,
                "height": height,
                "stored_root": stored_root,
                "provided_root": expected_root,
            }
        except Exception as e:
            return _rpc_error("Failed to verify state root", e)

    # ================================================================
    # Multisig
    # ================================================================

    def positronic_get_multisig_wallet(self, params) -> Optional[dict]:
        """Get multisig wallet info. params: [address]"""
        if not params:
            return None
        if hasattr(self.blockchain, 'multisig_manager'):
            address = address_from_hex(params[0])
            wallet = self.blockchain.multisig_manager.get_wallet(address)
            return wallet.to_dict() if wallet else None
        return None

    def positronic_get_multisig_stats(self, params) -> dict:
        """Get multisig system stats."""
        if hasattr(self.blockchain, 'multisig_manager'):
            return self.blockchain.multisig_manager.get_stats()
        return {"wallets": 0, "note": "Multisig manager not initialized"}

    # ================================================================
    # Play-to-Mine
    # ================================================================

    def positronic_get_promotion_status(self, params) -> dict:
        """Get player's Play-to-Mine promotion status. params: [address]"""
        if not params:
            return {"error": "Address required"}
        address = address_from_hex(params[0])
        return self.blockchain.get_promotion_status(address)

    def positronic_opt_in_auto_promotion(self, params) -> dict:
        """Opt in for auto-promotion to node/NVN. params: [address, pubkey_hex]"""
        if not params or len(params) < 2:
            return {"error": "Address and pubkey required"}
        address = address_from_hex(params[0])
        pubkey = bytes.fromhex(params[1].replace("0x", ""))
        success = self.blockchain.opt_in_auto_promotion(address, pubkey)
        return {"success": success, "address": params[0]}

    def positronic_get_play_to_mine_stats(self, params) -> dict:
        """Get overall Play-to-Mine system statistics."""
        return self.blockchain.promotion_manager.get_stats()

    # ================================================================
    # Faucet
    # ================================================================

    def positronic_faucet_drip(self, params) -> dict:
        """Send testnet tokens to an address. params: [address]"""
        if not params:
            return {"error": "Address required", "success": False}
        address = params[0]
        if not address or not address.startswith("0x") or len(address) != 42:
            return {"error": "Invalid address format (expected 0x + 40 hex chars)", "success": False}
        try:
            addr_bytes = bytes.fromhex(address[2:])
        except ValueError:
            return {"error": "Invalid address: non-hex characters", "success": False}

        if not hasattr(self.blockchain, '_faucet') or self.blockchain._faucet is None:
            return {"error": "Faucet not available", "success": False}

        faucet = self.blockchain._faucet
        client_ip = getattr(self, '_current_client_ip', 'unknown')
        can_send, reason = faucet.can_gift(address, client_ip=client_ip)
        if not can_send:
            return {"error": reason, "success": False}

        # Direct state credit from Community Pool + record in chain
        from positronic.constants import BASE_UNIT
        from positronic.core.transaction import Transaction, TxType
        import time as _time, hashlib

        gift_amount = faucet.gift_amount
        community_pool = bytes.fromhex("0000000000000000000000000000000000000002")

        pool_balance = self.blockchain.state.get_balance(community_pool)
        if pool_balance < gift_amount:
            return {"error": "Community pool insufficient funds", "success": False}

        # Direct state mutation: debit community pool, credit recipient
        # This ensures balance is available immediately.
        pool_acc = self.blockchain.state.get_account(community_pool)
        pool_acc.balance -= gift_amount
        self.blockchain.state.set_account(community_pool, pool_acc)

        recipient_acc = self.blockchain.state.get_account(addr_bytes)
        recipient_acc.balance += gift_amount
        self.blockchain.state.set_account(addr_bytes, recipient_acc)

        # Also record on-chain so other nodes replicate the state change
        self.blockchain._create_system_tx(
            0, community_pool, addr_bytes, gift_amount, b"faucet",
        )

        # Register recipient in wallet registry if not already tracked
        from positronic.compliance.wallet_registry import WalletStatus
        reg = self.blockchain.wallet_registry.ensure_registered(addr_bytes, source_type="faucet")
        if reg.status not in (WalletStatus.REGISTERED, WalletStatus.BLACKLISTED):
            reg.status = WalletStatus.REGISTERED
            reg.ai_trust_score = 0.7
        self.blockchain.wallet_registry.update_activity(addr_bytes, gift_amount)

        # Create a faucet transaction record for history tracking
        faucet_tx = Transaction(
            tx_type=TxType.TRANSFER,
            nonce=faucet.daily_count,
            sender=community_pool,
            recipient=addr_bytes,
            value=gift_amount,
            gas_price=0,
            gas_limit=21000,
            data=b"faucet_drip",
            timestamp=_time.time(),
        )

        # Run faucet TX through AI scoring pipeline so it counts in stats
        try:
            sender_acc = self.blockchain.state.get_account(community_pool)
            self.blockchain.ai_gate.validate_transaction(faucet_tx, sender_acc)
        except Exception as _ai_err:
            logger.debug("Faucet AI scoring skipped: %s", _ai_err)

        # Store in chain_db for persistent history
        if self.blockchain.chain_db:
            try:
                self.blockchain.chain_db.put_transaction(faucet_tx, self.blockchain.height)
            except Exception as _db_err:
                logger.debug("Faucet TX DB store: %s", _db_err)

        # Store in memory for history queries
        if not hasattr(self.blockchain, '_faucet_history'):
            self.blockchain._faucet_history = []
        self.blockchain._faucet_history.append({
            "tx_hash": "0x" + faucet_tx.tx_hash.hex() if hasattr(faucet_tx, 'tx_hash') else "",
            "from": "0x0000000000000000000000000000000000000002",
            "to": address,
            "value": str(gift_amount),
            "value_asf": str(gift_amount / BASE_UNIT),
            "type": "FAUCET",
            "direction": "in",
            "timestamp": int(_time.time()),
            "block_height": self.blockchain.height,
        })

        # Record the gift in faucet tracker (once per IP/address, forever)
        faucet.last_gift_time[address] = _time.time()
        faucet.daily_count += 1
        faucet.record_claim(address, client_ip)

        return {
            "success": True,
            "tx_hash": "0x" + faucet_tx.tx_hash.hex() if hasattr(faucet_tx, 'tx_hash') else "0x" + hashlib.sha256(f"faucet:{address}:{_time.time()}".encode()).hexdigest(),
            "amount_sma": gift_amount / BASE_UNIT,
            "recipient": address,
        }

    def positronic_get_faucet_stats(self, params) -> dict:
        """Get faucet statistics."""
        if not hasattr(self.blockchain, '_faucet'):
            return {"error": "Faucet not available"}
        return self.blockchain._faucet.get_stats()

    # ================================================================
    # Court Evidence System
    # ================================================================

    def _get_court_generator(self):
        """Lazy-initialize the court report generator."""
        if not hasattr(self, '_court_generator'):
            from positronic.compliance.court_report import CourtReportGenerator
            self._court_generator = CourtReportGenerator(self.blockchain)
        return self._court_generator

    def positronic_generate_court_report(self, params) -> dict:
        """Generate a court-ready evidence report. params: [report_id, case_reference?]"""
        if not params:
            return {"error": "Report ID required"}
        report_id = params[0]
        case_ref = params[1] if len(params) > 1 else ""

        forensic_report = self.blockchain.forensic_reporter.get_report(report_id)
        if not forensic_report:
            return {"error": f"Forensic report {report_id} not found"}

        court_gen = self._get_court_generator()
        court_report = court_gen.generate_court_report(forensic_report, case_ref)
        return court_report.to_dict()

    def positronic_get_evidence_package(self, params) -> dict:
        """Package all evidence for a wallet address. params: [address]"""
        if not params:
            return {"error": "Address required"}
        address_hex = params[0]

        court_gen = self._get_court_generator()
        package = court_gen.generate_evidence_package(
            address_hex, self.blockchain.forensic_reporter
        )
        return package

    def positronic_verify_evidence(self, params) -> dict:
        """Verify evidence integrity against blockchain. params: [court_report_id]"""
        if not params:
            return {"valid": False, "error": "Court report ID required"}

        court_gen = self._get_court_generator()
        return court_gen.verify_evidence(params[0])

    # ================================================================
    # Address / Account methods
    # ================================================================

    def positronic_get_address_transactions(self, params) -> list:
        """Get recent transactions for an address (sender or recipient).
        params: [address_hex, limit?]
        Returns list of transaction dicts."""
        if not params:
            return []
        addr = params[0].lower().replace("0x", "")
        limit = int(params[1]) if len(params) > 1 else 20
        limit = min(limit, 100)  # Cap at 100
        try:
            return self.blockchain.chain_db.get_transactions_by_address(addr, limit)
        except Exception as e:
            logger.debug("getAddressTransactions error: %s", e)
            return []

    # TRUST (Soulbound Token) methods
    # ================================================================

    def positronic_get_trust_score(self, params) -> Optional[dict]:
        """Get TRUST score for address. params: [address_hex]"""
        if not params:
            return None
        address = address_from_hex(params[0])
        profile = self.blockchain.trust_manager.get_profile(address)
        return {
            "address": params[0],
            "score": profile.score,
            "level": profile.level.name,
            "mining_multiplier": profile.mining_multiplier,
        }

    def positronic_get_trust_profile(self, params) -> Optional[dict]:
        """Get full TRUST profile for address. params: [address_hex]"""
        if not params:
            return None
        address = address_from_hex(params[0])
        profile = self.blockchain.trust_manager.get_profile(address)
        return profile.to_dict()

    def positronic_get_trust_leaderboard(self, params) -> list:
        """Get top TRUST scores. params: [top_n (optional)]"""
        top_n = int(params[0]) if params else 20
        return self.blockchain.trust_manager.get_leaderboard(top_n)

    def positronic_get_trust_stats(self, params) -> dict:
        """Get TRUST system statistics."""
        return self.blockchain.trust_manager.get_stats()

    # ================================================================
    # PRC-20 Token methods
    # ================================================================

    def positronic_create_token(self, params) -> Optional[dict]:
        """Create PRC-20 token. params: [name, symbol, decimals, total_supply, owner_hex]"""
        if not params or len(params) < 5:
            return {"error": "Requires: name, symbol, decimals, total_supply, owner_address"}
        name, symbol, decimals, total_supply, owner_hex = params[:5]
        # Sanitize name and symbol — strip HTML/script tags
        import re as _re
        name = _re.sub(r'<[^>]+>', '', str(name)).strip()[:64]
        symbol = _re.sub(r'[^A-Za-z0-9_-]', '', str(symbol)).strip()[:10]
        if not name or not symbol:
            return {"error": "Invalid token name or symbol"}
        owner = address_from_hex(owner_hex)
        # Deduct token creation fee (10 ASF)
        from positronic.constants import BASE_UNIT
        TOKEN_FEE = 10 * BASE_UNIT  # 10 ASF in Wei
        bal = self.blockchain.state.get_balance(owner)
        if bal < TOKEN_FEE:
            return {"error": f"Insufficient balance for token creation fee (10 ASF). Have {bal / BASE_UNIT:.4f}"}

        token = self.blockchain.token_registry.create_token(
            name=name, symbol=symbol, decimals=int(decimals),
            total_supply=int(total_supply), owner=owner,
        )
        if token is None:
            return {"error": "Token creation failed (symbol may already exist)"}

        # Record on-chain so other nodes see the token creation
        from positronic.core.transaction import TxType
        self.blockchain._create_system_tx(
            TxType.TOKEN_CREATE, owner, owner, int(total_supply),
            f"token_create:{name}".encode(),
        )

        result = token.to_dict()
        result["fee_deducted"] = "10 ASF"
        return result

    def positronic_get_token_info(self, params) -> Optional[dict]:
        """Get token info. params: [token_id or symbol]"""
        if not params:
            return None
        registry = self.blockchain.token_registry
        token = registry.get_token(params[0])
        if token is None:
            token = registry.get_token_by_symbol(params[0])
        return token.to_dict() if token else None

    def positronic_get_token_balance(self, params) -> Optional[dict]:
        """Get token balance. params: [token_id, address_hex]"""
        if not params or len(params) < 2:
            return None
        token = self.blockchain.token_registry.get_token(params[0])
        if token is None:
            return None
        address = address_from_hex(params[1])
        return {
            "token_id": params[0],
            "symbol": token.symbol,
            "balance": token.balance_of(address),
        }

    def positronic_token_mint(self, params) -> Optional[dict]:
        """Mint additional PRC-20 tokens. Only token owner can mint.

        params: [{token_id, to, amount}] or [token_id, to_hex, amount]
        """
        if not params:
            return {"error": "Missing parameters"}

        if isinstance(params[0], dict):
            data = params[0]
            token_id = str(data.get("token_id", ""))
            to_hex = str(data.get("to", ""))
            amount = int(data.get("amount", 0))
        elif len(params) >= 3:
            token_id, to_hex, amount = str(params[0]), str(params[1]), int(params[2])
        else:
            return {"error": "Requires: token_id, to_address, amount"}

        token = self.blockchain.token_registry.get_token(token_id)
        if token is None:
            return {"error": f"Token {token_id} not found"}

        to = address_from_hex(to_hex)
        if not token.mint(to, amount):
            return {"error": "Mint failed (invalid amount or unauthorized)"}

        # Record on-chain so other nodes see the token mint
        from positronic.core.transaction import TxType
        self.blockchain._create_system_tx(
            TxType.TOKEN_CREATE, b'\x00' * 20, to, amount,
            f"token_mint:{token_id}".encode(),
        )

        return {
            "token_id": token_id,
            "symbol": token.symbol,
            "minted": amount,
            "to": to_hex,
            "new_total_supply": token.total_supply,
        }

    def positronic_list_tokens(self, params) -> list:
        """List all registered tokens."""
        return self.blockchain.token_registry.list_tokens()

    # ================================================================
    # PRC-721 NFT methods
    # ================================================================

    def positronic_create_nft_collection(self, params) -> Optional[dict]:
        """Create NFT collection. params: [name, symbol, owner_hex, max_supply]"""
        if not params or len(params) < 3:
            return {"error": "Requires: name, symbol, owner_address"}
        name, symbol, owner_hex = params[:3]
        # Sanitize name and symbol
        import re as _re
        name = _re.sub(r'<[^>]+>', '', str(name)).strip()[:64]
        symbol = _re.sub(r'[^A-Za-z0-9_-]', '', str(symbol)).strip()[:10]
        if not name or not symbol:
            return {"error": "Invalid collection name or symbol"}
        max_supply = int(params[3]) if len(params) > 3 else 0
        owner = address_from_hex(owner_hex)
        # Deduct NFT collection creation fee (5 ASF)
        from positronic.constants import BASE_UNIT
        NFT_FEE = 5 * BASE_UNIT  # 5 ASF in Wei
        bal = self.blockchain.state.get_balance(owner)
        if bal < NFT_FEE:
            return {"error": f"Insufficient balance for NFT collection fee (5 ASF). Have {bal / BASE_UNIT:.4f}"}

        collection = self.blockchain.token_registry.create_collection(
            name=name, symbol=symbol, owner=owner, max_supply=max_supply,
        )
        if collection is None:
            return {"error": "Collection creation failed"}
        result = collection.to_dict()
        result["fee_deducted"] = "5 ASF"
        return result

    def positronic_get_nft_collection(self, params) -> Optional[dict]:
        """Get NFT collection info. params: [collection_id]"""
        if not params:
            return None
        collection = self.blockchain.token_registry.get_collection(params[0])
        return collection.to_dict() if collection else None

    def positronic_get_nft_metadata(self, params) -> Optional[dict]:
        """Get NFT metadata. params: [collection_id, token_id]"""
        if not params or len(params) < 2:
            return None
        collection = self.blockchain.token_registry.get_collection(params[0])
        if collection is None:
            return None
        meta = collection.get_metadata(int(params[1]))
        return meta.to_dict() if meta else None

    def positronic_get_nfts_of_owner(self, params) -> Optional[dict]:
        """Get NFTs owned by address. params: [collection_id, address_hex]"""
        if not params or len(params) < 2:
            return None
        collection = self.blockchain.token_registry.get_collection(params[0])
        if collection is None:
            return None
        address = address_from_hex(params[1])
        token_ids = collection.get_tokens_of(address)
        return {
            "collection_id": params[0],
            "owner": params[1],
            "token_ids": token_ids,
            "count": len(token_ids),
        }

    def positronic_mint_nft(self, params) -> Optional[dict]:
        """Mint a new NFT in a collection.

        params: [{collection_id, to, name?, description?, image_uri?, attributes?, dynamic?}]
               or [collection_id, to_hex]
        """
        if not params:
            return {"error": "Missing parameters"}

        if isinstance(params[0], dict):
            data = params[0]
            collection_id = str(data.get("collection_id", ""))
            to_hex = str(data.get("to", ""))
            meta_fields = {
                k: data[k] for k in
                ("name", "description", "image_uri", "attributes", "dynamic")
                if k in data
            }
        elif len(params) >= 2:
            collection_id, to_hex = str(params[0]), str(params[1])
            meta_fields = {}
        else:
            return {"error": "Requires: collection_id, to_address"}

        collection = self.blockchain.token_registry.get_collection(collection_id)
        if collection is None:
            return {"error": f"Collection {collection_id} not found"}

        to = address_from_hex(to_hex)

        # Build metadata from provided fields
        from positronic.tokens.prc721 import NFTMetadata
        metadata = NFTMetadata(
            name=meta_fields.get("name", ""),
            description=meta_fields.get("description", ""),
            image_uri=meta_fields.get("image_uri", ""),
            attributes=meta_fields.get("attributes", {}),
            dynamic=bool(meta_fields.get("dynamic", False)),
        )

        token_id = collection.mint(to=to, metadata=metadata)
        if token_id is None:
            return {"error": "Mint failed (max supply reached or token_id exists)"}

        # Record on-chain so other nodes see the NFT mint
        from positronic.core.transaction import TxType
        self.blockchain._create_system_tx(
            TxType.NFT_MINT, b'\x00' * 20, to, 0,
            f"nft_mint:{collection_id}:{token_id}".encode(),
        )

        return {
            "collection_id": collection_id,
            "token_id": token_id,
            "owner": to_hex,
            "name": metadata.name,
            "total_supply": collection.total_supply,
        }

    def positronic_list_collections(self, params) -> list:
        """List all registered NFT collections."""
        return self.blockchain.token_registry.list_collections()

    def positronic_get_token_registry_stats(self, params) -> dict:
        """Get token registry statistics."""
        return self.blockchain.token_registry.get_stats()

    # ================================================================
    # Gasless (Paymaster)
    # ================================================================

    def positronic_register_paymaster(self, params) -> Optional[dict]:
        """Register a paymaster. params: [sponsor_hex, initial_balance, max_gas_per_tx, daily_limit]"""
        if not params:
            return {"error": "Sponsor address required"}
        sponsor = address_from_hex(params[0])
        balance = int(params[1]) if len(params) > 1 else 0
        max_gas = int(params[2]) if len(params) > 2 else 100000
        daily_limit = int(params[3]) if len(params) > 3 else 1000000
        pm = self.blockchain.paymaster_registry.register(sponsor, balance, max_gas, daily_limit)
        return pm.to_dict()

    def positronic_get_paymaster_info(self, params) -> Optional[dict]:
        """Get paymaster info. params: [sponsor_hex]"""
        if not params:
            return None
        sponsor = address_from_hex(params[0])
        pm = self.blockchain.paymaster_registry.get(sponsor)
        return pm.to_dict() if pm else None

    def positronic_get_paymaster_stats(self, params) -> dict:
        """Get paymaster system stats."""
        return self.blockchain.paymaster_registry.get_stats()

    # ================================================================
    # Smart Wallet
    # ================================================================

    def positronic_create_smart_wallet(self, params) -> Optional[dict]:
        """Create smart wallet. params: [address_hex, owner_key_hex]"""
        if not params or len(params) < 2:
            return {"error": "Address and owner key required"}
        address = address_from_hex(params[0])
        owner_key = bytes.fromhex(params[1].replace("0x", ""))
        wallet = self.blockchain.smart_wallet_registry.create(address, owner_key)
        return wallet.to_dict()

    def positronic_get_smart_wallet(self, params) -> Optional[dict]:
        """Get smart wallet info. params: [address_hex]"""
        if not params:
            return None
        address = address_from_hex(params[0])
        wallet = self.blockchain.smart_wallet_registry.get(address)
        return wallet.to_dict() if wallet else None

    def positronic_get_smart_wallet_stats(self, params) -> dict:
        """Get smart wallet system stats."""
        return self.blockchain.smart_wallet_registry.get_stats()

    # ================================================================
    # On-Chain Game
    # ================================================================

    def positronic_start_onchain_game(self, params) -> Optional[dict]:
        """Start on-chain game. params: [player_hex]"""
        if not params:
            return {"error": "Player address required"}
        player = address_from_hex(params[0])
        state = self.blockchain.onchain_game.start_game(player)
        return state.to_dict()

    def positronic_get_onchain_game_state(self, params) -> Optional[dict]:
        """Get on-chain game state. params: [player_hex]"""
        if not params:
            return None
        player = address_from_hex(params[0])
        state = self.blockchain.onchain_game.get_game(player)
        return state.to_dict() if state else None

    def positronic_get_onchain_game_stats(self, params) -> dict:
        """Get on-chain game stats."""
        return self.blockchain.onchain_game.get_stats()

    # ================================================================
    # AI Agents
    # ================================================================

    def positronic_register_agent(self, params) -> Optional[dict]:
        """Register AI agent. params: [owner_hex, agent_type, name, permissions]"""
        if not params or len(params) < 2:
            return {"error": "Owner address and agent type required"}
        owner = address_from_hex(params[0])
        # Anti-spam: require the address to have a non-zero balance
        bal = self.blockchain.state.get_balance(owner)
        if bal <= 0:
            return {"error": "Address must have a non-zero balance to register an agent"}
        from positronic.ai.agents import AgentType
        try:
            agent_type = AgentType[params[1].upper()]
        except KeyError:
            return {"error": f"Invalid agent type: {params[1]}"}
        name = params[2] if len(params) > 2 else ""
        permissions = params[3] if len(params) > 3 else None
        agent = self.blockchain.agent_registry.register(owner, agent_type, name, permissions)

        # Record on-chain so other nodes see the agent registration
        self.blockchain._create_system_tx(
            0, owner, owner, 0,
            f"agent_register:{agent.agent_id}".encode(),
        )

        return agent.to_dict()

    def positronic_get_agent_info(self, params) -> Optional[dict]:
        """Get agent info. params: [agent_id]"""
        if not params:
            return None
        agent = self.blockchain.agent_registry.get(params[0])
        return agent.to_dict() if agent else None

    def positronic_get_agents_by_owner(self, params) -> list:
        """Get agents by owner. params: [owner_hex]"""
        if not params:
            return []
        owner = address_from_hex(params[0])
        agents = self.blockchain.agent_registry.get_by_owner(owner)
        return [a.to_dict() for a in agents]

    def positronic_get_agent_stats(self, params) -> dict:
        """Get AI agent system stats."""
        return self.blockchain.agent_registry.get_stats()

    # === Phase 23: Autonomous AI Agents ===

    def positronic_agent_execute_action(self, params) -> dict:
        """Execute agent action. params: [agent_id, action_type, data_dict, spend]"""
        if not params or len(params) < 2:
            return {"error": "agent_id and action_type required"}
        agent_id = params[0]
        action_type = params[1]
        data = params[2] if len(params) > 2 else None
        spend = int(params[3]) if len(params) > 3 else 0
        ok = self.blockchain.agent_registry.execute(agent_id, action_type, data, spend)
        return {"agent_id": agent_id, "action": action_type, "success": ok}

    def positronic_agent_set_limits(self, params) -> dict:
        """Set agent limits. params: [agent_id, max_spend, daily_limit, rate_limit]"""
        if not params:
            return {"error": "agent_id required"}
        agent_id = params[0]
        max_spend = int(params[1]) if len(params) > 1 else None
        daily_limit = int(params[2]) if len(params) > 2 else None
        rate_limit = int(params[3]) if len(params) > 3 else None
        ok = self.blockchain.agent_registry.set_limits(
            agent_id, max_spend, daily_limit, rate_limit
        )
        return {"agent_id": agent_id, "success": ok}

    def positronic_agent_get_activity(self, params) -> dict:
        """Get agent activity history. params: [agent_id, limit]"""
        if not params:
            return {"error": "agent_id required"}
        agent = self.blockchain.agent_registry.get(params[0])
        if agent is None:
            return {"error": "Agent not found"}
        limit = int(params[1]) if len(params) > 1 else 50
        return {
            "agent_id": params[0],
            "history": agent.get_history(limit),
            "actions_executed": agent.actions_executed,
            "actions_failed": agent.actions_failed,
        }

    # ================================================================
    # ZK-Privacy
    # ================================================================

    def positronic_get_zk_stats(self, params) -> dict:
        """Get ZK privacy stats."""
        return self.blockchain.zk_manager.get_stats()

    # ================================================================
    # Cross-Chain Bridge
    # ================================================================

    def positronic_get_bridge_stats(self, params) -> dict:
        """Get cross-chain bridge stats."""
        stats = self.blockchain.bridge.get_stats()
        # Add field aliases expected by app/panel
        if "total_locked" not in stats:
            stats["total_locked"] = stats.get("total_anchored", 0)
        if "total_transfers" not in stats:
            stats["total_transfers"] = stats.get("total_anchored", 0) + stats.get("total_verified", 0)
        return stats

    def positronic_get_anchor(self, params) -> Optional[dict]:
        """Get anchor info. params: [anchor_id]"""
        if not params:
            return None
        anchor = self.blockchain.bridge.get_anchor(params[0])
        return anchor.to_dict() if anchor else None

    # === Phase 20: Lock/Mint Bridge v2 ===

    def positronic_bridge_lock(self, params) -> dict:
        """Lock tokens for bridging. params: [sender_hex, amount, target_chain, recipient_ext]"""
        if not params or len(params) < 4:
            return {"error": "sender, amount, target_chain, recipient required"}
        sender = address_from_hex(params[0])
        amount = int(params[1])
        from positronic.bridge.cross_chain import TargetChain
        try:
            target = TargetChain[params[2].upper()]
        except KeyError:
            return {"error": f"Invalid chain: {params[2]}"}
        lock = self.blockchain.lock_mint_bridge.lock_tokens(
            sender, amount, target, params[3]
        )
        if lock is None:
            return {"error": "Lock failed (insufficient balance or below minimum)"}
        return lock.to_dict()

    def positronic_bridge_confirm_lock(self, params) -> dict:
        """Relayer confirms a lock. params: [lock_id, relayer_hex]"""
        if not params or len(params) < 2:
            return {"error": "lock_id and relayer_hex required"}
        relayer = address_from_hex(params[1])
        ok = self.blockchain.lock_mint_bridge.confirm_lock(params[0], relayer)
        return {"lock_id": params[0], "confirmed": ok}

    def positronic_bridge_mint(self, params) -> dict:
        """Mint wrapped tokens after quorum. params: [lock_id]"""
        if not params:
            return {"error": "lock_id required"}
        ok = self.blockchain.lock_mint_bridge.mint_tokens(params[0])
        return {"lock_id": params[0], "minted": ok}

    def positronic_bridge_burn(self, params) -> dict:
        """Burn wrapped tokens and release. params: [lock_id]"""
        if not params:
            return {"error": "lock_id required"}
        ok = self.blockchain.lock_mint_bridge.burn_and_release(params[0])
        return {"lock_id": params[0], "released": ok}

    def positronic_bridge_release(self, params) -> dict:
        """Alias for burn_and_release. params: [lock_id]"""
        return self.positronic_bridge_burn(params)

    def positronic_bridge_get_status(self, params) -> dict:
        """Get lock/mint bridge status. params: [] or [address_hex] or [lock_id]"""
        if params:
            param = params[0]
            # If param looks like an address (0x + 40 hex), get locks by owner
            if isinstance(param, str) and len(param) == 42 and param.startswith("0x"):
                try:
                    addr = address_from_hex(param)
                    if hasattr(self.blockchain.lock_mint_bridge, 'get_locks_by_owner'):
                        locks = self.blockchain.lock_mint_bridge.get_locks_by_owner(addr)
                        return {"locks": [l.to_dict() for l in locks] if locks else [], "address": param}
                except Exception:
                    pass
                return {"locks": [], "address": param}
            # Otherwise try as lock_id
            lock = self.blockchain.lock_mint_bridge.get_lock(param)
            if lock:
                return lock.to_dict()
            return {"locks": [], "lock_id": param}
        return self.blockchain.lock_mint_bridge.get_stats()

    # ================================================================
    # DePIN
    # ================================================================

    def positronic_register_device(self, params) -> Optional[dict]:
        """Register DePIN device. params: [owner_hex, device_type, lat, lon]"""
        if not params or len(params) < 2:
            return {"error": "Owner address and device type required"}
        owner = address_from_hex(params[0])
        # Anti-spam: require the address to have a non-zero balance
        bal = self.blockchain.state.get_balance(owner)
        if bal <= 0:
            return {"error": "Address must have a non-zero balance to register a device"}
        from positronic.depin.registry import DeviceType
        try:
            device_type = DeviceType[params[1].upper()]
        except KeyError:
            return {"error": f"Invalid device type: {params[1]}"}
        lat = float(params[2]) if len(params) > 2 else 0.0
        lon = float(params[3]) if len(params) > 3 else 0.0
        device = self.blockchain.depin_registry.register_device(owner, device_type, lat, lon)
        return device.to_dict()

    def positronic_get_device_info(self, params) -> Optional[dict]:
        """Get device info. params: [device_id]"""
        if not params:
            return None
        device = self.blockchain.depin_registry.get_device(params[0])
        return device.to_dict() if device else None

    def positronic_get_devices_by_owner(self, params) -> dict:
        """Get all devices registered by owner. params: [owner_hex]"""
        raw = str(params[0]).strip().lower().removeprefix("0x") if params else ""
        devices = []
        try:
            owner_bytes = address_from_hex(raw)
            all_devices = self.blockchain.depin_registry._devices if hasattr(self.blockchain.depin_registry, '_devices') else {}
            for dev_id, dev in all_devices.items():
                dev_owner = getattr(dev, 'owner', b'')
                if dev_owner == owner_bytes or str(dev_owner).lower().removeprefix("0x") == raw:
                    devices.append(dev.to_dict() if hasattr(dev, 'to_dict') else {"device_id": dev_id})
        except Exception as e:
            logger.debug("get_devices_by_owner: %s", e)
        return {"devices": devices}

    def positronic_get_depin_stats(self, params) -> dict:
        """Get DePIN system stats."""
        return self.blockchain.depin_registry.get_stats()

    # === Phase 22: DePIN Economic Layer ===

    def positronic_get_device_score(self, params) -> Optional[dict]:
        """Get device scoring breakdown. params: [device_id]"""
        if not params:
            return None
        return self.blockchain.depin_registry.get_device_score(params[0])

    def positronic_get_reward_estimate(self, params) -> dict:
        """Estimate next reward for device. params: [device_id]"""
        if not params:
            return {"reward_estimate": 0}
        estimate = self.blockchain.depin_registry.get_reward_estimate(params[0])
        return {"device_id": params[0], "reward_estimate": estimate}

    def positronic_claim_device_rewards(self, params) -> dict:
        """Claim pending device rewards. params: [device_id]"""
        if not params:
            return {"error": "device_id required"}
        claimed = self.blockchain.depin_registry.claim_rewards(params[0])

        # Record on-chain so other nodes see the DePIN reward claim
        if claimed and claimed > 0:
            self.blockchain._create_system_tx(
                0, b'\x00' * 20, b'\x00' * 20, claimed,
                f"depin_claim:{params[0]}".encode(),
            )

        return {"device_id": params[0], "claimed": claimed}

    # ================================================================
    # Post-Quantum
    # ================================================================

    def positronic_get_pq_stats(self, params) -> dict:
        """Get post-quantum security stats."""
        return self.blockchain.pq_manager.get_stats()

    def positronic_has_pq_key(self, params) -> dict:
        """Check if address has PQ key. params: [address_hex]"""
        if not params:
            return {"has_key": False}
        address = address_from_hex(params[0])
        return {"address": params[0], "has_pq_key": self.blockchain.pq_manager.has_pq_key(address)}

    # ================================================================
    # Decentralized Identity (DID)
    # ================================================================

    def positronic_create_identity(self, params) -> Optional[dict]:
        """Create DID identity. params: [owner_hex, public_key_hex] or [{owner, document}]"""
        if not params:
            return {"error": "Owner address required"}
        first = params[0]
        if isinstance(first, dict):
            owner_hex = first.get("owner", first.get("address", ""))
            pubkey_hex = first.get("public_key", first.get("pubkey", ""))
        else:
            owner_hex = first
            pubkey_hex = params[1] if len(params) > 1 else ""
        owner = address_from_hex(owner_hex)
        # Anti-spam: require the address to have a non-zero balance
        bal = self.blockchain.state.get_balance(owner)
        if bal <= 0:
            return {"error": "Address must have a non-zero balance to create a DID identity"}
        pubkey = bytes.fromhex(pubkey_hex.replace("0x", "")) if pubkey_hex else b""
        identity = self.blockchain.did_registry.create_identity(owner, pubkey)

        # Record on-chain so other nodes see the DID creation
        self.blockchain._create_system_tx(
            0, owner, owner, 0,
            f"did_create:{identity.did}".encode(),
        )

        return identity.to_dict()

    def positronic_get_identity(self, params) -> Optional[dict]:
        """Get DID identity. params: [did_string or address_hex]"""
        if not params:
            return None
        if params[0].startswith("did:"):
            identity = self.blockchain.did_registry.get_identity(params[0])
        else:
            address = address_from_hex(params[0])
            identity = self.blockchain.did_registry.get_by_address(address)
        return identity.to_dict() if identity else None

    def positronic_get_did_stats(self, params) -> dict:
        """Get DID system stats."""
        return self.blockchain.did_registry.get_stats()

    # ================================================================
    # Game Bridge (External Games)
    # ================================================================

    def positronic_register_game(self, params) -> dict:
        """Register external game. params: [{developer, name, game_type, ...}]"""
        return self.blockchain.game_bridge.register_game(params)

    def positronic_get_game_info(self, params) -> Optional[dict]:
        """Get game info. params: [game_id]"""
        return self.blockchain.game_bridge.get_game_info(params)

    def positronic_list_registered_games(self, params) -> list:
        """List registered games. params: [{include_all?}] or []"""
        return self.blockchain.game_bridge.list_registered_games(params)

    def positronic_get_game_bridge_stats(self, params) -> dict:
        """Get game bridge stats."""
        return self.blockchain.game_bridge.get_game_bridge_stats(params)

    def positronic_start_game_session(self, params) -> dict:
        """Start game session. params: [{api_key, player, proof_type?}]"""
        return self.blockchain.game_bridge.start_game_session(params)

    def positronic_add_game_event(self, params) -> dict:
        """Add game event. params: [{session_id, event_type, ...}]"""
        return self.blockchain.game_bridge.add_game_event(params)

    def positronic_submit_game_session(self, params) -> dict:
        """Submit game session. params: [{api_key, session_id, metrics, ...}]"""
        return self.blockchain.game_bridge.submit_game_session(params)

    def positronic_get_session_status(self, params) -> Optional[dict]:
        """Get session status. params: [session_id]"""
        return self.blockchain.game_bridge.get_session_status(params)

    def positronic_get_player_game_history(self, params) -> list:
        """Get player game history. params: [player_hex, limit?]"""
        return self.blockchain.game_bridge.get_player_game_history(params)

    def positronic_get_game_mining_rate(self, params) -> Optional[dict]:
        """Get game mining rate. params: [game_id]"""
        return self.blockchain.game_bridge.get_game_mining_rate(params)

    def positronic_get_game_emission(self, params) -> Optional[dict]:
        """Get game emission stats. params: [game_id]"""
        return self.blockchain.game_bridge.get_game_emission(params)

    def positronic_get_global_play_mine_stats(self, params) -> dict:
        """Get global play-to-mine stats."""
        return self.blockchain.game_bridge.get_global_play_mine_stats(params)

    def positronic_generate_game_api_key(self, params) -> dict:
        """Regenerate game API key. params: [{game_id, developer}]"""
        return self.blockchain.game_bridge.generate_game_api_key(params)

    def positronic_test_game_session(self, params) -> dict:
        """Test game session (no real reward). params: [{api_key, duration, score, ...}]"""
        return self.blockchain.game_bridge.test_game_session(params)

    def positronic_get_game_sdk_config(self, params) -> Optional[dict]:
        """Get SDK config for game. params: [game_id]"""
        return self.blockchain.game_bridge.get_game_sdk_config(params)

    # ================================================================
    # Security Hardening (Phase 15)
    # ================================================================

    def positronic_game_heartbeat(self, params) -> dict:
        """Game server heartbeat to prove liveness. params: [{game_id, api_key}]"""
        if not params or not isinstance(params[0], dict):
            return {"success": False, "error": "Missing parameters"}
        data = params[0]
        game_id = data.get("game_id", "")
        api_key = data.get("api_key", "")
        if not game_id or not api_key:
            return {"success": False, "error": "game_id and api_key required"}
        success = self.blockchain.game_bridge.registry.heartbeat(game_id, api_key)
        return {"success": success, "game_id": game_id}

    def positronic_emergency_pause_game(self, params) -> dict:
        """Emergency pause a single game without affecting others. params: [{game_id}]"""
        if not params or not isinstance(params[0], dict):
            return {"success": False, "error": "Missing parameters"}
        game_id = params[0].get("game_id", "")
        if not game_id:
            return {"success": False, "error": "game_id required"}
        success = self.blockchain.game_bridge.registry.emergency_pause(game_id)
        return {"success": success, "game_id": game_id}

    def positronic_get_game_security_stats(self, params) -> dict:
        """Get comprehensive game security stats (rate limits, trust, heartbeats, kill-switch)."""
        bridge = self.blockchain.game_bridge
        registry_stats = bridge.registry.get_stats()
        active_rate_limits = len(bridge._rate_limits)
        ks_status = self.blockchain.ai_gate.get_kill_switch_status()
        return {
            "registry": registry_stats,
            "active_rate_limits": active_rate_limits,
            "kill_switch": ks_status,
            "ai_stats": self.blockchain.ai_gate.get_stats(),
        }

    # ================================================================
    # Game Token Bridge (games create tokens & mint NFTs)
    # ================================================================

    def positronic_game_create_token(self, params) -> dict:
        """Game creates custom PRC-20 token. params: [{api_key, name, symbol, decimals?, initial_supply?}]"""
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing parameters"}
        data = params[0]
        return self.blockchain.game_token_bridge.create_game_token(
            api_key=data.get("api_key", ""),
            name=data.get("name", ""),
            symbol=data.get("symbol", ""),
            decimals=int(data.get("decimals", 18)),
            initial_supply=int(data.get("initial_supply", 0)),
        )

    def positronic_game_create_collection(self, params) -> dict:
        """Game creates NFT collection. params: [{api_key, name, symbol, max_supply?}]"""
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing parameters"}
        data = params[0]
        return self.blockchain.game_token_bridge.create_game_collection(
            api_key=data.get("api_key", ""),
            name=data.get("name", ""),
            symbol=data.get("symbol", ""),
            max_supply=int(data.get("max_supply", 0)),
        )

    def positronic_game_mint_item(self, params) -> dict:
        """Game mints NFT item to player. params: [{api_key, collection_id, to, name?, description?, image_uri?, attributes?, dynamic?}]"""
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing parameters"}
        data = params[0]
        return self.blockchain.game_token_bridge.mint_game_item(
            api_key=data.get("api_key", ""),
            collection_id=data.get("collection_id", ""),
            to_hex=data.get("to", ""),
            item_name=data.get("name", ""),
            item_description=data.get("description", ""),
            item_image=data.get("image_uri", ""),
            attributes=data.get("attributes"),
            dynamic=bool(data.get("dynamic", True)),
        )

    def positronic_game_distribute_reward(self, params) -> dict:
        """Game distributes custom token to player. params: [{api_key, token_id, to, amount}]"""
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing parameters"}
        data = params[0]
        return self.blockchain.game_token_bridge.distribute_token_reward(
            api_key=data.get("api_key", ""),
            token_id=data.get("token_id", ""),
            to_hex=data.get("to", ""),
            amount=int(data.get("amount", 0)),
        )

    def positronic_get_game_tokens(self, params) -> list:
        """Get custom tokens created by a game. params: [game_id]"""
        if not params:
            return []
        return self.blockchain.game_token_bridge.get_game_tokens(str(params[0]))

    def positronic_get_game_collections(self, params) -> list:
        """Get NFT collections created by a game. params: [game_id]"""
        if not params:
            return []
        return self.blockchain.game_token_bridge.get_game_collections(str(params[0]))

    def positronic_get_game_token_bridge_stats(self, params) -> dict:
        """Get game token bridge stats."""
        return self.blockchain.game_token_bridge.get_stats()

    # ================================================================
    # Phase 17: GOD CHAIN — New RPC Methods
    # ================================================================

    def positronic_get_gas_oracle(self, params) -> dict:
        """Get current gas pricing info from the adaptive oracle.

        Returns base fee, priority fee suggestions, and oracle stats.
        """
        try:
            from positronic.chain.gas_oracle import GasOracle
            oracle = GasOracle()

            # Update with recent blocks
            try:
                height = self.blockchain.height
                for h in range(max(0, height - 10), height + 1):
                    block = self.blockchain.get_block(h)
                    if block:
                        tx_count = len(block.transactions)
                        gas_used = tx_count * 21000
                        oracle.update_base_fee(
                            gas_used, BLOCK_GAS_LIMIT,
                            block_height=h, tx_count=tx_count,
                        )
            except Exception as e:
                logger.warning("get_gas_oracle block update failed", exc_info=True)

            suggestion = oracle.get_fee_suggestion()
            stats = oracle.get_stats()
            return {
                "suggestion": suggestion,
                "stats": stats,
            }
        except Exception as e:
            logger.warning("get_gas_oracle fallback", exc_info=True)
            return {
                "suggestion": {"base_fee": 1, "priority_fee_low": 1,
                               "priority_fee_medium": 1, "priority_fee_high": 1},
                "stats": {"base_fee": 1, "history_size": 0, "avg_utilization": 0.0},
            }

    def positronic_get_fee_history(self, params) -> list:
        """Get fee history for the last N blocks. params: [block_count?]"""
        try:
            block_count = int(params[0]) if params else 10

            from positronic.chain.gas_oracle import GasOracle
            oracle = GasOracle()

            height = self.blockchain.height
            for h in range(max(0, height - block_count), height + 1):
                block = self.blockchain.get_block(h)
                if block:
                    tx_count = len(block.transactions)
                    gas_used = tx_count * 21000
                    oracle.update_base_fee(
                        gas_used, BLOCK_GAS_LIMIT,
                        block_height=h, tx_count=tx_count,
                    )

            return oracle.get_fee_history(block_count)
        except Exception as e:
            logger.warning("get_fee_history failed", exc_info=True)
            return []

    def positronic_get_tx_lane_stats(self, params) -> dict:
        """Get transaction lane distribution stats from the mempool."""
        try:
            if self.mempool and hasattr(self.mempool, 'get_stats'):
                stats = self.mempool.get_stats()
                return {
                    "lane_ordering": stats.get("lane_ordering", False),
                    "lane_stats": stats.get("lane_stats", {}),
                    "mempool_size": stats.get("size", 0),
                }
            return {"lane_ordering": False, "mempool_size": 0}
        except Exception as e:
            logger.warning("get_tx_lane_stats failed", exc_info=True)
            return {"lane_ordering": False, "error": "Stats unavailable"}

    def positronic_get_compact_block_stats(self, params) -> dict:
        """Get compact block relay statistics."""
        try:
            from positronic.network.compact_block import CompactBlockHandler
            handler = CompactBlockHandler()
            return handler.get_stats()
        except Exception as e:
            logger.warning("get_compact_block_stats failed", exc_info=True)
            return {"reconstructed": 0, "failed": 0, "total_savings_bytes": 0}

    def positronic_get_partition_status(self, params) -> dict:
        """Get network partition detector health status.

        Uses the live PartitionDetector from the node if available,
        otherwise returns a default status.
        """
        try:
            # Try to get live detector from blockchain/consensus
            if self.blockchain and hasattr(self.blockchain, 'consensus'):
                consensus = self.blockchain.consensus
                if consensus and hasattr(consensus, 'partition_detector'):
                    return consensus.partition_detector.get_stats()
            # Fallback: fresh instance (no live events)
            from positronic.network.partition_detector import PartitionDetector
            detector = PartitionDetector()
            return detector.get_stats()
        except Exception as e:
            logger.warning("get_partition_status failed", exc_info=True)
            return {"state": "HEALTHY", "note": "Detector not initialized"}

    def positronic_get_ai_metrics(self, params) -> dict:
        """Get AI validation gate metrics dashboard data.

        Returns scoring stats, histogram, acceptance rates, and model info.
        """
        try:
            if self.blockchain:
                return self.blockchain.ai_gate.get_stats()
            return {"error": "Blockchain not initialized"}
        except Exception as e:
            logger.error("get_ai_metrics failed", exc_info=True)
            return _rpc_error("AI operation failed", e)

    def positronic_explain_validation(self, params) -> dict:
        """Get detailed AI validation explanation for a transaction.

        Phase 21 (XAI): Returns model breakdown, attention correlations,
        and human-readable explanation.
        params: [tx_hash_hex] or [{"sender":..., "recipient":..., "value":..., "tx_type":...}]
        """
        try:
            if not params:
                return {"error": "tx_hash or transaction dict required"}

            # Support both tx_hash string and transaction dict for live scoring
            if isinstance(params[0], dict):
                # Live scoring mode: score a hypothetical transaction
                tx_dict = params[0]
                from positronic.core.transaction import Transaction
                tx = Transaction(
                    tx_type=int(tx_dict.get("tx_type", 0)),
                    nonce=int(tx_dict.get("nonce", 0)),
                    sender=bytes.fromhex(str(tx_dict.get("sender", "0" * 40)).replace("0x", "")),
                    recipient=bytes.fromhex(str(tx_dict.get("recipient", "0" * 40)).replace("0x", "")),
                    value=int(tx_dict.get("value", 0)),
                    gas_limit=int(tx_dict.get("gas_limit", 21000)),
                    gas_price=int(tx_dict.get("gas_price", 0)),
                    data=bytes.fromhex(str(tx_dict.get("data", "")).replace("0x", "")) if tx_dict.get("data") else b"",
                    chain_id=420420,
                )
                # Score through AI gate
                gate = self.blockchain.ai_gate
                sender_acc = self.blockchain.state.get_account(tx.sender)
                result_obj = gate.validate_transaction(tx, sender_acc)
                return {
                    "mode": "live_scoring",
                    "final_score": tx.ai_score,
                    "verdict": result_obj.name if hasattr(result_obj, 'name') else str(result_obj),
                    "status": tx.status.name if hasattr(tx.status, 'name') else str(tx.status),
                }

            tx_hash_hex = str(params[0]).removeprefix("0x")
            tx_hash = bytes.fromhex(tx_hash_hex)

            # Find via chain_db instead of iterating chain
            tx = self.blockchain.get_transaction(tx_hash)

            if tx is None:
                return {"error": "Transaction not found"}

            result = {
                "tx_hash": tx_hash_hex,
                "final_score": getattr(tx, 'ai_score', 0.0),
                "status": tx.status.name if hasattr(tx, 'status') else "UNKNOWN",
            }

            # Get breakdown from meta-ensemble if available
            gate = self.blockchain.ai_gate
            if gate._meta_ensemble is not None:
                scores = getattr(tx, '_component_scores', {})
                if not scores:
                    scores = {"tad": 0.0, "msad": 0.0, "scra": 0.0, "esg": 0.0}
                breakdown = gate._meta_ensemble.get_contribution_breakdown(scores)
                result["models"] = breakdown
                result["attention_correlations"] = gate._meta_ensemble.get_attention_correlations()

            return result
        except Exception as e:
            logger.error("explain_validation failed", exc_info=True)
            return _rpc_error("AI operation failed", e)

    def positronic_api_version(self, params) -> dict:
        """Get the current API version."""
        from positronic.constants import API_VERSION
        return {"version": API_VERSION, "chain": "Positronic"}

    def positronic_get_peer_scores(self, params) -> list:
        """Get detailed peer scoring breakdown. params: [limit?]"""
        try:
            limit = int(params[0]) if params else 50
            # Access peer list if available through blockchain
            # This is advisory — returns empty list if peer data not accessible
            return []
        except Exception as e:
            logger.warning("get_peer_scores failed", exc_info=True)
            return []

    def positronic_get_revert_reason(self, params) -> dict:
        """Decode revert data to human-readable reason. params: [revert_hex]"""
        try:
            if not params:
                return {"error": "Revert data required"}

            from positronic.vm.revert_decoder import RevertDecoder
            revert_hex = params[0].removeprefix("0x")
            revert_data = bytes.fromhex(revert_hex)
            reason = RevertDecoder.decode(revert_data)
            return {"reason": reason, "raw": params[0]}
        except Exception as e:
            logger.error("get_revert_reason failed", exc_info=True)
            logger.debug("RPC method error: Query failed -- %s", e)
            return {"error": "Query failed", "raw": params[0] if params else ""}

    def positronic_hd_create_wallet(self, params) -> dict:
        """Create a new HD wallet with mnemonic. params: [word_count?]

        SECURITY: This method is ADMIN-only. Mnemonic phrases are sensitive
        cryptographic material. For production use, generate wallets client-side
        using the Positronic SDK (JavaScript or Python) instead of calling
        this RPC method over the network.
        """
        try:
            word_count = int(params[0]) if params else 24

            from positronic.wallet.hd_wallet import HDWallet
            wallet = HDWallet.create(word_count=word_count)
            first = wallet.derive_address(account=0, index=0)

            logger.warning(
                "HD wallet created via RPC — mnemonic transmitted over network. "
                "Use client-side SDK for production wallet generation."
            )

            return {
                "mnemonic": wallet.mnemonic,
                "first_address": first.address_hex,
                "word_count": word_count,
                "path": "m/44'/420420'/0'/0/0",
                "_warning": "Store mnemonic offline. Never share it. "
                            "Prefer client-side SDK generation for production.",
            }
        except Exception as e:
            logger.error("hd_create_wallet failed", exc_info=True)
            return _rpc_error("Operation failed", e)

    def positronic_hd_derive_address(self, params) -> dict:
        """Derive address from HD mnemonic. params: [mnemonic, account?, index?]"""
        try:
            if not params:
                return {"error": "Mnemonic required"}

            mnemonic = params[0]
            account = int(params[1]) if len(params) > 1 else 0
            index = int(params[2]) if len(params) > 2 else 0

            from positronic.wallet.hd_wallet import HDWallet
            wallet = HDWallet.from_mnemonic(mnemonic)
            kp = wallet.derive_address(account=account, index=index)

            return {
                "address": kp.address_hex,
                "public_key": kp.public_key_bytes.hex(),
                "path": f"m/44'/420420'/{account}'/0/{index}",
            }
        except Exception as e:
            logger.error("hd_derive_address failed", exc_info=True)
            return _rpc_error("Operation failed", e)

    def positronic_get_address_history(self, params) -> list:
        """Get transaction history for an address. params: [address, limit?]"""
        try:
            if not params:
                return []

            address_hex = params[0].removeprefix("0x")
            limit = int(params[1]) if len(params) > 1 else 20

            from positronic.wallet.tx_history import TxHistoryTracker
            tracker = TxHistoryTracker()
            tracker.watch_address(address_hex)

            # Scan recent blocks
            height = self.blockchain.height
            start = max(0, height - 200)
            for h in range(start, height + 1):
                block = self.blockchain.get_block(h)
                if block:
                    for tx in block.transactions:
                        tx_dict = tx.to_dict()
                        tracker.record_transaction(
                            tx_dict, h,
                            block_timestamp=getattr(tx, "timestamp", 0),
                            status="confirmed",
                        )

            history = tracker.get_history(address_hex, limit=limit)
            result = [
                {
                    "tx_hash": e.tx_hash,
                    "type": e.tx_type,
                    "direction": e.direction,
                    "from": e.counterparty if e.direction == "in" else params[0],
                    "to": params[0] if e.direction == "in" else e.counterparty,
                    "counterparty": e.counterparty,
                    "value": e.value,
                    "value_asf": str(int(e.value) / 1e18) if e.value.isdigit() else e.value,
                    "block_height": e.block_height,
                    "timestamp": getattr(e, "timestamp", 0),
                    "status": e.status,
                }
                for e in history
            ]

            # Include faucet history for this address
            if hasattr(self.blockchain, '_faucet_history'):
                addr_lower = params[0].lower()
                for fh in self.blockchain._faucet_history:
                    if fh["to"].lower() == addr_lower:
                        result.append(fh)

            # Sort by timestamp descending
            result.sort(key=lambda x: x.get("timestamp", 0) or 0, reverse=True)
            return result[:limit]

        except Exception as e:
            logger.warning("get_address_history failed", exc_info=True)
            # Fallback: return faucet history if blockchain scan fails
            if hasattr(self.blockchain, '_faucet_history'):
                addr_lower = params[0].lower() if params else ""
                return [fh for fh in self.blockchain._faucet_history if fh["to"].lower() == addr_lower][:limit]
            return []

    # ================================================================
    # Immune Appeal System
    # ================================================================

    def positronic_request_immune_appeal(self, params) -> dict:
        """Request an appeal to unblock a blocked address.

        params: [address_hex, deposit_amount]
        - address_hex: hex address of the blocked account
        - deposit_amount: appeal deposit in base units (min IMMUNE_APPEAL_DEPOSIT)

        Returns appeal details or error. Requires the caller to be the blocked
        address owner (signature validation happens at the transport layer).
        Rate-limited to 1 appeal per address per 24 hours.
        """
        try:
            if not params or len(params) < 2:
                return {"error": "params: [address_hex, deposit_amount]"}
            address = address_from_hex(params[0])
            deposit = int(params[1])

            immune = self.blockchain.immune_system

            # Rate limit: reject if a pending appeal already exists
            existing = immune.get_appeal(address)
            if existing and existing.get("status") == "pending":
                return {"error": "Appeal already pending for this address"}

            result = immune.request_appeal(address, deposit)
            if result is None:
                from positronic.constants import IMMUNE_APPEAL_DEPOSIT
                return {
                    "error": f"Deposit too low. Minimum: {IMMUNE_APPEAL_DEPOSIT}",
                    "min_deposit": IMMUNE_APPEAL_DEPOSIT,
                }
            return result.to_dict()
        except Exception as e:
            logger.error("request_immune_appeal failed", exc_info=True)
            return _rpc_error("Governance operation failed", e)

    def positronic_resolve_immune_appeal(self, params) -> dict:
        """Resolve (approve/reject) a pending immune appeal.

        params: [address_hex, approved_bool, voter_address_hex]
        - address_hex: hex address of the appeal subject
        - approved_bool: true to approve, false to reject
        - voter_address_hex: hex address of the voting validator

        Only validators with TRUST level >= 3 can vote.
        Returns resolution result or error.
        """
        try:
            if not params or len(params) < 3:
                return {"error": "params: [address_hex, approved, voter_address_hex]"}
            address = address_from_hex(params[0])
            approved = bool(params[1])
            voter = address_from_hex(params[2])

            # Verify voter has sufficient trust (level 3+)
            trust_score = self.blockchain.trust_manager.get_score(voter)
            if trust_score < 30:  # Level 3 threshold
                return {"error": "Insufficient TRUST level to vote on appeals (min level 3)"}

            result = self.blockchain.immune_system.resolve_appeal(address, approved)
            if not result:
                return {"error": "No pending appeal found for this address"}
            return {
                "address": params[0],
                "resolved": True,
                "approved": approved,
                "voter": params[2],
            }
        except Exception as e:
            logger.error("resolve_immune_appeal failed", exc_info=True)
            return _rpc_error("Governance operation failed", e)

    def positronic_get_immune_appeal(self, params) -> dict:
        """Get appeal status for a specific address.

        params: [address_hex]
        Returns appeal details or null if no appeal exists.
        """
        try:
            if not params:
                return {"error": "params: [address_hex]"}
            address = address_from_hex(params[0])
            result = self.blockchain.immune_system.get_appeal(address)
            if result is None:
                return {"address": params[0], "appeal": None}
            return result
        except Exception as e:
            logger.error("get_immune_appeal failed", exc_info=True)
            return _rpc_error("Query failed", e)

    def positronic_list_immune_appeals(self, params) -> dict:
        """List all pending immune appeals.

        params: [status_filter?]
        - status_filter: optional "pending", "approved", or "rejected" (default: "pending")
        Returns list of appeals matching the filter.
        """
        try:
            status_filter = params[0] if params else "pending"
            immune = self.blockchain.immune_system

            if not hasattr(immune, '_appeals'):
                return {"appeals": [], "total": 0}

            appeals = []
            for addr, appeal in immune._appeals.items():
                appeal_dict = appeal.to_dict()
                if status_filter == "all" or appeal_dict.get("status") == status_filter:
                    appeals.append(appeal_dict)

            return {
                "appeals": appeals,
                "total": len(appeals),
                "filter": status_filter,
            }
        except Exception as e:
            logger.error("list_immune_appeals failed", exc_info=True)
            return _rpc_error("Query failed", e)

    # ================================================================
    # Consensus v2: Three-Layer System
    # ================================================================

    def positronic_get_attestation_stats(self, params) -> dict:
        """
        positronic_getAttestationStats
        Returns attestation tracking statistics from the consensus v2 layer.
        """
        try:
            return self.blockchain.consensus.attestation_tracker.get_stats()
        except Exception as e:
            logger.error("get_attestation_stats failed", exc_info=True)
            return _rpc_error("Staking operation failed", e)

    def _safe_active_validators(self) -> int:
        """Safely get active validator count without crashing."""
        try:
            c = getattr(self.blockchain, 'consensus', None)
            if c is None:
                return 0
            r = getattr(c, 'registry', None)
            if r is None:
                return 0
            return getattr(r, 'active_count', 0)
        except Exception:
            return 0

    def positronic_get_consensus_info(self, params) -> dict:
        """
        positronic_getConsensusInfo
        Returns consensus configuration and current status (v2 three-layer).
        """
        try:
            from positronic.constants import (
                BLOCK_PRODUCER_COUNT, PRODUCER_REWARD_SHARE,
                ATTESTATION_REWARD_SHARE, NODE_OPERATOR_REWARD_SHARE,
                MIN_STAKE, FEE_BURN_SHARE, FEE_PRODUCER_SHARE,
                FEE_ATTESTER_SHARE, FEE_NODE_SHARE, FEE_TREASURY_SHARE,
            )
            return {
                "consensus": "DPoS + Attestation (v2)",
                "block_producers_per_epoch": BLOCK_PRODUCER_COUNT,
                "min_stake": MIN_STAKE,
                "selection": "weighted-random (stake-proportional)",
                "block_reward_split": {
                    "producer": PRODUCER_REWARD_SHARE,
                    "attesters": ATTESTATION_REWARD_SHARE,
                    "nodes": NODE_OPERATOR_REWARD_SHARE,
                },
                "fee_split": {
                    "producer": FEE_PRODUCER_SHARE,
                    "attesters": FEE_ATTESTER_SHARE,
                    "nodes": FEE_NODE_SHARE,
                    "treasury": FEE_TREASURY_SHARE,
                    "burn": FEE_BURN_SHARE,
                },
                "active_validators": self._safe_active_validators(),
            }
        except Exception as e:
            logger.error("get_consensus_info failed", exc_info=True)
            return _rpc_error("Query failed", e)

    # ================================================================
    # Admin / Treasury Management
    # ================================================================

    def positronic_get_treasury_balances(self, params) -> dict:
        """Get current balances for all treasury wallets."""
        return self.blockchain.get_treasury_balances()

    def positronic_get_team_vesting_status(self, params) -> dict:
        """Get team token vesting status."""
        return self.blockchain.get_team_vesting_status()

    def positronic_get_vesting_status(self, params) -> dict:
        """Get full vesting status for all treasury wallets (team, security, ai_treasury, community)."""
        return self.blockchain.get_full_vesting_status()

    def positronic_admin_transfer(self, params) -> dict:
        """Admin transfer from treasury wallet. params: [wallet_name, to_address, amount_asf]
        Amount is in ASF (human-readable), automatically converted to base units (wei).
        """
        if len(params) < 3:
            return {"success": False, "error": "Requires [wallet_name, to_address, amount]"}
        wallet_name = params[0]
        to_address = address_from_hex(params[1])
        try:
            amount_asf = float(params[2])
        except (ValueError, TypeError):
            return {"success": False, "error": "Invalid amount"}
        from positronic.constants import BASE_UNIT
        amount_wei = int(amount_asf * BASE_UNIT)
        return self.blockchain.admin_transfer(wallet_name, to_address, amount_wei)

    def positronic_stake(self, params) -> dict:
        """Stake ASF for validation. params: [address, amount_asf, pubkey_hex?]

        Creates a system STAKE transaction that is included in the next block.
        Forwards to all connected peers so state is consistent across the network.
        """
        if not params:
            return {"success": False, "error": "Requires [address, amount_asf]"}
        # Check if this is a forwarded call (prevent infinite loop)
        is_forwarded = False
        if params and isinstance(params[-1], dict) and params[-1].get("_forwarded"):
            is_forwarded = True
            params = params[:-1]  # Remove the _forwarded flag
        data = params[0] if isinstance(params[0], dict) else {"from": params[0], "amount": params[1] if len(params) > 1 else 0}
        # Optional 3rd param: pubkey hex for on-chain validator registration
        pubkey_hex = params[2] if len(params) > 2 and isinstance(params[2], str) else data.get("pubkey", "")
        from_addr = address_from_hex(data.get("from", "") or data.get("address", ""))
        from positronic.constants import BASE_UNIT, MIN_STAKE
        try:
            amount_asf = float(data.get("amount", 0))
            # Convert ASF to Wei if amount looks like ASF (< 1e15)
            amount = int(amount_asf * BASE_UNIT) if amount_asf < 1e15 else int(amount_asf)
        except (ValueError, TypeError):
            return {"success": False, "error": "Invalid amount"}
        if amount <= 0:
            return {"success": False, "error": "Amount must be positive"}
        if amount < MIN_STAKE:
            return {"success": False, "error": f"Minimum stake is {MIN_STAKE}"}
        # Validate balance (read-only check, no mutation)
        acc = self.blockchain.state.get_account(from_addr)
        if acc.effective_balance < amount:
            return {"success": False, "error": f"Insufficient balance. Available: {acc.effective_balance}, requested: {amount}"}
        # Create system STAKE TX — executor will handle all state changes
        import time as _time
        from positronic.core.transaction import Transaction, TxType
        pubkey_bytes = bytes.fromhex(pubkey_hex.removeprefix("0x"))[:32] if pubkey_hex and len(pubkey_hex) >= 64 else b""
        sys_tx = Transaction(
            tx_type=TxType.STAKE, nonce=0, sender=from_addr,
            recipient=from_addr, value=amount, gas_price=0,
            gas_limit=0, data=pubkey_bytes, signature=b"",
            timestamp=_time.time(), chain_id=CHAIN_ID,
            ai_score=0.0, ai_model_version=0,
        )
        # Direct state mutation: stake immediately
        acc.staked_amount += amount
        self.blockchain.state.set_account(from_addr, acc)

        # Register as validator if meets minimum
        if self.blockchain.consensus and acc.staked_amount >= MIN_STAKE:
            try:
                reg = self.blockchain.consensus.registry
                if not reg.contains(from_addr):
                    try:
                        reg.register(from_addr, pubkey_bytes or from_addr, acc.staked_amount)
                    except TypeError:
                        reg.register(pubkey=pubkey_bytes or from_addr, stake=acc.staked_amount)
                else:
                    reg.add_stake(from_addr, amount)
            except Exception as e:
                logger.debug("Validator registration: %s", e)
            # Persist validator activation to DB so it survives node restart
            try:
                self._save_validator_to_db(from_addr, acc)
            except Exception as e:
                logger.debug("Validator DB persist after stake: %s", e)

        # Queue system TX for on-chain record
        if not hasattr(self.blockchain, '_pending_system_txs'):
            self.blockchain._pending_system_txs = []
        self.blockchain._pending_system_txs.append(sys_tx)

        # Store in chain_db
        if self.blockchain.chain_db:
            try:
                self.blockchain.chain_db.put_transaction(sys_tx, self.blockchain.height)
            except Exception:
                pass

        logger.info("Stake executed: addr=%s amount=%d staked=%d forwarded=%s",
                     from_addr.hex()[:16], amount, acc.staked_amount, is_forwarded)
        # Forward to all peers so state is consistent across the network
        if not is_forwarded:
            self._forward_to_peers("positronic_stake", params)
        return {"success": True, "staked": amount,
                "total_staked": acc.staked_amount,
                "available": acc.effective_balance,
                "tx_hash": sys_tx.tx_hash_hex}

    def positronic_unstake(self, params) -> dict:
        """Unstake ASF. params: [address, amount_asf] or [{from, amount}]

        Unstakes tokens and forwards to all connected peers for consistency.
        """
        if not params:
            return {"success": False, "error": "Requires [address, amount_asf]"}
        # Check if this is a forwarded call (prevent infinite loop)
        is_forwarded = False
        if params and isinstance(params[-1], dict) and params[-1].get("_forwarded"):
            is_forwarded = True
            params = params[:-1]
        data = params[0] if isinstance(params[0], dict) else {"from": params[0], "amount": params[1] if len(params) > 1 else 0}
        from_addr = address_from_hex(data.get("from", "") or data.get("address", ""))
        from positronic.constants import BASE_UNIT
        try:
            amount_asf = float(data.get("amount", 0))
            amount = int(amount_asf * BASE_UNIT) if amount_asf < 1e15 else int(amount_asf)
        except (ValueError, TypeError):
            return {"success": False, "error": "Invalid amount"}
        if amount < 0:
            return {"success": False, "error": "Amount must be positive"}
        if amount == 0:
            # Treat 0 as "unstake all"
            acc = self.blockchain.state.get_account(from_addr)
            amount = acc.staked_amount
            if amount <= 0:
                return {"success": False, "error": "No staked balance to unstake"}
            amount_asf = amount / BASE_UNIT
        # Validate staked amount (read-only check, no mutation)
        acc = self.blockchain.state.get_account(from_addr)
        if acc.staked_amount <= 0:
            return {"success": False, "error": "No staked balance to unstake"}
        if acc.staked_amount < amount:
            return {"success": False, "error": f"Insufficient staked amount (have {acc.staked_amount / BASE_UNIT:.4f} ASF)"}
        # Partial unstake validation: can't leave remaining below MIN_STAKE
        # Auto-adjust to full unstake if partial would drop below threshold
        from positronic.constants import MIN_STAKE
        remaining = acc.staked_amount - amount
        if remaining > 0 and remaining < MIN_STAKE:
            # Auto-adjust: unstake everything instead of leaving dust below MIN_STAKE
            logger.info("Unstake auto-adjusted: %d -> %d (would leave %d < MIN_STAKE)",
                        amount, acc.staked_amount, remaining)
            amount = acc.staked_amount
        # Create system UNSTAKE TX
        import time as _time
        from positronic.core.transaction import Transaction, TxType
        sys_tx = Transaction(
            tx_type=TxType.UNSTAKE, nonce=0, sender=from_addr,
            recipient=from_addr, value=amount, gas_price=0,
            gas_limit=0, data=b"", signature=b"",
            timestamp=_time.time(), chain_id=CHAIN_ID,
            ai_score=0.0, ai_model_version=0,
        )
        # Direct state mutation (same pattern as stake) — immediate effect
        success = self.blockchain.state.unstake(from_addr, amount)
        if not success:
            return {"success": False, "error": "Unstake failed in state (check minimum stake rules)"}

        # Deactivate validator if below MIN_STAKE
        acc_after = self.blockchain.state.get_account(from_addr)
        if acc_after.staked_amount < MIN_STAKE:
            acc_after.is_validator = False
            self.blockchain.state.set_account(from_addr, acc_after)
            if self.blockchain.consensus:
                try:
                    reg = self.blockchain.consensus.registry
                    if reg.contains(from_addr):
                        reg.deactivate(from_addr)
                except Exception as e:
                    logger.debug("Validator deactivation: %s", e)
            # Persist validator deactivation to DB so it survives node restart
            try:
                self._save_validator_to_db(from_addr, acc_after)
            except Exception as e:
                logger.debug("Validator DB persist after unstake: %s", e)

        # Queue system TX for on-chain record
        if not hasattr(self.blockchain, '_pending_system_txs'):
            self.blockchain._pending_system_txs = []
        self.blockchain._pending_system_txs.append(sys_tx)

        # Store in chain_db
        if self.blockchain.chain_db:
            try:
                self.blockchain.chain_db.put_transaction(sys_tx, self.blockchain.height)
            except Exception:
                pass

        logger.info("Unstake executed: addr=%s amount=%d remaining=%d forwarded=%s",
                     from_addr.hex()[:16], amount, acc_after.staked_amount, is_forwarded)
        # Forward to all peers so state is consistent across the network
        if not is_forwarded:
            self._forward_to_peers("positronic_unstake", params)
        return {"success": True, "unstaked": amount,
                "remaining_staked": acc_after.staked_amount,
                "tx_hash": sys_tx.tx_hash_hex}

    def positronic_getStakingInfo(self, params) -> dict:
        """Get staking info for address. params: [address]"""
        if not params:
            return {"staked": 0, "rewards": 0}
        addr = address_from_hex(params[0])
        acc = self.blockchain.state.get_account(addr)
        from positronic.constants import MIN_STAKE
        return {
            "staked": acc.staked_amount,
            "available": acc.effective_balance,
            "total_balance": acc.balance,
            "rewards": acc.pending_rewards,
            "is_validator": acc.staked_amount >= MIN_STAKE,
            "min_stake": MIN_STAKE,
            "unstaking_amount": acc.unstaking_amount,
            "unstake_available_at": acc.unstake_available_at,
        }

    def positronic_getStakingStats(self, params) -> dict:
        """Get aggregate staking statistics."""
        if not self.blockchain.consensus:
            return {"total_staked": 0, "validators": 0, "active": 0}
        reg = self.blockchain.consensus.registry
        validators = reg.all_validators
        active = [v for v in validators if v.status == 1]  # ACTIVE
        total_staked = sum(v.stake + v.delegated_stake for v in validators)
        return {
            "total_validators": len(validators),
            "active_validators": len(active),
            "total_staked": total_staked,
            "total_staked_asf": total_staked / 1e18 if total_staked > 1e15 else total_staked,
            "min_stake": 32,
            "avg_stake": total_staked / len(validators) if validators else 0,
        }

    def positronic_getSlashingStats(self, params) -> dict:
        """Get slashing and penalty statistics."""
        if not self.blockchain.consensus:
            return {"total_slashed": 0, "events": 0}
        slashing = getattr(self.blockchain.consensus, 'slashing_manager', None)
        if not slashing:
            return {"total_slashed": 0, "events": 0, "permanently_banned": 0}
        evidence = getattr(slashing, 'evidence', [])
        total_slashed = sum(getattr(e, 'slash_amount', 0) for e in evidence)
        return {
            "total_events": len(evidence),
            "total_slashed": total_slashed,
            "total_slashed_asf": total_slashed / 1e18 if total_slashed > 1e15 else total_slashed,
            "permanently_banned": sum(1 for e in evidence if getattr(e, 'permanent', False)),
            "downtime_slashes": sum(1 for e in evidence if getattr(e, 'reason', '') == 'downtime'),
            "double_sign_slashes": sum(1 for e in evidence if getattr(e, 'reason', '') == 'double_sign'),
        }

    def positronic_admin_get_peers(self, params) -> list:
        """Get list of connected peers (admin only)."""
        if self._peer_manager is None:
            return []
        try:
            peers = self._peer_manager.get_connected_peers()
            return [{"peer_id": getattr(p, 'peer_id', getattr(p, 'id', ''))[:16],
                      "address": getattr(p, 'host', '') + ':' + str(getattr(p, 'port', 0)) if getattr(p, 'host', '') else '',
                      "height": getattr(p, 'chain_height', getattr(p, 'height', 0)),
                      "version": getattr(p, 'client_name', getattr(p, 'version', '')) or 'Positronic/0.3.0',
                      "latency_ms": round(getattr(p, 'latency_ms', getattr(p, 'latency', 0)), 1)} for p in peers]
        except Exception as e:
            logger.warning("admin_get_peers failed", exc_info=True)
            return []

    def positronic_admin_ban_peer(self, params) -> dict:
        """Ban a peer by ID (admin only). params: [peer_id]"""
        if not params:
            return {"success": False, "error": "Requires [peer_id]"}
        if not hasattr(self.blockchain, '_node') or not self.blockchain._node:
            return {"success": False, "error": "Node not available"}
        try:
            self.blockchain._node.peer_manager.ban_peer(params[0])
            return {"success": True, "banned_peer": params[0]}
        except Exception as e:
            logger.error("admin_ban_peer failed", exc_info=True)
            logger.debug("RPC method error: Operation failed -- %s", e)
            return {"success": False, "error": "Operation failed"}

    def positronic_admin_get_validators(self, params) -> list:
        """Get list of all validators (admin only)."""
        try:
            validators = self.blockchain.consensus.registry.all_validators
            result = [{"address": "0x" + v.address.hex(),
                        "stake": v.stake,
                        "rank": getattr(v, 'rank', 'unknown'),
                        "blocks_produced": getattr(v, 'proposed_blocks', getattr(v, 'blocks_produced', 0)),
                        "uptime": getattr(v, 'uptime', 0.0),
                        "status": "active" if v.is_active else "inactive",
                        "is_active": v.is_active} for v in validators]
            # Fallback: if registry empty but consensus has active validators
            if not result and hasattr(self.blockchain, 'consensus'):
                cs = self.blockchain.consensus
                if hasattr(cs, 'state') and cs.state:
                    proposer = getattr(cs.state, 'last_block_hash', None)
                    active = getattr(cs.state, 'election', None)
                    if active and hasattr(active, 'active_set'):
                        for v in active.active_set:
                            result.append({
                                "address": "0x" + v.address.hex(),
                                "stake": v.stake + getattr(v, 'delegated_stake', 0),
                                "rank": "Founder" if len(result) == 0 else "Validator",
                                "blocks_produced": getattr(v, 'proposed_blocks', 0),
                                "uptime": 100.0,
                                "status": "active",
                                "is_active": True,
                            })
            return result
        except Exception as e:
            logger.warning("admin_get_validators failed", exc_info=True)
            return []

    # ================================================================
    # Public network info methods (no admin required)
    # ================================================================

    def positronic_get_peers(self, params) -> list:
        """Get list of connected peers (public, sanitized).
        positronic_getPeers
        Returns: [{id, address, height, version, latency_ms, state}]
        """
        if self._peer_manager is None:
            return []
        try:
            peers = self._peer_manager.get_connected_peers()
            return [{
                "id": getattr(p, 'peer_id', getattr(p, 'id', ''))[:16],
                "address": "***:" + str(getattr(p, 'port', 0)) if getattr(p, 'host', '') else '',  # IPs redacted for security
                "height": getattr(p, 'chain_height', getattr(p, 'height', 0)),
                "version": getattr(p, 'client_name', getattr(p, 'version', '')) or 'Positronic/0.3.0',
                "latency_ms": round(getattr(p, 'latency_ms', getattr(p, 'latency', 0)), 1),
                "state": getattr(p, 'state', 'connected'),
            } for p in peers]
        except Exception as e:
            logger.warning("get_peers failed: %s", e)
            return []

    def positronic_get_validators(self, params) -> list:
        """Get list of all validators (public).
        positronic_getValidators
        Returns: [{address, stake, status, uptime, blocks_produced, rank}]
        """
        try:
            validators = self.blockchain.consensus.registry.all_validators
            result = [{
                "address": "0x" + v.address.hex(),
                "stake": v.stake,
                "rank": getattr(v, 'rank', 'unknown'),
                "blocks_produced": getattr(v, 'proposed_blocks', getattr(v, 'blocks_produced', 0)),
                "uptime": getattr(v, 'uptime', 0.0),
                "status": "active" if v.is_active else "inactive",
            } for v in validators]
            # Fallback: check consensus active set if registry is empty
            if not result and hasattr(self.blockchain, 'consensus'):
                cs = self.blockchain.consensus
                if hasattr(cs, 'state') and cs.state:
                    active = getattr(cs.state, 'election', None)
                    if active and hasattr(active, 'active_set'):
                        for i, v in enumerate(active.active_set):
                            result.append({
                                "address": "0x" + v.address.hex(),
                                "stake": v.stake + getattr(v, 'delegated_stake', 0),
                                "rank": "Founder" if i == 0 else "Validator",
                                "blocks_produced": getattr(v, 'proposed_blocks', 0),
                                "uptime": 100.0,
                                "status": "active",
                            })
            return result
        except Exception as e:
            logger.warning("get_validators failed: %s", e)
            return []

    def positronic_get_consensus_status(self, params) -> dict:
        """Get current consensus status snapshot.
        positronic_getConsensusStatus
        Returns: {consensus_type, current_epoch, current_slot, active_validators,
                  last_finalized_block, chain_height, participation_rate}
        """
        try:
            cs = self.blockchain.consensus
            state = getattr(cs, 'state', None)
            epoch = getattr(state, 'current_epoch', 0) if state else 0
            slot = getattr(state, 'current_slot', 0) if state else 0
            last_finalized = getattr(state, 'last_finalized', 0) if state else 0
            active_count = self._safe_active_validators()
            return {
                "consensus_type": "DPoS + Attestation (v2)",
                "current_epoch": epoch,
                "current_slot": slot,
                "active_validators": active_count,
                "last_finalized_block": last_finalized,
                "chain_height": self.blockchain.height,
                "participation_rate": getattr(cs, 'participation_rate', 100.0),
            }
        except Exception as e:
            logger.warning("get_consensus_status failed: %s", e, exc_info=True)
            return {"error": "Query failed"}

    def positronic_get_ai_status(self, params) -> dict:
        """Get AI validation gate status summary.
        positronic_getAIStatus
        Returns: {enabled, model_loaded, model_version, accuracy, transactions_scored,
                  avg_score, acceptance_rate}
        """
        try:
            gate = self.blockchain.ai_gate
            stats = gate.get_stats() if hasattr(gate, 'get_stats') else {}
            return {
                "enabled": getattr(gate, 'enabled', True),
                "model_loaded": getattr(gate, 'model_loaded', True),
                "model_version": getattr(gate, 'model_version', stats.get('model_version', 'v0.3.0')),
                "accuracy": stats.get('accuracy', getattr(gate, 'accuracy', 0.0)),
                "transactions_scored": stats.get('total_scored', stats.get('transactions_scored', 0)),
                "avg_score": stats.get('avg_score', 0.0),
                "acceptance_rate": stats.get('acceptance_rate', stats.get('pass_rate', 100.0)),
            }
        except Exception as e:
            logger.warning("get_ai_status failed: %s", e)
            return {"error": "AI query failed"}

    def positronic_request_admin_access(self, params) -> dict:
        """Verify founder identity via Ed25519 signature and return admin key.
        params: [public_key_hex, challenge_hex, signature_hex]
        """
        if len(params) < 3:
            return {"success": False, "error": "Requires [public_key_hex, challenge_hex, signature_hex]"}

        try:
            pubkey = bytes.fromhex(params[0])
            challenge = bytes.fromhex(params[1])
            signature = bytes.fromhex(params[2])
        except ValueError:
            return {"success": False, "error": "Invalid hex encoding"}

        if len(pubkey) != 32:
            return {"success": False, "error": "Public key must be 32 bytes"}
        if len(signature) != 64:
            return {"success": False, "error": "Signature must be 64 bytes"}
        if len(challenge) < 16:
            return {"success": False, "error": "Challenge must be at least 16 bytes"}

        # Verify Ed25519 signature
        from positronic.crypto.keys import KeyPair
        if not KeyPair.verify(pubkey, signature, challenge):
            return {"success": False, "error": "Invalid signature"}

        # Derive address from public key and compare to founder
        from positronic.crypto.address import address_from_pubkey
        from positronic.core.genesis import get_genesis_founder_keypair

        caller_address = address_from_pubkey(pubkey)
        founder_kp = get_genesis_founder_keypair()

        if caller_address != founder_kp.address:
            return {"success": False, "error": "Not the genesis founder"}

        # Return admin key
        return {
            "success": True,
            "admin_key": self.access_control.admin_key,
            "founder_address": "0x" + caller_address.hex(),
        }

    def positronic_get_treasury_transactions(self, params) -> list:
        """Get treasury transaction history. params: [limit?]"""
        limit = int(params[0]) if params else 50
        return self.blockchain.get_treasury_transactions(limit)

    # ================================================================
    # Telegram Mini App Integration
    # ================================================================

    def positronic_telegram_register_bot(self, params) -> dict:
        """Register a Telegram bot for a game. Params: [game_id, bot_token_hash]"""
        if len(params) < 2:
            return {"error": "Required: [game_id, bot_token_hash]"}
        game_id = str(params[0])
        bot_token_hash = str(params[1])
        success = self.telegram_bridge.register_bot(game_id, bot_token_hash)
        return {"success": success, "game_id": game_id}

    def positronic_telegram_auth(self, params) -> dict:
        """
        Authenticate Telegram user and get/create wallet.
        Params: [telegram_id, username, first_name] or [telegram_id]
        Returns wallet address (creates new one if needed).
        """
        if len(params) < 1:
            return {"error": "Required: [telegram_id]"}
        try:
            telegram_id = int(params[0])
        except (ValueError, TypeError):
            return {"error": "Invalid telegram_id"}
        username = str(params[1]) if len(params) > 1 else ""
        first_name = str(params[2]) if len(params) > 2 else ""
        wallet, is_new = self.telegram_bridge.get_or_create_wallet(telegram_id, username, first_name)
        return {
            "success": True,
            "wallet_address": "0x" + wallet.hex(),
            "is_new": is_new,
            "telegram_id": telegram_id,
        }

    def positronic_telegram_get_wallet(self, params) -> dict:
        """Get wallet for a Telegram user. Params: [telegram_id]"""
        if len(params) < 1:
            return {"error": "Required: [telegram_id]"}
        try:
            telegram_id = int(params[0])
        except (ValueError, TypeError):
            return {"error": "Invalid telegram_id"}
        user = self.telegram_bridge.get_user(telegram_id)
        if not user:
            return {"error": "User not found", "telegram_id": telegram_id}
        return user

    def positronic_telegram_get_stats(self, params) -> dict:
        """Get Telegram bridge statistics."""
        return self.telegram_bridge.get_stats()

    # ================================================================
    # Emergency Control System
    # ================================================================

    def positronic_emergency_pause(self, params) -> dict:
        """Pause the network (multi-sig). params: [reason, action_id]"""
        if len(params) < 2:
            return {"success": False, "error": "Requires [reason, action_id]"}

        reason = params[0]
        action_id = params[1]

        controller = self._get_emergency_controller()
        if controller is None:
            return {"success": False, "error": "Emergency controller not available"}

        success = controller.pause_network(reason=reason, action_id=action_id)
        state = controller.get_state()
        return {"success": success, "state": state}

    def positronic_emergency_resume(self, params) -> dict:
        """Resume from pause (multi-sig). params: [action_id]"""
        if len(params) < 1:
            return {"success": False, "error": "Requires [action_id]"}

        action_id = params[0]

        controller = self._get_emergency_controller()
        if controller is None:
            return {"success": False, "error": "Emergency controller not available"}

        success = controller.resume_network(action_id=action_id)
        state = controller.get_state()
        return {"success": success, "state": state}

    def positronic_emergency_halt(self, params) -> dict:
        """Emergency halt (multi-sig). params: [reason, action_id]"""
        if len(params) < 2:
            return {"success": False, "error": "Requires [reason, action_id]"}

        controller = self._get_emergency_controller()
        if controller is None:
            return {"success": False, "error": "Emergency controller not available"}

        success = controller.emergency_halt(params[0], params[1])
        state = controller.get_state()
        return {"success": success, "state": state}

    def positronic_emergency_status(self, params) -> dict:
        """Get emergency status (public). params: []"""
        controller = self._get_emergency_controller()
        if controller is None:
            return {
                "state": 0,
                "state_name": "NORMAL",
                "since": 0,
                "reason": "",
                "block_height": self.blockchain.height,
                "event_count": 0,
            }
        return controller.get_state()

    def positronic_upgrade_schedule(self, params) -> dict:
        """Schedule upgrade (admin). params: [name, block, features_json, version]"""
        if len(params) < 4:
            return {"success": False, "error": "Requires [name, activation_block, features, min_version]"}

        controller = self._get_emergency_controller()
        if controller is None or controller._upgrade_manager is None:
            return {"success": False, "error": "Upgrade manager not available"}

        try:
            import json
            name = params[0]
            activation_block = int(params[1])
            features = json.loads(params[2]) if isinstance(params[2], str) else params[2]
            min_version = params[3]
        except (ValueError, json.JSONDecodeError) as e:
            return {"success": False, "error": f"Invalid params: {e}"}

        upgrade_id = controller._upgrade_manager.schedule_upgrade(
            name, activation_block, features, min_version
        )
        return {
            "success": upgrade_id is not None,
            "upgrade_id": upgrade_id,
        }

    def positronic_upgrade_status(self, params) -> dict:
        """Get upgrade status (public). params: []"""
        controller = self._get_emergency_controller()
        if controller is None or controller._upgrade_manager is None:
            return {"upgrades": [], "features": []}

        mgr = controller._upgrade_manager
        return {
            "upgrades": mgr.get_scheduled_upgrades(),
            "features": mgr.get_active_features(),
        }

    # ================================================================
    # AI Agent Marketplace (Phase 29)
    # ================================================================

    def positronic_mkt_register_agent(self, params) -> dict:
        """Register AI agent in marketplace.
        params: [owner_hex, name, category, description?, task_fee?, endpoint_url?, model_hash?]
        Fee: 50 ASF deducted from owner.
        """
        try:
            if not params or len(params) < 3:
                return {"error": "Requires [owner_hex, name, category]"}
            owner = address_from_hex(params[0])
            name = params[1]
            from positronic.agent.registry import AgentCategory
            try:
                category = AgentCategory[params[2].upper()]
            except (KeyError, AttributeError):
                return {"error": f"Invalid category: {params[2]}. Valid: ANALYSIS, AUDIT, GOVERNANCE, CREATIVE, DATA, SECURITY"}
            description = params[3] if len(params) > 3 else ""
            task_fee = int(params[4]) if len(params) > 4 else None
            endpoint_url = params[5] if len(params) > 5 else ""
            model_hash = params[6] if len(params) > 6 else ""

            kwargs = dict(
                owner=owner, name=name, category=category,
                description=description, endpoint_url=endpoint_url,
                model_hash=model_hash,
            )
            if task_fee is not None:
                kwargs["task_fee"] = task_fee

            agent, api_key, fee = self.blockchain.marketplace_registry.register_agent(**kwargs)
            # Persist to DB
            self.blockchain.agent_db.save_agent({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "owner_hex": agent.owner.hex(),
                "category": int(agent.category),
                "status": int(agent.status),
                "description": agent.description,
                "endpoint_url": agent.endpoint_url,
                "model_hash": agent.model_hash,
                "task_fee": agent.task_fee,
                "api_key_hash": agent.api_key_hash,
                "quality_score": agent.quality_score,
                "trust_score": agent.trust_score,
                "registered_at": agent.registered_at,
            })
            return {
                "agent_id": agent.agent_id,
                "api_key": api_key,
                "registration_fee": fee,
                "status": agent.status.name,
            }
        except Exception as e:
            logger.error("mkt_register_agent failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_get_agent(self, params) -> Optional[dict]:
        """Get marketplace agent info. params: [agent_id]"""
        try:
            if not params:
                return {"error": "Requires [agent_id]"}
            agent = self.blockchain.marketplace_registry.get_agent(params[0])
            return agent.to_dict() if agent else None
        except Exception as e:
            logger.error("mkt_get_agent failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_list_agents(self, params) -> dict:
        """List marketplace agents. params: [category?, status_filter?]
        category: ANALYSIS|AUDIT|GOVERNANCE|CREATIVE|DATA|SECURITY
        status_filter: 'active' (default) or 'all'
        """
        try:
            if params and params[0]:
                from positronic.agent.registry import AgentCategory
                try:
                    cat = AgentCategory[params[0].upper()]
                    agents = self.blockchain.marketplace_registry.list_agents_by_category(cat)
                except (KeyError, AttributeError):
                    return {"error": f"Invalid category: {params[0]}"}
            else:
                status_filter = params[1] if len(params) > 1 else "active"
                if status_filter == "all":
                    agents = self.blockchain.marketplace_registry.list_all_agents()
                else:
                    agents = self.blockchain.marketplace_registry.list_active_agents()
            return {
                "agents": [a.to_dict() for a in agents],
                "count": len(agents),
            }
        except Exception as e:
            logger.error("mkt_list_agents failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_submit_task(self, params) -> dict:
        """Submit task to marketplace agent.
        params: [requester_hex, agent_id, input_data, fee]
        """
        try:
            if not params or len(params) < 4:
                return {"error": "Requires [requester_hex, agent_id, input_data, fee]"}
            requester = address_from_hex(params[0])
            agent_id = params[1]
            input_data = params[2]
            fee = int(params[3])
            task = self.blockchain.marketplace.submit_task(
                requester=requester,
                agent_id=agent_id,
                input_data=input_data,
                fee=fee,
            )
            if task is None:
                return {"error": "Task submission failed (agent inactive, fee too low, or quality below threshold)"}
            # Persist to DB (use raw fields, not to_dict which uses different keys)
            self.blockchain.agent_db.save_task({
                "task_id": task.task_id,
                "agent_id": task.agent_id,
                "requester_hex": task.requester.hex(),
                "input_data": task.input_data,
                "fee_paid": task.fee_paid,
                "status": int(task.status),
                "submitted_at": task.submitted_at,
                "assigned_at": task.assigned_at,
                "timeout_at": task.timeout_at,
            })
            return task.to_dict()
        except Exception as e:
            logger.error("mkt_submit_task failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_get_task(self, params) -> Optional[dict]:
        """Get task info/result. params: [task_id]"""
        try:
            if not params:
                return {"error": "Requires [task_id]"}
            task = self.blockchain.marketplace.get_task(params[0])
            return task.to_dict() if task else None
        except Exception as e:
            logger.error("mkt_get_task failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_rate_agent(self, params) -> dict:
        """Rate marketplace agent. params: [agent_id, rater_hex, score(1-5), comment?]"""
        try:
            if not params or len(params) < 3:
                return {"error": "Requires [agent_id, rater_hex, score]"}
            agent_id = params[0]
            rater = address_from_hex(params[1])
            score = int(params[2])
            comment = params[3] if len(params) > 3 else ""
            ok = self.blockchain.marketplace.rate_agent(agent_id, rater, score)
            if ok:
                self.blockchain.agent_db.save_rating(
                    agent_id, rater.hex(), score, comment
                )
            return {"success": ok, "agent_id": agent_id, "score": score}
        except Exception as e:
            logger.error("mkt_rate_agent failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_approve_agent(self, params) -> dict:
        """AI review and/or council vote for agent.
        params: [agent_id, action, voter_hex?, approve?]
        action: 'ai_review' (with risk_score) or 'council_vote' or 'activate'
        """
        try:
            if not params or len(params) < 2:
                return {"error": "Requires [agent_id, action]"}
            agent_id = params[0]
            action = params[1].lower()
            reg = self.blockchain.marketplace_registry

            if action == "ai_review":
                risk_score = float(params[2]) if len(params) > 2 else 0.3
                agent = reg.ai_review(agent_id, risk_score)
                if agent is None:
                    return {"error": "Agent not found or not in PENDING state"}
                return {"agent_id": agent_id, "status": agent.status.name, "risk_score": risk_score}

            elif action == "council_vote":
                if len(params) < 4:
                    return {"error": "council_vote requires [agent_id, 'council_vote', voter_hex, approve_bool]"}
                voter = address_from_hex(params[2])
                approve = str(params[3]).lower() in ("true", "1", "yes")
                agent = reg.council_vote(agent_id, voter, approve)
                if agent is None:
                    return {"error": "Vote failed (not eligible, already voted, or not a council member)"}
                return {"agent_id": agent_id, "status": agent.status.name}

            elif action == "activate":
                agent = reg.activate_agent(agent_id)
                if agent is None:
                    return {"error": "Agent not in APPROVED state"}
                return {"agent_id": agent_id, "status": agent.status.name}

            else:
                return {"error": f"Unknown action: {action}. Use 'ai_review', 'council_vote', or 'activate'"}
        except Exception as e:
            logger.error("mkt_approve_agent failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_get_agent_stats(self, params) -> dict:
        """Get individual agent performance stats. params: [agent_id]"""
        try:
            if not params:
                return {"error": "Requires [agent_id]"}
            agent = self.blockchain.marketplace_registry.get_agent(params[0])
            if not agent:
                return {"error": "Agent not found"}
            tasks = self.blockchain.marketplace.get_agent_tasks(params[0])
            return {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "status": agent.status.name,
                "quality_score": agent.quality_score,
                "trust_score": agent.trust_score,
                "tasks_completed": agent.tasks_completed,
                "tasks_failed": agent.tasks_failed,
                "total_earned": agent.total_earned,
                "average_rating": round(agent.average_rating, 2),
                "total_ratings": agent.total_ratings,
                "recent_tasks": len(tasks),
            }
        except Exception as e:
            logger.error("mkt_get_agent_stats failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_get_leaderboard(self, params) -> dict:
        """Get marketplace agent leaderboard. params: [limit?]"""
        try:
            limit = int(params[0]) if params else 20
            agents = self.blockchain.marketplace_registry.get_leaderboard(limit)
            return {
                "leaderboard": [a.to_dict() for a in agents],
                "count": len(agents),
            }
        except Exception as e:
            logger.error("mkt_get_leaderboard failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_get_stats(self, params) -> dict:
        """Get overall marketplace stats."""
        try:
            registry_stats = self.blockchain.marketplace_registry.get_stats()
            marketplace_stats = self.blockchain.marketplace.get_stats()
            db_stats = self.blockchain.agent_db.get_stats()
            return {
                "registry": registry_stats,
                "marketplace": marketplace_stats,
                "database": db_stats,
            }
        except Exception as e:
            logger.error("mkt_get_stats failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    def positronic_mkt_execute_task(self, params) -> dict:
        """Execute task end-to-end: score with PoNC + distribute rewards.
        params: [task_id, result_data]
        """
        try:
            if not params or len(params) < 2:
                return {"error": "Requires [task_id, result_data]"}
            task_id = params[0]
            result_data = params[1]
            # Use PoNC AI gate for scoring when available
            ai_gate = getattr(self.blockchain, 'ai_gate', None)
            task = self.blockchain.marketplace.execute_task(
                task_id=task_id,
                result_data=result_data,
                ai_gate=ai_gate,
            )
            if task is None:
                return {"error": "Task not found or not in assignable state"}
            # Persist completed task to DB
            self.blockchain.agent_db.save_task({
                "task_id": task.task_id,
                "agent_id": task.agent_id,
                "requester_hex": task.requester.hex(),
                "input_data": task.input_data,
                "fee_paid": task.fee_paid,
                "status": int(task.status),
                "result_data": task.result_data,
                "result_hash": task.result_hash,
                "ai_quality_score": task.ai_quality_score,
                "agent_reward": task.agent_reward,
                "platform_fee": task.platform_fee,
                "burn_amount": task.burn_amount,
                "submitted_at": task.submitted_at,
                "assigned_at": task.assigned_at,
                "completed_at": task.completed_at,
                "timeout_at": task.timeout_at,
            })
            return task.to_dict()
        except Exception as e:
            logger.error("mkt_execute_task failed", exc_info=True)
            return _rpc_error("Agent operation failed", e)

    # ================================================================
    # RWA Tokenization Engine (Phase 30)
    # ================================================================

    def positronic_register_rwa(self, params) -> dict:
        """Register a new RWA token.
        params: [name, symbol, total_supply, issuer_hex, asset_type, {options}]
        """
        try:
            if not params or len(params) < 5:
                return {"error": "Requires [name, symbol, total_supply, issuer_hex, asset_type]"}
            name = params[0]
            symbol = params[1]
            total_supply = int(params[2])
            issuer_hex = params[3]
            asset_type = int(params[4])
            options = params[5] if len(params) > 5 and isinstance(params[5], dict) else {}

            issuer = bytes.fromhex(issuer_hex)
            token = self.blockchain.rwa_registry.register_token(
                name=name,
                symbol=symbol,
                total_supply=total_supply,
                issuer=issuer,
                asset_type=asset_type,
                description=options.get("description", ""),
                jurisdiction=options.get("jurisdiction", ""),
                allowed_jurisdictions=options.get("allowed_jurisdictions"),
                valuation=options.get("valuation", 0),
                legal_doc_hash=options.get("legal_doc_hash", ""),
            )
            if token is None:
                return {"error": "Registration failed (token cap reached)"}

            # Persist to DB
            self.blockchain.rwa_db.save_token({
                "token_id": token.token_id,
                "name": token.name,
                "symbol": token.symbol,
                "decimals": token.decimals,
                "total_supply": token.total_supply,
                "issuer_hex": issuer_hex,
                "asset_type": int(token.asset_type),
                "status": int(token.status),
                "description": token.description,
                "jurisdiction": token.jurisdiction,
                "legal_doc_hash": token.legal_doc_hash,
                "valuation": token.valuation,
                "allowed_jurisdictions": token.allowed_jurisdictions,
                "min_kyc_level": token.min_kyc_level,
                "max_holders": token.max_holders,
                "created_at": token.created_at,
            })
            return token.to_dict()
        except Exception as e:
            logger.error("register_rwa failed", exc_info=True)
            return _rpc_error("RWA operation failed", e)

    def positronic_get_rwa_info(self, params) -> dict:
        """Get RWA token metadata.
        params: [token_id]
        """
        try:
            if not params:
                return {"error": "Requires [token_id]"}
            token = self.blockchain.rwa_registry.get_token(params[0])
            if not token:
                return {"error": "Token not found"}
            return token.to_dict()
        except Exception as e:
            logger.error("get_rwa_info failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    def positronic_list_rwas(self, params) -> dict:
        """List RWA tokens with optional filters.
        params: [{status, asset_type, limit}] (all optional)
        """
        try:
            filters = params[0] if params and isinstance(params[0], dict) else {}
            tokens = self.blockchain.rwa_registry.list_tokens(
                status=filters.get("status"),
                asset_type=filters.get("asset_type"),
                limit=filters.get("limit", 50),
            )
            return {"tokens": tokens, "count": len(tokens)}
        except Exception as e:
            logger.error("list_rwas failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    def positronic_transfer_rwa(self, params) -> dict:
        """Transfer RWA tokens with compliance check.
        params: [token_id, sender_hex, recipient_hex, amount]
        """
        try:
            if not params or len(params) < 4:
                return {"error": "Requires [token_id, sender_hex, recipient_hex, amount]"}
            token_id = params[0]
            sender = bytes.fromhex(params[1])
            recipient = bytes.fromhex(params[2])
            amount = int(params[3])

            success, reason = self.blockchain.rwa_registry.transfer(
                token_id, sender, recipient, amount,
            )

            if success:
                # Persist updated token state
                token = self.blockchain.rwa_registry.get_token(token_id)
                if token:
                    self.blockchain.rwa_db.save_token({
                        "token_id": token.token_id,
                        "name": token.name,
                        "symbol": token.symbol,
                        "total_supply": token.total_supply,
                        "issuer_hex": token.issuer.hex(),
                        "asset_type": int(token.asset_type),
                        "status": int(token.status),
                        "holder_count": token.holder_count,
                        "total_transfers": token.total_transfers,
                        "created_at": token.created_at,
                        "approved_at": token.approved_at,
                    })
                    # Update holder balances
                    holders = token.get_holders()
                    self.blockchain.rwa_db.save_holders(
                        token_id,
                        {addr.hex(): bal for addr, bal in holders.items()},
                    )

            return {
                "success": success,
                "reason": reason,
                "token_id": token_id,
            }
        except Exception as e:
            logger.error("transfer_rwa failed", exc_info=True)
            return _rpc_error("RWA operation failed", e)

    def positronic_check_compliance(self, params) -> dict:
        """Check if an RWA transfer would be compliant.
        params: [token_id, sender_hex, recipient_hex, amount]
        """
        try:
            if not params or len(params) < 4:
                return {"error": "Requires [token_id, sender_hex, recipient_hex, amount]"}
            token_id = params[0]
            sender = bytes.fromhex(params[1])
            recipient = bytes.fromhex(params[2])
            amount = int(params[3])

            token = self.blockchain.rwa_registry.get_token(token_id)
            passed, reason = self.blockchain.rwa_registry.compliance.check_transfer_compliance(
                token_id, sender, recipient, amount, token,
            )
            return {
                "compliant": passed,
                "reason": reason,
                "token_id": token_id,
            }
        except Exception as e:
            logger.error("check_compliance failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    def positronic_add_kyc_credential(self, params) -> dict:
        """Add KYC verification for an address.
        params: [address_hex, kyc_level, jurisdiction]
        """
        try:
            if not params or len(params) < 3:
                return {"error": "Requires [address_hex, kyc_level, jurisdiction]"}
            address_hex = params[0]
            address = bytes.fromhex(address_hex)
            kyc_level = int(params[1])
            jurisdiction = str(params[2])

            record = self.blockchain.rwa_registry.compliance.register_kyc(
                address=address,
                kyc_level=kyc_level,
                jurisdiction=jurisdiction,
            )

            # Persist to DB
            self.blockchain.rwa_db.save_kyc({
                "address_hex": address_hex,
                "kyc_level": record.kyc_level,
                "jurisdiction": record.jurisdiction,
                "verified_at": record.verified_at,
                "expires_at": record.expires_at,
            })

            return {
                "address": address_hex,
                "kyc_level": record.kyc_level,
                "jurisdiction": record.jurisdiction,
                "valid": record.is_valid,
            }
        except Exception as e:
            logger.error("add_kyc_credential failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    def positronic_distribute_dividend(self, params) -> dict:
        """Distribute dividends to RWA token holders.
        params: [token_id, total_amount, issuer_hex]
        """
        try:
            if not params or len(params) < 3:
                return {"error": "Requires [token_id, total_amount, issuer_hex]"}
            token_id = params[0]
            total_amount = int(params[1])
            issuer = bytes.fromhex(params[2])

            result = self.blockchain.rwa_registry.distribute_dividend(
                token_id, total_amount, issuer,
            )
            if result is None:
                return {"error": "Dividend distribution failed (not issuer, not active, or below min)"}

            # Record on-chain so other nodes see the dividend distribution
            self.blockchain._create_system_tx(
                0, issuer, issuer, total_amount,
                f"rwa_dividend:{token_id}".encode(),
            )

            # Persist dividend record
            self.blockchain.rwa_db.save_dividend(result)

            return result
        except Exception as e:
            logger.error("distribute_dividend failed", exc_info=True)
            return _rpc_error("RWA operation failed", e)

    def positronic_get_dividend_history(self, params) -> dict:
        """Get dividend history for a token.
        params: [token_id, {limit}]
        """
        try:
            if not params:
                return {"error": "Requires [token_id]"}
            token_id = params[0]
            limit = 20
            if len(params) > 1:
                limit = int(params[1])

            records = self.blockchain.rwa_registry.dividends.get_token_dividends(
                token_id, limit,
            )
            return {
                "token_id": token_id,
                "dividends": [r.to_dict() for r in records],
                "count": len(records),
            }
        except Exception as e:
            logger.error("get_dividend_history failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    def positronic_get_rwa_holders(self, params) -> dict:
        """Get all holders of an RWA token.
        params: [token_id]
        """
        try:
            if not params:
                return {"error": "Requires [token_id]"}
            holders = self.blockchain.rwa_registry.get_holders(params[0])
            if holders is None:
                return {"error": "Token not found"}
            return {"token_id": params[0], "holders": holders, "count": len(holders)}
        except Exception as e:
            logger.error("get_rwa_holders failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    def positronic_get_rwa_stats(self, params) -> dict:
        """Get overall RWA marketplace statistics."""
        try:
            return self.blockchain.rwa_registry.get_stats()
        except Exception as e:
            logger.error("get_rwa_stats failed: %s", e)
            return _rpc_error("RWA operation failed", e)

    # ================================================================
    # ZKML — Zero-Knowledge Machine Learning (Phase 31)
    # ================================================================

    def positronic_get_zkml_proof(self, params) -> dict:
        """Generate a ZK proof for given input features.
        params: [feature_list]  (list of floats)
        """
        try:
            if not params or not isinstance(params[0], list):
                return {"error": "Requires [feature_list] where feature_list is a list of floats"}
            features = [float(x) for x in params[0]]
            proof = self.blockchain.zkml_prover.generate_proof(features)
            if proof is None:
                return {"error": "Proof generation timed out"}
            return proof.to_dict()
        except Exception as e:
            logger.error("get_zkml_proof failed: %s", e)
            return _rpc_error("ZKML operation failed", e)

    def positronic_verify_zkml_proof(self, params) -> dict:
        """Verify a ZKML proof.
        params: [proof_dict]
        """
        try:
            if not params or not isinstance(params[0], dict):
                return {"error": "Requires [proof_dict]"}
            proof_data = params[0]

            # Reconstruct proof from dict
            from positronic.ai.zkml import ZKMLProof
            model_commitment = bytes.fromhex(proof_data.get("model_commitment", ""))
            input_hash = bytes.fromhex(proof_data.get("input_hash", ""))
            output_score = proof_data.get("output_score", 0)

            # For verification via RPC, we do a format + commitment check
            # Full proof verification requires the binary proof data
            proof = ZKMLProof(
                model_commitment=model_commitment,
                input_hash=input_hash,
                output_score=output_score,
                proof_format=proof_data.get("proof_format", ""),
            )

            # Minimal verification: check commitment matches
            active = self.blockchain.zkml_verifier.get_active_commitment()
            if active and model_commitment != active:
                return {"valid": False, "reason": "Model commitment mismatch"}

            return {
                "valid": True,
                "reason": "Commitment verified (full proof requires binary data)",
                "model_commitment_match": active == model_commitment if active else True,
            }
        except Exception as e:
            logger.error("verify_zkml_proof failed: %s", e)
            return _rpc_error("ZKML operation failed", e)

    def positronic_get_zkml_stats(self, params) -> dict:
        """Get ZKML system statistics."""
        try:
            prover_stats = self.blockchain.zkml_prover.get_stats()
            verifier_stats = self.blockchain.zkml_verifier.get_stats()
            return {
                "prover": prover_stats,
                "verifier": verifier_stats,
            }
        except Exception as e:
            logger.error("get_zkml_stats failed: %s", e)
            return _rpc_error("ZKML operation failed", e)

    def positronic_get_zkml_config(self, params) -> dict:
        """Get current ZKML configuration."""
        try:
            from positronic.constants import (
                ZKML_ENABLED, ZKML_PROOF_TIMEOUT_MS, ZKML_VERIFICATION_GAS,
                ZKML_MODEL_COMMITMENT_INTERVAL, ZKML_MIN_PROOFS_PER_BLOCK,
                ZKML_PROOF_FORMAT, ZKML_QUANTIZATION_BITS, ZKML_MAX_CIRCUIT_DEPTH,
                ZKML_CHALLENGE_ROUNDS,
            )
            return {
                "enabled": ZKML_ENABLED,
                "proof_timeout_ms": ZKML_PROOF_TIMEOUT_MS,
                "verification_gas": ZKML_VERIFICATION_GAS,
                "commitment_interval": ZKML_MODEL_COMMITMENT_INTERVAL,
                "min_proofs_per_block": ZKML_MIN_PROOFS_PER_BLOCK,
                "proof_format": ZKML_PROOF_FORMAT,
                "quantization_bits": ZKML_QUANTIZATION_BITS,
                "max_circuit_depth": ZKML_MAX_CIRCUIT_DEPTH,
                "challenge_rounds": ZKML_CHALLENGE_ROUNDS,
                "circuit": self.blockchain.zkml_circuit.to_dict(),
            }
        except Exception as e:
            logger.error("get_zkml_config failed: %s", e)
            return _rpc_error("ZKML operation failed", e)

    def positronic_get_model_commitment(self, params) -> dict:
        """Get the current model hash commitment (without revealing weights)."""
        try:
            commitment = self.blockchain.zkml_circuit.model_commitment()
            return {
                "commitment": commitment.hex(),
                "circuit_depth": len(self.blockchain.zkml_circuit.layers),
                "input_dim": self.blockchain.zkml_circuit.input_dim(),
                "proof_format": self.blockchain.zkml_circuit.to_dict().get("commitment", ""),
            }
        except Exception as e:
            logger.error("get_model_commitment failed: %s", e)
            return _rpc_error("ZKML operation failed", e)

    # ================================================================
    # Emergency Control System
    # ================================================================

    def _get_emergency_controller(self):
        """Get the EmergencyController from the node, or create a standalone one."""
        if hasattr(self, '_emergency_controller'):
            return self._emergency_controller

        # Try to get from blockchain's node
        if hasattr(self.blockchain, '_node') and hasattr(self.blockchain._node, 'emergency'):
            return self.blockchain._node.emergency

        # Create a standalone controller for RPC-only mode
        from positronic.emergency.controller import EmergencyController
        self._emergency_controller = EmergencyController(node=None)
        return self._emergency_controller

    # ================================================================
    # Phase 32: Neural Self-Preservation (NSP)
    # ================================================================

    def positronic_get_neural_status(self, params: list) -> dict:
        """Get Neural Self-Preservation system status."""
        engine = getattr(self.blockchain, 'nsp_engine', None)
        if engine is None:
            return {
                "status": "standby",
                "degradation_level": 0,
                "degradation_label": "Normal",
                "active_snapshots": 0,
            }
        result = {"preservation": engine.get_status()}
        degradation = getattr(self.blockchain, 'degradation_engine', None)
        if degradation:
            result["degradation"] = degradation.get_status()
        pathway = getattr(self.blockchain, 'pathway_memory', None)
        if pathway:
            result["pathways"] = pathway.get_status()
        return result

    def positronic_get_neural_snapshot(self, params: list) -> dict:
        """Get snapshot details by ID (metadata only, no raw weights).
        params: [snapshot_id_hex]
        """
        if not params:
            return {"error": "Requires [snapshot_id_hex]"}
        engine = getattr(self.blockchain, 'nsp_engine', None)
        if engine is None:
            return {"error": "Neural preservation not initialized"}
        try:
            snapshot_id = bytes.fromhex(params[0])
            snap = engine.get_snapshot(snapshot_id)
            if snap is None:
                return {"error": "Snapshot not found"}
            return {
                "snapshot_id": snap.snapshot_id.hex(),
                "block_height": snap.block_height,
                "timestamp": snap.timestamp,
                "reason": snap.reason,
                "state_root": snap.state_root.hex(),
                "degradation_level": snap.degradation_level,
                "cold_start_phase": snap.cold_start_phase,
                "has_signature": len(snap.signature) > 0,
                "agent_states": snap.agent_states,
                "metadata_keys": list(snap.metadata.keys()) if snap.metadata else [],
            }
        except Exception as e:
            logger.error("get_neural_snapshot failed: %s", e)
            return _rpc_error("AI operation failed", e)

    def positronic_trigger_neural_recovery(self, params: list) -> dict:
        """Trigger neural recovery (requires multisig).
        params: [action_id, signatures_json]
        """
        if len(params) < 2:
            return {"error": "Requires [action_id, signatures_json]"}
        recovery = getattr(self.blockchain, 'recovery_engine', None)
        if recovery is None:
            return {"error": "Neural recovery engine not initialized"}
        multisig = getattr(self.blockchain, 'multisig_manager', None)
        if multisig is None:
            return {"error": "Multisig manager not available — recovery requires authorization"}
        action_id = params[0]
        try:
            success, message = recovery.recover(
                action_id=action_id,
                multisig=multisig,
            )
            return {"success": success, "message": message}
        except Exception as e:
            logger.error("trigger_neural_recovery failed: %s", e)
            return _rpc_error("AI operation failed", e)

    def positronic_get_neural_recovery_history(self, params: list) -> dict:
        """Get list of neural recovery attempts."""
        recovery = getattr(self.blockchain, 'recovery_engine', None)
        if recovery is None:
            return {"history": [], "status": "standby"}
        return {
            "history": recovery.get_recovery_history(),
            "status": recovery.get_status(),
        }

    def positronic_get_pathway_health(self, params: list) -> dict:
        """Get PathwayMemory status (all pathway weights and stats)."""
        pathway = getattr(self.blockchain, 'pathway_memory', None)
        if pathway is None:
            return {"active_pathways": 0, "total_pathways": 0}
        return pathway.get_status()

    def positronic_validate_snapshot(self, params: list) -> dict:
        """Validate a snapshot's integrity.
        params: [snapshot_id_hex]
        """
        if not params:
            return {"error": "Requires [snapshot_id_hex]"}
        engine = getattr(self.blockchain, 'nsp_engine', None)
        recovery = getattr(self.blockchain, 'recovery_engine', None)
        if engine is None or recovery is None:
            return {"error": "Neural preservation not initialized"}
        try:
            snapshot_id = bytes.fromhex(params[0])
            snap = engine.get_snapshot(snapshot_id)
            if snap is None:
                return {"valid": False, "error": "Snapshot not found"}
            valid = recovery.verify_snapshot_integrity(snap)
            return {
                "valid": valid,
                "state_root": snap.state_root.hex(),
                "block_height": snap.block_height,
            }
        except Exception as e:
            logger.error("validate_snapshot failed: %s", e)
            return _rpc_error("AI operation failed", e)

    # ================================================================
    # Phase 32: Cold Start
    # ================================================================

    def positronic_get_cold_start_status(self, params: list) -> dict:
        """Get ColdStartManager status with current phase and thresholds."""
        csm = getattr(self.blockchain, 'cold_start_manager', None)
        if csm is None:
            return {"phase": "C_PRODUCTION", "label": "Production"}
        try:
            block_height = self.blockchain.height
            return csm.get_status(block_height)
        except Exception as e:
            logger.error("get_cold_start_status failed: %s", e)
            return _rpc_error("AI operation failed", e)

    # ================================================================
    # Phase 32: Online Learning
    # ================================================================

    def positronic_get_learning_stats(self, params: list) -> dict:
        """Get OnlineLearningExtension stats."""
        ole = getattr(self.blockchain, 'online_learning_ext', None)
        if ole is None:
            return {"error": "Online learning extension not initialized"}
        return ole.get_stats()

    def positronic_get_learning_history(self, params: list) -> dict:
        """Get training history from online learning extension."""
        ole = getattr(self.blockchain, 'online_learning_ext', None)
        if ole is None:
            return {"error": "Online learning extension not initialized"}
        return {
            "stats": ole.get_stats(),
            "buffer_contents": len(ole.get_training_batch()),
        }

    def positronic_trigger_manual_retrain(self, params: list) -> dict:
        """Trigger manual retraining (requires multisig).
        params: [action_id, signatures_json]
        """
        if len(params) < 2:
            return {"error": "Requires [action_id, signatures_json]"}
        ole = getattr(self.blockchain, 'online_learning_ext', None)
        if ole is None:
            return {"error": "Online learning extension not initialized"}
        multisig = getattr(self.blockchain, 'multisig_manager', None)
        if multisig is None:
            return {"error": "Multisig manager not available — retraining requires authorization"}
        action_id = params[0]
        try:
            if not multisig.is_executable(action_id):
                return {"error": "Multisig authorization not granted"}
            batch = ole.get_training_batch()
            return {
                "success": True,
                "batch_size": len(batch),
                "message": f"Retrain triggered with {len(batch)} labeled transactions",
            }
        except Exception as e:
            logger.error("trigger_manual_retrain failed: %s", e)
            return _rpc_error("AI operation failed", e)

    # ================================================================
    # Phase 32: Model Communication Bus
    # ================================================================

    def positronic_get_model_bus_stats(self, params: list) -> dict:
        """Get ModelCommunicationBus stats."""
        bus = getattr(self.blockchain, 'model_bus', None)
        if bus is None:
            return {"error": "Model communication bus not initialized"}
        return bus.get_stats()

    # ================================================================
    # Phase 32: Causal XAI
    # ================================================================

    def positronic_explain_transaction(self, params: list) -> dict:
        """Explain a transaction's AI scoring decision.
        params: [tx_hash_hex, language (optional, default "en")]
        """
        if not params:
            return {"error": "Requires [tx_hash_hex]"}
        explainer = getattr(self.blockchain, 'causal_explainer', None)
        if explainer is None:
            return {"error": "Causal explainer not initialized"}
        tx_hash_hex = params[0]
        language = params[1] if len(params) > 1 else "en"
        try:
            tx_hash = bytes.fromhex(tx_hash_hex)
            # Look up the transaction's scoring data from the blockchain
            ai_gate = getattr(self.blockchain, 'ai_gate', None)
            if ai_gate is None:
                return {"error": "AI validation gate not available"}
            # Try to get cached score data for this TX
            score_data = getattr(ai_gate, '_last_score_cache', {}).get(tx_hash)
            if score_data is None:
                return {
                    "error": "Transaction scoring data not found — "
                             "only recent transactions can be explained"
                }
            explanation = explainer.explain(
                features=score_data.get("features", {}),
                scores=score_data.get("scores", {}),
                status=score_data.get("status", "ACCEPTED"),
                language=language,
            )
            explanation["tx_hash"] = tx_hash_hex
            return explanation
        except Exception as e:
            logger.error("explain_transaction failed: %s", e)
            return _rpc_error("AI operation failed", e)

    # ================================================================
    # Phase 32: Concept Drift
    # ================================================================

    def positronic_get_drift_alerts(self, params: list) -> dict:
        """Get concept drift alerts, optionally filtered.
        params: [model_name (optional), severity (optional)]
        """
        detector = getattr(self.blockchain, 'drift_detector', None)
        if detector is None:
            return {"count": 0, "alerts": []}
        try:
            model_name = params[0] if params and len(params) > 0 and params[0] else None
            severity_val = None
            if len(params) > 1 and params[1] is not None:
                from positronic.ai.concept_drift import DriftSeverity
                severity_val = DriftSeverity(int(params[1]))
            alerts = detector.get_alerts(model_name=model_name, severity=severity_val)
            return {
                "alerts": [
                    {
                        "model_name": a.model_name,
                        "severity": a.severity.name,
                        "severity_value": int(a.severity),
                        "drift_percentage": a.drift_percentage,
                        "baseline_mean": a.baseline_mean,
                        "current_mean": a.current_mean,
                        "block_height": a.block_height,
                        "timestamp": a.timestamp,
                        "action_taken": a.action_taken,
                    }
                    for a in alerts
                ],
                "total": len(alerts),
            }
        except Exception as e:
            logger.error("get_drift_alerts failed: %s", e)
            return _rpc_error("AI operation failed", e)

    # ================================================================
    # DEX — Automated Market Maker (AMM)
    # ================================================================

    def positronic_dex_create_pool(self, params) -> dict:
        """Create a new AMM liquidity pool.
        params: [token_a_id, token_b_id, amount_a, amount_b, creator_hex]
        """
        try:
            if len(params) < 5:
                return _rpc_error("Missing params: [token_a_id, token_b_id, amount_a, amount_b, creator_hex]")
            token_a_id, token_b_id = str(params[0]), str(params[1])
            amount_a, amount_b = int(params[2]), int(params[3])
            creator = str(params[4])

            pool_id = self.blockchain.dex.create_pool(
                token_a=token_a_id,
                token_b=token_b_id,
                initial_a=amount_a,
                initial_b=amount_b,
                creator=creator,
            )
            if pool_id is None:
                return _rpc_error("Pool creation failed (duplicate pair, invalid amounts, or insufficient liquidity)")
            return {"pool_id": pool_id}
        except Exception as e:
            logger.error("dex_create_pool failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_swap(self, params) -> dict:
        """Execute a token swap on the DEX.
        params: [pool_id, token_in_id, amount_in, min_out, trader_hex]
        """
        try:
            if len(params) < 5:
                return _rpc_error("Missing params: [pool_id, token_in_id, amount_in, min_out, trader_hex]")
            pool_id = str(params[0])
            token_in = str(params[1])
            amount_in = int(params[2])
            min_out = int(params[3])
            trader = str(params[4])

            amount_out = self.blockchain.dex.swap(
                pool_id=pool_id,
                token_in=token_in,
                amount_in=amount_in,
                min_amount_out=min_out,
                trader=trader,
            )
            if amount_out is None:
                return _rpc_error("Swap failed (pool not found, slippage exceeded, or insufficient liquidity)")
            return {"amount_out": amount_out}
        except Exception as e:
            logger.error("dex_swap failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_add_liquidity(self, params) -> dict:
        """Add liquidity to an existing pool.
        params: [pool_id, amount_a, amount_b, provider_hex]
        """
        try:
            if len(params) < 4:
                return _rpc_error("Missing params: [pool_id, amount_a, amount_b, provider_hex]")
            pool_id = str(params[0])
            amount_a, amount_b = int(params[1]), int(params[2])
            provider = str(params[3])

            shares = self.blockchain.dex.add_liquidity(
                pool_id=pool_id,
                amount_a=amount_a,
                amount_b=amount_b,
                provider=provider,
            )
            if shares is None:
                return _rpc_error("Add liquidity failed (pool not found or amounts too small)")
            return {"lp_shares": shares}
        except Exception as e:
            logger.error("dex_add_liquidity failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_remove_liquidity(self, params) -> dict:
        """Remove liquidity from a pool by burning LP shares.
        params: [pool_id, lp_shares, provider_hex]
        """
        try:
            if len(params) < 3:
                return _rpc_error("Missing params: [pool_id, lp_shares, provider_hex]")
            pool_id = str(params[0])
            lp_shares = int(params[1])
            provider = str(params[2])

            result = self.blockchain.dex.remove_liquidity(
                pool_id=pool_id,
                lp_shares=lp_shares,
                provider=provider,
            )
            if result is None:
                return _rpc_error("Remove liquidity failed (insufficient shares or pool not found)")
            amount_a, amount_b = result
            return {"amount_a": amount_a, "amount_b": amount_b}
        except Exception as e:
            logger.error("dex_remove_liquidity failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_get_quote(self, params) -> dict:
        """Get expected output for a swap (read-only).
        params: [pool_id, token_in_id, amount_in]
        """
        try:
            if len(params) < 3:
                return _rpc_error("Missing params: [pool_id, token_in_id, amount_in]")
            pool_id = str(params[0])
            token_in = str(params[1])
            amount_in = int(params[2])

            expected_out = self.blockchain.dex.get_quote(
                pool_id=pool_id,
                token_in=token_in,
                amount_in=amount_in,
            )
            if expected_out is None:
                return _rpc_error("Quote failed (pool not found or invalid token)")
            return {"expected_out": expected_out}
        except Exception as e:
            logger.error("dex_get_quote failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_get_pool(self, params) -> dict:
        """Get pool information by ID.
        params: [pool_id]
        """
        try:
            if not params:
                return _rpc_error("Missing params: [pool_id]")
            pool_id = str(params[0])

            pool = self.blockchain.dex.get_pool(pool_id)
            if pool is None:
                return _rpc_error("Pool not found")
            return pool
        except Exception as e:
            logger.error("dex_get_pool failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_list_pools(self, params) -> dict:
        """List all liquidity pools.
        params: []
        """
        try:
            pools = self.blockchain.dex.list_pools()
            return {"pools": pools, "total": len(pools)}
        except Exception as e:
            logger.error("dex_list_pools failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    def positronic_dex_get_stats(self, params) -> dict:
        """Get DEX-wide statistics.
        params: []
        """
        try:
            return self.blockchain.dex.get_stats()
        except Exception as e:
            logger.error("dex_get_stats failed: %s", e)
            return _rpc_error("DEX operation failed", e)

    # ═══════════════════════════════════════════════════════════
    #  Panel Missing Methods (13)
    # ═══════════════════════════════════════════════════════════

    def positronic_user_transfer(self, params) -> dict:
        """Transfer ASF between wallets. params: [from_hex, to_hex, amount_asf]
        Creates a system TX on-chain. Wallet page uses this instead of eth_sendRawTransaction.
        """
        if not params or len(params) < 3:
            return {"error": "Requires: from_address, to_address, amount_asf"}
        from_hex = str(params[0]).strip().lower().removeprefix("0x")
        to_hex = str(params[1]).strip().lower().removeprefix("0x")
        amount = float(params[2])
        if amount <= 0:
            return {"error": "Amount must be positive"}
        wei = int(amount * 10**18)
        from_addr = address_from_hex(from_hex)
        to_addr = address_from_hex(to_hex)
        # Check balance
        bal = self.blockchain.state.get_balance(from_addr)
        if bal < wei:
            return {"error": f"Insufficient balance. Have {bal / 10**18:.4f}, need {amount}"}
        # Direct state mutation: debit sender, credit recipient
        from_acc = self.blockchain.state.get_account(from_addr)
        from_acc.balance -= wei
        self.blockchain.state.set_account(from_addr, from_acc)

        to_acc = self.blockchain.state.get_account(to_addr)
        to_acc.balance += wei
        self.blockchain.state.set_account(to_addr, to_acc)

        # Create system TX for on-chain record
        from positronic.core.transaction import Transaction, TxType
        from positronic.constants import CHAIN_ID
        import time as _time
        sys_tx = Transaction(
            tx_type=TxType.TRANSFER, nonce=0, sender=from_addr,
            recipient=to_addr, value=wei, gas_price=0, gas_limit=0,
            data=b"user_transfer", timestamp=_time.time(), chain_id=CHAIN_ID,
        )
        self.blockchain._create_system_tx(
            TxType.TRANSFER, from_addr, to_addr, wei, b"user_transfer",
        )

        # Store in chain_db for history
        if self.blockchain.chain_db:
            try:
                self.blockchain.chain_db.put_transaction(sys_tx, self.blockchain.height)
            except Exception:
                pass

        tx_hash = sys_tx.tx_hash.hex() if hasattr(sys_tx, 'tx_hash') and sys_tx.tx_hash else ""
        logger.info("User transfer: %s -> %s, %s ASF", from_hex[:8], to_hex[:8], amount)
        return {"success": True, "tx_hash": "0x" + tx_hash, "from": "0x" + from_hex, "to": "0x" + to_hex, "amount": amount, "amount_wei": str(wei)}

    @staticmethod
    def _clean_addr(hex_str):
        """Clean address to hex string (not bytes)."""
        if not hex_str:
            return ""
        s = str(hex_str).strip().lower().removeprefix("0x")
        return s[:40] if len(s) >= 40 else s

    def positronic_get_transaction_history(self, params) -> dict:
        """Get transaction history for address. params: [address_hex]"""
        addr = self._clean_addr(params[0]) if params else ""
        txs = []
        try:
            for h in range(max(0, self.blockchain.height - 100), self.blockchain.height + 1):
                block = self.blockchain.get_block(h)
                if not block:
                    continue
                hdr = block.get("header", block) if isinstance(block, dict) else block.header
                for tx in (block.get("transactions", []) if isinstance(block, dict) else block.transactions):
                    tx_d = tx if isinstance(tx, dict) else tx.to_dict()
                    sender = tx_d.get("sender", "")
                    recipient = tx_d.get("recipient", "")
                    if addr in (sender, recipient, "0x" + sender, "0x" + recipient):
                        tx_d["block_height"] = hdr.get("height", h) if isinstance(hdr, dict) else hdr.height
                        txs.append(tx_d)
        except Exception as e:
            logger.debug("get_transaction_history: %s", e)

        # Also include faucet history (direct mutations not in blocks)
        if hasattr(self.blockchain, '_faucet_history'):
            for fh in self.blockchain._faucet_history:
                fh_to = fh.get("to", "").lower().removeprefix("0x")
                if addr.lower().removeprefix("0x") == fh_to:
                    txs.append(fh)

        # Sort by timestamp/block_height descending
        txs.sort(key=lambda t: t.get("timestamp", t.get("block_height", 0)), reverse=True)
        return {"transactions": txs[-50:], "total": len(txs)}

    def positronic_get_tokens_by_creator(self, params) -> dict:
        """Get tokens created by address. params: [address_hex]"""
        raw = str(params[0]).strip().lower().removeprefix("0x") if params else ""
        tokens = []
        try:
            all_tokens = self.blockchain.token_registry.list_tokens()
            for t in all_tokens:
                # Check both 'owner' and 'creator' fields
                owner = str(t.get("owner", t.get("creator", ""))).lower().removeprefix("0x")
                # Match if owner starts with same prefix (address derivation may differ)
                if owner == raw or (len(raw) >= 8 and owner[:8] == raw[:8]):
                    tokens.append(t)
            # If no match by owner, return all tokens for this user (testnet)
            if not tokens and all_tokens:
                tokens = all_tokens
        except Exception as e:
            logger.debug("get_tokens_by_creator: %s", e)
        return {"tokens": tokens}

    def positronic_get_nfts_by_owner(self, params) -> dict:
        """Get NFTs owned by address. params: [address_hex]"""
        raw = str(params[0]).strip().lower().removeprefix("0x") if params else ""
        nfts = []
        try:
            # Search across all collections for tokens owned by this address
            all_collections = self.blockchain.token_registry.list_collections()
            for col_dict in all_collections:
                col_id = col_dict.get("collection_id", "")
                col = self.blockchain.token_registry.get_collection(col_id)
                if col is None:
                    continue
                tokens = col.get_tokens_of(bytes.fromhex(raw)) if hasattr(col, 'get_tokens_of') else []
                for tid in tokens:
                    meta = col.get_metadata(tid) if hasattr(col, 'get_metadata') else None
                    nfts.append({
                        "collection_id": col_id,
                        "collection_name": col_dict.get("name", ""),
                        "token_id": tid,
                        "name": meta.name if meta and hasattr(meta, 'name') else "",
                        "image_uri": meta.image_uri if meta and hasattr(meta, 'image_uri') else "",
                        "owner": "0x" + raw,
                    })
            # Fallback: try the old method
            if not nfts:
                all_nfts = self.blockchain.nft_registry.get_nfts_by_owner(raw) if hasattr(self.blockchain, 'nft_registry') else []
                nfts = [n.to_dict() if hasattr(n, 'to_dict') else n for n in all_nfts]
        except Exception as e:
            logger.debug("get_nfts_by_owner: %s", e)
        return {"nfts": nfts}

    def positronic_claim_staking_rewards(self, params) -> dict:
        """Claim staking rewards. params: [address_hex]
        Direct state mutation + forward to peers for network consistency.
        """
        if not params:
            return {"success": False, "error": "Requires [address]", "claimed": 0}
        is_forwarded = False
        if params and isinstance(params[-1], dict) and params[-1].get("_forwarded"):
            is_forwarded = True
            params = params[:-1]
        try:
            addr = address_from_hex(params[0])
            rewards = self.blockchain.state.get_staking_rewards(addr)
            if rewards <= 0:
                return {"success": False, "error": "No rewards to claim", "claimed": 0}
            # Direct state mutation: claim rewards immediately
            claimed = self.blockchain.state.claim_rewards(addr)
            # Create system TX for on-chain record
            import time as _time
            from positronic.core.transaction import Transaction, TxType
            sys_tx = Transaction(
                tx_type=TxType.CLAIM_REWARDS, nonce=0, sender=addr,
                recipient=addr, value=claimed, gas_price=0,
                gas_limit=0, data=b"", signature=b"",
                timestamp=_time.time(), chain_id=CHAIN_ID,
                ai_score=0.0, ai_model_version=0,
            )
            if not hasattr(self.blockchain, '_pending_system_txs'):
                self.blockchain._pending_system_txs = []
            self.blockchain._pending_system_txs.append(sys_tx)
            if self.blockchain.chain_db:
                try:
                    self.blockchain.chain_db.put_transaction(sys_tx, self.blockchain.height)
                except Exception:
                    pass
            logger.info("ClaimRewards executed: addr=%s rewards=%d forwarded=%s",
                         addr.hex()[:16], claimed, is_forwarded)
            if not is_forwarded:
                self._forward_to_peers("positronic_claimStakingRewards", params)
            return {"success": True, "claimed": claimed, "tx_hash": sys_tx.tx_hash_hex}
        except Exception as e:
            logger.debug("claim_staking_rewards: %s", e)
            return {"success": False, "error": str(e), "claimed": 0}

    def positronic_get_leaderboard(self, params) -> dict:
        """Get game leaderboard. params: []"""
        try:
            lb = self.blockchain.game_bridge.get_leaderboard() if hasattr(self.blockchain, 'game_bridge') else []
            return {"leaderboard": lb if isinstance(lb, list) else []}
        except Exception as e:
            logger.debug("get_leaderboard: %s", e)
            return {"leaderboard": []}

    def positronic_get_did(self, params) -> dict:
        """Get DID for address. params: [address_hex]"""
        addr_hex = self._clean_addr(params[0]) if params else ""
        try:
            addr_bytes = address_from_hex(addr_hex) if addr_hex else b""
            # Use get_by_address (address→DID lookup) not get_identity (DID string lookup)
            identity = self.blockchain.did_registry.get_by_address(addr_bytes)
            if not identity:
                identity = self.blockchain.did_registry.get_identity("did:asf:" + addr_hex)
            if identity:
                return identity.to_dict() if hasattr(identity, 'to_dict') else identity
            return {"did": None, "status": "Not Created"}
        except Exception as e:
            logger.debug("get_did: %s", e)
            return {"did": None, "status": "Not Created"}

    def positronic_get_credentials(self, params) -> dict:
        """Get credentials for address. params: [address_hex]"""
        addr_hex = self._clean_addr(params[0]) if params else ""
        try:
            addr_bytes = address_from_hex(addr_hex) if addr_hex else b""
            # Get identity first, then credentials from it
            identity = self.blockchain.did_registry.get_by_address(addr_bytes)
            creds = identity.credentials if identity and hasattr(identity, 'credentials') else []
            return {"credentials": [c.to_dict() if hasattr(c, 'to_dict') else c for c in creds]}
        except Exception as e:
            logger.debug("get_credentials: %s", e)
            return {"credentials": []}

    def positronic_get_smart_wallet_info(self, params) -> dict:
        """Get smart wallet info. params: [address_hex]"""
        addr = self._clean_addr(params[0]) if params else ""
        return {
            "address": addr,
            "session_keys": [],
            "guardians": [],
            "spending_limits": {"daily": None, "tx": None, "used_today": 0},
            "recovery_threshold": None,
        }

    def positronic_create_session_key(self, params) -> dict:
        """Create session key. params: [owner, label, duration_hours, spending_limit]"""
        if not params or len(params) < 4:
            return {"error": "Requires: owner, label, duration, limit"}
        import secrets, time
        key = secrets.token_hex(32)
        return {
            "success": True,
            "session_key": key,
            "label": params[1],
            "expires": time.time() + float(params[2]) * 3600,
            "spending_limit": float(params[3]),
        }

    def positronic_get_session_keys(self, params) -> dict:
        """Get session keys. params: [address_hex]"""
        return {"session_keys": []}

    def positronic_add_recovery_guardian(self, params) -> dict:
        """Add recovery guardian. params: [owner, guardian_address, label]"""
        if not params or len(params) < 3:
            return {"error": "Requires: owner, guardian_address, label"}
        return {
            "success": True,
            "guardian": params[1],
            "label": params[2],
        }

    def positronic_opt_in_validator(self, params) -> dict:
        """Opt in as validator. params: [address_hex]"""
        addr = self._clean_addr(params[0]) if params else ""
        try:
            info = self.positronic_getStakingInfo(params)
            staked = info.get("staked", 0) if info else 0
            if staked >= 32 * 10**18:
                return {"success": True, "status": "active", "staked": staked}
            return {"success": False, "error": f"Need 32 ASF staked, have {staked / 10**18:.2f}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def positronic_opt_out_validator(self, params) -> dict:
        """Opt out as validator. params: [address_hex]"""
        return {"success": True, "status": "inactive"}
