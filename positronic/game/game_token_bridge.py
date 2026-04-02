"""
Positronic - Game-to-Token Bridge Orchestration Layer
Allows registered games to create custom PRC-20 tokens and mint PRC-721 NFTs
for in-game items, achievements, and reward distribution.

Flow:
  1. Game registers via positronic_registerGame (existing)
  2. Game creates a custom token via positronic_gameCreateToken
  3. Game mints NFTs for items via positronic_gameMintItem
  4. Game distributes custom tokens via positronic_gameDistributeReward
"""

import time
from typing import Any, Dict, List, Optional

from positronic.game.game_registry import GameRegistry, GameInfo, GameStatus
from positronic.tokens.registry import TokenRegistry
from positronic.tokens.prc20 import PRC20Token
from positronic.tokens.prc721 import PRC721Collection, NFTMetadata
from positronic.utils.logging import get_logger

logger = get_logger("game_token_bridge")

# Limits
MAX_GAME_TOKENS = 3          # Max custom tokens per game
MAX_GAME_COLLECTIONS = 5     # Max NFT collections per game
MINT_COOLDOWN = 1.0           # Min seconds between mints per game


class GameTokenBridge:
    """
    Orchestrates game↔token interactions.

    Each registered game can:
    - Create up to 3 custom PRC-20 tokens (e.g., in-game currency)
    - Create up to 5 PRC-721 NFT collections (e.g., items, skins)
    - Mint NFTs for players (game items as NFTs)
    - Distribute custom tokens as rewards

    All operations require valid API key authentication.
    """

    def __init__(
        self,
        game_registry: GameRegistry,
        token_registry: TokenRegistry,
        game_db=None,
        blockchain=None,
    ):
        self.games = game_registry
        self.tokens = token_registry
        self._db = game_db  # Optional GameDatabase for persistence
        self._blockchain = blockchain  # for recording system TXs on-chain

        # Track game→token/collection mappings
        self._game_tokens: Dict[str, List[str]] = {}        # game_id → [token_id]
        self._game_collections: Dict[str, List[str]] = {}   # game_id → [collection_id]
        self._last_mint: Dict[str, float] = {}               # game_id → timestamp

        # Load mappings from database if available
        if self._db is not None:
            self._load_mappings_from_db()

    def _load_mappings_from_db(self):
        """Load game→token/collection mappings from database."""
        try:
            # Get all games that have tokens/collections
            for game_data in self._db.load_all_games():
                game_id = game_data["game_id"]
                token_ids = self._db.get_game_token_ids(game_id)
                if token_ids:
                    self._game_tokens[game_id] = token_ids
                col_ids = self._db.get_game_collection_ids(game_id)
                if col_ids:
                    self._game_collections[game_id] = col_ids
            logger.info("loaded_mappings_from_db", extra={
                "tokens": sum(len(v) for v in self._game_tokens.values()),
                "collections": sum(len(v) for v in self._game_collections.values()),
            })
        except Exception as e:
            logger.error("load_mappings_failed", exc_info=e)

    def _authenticate(self, api_key: str) -> Optional[GameInfo]:
        """Authenticate game via API key. Returns GameInfo or None."""
        return self.games.authenticate(api_key)

    # ---- Game Token Creation ----

    def create_game_token(
        self,
        api_key: str,
        name: str,
        symbol: str,
        decimals: int = 18,
        initial_supply: int = 0,
    ) -> dict:
        """Create a custom PRC-20 token for a registered game.

        The game's developer address becomes the token owner.
        """
        game = self._authenticate(api_key)
        if not game:
            return {"error": "Invalid API key or game not active"}

        # Check per-game limit
        existing = self._game_tokens.get(game.game_id, [])
        if len(existing) >= MAX_GAME_TOKENS:
            return {"error": f"Game can have max {MAX_GAME_TOKENS} custom tokens"}

        # Prefix symbol with game context for uniqueness
        token = self.tokens.create_token(
            name=name,
            symbol=symbol,
            decimals=decimals,
            total_supply=initial_supply,
            owner=game.developer,
        )
        if token is None:
            return {"error": "Token creation failed (symbol may already exist)"}

        # Record on-chain so other nodes see the game token creation
        if self._blockchain is not None and hasattr(self._blockchain, '_create_system_tx'):
            self._blockchain._create_system_tx(
                9, game.developer, game.developer, initial_supply,
                f"game_token_create:{name}".encode(),
            )

        # Track mapping + persist
        existing.append(token.token_id)
        self._game_tokens[game.game_id] = existing
        if self._db is not None:
            try:
                self._db.save_token_mapping(game.game_id, token.token_id)
            except Exception as e:
                logger.error("save_token_mapping_failed", exc_info=e)

        logger.info(
            "game_token_created",
            extra={
                "game_id": game.game_id,
                "token_id": token.token_id,
                "symbol": symbol,
            },
        )

        return {
            "token_id": token.token_id,
            "name": name,
            "symbol": symbol,
            "decimals": decimals,
            "initial_supply": initial_supply,
            "game_id": game.game_id,
        }

    # ---- Game NFT Collection Creation ----

    def create_game_collection(
        self,
        api_key: str,
        name: str,
        symbol: str,
        max_supply: int = 0,
    ) -> dict:
        """Create a PRC-721 NFT collection for game items."""
        game = self._authenticate(api_key)
        if not game:
            return {"error": "Invalid API key or game not active"}

        existing = self._game_collections.get(game.game_id, [])
        if len(existing) >= MAX_GAME_COLLECTIONS:
            return {"error": f"Game can have max {MAX_GAME_COLLECTIONS} NFT collections"}

        collection = self.tokens.create_collection(
            name=name,
            symbol=symbol,
            owner=game.developer,
            max_supply=max_supply,
        )
        if collection is None:
            return {"error": "Collection creation failed"}

        existing.append(collection.collection_id)
        self._game_collections[game.game_id] = existing
        if self._db is not None:
            try:
                self._db.save_collection_mapping(game.game_id, collection.collection_id)
            except Exception as e:
                logger.error("save_collection_mapping_failed", exc_info=e)

        logger.info(
            "game_collection_created",
            extra={
                "game_id": game.game_id,
                "collection_id": collection.collection_id,
                "symbol": symbol,
            },
        )

        return {
            "collection_id": collection.collection_id,
            "name": name,
            "symbol": symbol,
            "max_supply": max_supply,
            "game_id": game.game_id,
        }

    # ---- Mint Game Item as NFT ----

    def mint_game_item(
        self,
        api_key: str,
        collection_id: str,
        to_hex: str,
        item_name: str = "",
        item_description: str = "",
        item_image: str = "",
        attributes: Optional[dict] = None,
        dynamic: bool = True,
    ) -> dict:
        """Mint a game item as an NFT to a player's address.

        Dynamic=True allows the game to update item metadata later
        (e.g., weapon level-ups, skin changes).
        """
        game = self._authenticate(api_key)
        if not game:
            return {"error": "Invalid API key or game not active"}

        # Verify collection belongs to this game
        game_cols = self._game_collections.get(game.game_id, [])
        if collection_id not in game_cols:
            return {"error": "Collection does not belong to this game"}

        # Rate limit minting
        now = time.time()
        last = self._last_mint.get(game.game_id, 0)
        if now - last < MINT_COOLDOWN:
            return {"error": "Mint cooldown — try again in a moment"}

        collection = self.tokens.get_collection(collection_id)
        if collection is None:
            return {"error": f"Collection {collection_id} not found"}

        try:
            to = bytes.fromhex(to_hex)
        except ValueError:
            return {"error": "Invalid player address"}

        metadata = NFTMetadata(
            name=item_name,
            description=item_description,
            image_uri=item_image,
            attributes=attributes or {},
            dynamic=dynamic,
        )

        token_id = collection.mint(to=to, metadata=metadata)
        if token_id is None:
            return {"error": "Mint failed (max supply reached)"}

        # Record on-chain so other nodes see the game item mint
        if self._blockchain is not None and hasattr(self._blockchain, '_create_system_tx'):
            self._blockchain._create_system_tx(
                11, b'\x00' * 20, to, 0,
                f"game_nft_mint:{collection_id}:{token_id}".encode(),
            )

        self._last_mint[game.game_id] = now

        logger.info(
            "game_item_minted",
            extra={
                "game_id": game.game_id,
                "collection_id": collection_id,
                "token_id": token_id,
                "to": to_hex,
            },
        )

        return {
            "collection_id": collection_id,
            "token_id": token_id,
            "owner": to_hex,
            "item_name": item_name,
            "game_id": game.game_id,
            "total_minted": collection._total_minted,
        }

    # ---- Distribute Custom Token Rewards ----

    def distribute_token_reward(
        self,
        api_key: str,
        token_id: str,
        to_hex: str,
        amount: int,
    ) -> dict:
        """Distribute custom PRC-20 tokens to a player as game reward.

        The token must belong to this game and the game's developer
        must have sufficient balance (or minting rights).
        """
        game = self._authenticate(api_key)
        if not game:
            return {"error": "Invalid API key or game not active"}

        # Verify token belongs to this game
        game_tokens = self._game_tokens.get(game.game_id, [])
        if token_id not in game_tokens:
            return {"error": "Token does not belong to this game"}

        token = self.tokens.get_token(token_id)
        if token is None:
            return {"error": f"Token {token_id} not found"}

        try:
            to = bytes.fromhex(to_hex)
        except ValueError:
            return {"error": "Invalid player address"}

        if amount <= 0:
            return {"error": "Amount must be positive"}

        # Mint new tokens to the player (game owner has minting rights)
        if not token.mint(to, amount):
            return {"error": "Token mint failed"}

        # Record on-chain so other nodes see the game token distribution
        if self._blockchain is not None and hasattr(self._blockchain, '_create_system_tx'):
            self._blockchain._create_system_tx(
                9, b'\x00' * 20, to, amount,
                f"game_token_dist:{token_id}".encode(),
            )

        logger.info(
            "game_token_distributed",
            extra={
                "game_id": game.game_id,
                "token_id": token_id,
                "to": to_hex,
                "amount": amount,
            },
        )

        return {
            "token_id": token_id,
            "symbol": token.symbol,
            "to": to_hex,
            "amount": amount,
            "new_balance": token.balance_of(to),
            "game_id": game.game_id,
        }

    # ---- Queries ----

    def get_game_tokens(self, game_id: str) -> List[dict]:
        """Get all custom tokens created by a game."""
        token_ids = self._game_tokens.get(game_id, [])
        result = []
        for tid in token_ids:
            token = self.tokens.get_token(tid)
            if token:
                result.append(token.to_dict())
        return result

    def get_game_collections(self, game_id: str) -> List[dict]:
        """Get all NFT collections created by a game."""
        col_ids = self._game_collections.get(game_id, [])
        result = []
        for cid in col_ids:
            col = self.tokens.get_collection(cid)
            if col:
                result.append(col.to_dict())
        return result

    def get_stats(self) -> dict:
        """Get game token bridge statistics."""
        total_tokens = sum(len(v) for v in self._game_tokens.values())
        total_collections = sum(len(v) for v in self._game_collections.values())
        return {
            "games_with_tokens": len(self._game_tokens),
            "games_with_collections": len(self._game_collections),
            "total_game_tokens": total_tokens,
            "total_game_collections": total_collections,
        }
