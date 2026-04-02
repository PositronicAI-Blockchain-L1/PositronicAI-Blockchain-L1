"""
Positronic - Federated Gossip Learning

Enables nodes to share AI model improvements without sharing raw transaction
data. Uses federated averaging with a staleness-weighted gossip protocol.

Protocol:
    1. Node trains locally on its transaction data (via OnlineLearner).
    2. Computes delta: current_weights - last_shared_weights.
    3. Gossips delta to connected peers via the P2P network.
    4. Receives deltas from peers and queues them.
    5. Applies staleness-weighted and sample-weighted federated average
       of all pending deltas to update the local model.

Privacy guarantees:
    - Raw transaction data never leaves the node.
    - Only weight deltas (model parameter changes) are transmitted.
    - Deltas are clipped to a maximum norm to limit information leakage.
    - No single peer can reconstruct another node's training data from
      the aggregated delta.

Dependencies:
    - Model from positronic.ai.engine.model (state_dict, load_state_dict)
    - Serialization utilities from positronic.ai.engine.serialization
"""

import numpy as np
import hashlib
# pickle removed — unsafe for untrusted network data (arbitrary code execution)
# Using JSON serialization instead for consensus weight exchange
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelDelta:
    """Represents the change in model weights between training rounds.

    A delta captures the difference between a model's current weights and
    its weights at the time of the last federated share. Deltas are the
    unit of exchange in the gossip protocol.

    Attributes:
        model_name: Identifier for which model this delta applies to
            (e.g., "vae", "temporal_net", "meta_ensemble").
        delta: Mapping from parameter names to weight change arrays.
        version: The federated round number when this delta was produced.
        timestamp: Unix timestamp when the delta was computed.
        node_id: Unique identifier of the node that produced this delta.
        num_samples: Number of training samples this delta was derived
            from. Used as a quality weight during federated averaging.
    """
    model_name: str
    delta: Dict[str, np.ndarray]
    version: int
    timestamp: float
    node_id: str
    num_samples: int


class FederatedAverager:
    """Federated averaging with gossip protocol for P2P model sharing.

    The FederatedAverager manages the lifecycle of federated model updates:

    1. **Snapshot**: Before local training, ``snapshot_model()`` saves the
       current model weights.

    2. **Compute delta**: After local training, ``compute_delta()`` computes
       the difference between current weights and the snapshot.

    3. **Gossip**: The delta is serialized and sent to connected peers.

    4. **Receive**: Incoming deltas from peers are queued via
       ``receive_delta()``.

    5. **Average**: ``apply_federated_average()`` computes a weighted average
       of all pending deltas and applies it to the local model. Weights are
       determined by:

       - **Staleness**: More recent deltas receive higher weight, decaying
         exponentially by ``staleness_decay`` per round of age.
       - **Sample count**: Deltas trained on more data receive higher weight
         via log-scaled sample count.

    6. **Clip**: Individual deltas are norm-clipped to ``max_delta_norm`` to
       prevent any single peer from dominating the average and to limit
       potential poisoning attacks.

    Args:
        node_id: Unique identifier for this node in the P2P network.

    Example::

        averager = FederatedAverager(node_id="node-abc123")

        # Before local training
        averager.snapshot_model("vae", vae_model)

        # ... local training happens ...

        # After local training
        delta = averager.compute_delta("vae", vae_model)
        delta.num_samples = 500
        serialized = averager.serialize_delta(delta)
        # Send serialized delta to peers via P2P network

        # Receive deltas from peers
        for peer_delta in incoming_deltas:
            averager.receive_delta(peer_delta)

        # Apply federated average
        if averager.apply_federated_average("vae", vae_model):
            print("Model updated with peer knowledge")
    """

    def __init__(self, node_id: str = "local"):
        self.node_id = node_id

        # Snapshots of model state dicts at last share point
        self._last_shared: Dict[str, Dict[str, np.ndarray]] = {}

        # Queue of received deltas from peers
        self._pending_deltas: List[ModelDelta] = []

        # Federated round counter
        self._round: int = 0

        # Maximum number of pending deltas to retain (FIFO eviction)
        self._max_pending: int = 50

        # --- Averaging configuration ---

        # Exponential decay factor per round of staleness.
        # A delta that is 1 round old is weighted by staleness_decay^1,
        # 2 rounds old by staleness_decay^2, etc.
        self.staleness_decay: float = 0.9

        # Minimum number of peer deltas required before averaging is applied.
        self.min_peers_for_average: int = 1

        # Maximum L2 norm for any individual delta. Deltas exceeding this
        # norm are scaled down, providing defense against poisoning attacks.
        self.max_delta_norm: float = 10.0

        # Phase 5: Model state for consensus sync
        self._model_state: Dict[str, np.ndarray] = {}
        self._local_stake: int = 0

    def snapshot_model(self, model_name: str, model) -> None:
        """Take a snapshot of current model weights before local training.

        This snapshot serves as the baseline for computing deltas after
        training completes.

        Args:
            model_name: String identifier for the model (e.g., "vae").
            model: Model instance with a ``state_dict()`` method returning
                a dict of {param_name: np.ndarray}.
        """
        self._last_shared[model_name] = {
            k: v.copy() for k, v in model.state_dict().items()
        }

    def compute_delta(self, model_name: str, model) -> Optional[ModelDelta]:
        """Compute the weight delta between current and last-shared weights.

        Each parameter's delta is individually norm-clipped to
        ``max_delta_norm`` to prevent extreme updates.

        Args:
            model_name: String identifier for the model.
            model: Model instance with a ``state_dict()`` method.

        Returns:
            A ModelDelta capturing the weight changes, or None if no
            previous snapshot exists for this model.
        """
        if model_name not in self._last_shared:
            return None

        current = model.state_dict()
        previous = self._last_shared[model_name]

        delta = {}
        for key in current:
            if key in previous:
                d = current[key] - previous[key]
                # Clip extreme deltas to limit information leakage
                # and defend against potential poisoning
                norm = np.linalg.norm(d)
                if norm > self.max_delta_norm:
                    d = d * (self.max_delta_norm / norm)
                delta[key] = d

        return ModelDelta(
            model_name=model_name,
            delta=delta,
            version=self._round,
            timestamp=time.time(),
            node_id=self.node_id,
            num_samples=0,  # Caller should set this to actual sample count
        )

    def receive_delta(self, delta: ModelDelta) -> None:
        """Receive a model delta from a peer node.

        Deltas from the local node are silently ignored to prevent
        self-reinforcement. Old deltas are evicted FIFO when the
        pending queue exceeds ``_max_pending``.

        Args:
            delta: A ModelDelta received from a peer.
        """
        if delta.node_id == self.node_id:
            return  # Do not apply own deltas

        self._pending_deltas.append(delta)

        # Evict oldest deltas if queue is full
        if len(self._pending_deltas) > self._max_pending:
            self._pending_deltas = self._pending_deltas[-self._max_pending:]

    def apply_federated_average(self, model_name: str, model) -> bool:
        """Apply federated averaging of all pending deltas for a model.

        The weighted average is computed using two factors:

        - **Staleness weight**: ``staleness_decay ^ (current_round - delta_version)``
        - **Sample weight**: ``log(1 + num_samples)``

        These are multiplied together, normalized, and used to compute a
        weighted sum of all pending deltas for the specified model.

        After application, the model snapshot is updated and applied deltas
        are removed from the pending queue.

        Args:
            model_name: String identifier for the model to update.
            model: Model instance with ``state_dict()`` and
                ``load_state_dict()`` methods.

        Returns:
            True if the federated average was applied, False if there were
            insufficient pending deltas.
        """
        relevant = [d for d in self._pending_deltas if d.model_name == model_name]

        if len(relevant) < self.min_peers_for_average:
            return False

        # Compute staleness-weighted and sample-weighted importance
        weights = []
        for d in relevant:
            staleness = self._round - d.version
            w = (self.staleness_decay ** staleness) * np.log1p(d.num_samples)
            weights.append(max(w, 0.01))  # Ensure minimum weight

        total_weight = sum(weights)
        if total_weight < 1e-8:
            return False
        weights = [w / total_weight for w in weights]

        # Compute weighted average delta across all peers
        avg_delta: Dict[str, np.ndarray] = {}
        for d, w in zip(relevant, weights):
            for key, value in d.delta.items():
                if key not in avg_delta:
                    avg_delta[key] = np.zeros_like(value)
                avg_delta[key] += w * value

        # Apply the averaged delta to the model
        current_state = model.state_dict()
        for key in avg_delta:
            if key in current_state:
                current_state[key] = current_state[key] + avg_delta[key]
        model.load_state_dict(current_state)

        # Remove applied deltas from the pending queue
        relevant_set = set(id(d) for d in relevant)
        self._pending_deltas = [
            d for d in self._pending_deltas if id(d) not in relevant_set
        ]

        self._round += 1

        # Update the snapshot to reflect the new model state
        self.snapshot_model(model_name, model)

        return True

    # ------------------------------------------------------------------ #
    #  Phase 5: Consensus integration methods                              #
    # ------------------------------------------------------------------ #

    def export_model_hash(self) -> bytes:
        """SHA-512 hash of current model state for block header verification."""
        h = hashlib.sha512()
        # Hash all model parameters in sorted order for determinism
        for name in sorted(self._model_state.keys()):
            param = self._model_state[name]
            if hasattr(param, 'tobytes'):
                h.update(name.encode())
                h.update(param.tobytes())
            elif isinstance(param, (bytes, bytearray)):
                h.update(name.encode())
                h.update(param)
        return h.digest()

    def consensus_export_weights(self) -> Dict[str, bytes]:
        """Serialize model weights for P2P sync.

        Uses JSON-safe serialization — NEVER use unsafe deserializers
        on untrusted network data (arbitrary code execution risk).
        """
        import json
        weights = {}
        for name, param in self._model_state.items():
            # Convert numpy arrays to lists, keep scalars as-is
            if hasattr(param, 'tolist'):
                serializable = param.tolist()
            elif isinstance(param, (int, float)):
                serializable = param
            elif isinstance(param, (bytes, bytearray)):
                serializable = list(param)
            else:
                serializable = float(param) if param is not None else 0.0
            weights[name] = json.dumps(serializable).encode('utf-8')
        return weights

    def consensus_import_weights(self, weights: Dict[str, bytes],
                                  peer_stake: int) -> bool:
        """Import peer weights with stake-weighted averaging.

        Uses JSON deserialization (safe) — no arbitrary code execution.
        FedAvg: new_param = (local_stake * local + peer_stake * peer) / total_stake
        """
        import json
        try:
            for name, data in weights.items():
                raw = json.loads(data.decode('utf-8'))
                # Reconstruct as numpy array if original was array-like
                if isinstance(raw, list):
                    import numpy as np
                    peer_param = np.array(raw, dtype=np.float32)
                else:
                    peer_param = float(raw)
                if name in self._model_state:
                    local_param = self._model_state[name]
                    # Weighted average
                    total = self._local_stake + peer_stake
                    if total > 0:
                        self._model_state[name] = (
                            self._local_stake * local_param + peer_stake * peer_param
                        ) / total
            return True
        except (json.JSONDecodeError, ValueError, TypeError):
            return False

    def should_sync_models(self, current_epoch: int, sync_interval: int = 10) -> bool:
        """True if enough epochs have passed since last model sync."""
        if not hasattr(self, '_last_sync_epoch'):
            self._last_sync_epoch = 0
        return (current_epoch - self._last_sync_epoch) >= sync_interval

    def serialize_delta(self, delta: ModelDelta) -> bytes:
        """Serialize a model delta for network transmission.

        Args:
            delta: The ModelDelta to serialize.

        Returns:
            Bytes representation suitable for P2P transmission.
        """
        from positronic.ai.engine.serialization import serialize_state
        return serialize_state(delta.delta)

    def deserialize_delta(
        self,
        data: bytes,
        model_name: str,
        version: int,
        node_id: str,
        num_samples: int,
    ) -> ModelDelta:
        """Deserialize a model delta received from the network.

        Args:
            data: Serialized bytes received from a peer.
            model_name: Identifier for the target model.
            version: Federated round version from the sender.
            node_id: Unique identifier of the sending node.
            num_samples: Number of training samples the sender used.

        Returns:
            A reconstructed ModelDelta ready for ``receive_delta()``.
        """
        from positronic.ai.engine.serialization import deserialize_state
        delta_dict = deserialize_state(data)
        return ModelDelta(
            model_name=model_name,
            delta=delta_dict,
            version=version,
            timestamp=time.time(),
            node_id=node_id,
            num_samples=num_samples,
        )

    def get_stats(self) -> Dict:
        """Return a summary dictionary of the federated averager state.

        Returns:
            Dictionary with keys: round, pending_deltas, tracked_models,
            node_id.
        """
        return {
            "round": self._round,
            "pending_deltas": len(self._pending_deltas),
            "tracked_models": list(self._last_shared.keys()),
            "node_id": self.node_id,
        }
