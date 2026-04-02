"""
Positronic - Graph Attention Network (GAT) for Smart Contract Risk Analysis (SCRA)

Analyzes contract interaction graphs to detect security risks such as reentrancy,
unauthorized access, flash-loan exploits, and other smart contract vulnerabilities.

Architecture:
    2x GATLayer (4-head attention) -> Global Attention Pooling -> Risk Classifier

Each GATLayer computes multi-head attention over the graph neighborhood:
    alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    h_i' = sigma(sum_j alpha_ij * Wh_j)

The global attention pooling layer learns to weight each node's contribution
to the graph-level representation, enabling risk classification over the
entire contract interaction topology.
"""

import numpy as np
from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm, Dropout
from positronic.ai.engine.activations import LeakyReLU, GELU, Sigmoid, Softmax
from positronic.ai.engine.initializers import xavier_normal


class GATLayer(Model):
    """
    Single Graph Attention Network layer with multi-head attention.

    For each node i, computes attention-weighted aggregation over its neighbors:
        1. Project features: Wh_i = W * h_i for all nodes
        2. Compute attention: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        3. Normalize: alpha_ij = softmax_j(e_ij) (only over neighbors)
        4. Aggregate: h_i' = sum_j(alpha_ij * Wh_j)

    Multiple independent attention heads are computed and their outputs
    are concatenated to form the final node representations.

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Total number of output features per node (split across heads).
        Must be divisible by ``num_heads``.
    num_heads : int
        Number of independent attention heads. Default: 4.
    dropout : float
        Dropout probability applied during training. Default: 0.1.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads

        # Per-head linear transforms and attention weight vectors
        self.W = []  # Node feature projection: (in_features) -> (out_per_head)
        self.a = []  # Attention vector: (2 * out_per_head) -> scalar
        for _ in range(num_heads):
            self.W.append(Dense(in_features, self.out_per_head, bias=False))
            attn_weight = Tensor(
                xavier_normal((2 * self.out_per_head, 1)),
                requires_grad=True,
            )
            self.a.append(attn_weight)

        self.leaky_relu = LeakyReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        """
        Forward pass through the graph attention layer.

        Computes multi-head attention over graph neighborhoods defined by the
        adjacency matrix, then concatenates head outputs.

        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape (num_nodes, in_features).
        adj : np.ndarray
            Binary adjacency matrix of shape (num_nodes, num_nodes). A value of 1
            at position (i, j) indicates an edge from node i to node j.

        Returns
        -------
        Tensor
            Updated node features of shape (num_nodes, out_features), where
            out_features = num_heads * out_per_head.
        """
        num_nodes = x.data.shape[0]
        head_outputs = []

        for k in range(self.num_heads):
            # Project node features for this head: (num_nodes, out_per_head)
            Wh = self.W[k].forward(x)

            # Decomposed attention computation for efficiency.
            # Instead of computing a^T [Wh_i || Wh_j] for all pairs,
            # split the attention vector a_k into two halves (a1, a2) so that:
            #   e_ij = (Wh_i @ a1) + (Wh_j @ a2)
            a_k = self.a[k]
            a1 = a_k.data[:self.out_per_head]   # first half: source node
            a2 = a_k.data[self.out_per_head:]    # second half: target node

            # Compute source and target attention components
            e_i = Wh.data @ a1  # (num_nodes, 1) - source contribution
            e_j = Wh.data @ a2  # (num_nodes, 1) - target contribution

            # Pairwise attention coefficients via broadcasting
            e = e_i + e_j.T  # (num_nodes, num_nodes)

            # LeakyReLU activation on raw attention coefficients
            e = np.where(e > 0, e, 0.01 * e)

            # Mask non-neighbors with large negative value (effectively -inf for softmax)
            e = np.where(adj > 0, e, -1e9)

            # Numerically stable softmax over neighbor dimension
            e_max = np.max(e, axis=1, keepdims=True)
            exp_e = np.exp(e - e_max)
            exp_e = exp_e * (adj > 0)  # zero out non-neighbor entries
            alpha = exp_e / (np.sum(exp_e, axis=1, keepdims=True) + 1e-8)

            # Attention-weighted aggregation of neighbor features
            h_prime = alpha @ Wh.data  # (num_nodes, out_per_head)
            head_outputs.append(h_prime)

        # Concatenate all head outputs along the feature dimension
        concat = np.concatenate(head_outputs, axis=-1)  # (num_nodes, out_features)

        # Wrap result as a Tensor with backward support
        out = Tensor(concat, requires_grad=True, _children=(x,))

        def _backward():
            if x.requires_grad:
                # Backpropagate gradient through the linear transforms of each head
                grad_x = np.zeros_like(x.data)
                head_grad_start = 0
                for k in range(self.num_heads):
                    head_grad = out.grad[:, head_grad_start:head_grad_start + self.out_per_head]
                    # Gradient flows through the attention-weighted sum and linear projection
                    grad_x += head_grad @ self.W[k].weight.data.T
                    head_grad_start += self.out_per_head
                x.grad = x.grad + grad_x

        out._backward = _backward

        return out


class GraphAttentionNet(Model):
    """
    Graph Attention Network for smart contract risk analysis.

    Takes a graph of contract interactions (nodes = addresses, edges = calls/transfers)
    and produces risk scores across multiple vulnerability categories. The network
    uses two stacked GAT layers to learn node representations that capture both
    local interaction patterns and multi-hop structural features, followed by
    a learned global attention pooling mechanism that aggregates node-level
    information into a graph-level risk assessment.

    Architecture
    ------------
        Input: Node features (num_nodes, node_feature_dim)
        -> GATLayer(node_dim, 64, heads=4) + LeakyReLU
        -> GATLayer(64, 64, heads=4) + LeakyReLU
        -> Global Attention Pooling (learned node importance weighting)
        -> Dense(64, 32) + GELU
        -> Dense(32, num_risk_types=8) + Sigmoid
        -> Risk scores per type

    Parameters
    ----------
    node_dim : int
        Number of input features per node. Default: 8.
    hidden_dim : int
        Hidden dimension for GAT layers and classification head. Default: 64.
    num_heads : int
        Number of attention heads in each GAT layer. Default: 4.

    Attributes
    ----------
    NODE_FEATURE_DIM : int
        Expected number of input node features (8).
    NUM_RISK_TYPES : int
        Number of risk categories in the output (8), matching the RiskType enum.

    Notes
    -----
    The 8 node features are:
        0. nonce: transaction count of the address
        1. balance_log: log-scaled balance
        2. is_contract: 1.0 if the address is a known contract
        3. interaction_count: log-scaled total interactions
        4. code_size_log: log-scaled contract bytecode size
        5. creation_age: age of the contract
        6. total_value_sent_log: log-scaled cumulative outbound value
        7. unique_interactions: number of distinct interaction partners
    """

    NODE_FEATURE_DIM = 8
    NUM_RISK_TYPES = 8  # Matches the RiskType enum

    def __init__(self, node_dim: int = 8, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()

        # Two stacked GAT layers for multi-hop neighborhood aggregation
        self.gat1 = GATLayer(node_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads)
        self.leaky_relu = LeakyReLU()

        # Learned global attention pooling: computes importance weight per node
        self.pool_attn = Dense(hidden_dim, 1)

        # Risk classification head
        self.fc1 = Dense(hidden_dim, hidden_dim // 2)
        self.act = GELU()
        self.fc2 = Dense(hidden_dim // 2, self.NUM_RISK_TYPES)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        """
        Forward pass through the graph attention network.

        Processes node features through two GAT layers, aggregates via global
        attention pooling, and classifies risk across multiple categories.

        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape (num_nodes, node_dim).
        adj : np.ndarray
            Adjacency matrix of shape (num_nodes, num_nodes). Binary values
            where 1 indicates a directed or undirected edge.

        Returns
        -------
        Tensor
            Risk score tensor of shape (1, NUM_RISK_TYPES) with values in [0, 1],
            one probability per risk category.
        """
        # First GAT layer with LeakyReLU activation
        h = self.gat1.forward(x, adj)
        h = self.leaky_relu(h)

        # Second GAT layer with LeakyReLU activation
        h = self.gat2.forward(h, adj)
        h = self.leaky_relu(h)

        # Global attention pooling: learn a scalar importance weight per node,
        # then compute a weighted sum of node representations
        attn_scores = self.pool_attn.forward(h)  # (num_nodes, 1)
        max_scores = np.max(attn_scores.data)
        exp_scores = np.exp(attn_scores.data - max_scores)
        attn_weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        pooled_data = np.sum(h.data * attn_weights, axis=0, keepdims=True)  # (1, hidden_dim)

        pooled = Tensor(pooled_data, requires_grad=True, _children=(h,))

        def _pool_bwd():
            if h.requires_grad:
                h.grad = h.grad + np.broadcast_to(pooled.grad, h.data.shape) * attn_weights

        pooled._backward = _pool_bwd

        # Risk classification: Dense -> GELU -> Dense -> Sigmoid
        out = self.fc1.forward(pooled)
        out = self.act(out)
        out = self.fc2.forward(out)
        out = self.sigmoid(out)

        return out

    def score(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> float:
        """
        Score a contract interaction graph for overall risk.

        Convenience method that runs a forward pass in eval mode and returns
        the maximum risk probability across all risk categories.

        Parameters
        ----------
        node_features : np.ndarray
            Node feature matrix of shape (num_nodes, node_dim).
        adj_matrix : np.ndarray
            Adjacency matrix of shape (num_nodes, num_nodes).

        Returns
        -------
        float
            Risk score in [0, 1], where higher values indicate greater risk.
            Computed as the maximum probability across all risk types.
        """
        self.eval()
        x = Tensor(node_features, requires_grad=False)
        risk_scores = self.forward(x, adj_matrix)
        self.train()
        return float(np.max(risk_scores.data))

    @staticmethod
    def build_graph_from_tx(tx, call_graph, interaction_count, known_contracts,
                            account_lookup=None):
        """
        Build a node feature matrix and adjacency matrix from transaction context.

        Constructs a local subgraph around the target transaction by including:
        - The sender and recipient addresses
        - Up to 5 one-hop neighbors of each from the call graph

        Parameters
        ----------
        tx : Transaction
            The target transaction. Must have ``sender`` and ``recipient`` attributes
            that are convertible to bytes.
        call_graph : dict
            Mapping from address (bytes) to list of addresses that have been called.
            Represents the known contract interaction topology.
        interaction_count : dict
            Mapping from address (bytes) to integer count of total interactions.
        known_contracts : set or dict
            Collection of addresses (bytes) known to be smart contracts.
        account_lookup : dict, optional
            Phase 16: Mapping from address (bytes) to dict with optional keys:
            ``nonce``, ``balance``, ``code_size``, ``first_seen``, ``total_value_sent``.
            When provided, fills placeholder features with real account data.
            When None, placeholders remain 0.0 (backward compatible).

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            - node_features: float32 array of shape (num_nodes, 8)
            - adj_matrix: float32 array of shape (num_nodes, num_nodes), symmetric
              with self-loops on the diagonal

        Notes
        -----
        The 8 node features per address are:
            0. nonce: transaction count of the address (from account_lookup)
            1. balance_log: log10(balance + 1) (from account_lookup)
            2. is_contract: 1.0 if address is in known_contracts
            3. interaction_count: log1p of total interaction count
            4. code_size_log: log10(code_size + 1) (from account_lookup)
            5. creation_age: normalized age since first seen (from account_lookup)
            6. total_value_sent_log: log10(cumulative sent + 1) (from account_lookup)
            7. unique_interactions: number of distinct call targets
        """
        # Collect nodes: sender, recipient, and their 1-hop neighbors
        nodes = set()
        nodes.add(bytes(tx.sender))
        nodes.add(bytes(tx.recipient))

        # Add 1-hop neighbors from the call graph (limited to 5 per seed node)
        for addr in [bytes(tx.sender), bytes(tx.recipient)]:
            if addr in call_graph:
                for neighbor in call_graph[addr][:5]:
                    nodes.add(bytes(neighbor))

        nodes = list(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # Build symmetric adjacency matrix from call graph edges
        adj = np.zeros((num_nodes, num_nodes))
        for src, targets in call_graph.items():
            if bytes(src) in node_to_idx:
                i = node_to_idx[bytes(src)]
                for tgt in targets:
                    if bytes(tgt) in node_to_idx:
                        j = node_to_idx[bytes(tgt)]
                        adj[i][j] = 1.0
                        adj[j][i] = 1.0  # undirected graph

        # Add self-loops (each node attends to itself)
        np.fill_diagonal(adj, 1.0)

        # Build node feature matrix
        features = np.zeros((num_nodes, 8))
        for addr, idx in node_to_idx.items():
            interactions = interaction_count.get(addr, 0)
            is_known = 1.0 if addr in known_contracts else 0.0

            # Phase 16: Fill features from account_lookup if available
            nonce = 0.0
            balance_log = 0.0
            code_size_log = 0.0
            creation_age = 0.0
            total_value_sent_log = 0.0

            if account_lookup and addr in account_lookup:
                acct = account_lookup[addr]
                nonce = float(acct.get("nonce", 0))
                balance = acct.get("balance", 0)
                balance_log = float(np.log10(balance + 1)) if balance > 0 else 0.0
                code_size = acct.get("code_size", 0)
                code_size_log = float(np.log10(code_size + 1)) if code_size > 0 else 0.0
                # Normalize creation_age: cap at 1.0 (365 days)
                first_seen = acct.get("first_seen", 0)
                if first_seen > 0:
                    import time as _time
                    age_days = (_time.time() - first_seen) / 86400.0
                    creation_age = min(age_days / 365.0, 1.0)
                total_sent = acct.get("total_value_sent", 0)
                total_value_sent_log = float(np.log10(total_sent + 1)) if total_sent > 0 else 0.0

            features[idx] = [
                nonce,                                      # nonce (from account data)
                balance_log,                                # balance_log (from account data)
                is_known,                                   # is_contract
                np.log1p(float(interactions)),               # interaction_count (log-scaled)
                code_size_log,                              # code_size_log (from account data)
                creation_age,                               # creation_age (from account data)
                total_value_sent_log,                       # total_value_sent_log (from account data)
                float(len(call_graph.get(addr, []))),       # unique_interactions
            ]

        return features.astype(np.float32), adj.astype(np.float32)
