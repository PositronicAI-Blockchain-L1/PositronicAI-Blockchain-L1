"""
Positronic - AI Model Architectures

Neural network model implementations for the Proof-of-Neural-Consensus
AI Validation Gate. Each model is built on the Positronic neural engine
(pure NumPy autograd) and serves a specific role in transaction validation.

Models:
    - VAE: Variational Autoencoder for Transaction Anomaly Detection (TAD)
    - TemporalAttentionNet: Temporal attention for MEV/Sandwich Detection (MSAD)
    - GraphAttentionNet: Graph attention for Contract Risk Analysis (SCRA)
    - LSTMAttentionNet: LSTM + attention for Economic Stability Guardian (ESG)
    - MetaEnsemble: Learned meta-ensemble for combining model scores
    - ContrastiveLearner: Self-supervised pre-training framework
"""

from positronic.ai.models.vae import VAE, VAEEncoder, VAEDecoder
from positronic.ai.models.temporal_attention import TemporalAttentionNet, TransformerBlock
from positronic.ai.models.graph_attention import GraphAttentionNet, GATLayer
from positronic.ai.models.lstm_attention import LSTMAttentionNet, TemporalAttention
from positronic.ai.models.meta_ensemble import MetaEnsemble
from positronic.ai.models.contrastive import ContrastiveLearner, ProjectionHead

__all__ = [
    # TAD - Transaction Anomaly Detector
    "VAE", "VAEEncoder", "VAEDecoder",
    # MSAD - MEV/Sandwich Attack Detector
    "TemporalAttentionNet", "TransformerBlock",
    # SCRA - Smart Contract Risk Analyzer
    "GraphAttentionNet", "GATLayer",
    # ESG - Economic Stability Guardian
    "LSTMAttentionNet", "TemporalAttention",
    # Meta-Ensemble
    "MetaEnsemble",
    # Pre-training
    "ContrastiveLearner", "ProjectionHead",
]
