"""
Positronic - Variational Autoencoder (VAE)
==========================================

Real neural network for transaction anomaly detection in the Transaction
Anomaly Detector (TAD) component of the AI Validation Gate.

Architecture
------------
    Encoder:  input(35) -> Dense(64) -> Dense(32) -> Dense(16) -> mu(8), logvar(8)
    Latent:   z = mu + std * epsilon  (reparameterization trick)
    Decoder:  z(8) -> Dense(16) -> Dense(32) -> Dense(64) -> output(35)

Each hidden layer uses GELU activation, LayerNorm, and Dropout(0.1).
The decoder output is passed through Sigmoid to bound reconstructions in [0, 1].

Anomaly Detection
-----------------
Anomaly scores are derived from reconstruction error (MSE between input and
output). The score is a sigmoid-transformed z-score relative to running
statistics of reconstruction errors observed during training:

    anomaly_score = sigmoid(z_score - 2.0)

This centers the detection threshold at 2 standard deviations above the
mean reconstruction error, producing scores in [0, 1] where higher values
indicate greater anomaly likelihood.

Loss Function
-------------
    L = MSE(x, x_hat) + beta * KL(q(z|x) || p(z))

where the KL divergence regularizer encourages the latent distribution
q(z|x) = N(mu, sigma^2) to stay close to the standard normal prior
p(z) = N(0, I). The beta hyperparameter controls the tradeoff between
reconstruction fidelity and latent regularization.

Dependencies
------------
All imports reference the Positronic pure-NumPy neural engine:
    - positronic.ai.engine.tensor.Tensor
    - positronic.ai.engine.model.Model
    - positronic.ai.engine.layers.Dense, LayerNorm, Dropout
    - positronic.ai.engine.activations.GELU, Sigmoid
    - positronic.ai.engine.initializers.xavier_normal
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict

from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm, Dropout
from positronic.ai.engine.activations import GELU, Sigmoid
from positronic.ai.engine.initializers import xavier_normal
from positronic.constants import AI_FEATURE_DIM, AI_FEATURE_DIM_V1


class VAEEncoder(Model):
    """Encoder network that maps transaction features to a latent distribution.

    The encoder compresses high-dimensional transaction features into a
    low-dimensional latent space parameterized by a mean vector (mu) and
    a log-variance vector (log_var). Together these define the approximate
    posterior distribution q(z|x) = N(mu, diag(exp(log_var))).

    Architecture::

        input -> [Dense -> LayerNorm -> GELU -> Dropout] x N -> mu, log_var

    Args:
        input_dim: Dimensionality of the input feature vector. Default: ``35``
            (Phase 16 expanded transaction feature set).
        hidden_dims: List of hidden layer sizes for the encoder stack.
            Default: ``[64, 32, 16]``.
        latent_dim: Dimensionality of the latent space. Default: ``8``.
    """

    def __init__(
        self,
        input_dim: int = AI_FEATURE_DIM,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder hidden layers
        self.layers: List[Dense] = []
        self.norms: List[LayerNorm] = []
        self.activation = GELU()
        self.dropout = Dropout(p=0.1)

        prev_dim = input_dim
        for h_dim in hidden_dims:
            dense = Dense(prev_dim, h_dim)
            # Apply Xavier normal initialization for GELU-compatible weights
            dense.weight.data = xavier_normal(dense.weight.data.shape)
            self.layers.append(dense)
            self.norms.append(LayerNorm(h_dim))
            prev_dim = h_dim

        # Latent distribution parameters: mu and log_var
        self.fc_mu = Dense(prev_dim, latent_dim)
        self.fc_mu.weight.data = xavier_normal(self.fc_mu.weight.data.shape)

        self.fc_log_var = Dense(prev_dim, latent_dim)
        self.fc_log_var.weight.data = xavier_normal(
            self.fc_log_var.weight.data.shape
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input features into latent distribution parameters.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Tuple of ``(mu, log_var)`` where each has shape
            ``(batch_size, latent_dim)``. Together they parameterize the
            approximate posterior q(z|x) = N(mu, diag(exp(log_var))).
        """
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = layer.forward(h)
            h = norm.forward(h)
            h = self.activation(h)
            h = self.dropout.forward(h)

        mu = self.fc_mu.forward(h)
        log_var = self.fc_log_var.forward(h)
        return mu, log_var


class VAEDecoder(Model):
    """Decoder network that maps latent vectors back to the input space.

    The decoder reconstructs transaction features from a latent code,
    producing a reconstruction that can be compared against the original
    input to measure anomaly severity.

    Architecture::

        z -> [Dense -> LayerNorm -> GELU -> Dropout] x N -> Dense -> Sigmoid

    Args:
        latent_dim: Dimensionality of the latent space. Default: ``8``.
        hidden_dims: List of hidden layer sizes for the decoder stack.
            Default: ``[16, 32, 64]`` (mirror of encoder).
        output_dim: Dimensionality of the output (must match input_dim
            of the encoder). Default: ``35``.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = AI_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 64]

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build decoder hidden layers
        self.layers: List[Dense] = []
        self.norms: List[LayerNorm] = []
        self.activation = GELU()
        self.dropout = Dropout(p=0.1)

        prev_dim = latent_dim
        for h_dim in hidden_dims:
            dense = Dense(prev_dim, h_dim)
            dense.weight.data = xavier_normal(dense.weight.data.shape)
            self.layers.append(dense)
            self.norms.append(LayerNorm(h_dim))
            prev_dim = h_dim

        # Output projection with sigmoid to bound values in [0, 1]
        self.output_layer = Dense(prev_dim, output_dim)
        self.output_layer.weight.data = xavier_normal(
            self.output_layer.weight.data.shape
        )
        self.sigmoid = Sigmoid()

    def forward(self, z: Tensor) -> Tensor:
        """Decode a latent vector into a reconstructed feature vector.

        Args:
            z: Latent tensor of shape ``(batch_size, latent_dim)``.

        Returns:
            Reconstructed tensor of shape ``(batch_size, output_dim)``
            with values in [0, 1] (sigmoid-bounded).
        """
        h = z
        for layer, norm in zip(self.layers, self.norms):
            h = layer.forward(h)
            h = norm.forward(h)
            h = self.activation(h)
            h = self.dropout.forward(h)

        reconstruction = self.output_layer.forward(h)
        reconstruction = self.sigmoid(reconstruction)
        return reconstruction


class VAE(Model):
    """Variational Autoencoder for transaction anomaly detection.

    This is the core model for the Transaction Anomaly Detector (TAD) in the
    Positronic AI Validation Gate. It learns a compressed representation of
    normal transaction patterns and detects anomalies through reconstruction
    error analysis.

    Architecture::

        Encoder: input(35) -> Dense(64) -> Dense(32) -> Dense(16) -> mu(8), logvar(8)
        Reparameterize: z = mu + std * epsilon  where epsilon ~ N(0, I)
        Decoder: z(8) -> Dense(16) -> Dense(32) -> Dense(64) -> output(35)

    Anomaly Scoring::

        anomaly_score = sigmoid(z_score(reconstruction_error) - 2.0)

    where the z-score is computed relative to the running mean and standard
    deviation of reconstruction errors observed during training. This places
    the decision boundary at approximately 2 standard deviations above the
    training-time mean error.

    Args:
        input_dim: Dimensionality of transaction features. Default: ``35``
            (Phase 16 expanded feature set).
        latent_dim: Dimensionality of the latent space. Default: ``8``.
        beta: Weight for the KL divergence term in the loss function.
            Default: ``1.0``. Values less than 1.0 produce a beta-VAE that
            prioritizes reconstruction quality over latent regularization.

    Example::

        vae = VAE(input_dim=35, latent_dim=8, beta=1.0)
        optimizer = Adam(vae.parameters(), lr=0.001)

        # Training
        loss = vae.train_step(batch, optimizer)

        # Anomaly detection
        score = vae.compute_anomaly_score(transaction_features)
        is_anomalous = score > 0.7
    """

    def __init__(
        self,
        input_dim: int = AI_FEATURE_DIM,
        latent_dim: int = 8,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Sub-networks
        self.encoder = VAEEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, output_dim=input_dim)

        # Running statistics for anomaly scoring (Welford's online algorithm)
        self._error_mean: float = 0.0
        self._error_var: float = 1.0
        self._error_count: int = 0

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Apply the reparameterization trick to sample from the latent distribution.

        Generates a latent sample z = mu + std * epsilon where epsilon is
        drawn from the standard normal distribution. This trick allows
        gradients to flow through the sampling operation by moving the
        stochasticity into epsilon (which does not depend on model parameters).

        Args:
            mu: Mean of the approximate posterior, shape ``(batch_size, latent_dim)``.
            log_var: Log-variance of the approximate posterior, same shape as mu.

        Returns:
            Sampled latent tensor z of shape ``(batch_size, latent_dim)``.
        """
        clamped_log_var = np.clip(log_var.data, -20, 20)
        std_data = np.exp(0.5 * clamped_log_var)
        if self.training:
            eps = np.random.randn(*std_data.shape)
        else:
            eps = np.zeros_like(std_data)  # Deterministic inference
        z_data = mu.data + std_data * eps

        # Construct tensor with a custom backward that propagates gradients
        # through the reparameterization:
        #   dz/d(mu) = 1
        #   dz/d(log_var) = 0.5 * std * eps
        z = Tensor(z_data, requires_grad=True, _children=(mu, log_var))

        def _backward() -> None:
            if mu.requires_grad:
                mu.grad = mu.grad + z.grad
            if log_var.requires_grad:
                log_var.grad = log_var.grad + z.grad * 0.5 * std_data * eps

        z._backward = _backward
        return z

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Full forward pass through encoder, reparameterization, and decoder.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Tuple of ``(reconstruction, mu, log_var)`` where:
                - reconstruction: Reconstructed input, shape ``(batch_size, input_dim)``.
                - mu: Latent mean, shape ``(batch_size, latent_dim)``.
                - log_var: Latent log-variance, shape ``(batch_size, latent_dim)``.
        """
        mu, log_var = self.encoder.forward(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder.forward(z)
        return reconstruction, mu, log_var

    def compute_loss(
        self,
        x: Tensor,
        reconstruction: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> Tensor:
        """Compute the VAE loss: reconstruction error + beta * KL divergence.

        The reconstruction loss is the mean squared error between the input
        and its reconstruction. The KL divergence measures how far the
        learned latent distribution q(z|x) deviates from the standard
        normal prior p(z) = N(0, I).

        Loss = MSE(x, x_hat) + beta * KL(q(z|x) || N(0, I))

        where KL = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))

        Args:
            x: Original input tensor.
            reconstruction: Reconstructed output tensor.
            mu: Latent mean tensor.
            log_var: Latent log-variance tensor.

        Returns:
            Scalar loss tensor with backward support for autograd.
        """
        # Reconstruction loss: mean squared error
        diff = reconstruction - x
        recon_loss = (diff * diff).mean()

        # KL divergence: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        clamped_lv = np.clip(log_var.data, -20, 20)
        kl_data = -0.5 * np.mean(
            1.0 + clamped_lv - mu.data ** 2 - np.exp(clamped_lv)
        )
        kl_loss = Tensor(
            np.array(kl_data), requires_grad=True, _children=(mu, log_var)
        )

        def _kl_backward() -> None:
            # Gradients of KL divergence w.r.t. mu and log_var:
            #   d(KL)/d(mu) = mu / N
            #   d(KL)/d(log_var) = 0.5 * (exp(log_var) - 1) / N
            # where N is the total number of elements.
            n = float(mu.data.size)
            if mu.requires_grad:
                mu.grad = mu.grad + kl_loss.grad * mu.data / n
            if log_var.requires_grad:
                log_var.grad = (
                    log_var.grad
                    + kl_loss.grad * 0.5 * (np.exp(np.clip(log_var.data, -20, 20)) - 1.0) / n
                )

        kl_loss._backward = _kl_backward

        # Combined loss with beta weighting on KL term
        total_loss = recon_loss + kl_loss * Tensor(np.array(self.beta))
        return total_loss

    def train_step(self, x_batch: np.ndarray, optimizer) -> float:
        """Execute one training step on a mini-batch.

        Performs a full forward-backward pass and parameter update:
        1. Forward pass through encoder, reparameterization, decoder.
        2. Compute VAE loss (reconstruction + KL divergence).
        3. Backpropagate gradients through the computational graph.
        4. Update parameters via the optimizer.
        5. Update running error statistics for anomaly scoring.

        Args:
            x_batch: Input mini-batch as a numpy array of shape
                ``(batch_size, input_dim)``.
            optimizer: Optimizer instance with ``zero_grad()`` and ``step()``
                methods (e.g., ``positronic.ai.engine.optimizers.Adam``).

        Returns:
            Scalar loss value as a Python float.
        """
        optimizer.zero_grad()

        x = Tensor(x_batch, requires_grad=False)
        recon, mu, log_var = self.forward(x)
        loss = self.compute_loss(x, recon, mu, log_var)

        loss.backward()
        optimizer.step()

        # Update running reconstruction error statistics for anomaly scoring
        recon_error = float(
            np.mean((x_batch - recon.data) ** 2, axis=-1).mean()
        )
        self._update_error_stats(recon_error)

        return float(loss.data)

    def compute_anomaly_score(self, x: np.ndarray) -> float:
        """Compute an anomaly score for a transaction feature vector.

        The score is based on the reconstruction error: transactions that
        differ significantly from the patterns learned during training will
        produce high reconstruction errors, indicating potential anomalies.

        The raw error is converted to a probability-like score via:
        1. Compute z-score relative to training error statistics.
        2. Apply a shifted sigmoid centered at z=2 (2 std devs above mean).

        Args:
            x: Transaction feature vector as a numpy array of shape
                ``(input_dim,)`` or ``(1, input_dim)``.

        Returns:
            Anomaly score in [0, 1]. Higher values indicate greater anomaly
            likelihood. Typical threshold: 0.5-0.7 depending on desired
            sensitivity.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_tensor = Tensor(x, requires_grad=False)

        # Run inference in eval mode (disables dropout)
        self.eval()
        recon, mu, log_var = self.forward(x_tensor)
        self.train()

        # Compute per-sample reconstruction error (MSE)
        error = float(np.mean((x - recon.data) ** 2))

        # Convert to anomaly score via z-score and shifted sigmoid
        if self._error_count > 10:
            std = max(np.sqrt(self._error_var), 1e-10)
            z = (error - self._error_mean) / std
            # Shifted sigmoid: center detection at z=2 (2 std devs above mean)
            score = 1.0 / (1.0 + np.exp(-z + 2.0))
        else:
            # Insufficient training data for reliable statistics; return
            # a moderate default score indicating uncertainty.
            score = 0.3

        return float(np.clip(score, 0.0, 1.0))

    def compute_anomaly_scores_batch(self, x: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for a batch of transactions.

        Vectorized version of :meth:`compute_anomaly_score` for efficient
        batch processing.

        Args:
            x: Batch of transaction features, shape ``(batch_size, input_dim)``.

        Returns:
            Array of anomaly scores, shape ``(batch_size,)``, each in [0, 1].
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_tensor = Tensor(x, requires_grad=False)

        self.eval()
        recon, mu, log_var = self.forward(x_tensor)
        self.train()

        # Per-sample MSE
        errors = np.mean((x - recon.data) ** 2, axis=-1)

        if self._error_count > 10:
            std = max(np.sqrt(self._error_var), 1e-10)
            z_scores = (errors - self._error_mean) / std
            scores = 1.0 / (1.0 + np.exp(-z_scores + 2.0))
        else:
            scores = np.full(errors.shape, 0.3)

        return np.clip(scores, 0.0, 1.0)

    def _update_error_stats(self, error: float) -> None:
        """Update running mean and variance of reconstruction errors.

        Uses Welford's online algorithm for numerically stable incremental
        computation of mean and variance without storing all past errors.

        Args:
            error: The latest reconstruction error observation.
        """
        self._error_count += 1
        if self._error_count == 1:
            self._error_mean = error
            self._error_var = 0.0
        else:
            delta = error - self._error_mean
            self._error_mean += delta / self._error_count
            delta2 = error - self._error_mean
            self._error_var += (
                delta * delta2 - self._error_var
            ) / self._error_count

    def get_latent(self, x: np.ndarray) -> np.ndarray:
        """Extract the latent representation (mean) for input features.

        Returns the mean of the approximate posterior q(z|x) without
        sampling, providing a deterministic embedding suitable for
        downstream analysis, clustering, or visualization.

        Args:
            x: Input features, shape ``(input_dim,)`` or ``(batch_size, input_dim)``.

        Returns:
            Latent mean vector(s), shape ``(batch_size, latent_dim)``.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_tensor = Tensor(x, requires_grad=False)
        self.eval()
        mu, log_var = self.encoder.forward(x_tensor)
        self.train()
        return mu.data

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct input features through the full encode-decode pipeline.

        Useful for inspecting what the model considers "normal" for a given
        input. Large differences between the input and reconstruction
        highlight the specific features that the model finds anomalous.

        Args:
            x: Input features, shape ``(input_dim,)`` or ``(batch_size, input_dim)``.

        Returns:
            Reconstructed features, same shape as input.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_tensor = Tensor(x, requires_grad=False)
        self.eval()
        recon, _, _ = self.forward(x_tensor)
        self.train()
        return recon.data

    def save(self) -> dict:
        """Serialize the VAE to a dictionary including architecture metadata.

        Extends the base :meth:`Model.save` with VAE-specific configuration
        and running error statistics needed for anomaly scoring.

        Returns:
            Dictionary with model type, architecture config, weights, and
            anomaly scoring statistics.
        """
        return {
            "model_type": "VAE",
            "version": self._version,
            "config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "beta": self.beta,
            },
            "state": self.state_dict(),
            "error_stats": {
                "mean": self._error_mean,
                "var": self._error_var,
                "count": self._error_count,
            },
            "num_parameters": self.num_parameters(),
        }

    @classmethod
    def load(cls, data: dict, migrate: bool = False) -> "VAE":
        """Reconstruct a VAE from a dictionary produced by :meth:`save`.

        Supports optional weight migration from 26-dim (v1) to 35-dim (v2)
        models. When ``migrate=True`` and the saved model has old dimensions,
        new feature dimensions are initialized with Xavier normal weights.

        Args:
            data: Serialized VAE dictionary.
            migrate: If True and saved dim is v1 (26), migrate to v2 (35).
                Defaults to False (exact restore).

        Returns:
            Restored VAE instance with weights and error statistics.
        """
        config = data["config"]
        saved_input_dim = config["input_dim"]

        # Phase 16: Optionally migrate from old to new dimension
        if (migrate and saved_input_dim == AI_FEATURE_DIM_V1
                and AI_FEATURE_DIM > AI_FEATURE_DIM_V1):
            model = cls(
                input_dim=AI_FEATURE_DIM,
                latent_dim=config["latent_dim"],
                beta=config["beta"],
            )
            # Load with strict=False to handle dimension mismatches
            model.load_state_dict(data["state"], strict=False)
            model._migrate_weights_v1_to_v2(data["state"])
        else:
            model = cls(
                input_dim=saved_input_dim,
                latent_dim=config["latent_dim"],
                beta=config["beta"],
            )
            model.load_state_dict(data["state"], strict=False)

        # Restore anomaly scoring statistics
        stats = data.get("error_stats", {})
        model._error_mean = stats.get("mean", 0.0)
        model._error_var = stats.get("var", 1.0)
        model._error_count = stats.get("count", 0)

        return model

    def _migrate_weights_v1_to_v2(self, old_state: dict) -> None:
        """Phase 16: Migrate encoder/decoder weights from 26-dim to 35-dim.

        Preserves the first 26 columns of the encoder's first layer weights
        and the first 26 rows of the decoder's output layer weights. New
        dimensions are initialized with Xavier normal values.

        Args:
            old_state: State dict from a 26-dim VAE.
        """
        old_dim = AI_FEATURE_DIM_V1
        new_dim = AI_FEATURE_DIM

        # Migrate encoder first layer (input_dim changes)
        if self.encoder.layers:
            first_layer = self.encoder.layers[0]
            old_w = first_layer.weight.data
            if old_w.shape[1] == new_dim:
                # Copy old weights for first 26 features, keep Xavier init for new 9
                old_key = f"encoder.layers.0.weight"
                for key, value in old_state.items():
                    if "layers" in key and ".0." in key and "weight" in key:
                        if hasattr(value, 'shape') and value.shape[1] == old_dim:
                            old_w[:, :old_dim] = value
                            break

        # Migrate decoder output layer (output_dim changes)
        out_layer = self.decoder.output_layer
        old_w = out_layer.weight.data
        if old_w.shape[0] == new_dim:
            for key, value in old_state.items():
                if "output_layer" in key and "weight" in key:
                    if hasattr(value, 'shape') and value.shape[0] == old_dim:
                        old_w[:old_dim, :] = value
                        break
            # Bias migration
            if hasattr(out_layer, 'bias') and out_layer.bias is not None:
                for key, value in old_state.items():
                    if "output_layer" in key and "bias" in key:
                        if hasattr(value, 'shape') and value.shape[0] == old_dim:
                            out_layer.bias.data[:old_dim] = value
                            break
