"""
Positronic - Self-Supervised Contrastive Learning
=================================================

Pre-training framework that learns meaningful representations of normal
transaction patterns without requiring any labeled data. This is used at
blockchain genesis to bootstrap the AI Validation Gate before any
fraud/anomaly labels are available.

Algorithm: NT-Xent (Normalized Temperature-scaled Cross Entropy)
----------------------------------------------------------------
1. Take a batch of transaction feature vectors.
2. Create two augmented "views" of each transaction by applying random
   feature masking and Gaussian noise.
3. Encode both views through a shared encoder network.
4. Project encoded representations into a contrastive embedding space.
5. Define positive pairs as the two views of the same transaction and
   negative pairs as views from different transactions.
6. Minimize NT-Xent loss, which pushes positive pairs together and
   negative pairs apart in the embedding space.

After pre-training, the encoder has learned a compressed representation
that captures the statistical structure of normal transactions, enabling
effective anomaly detection even without labeled examples.

Augmentation Strategy
---------------------
Transaction features are augmented with two complementary perturbations:

- **Random feature masking** (default 15%): Randomly zeroes out features,
  forcing the model to learn redundant representations that are robust
  to missing information.
- **Gaussian noise** (default std=0.05): Adds small perturbations,
  teaching the model to be invariant to minor numerical variations.

Dependencies
------------
All imports reference the Positronic pure-NumPy neural engine:
    - positronic.ai.engine.tensor.Tensor
    - positronic.ai.engine.model.Model
    - positronic.ai.engine.layers.Dense, LayerNorm
    - positronic.ai.engine.activations.GELU
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm
from positronic.ai.engine.activations import GELU


class ProjectionHead(Model):
    """MLP projection head for mapping encoder outputs to the contrastive space.

    A two-layer MLP that projects encoder representations into a
    lower-dimensional space where the contrastive loss is computed.
    Following the findings of Chen et al. (SimCLR, 2020), the projection
    head is only used during pre-training and discarded afterwards;
    the encoder representations are used for downstream tasks.

    Architecture::

        input -> Dense -> LayerNorm -> GELU -> Dense -> output

    Args:
        input_dim: Dimensionality of the encoder output. Default: ``8``
            (matching the VAE latent dimension).
        hidden_dim: Size of the hidden layer. Default: ``64``.
        output_dim: Dimensionality of the contrastive embedding space.
            Default: ``32``.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = Dense(input_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.act = GELU()
        self.fc2 = Dense(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Project encoder output into the contrastive embedding space.

        Args:
            x: Encoder output tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Projected embedding of shape ``(batch_size, output_dim)``.
        """
        h = self.fc1.forward(x)
        h = self.norm.forward(h)
        h = self.act(h)
        return self.fc2.forward(h)


class ContrastiveLearner(Model):
    """Self-supervised contrastive learning using NT-Xent loss.

    This framework enables pre-training the transaction encoder at
    blockchain genesis, before any labeled anomaly data is available.
    It learns representations by contrasting augmented views of the
    same transaction against views from different transactions.

    How it works:

    1. **Augmentation**: Each transaction in a batch is augmented twice
       with random masking and Gaussian noise, producing two "views"
       that are semantically similar but numerically different.

    2. **Encoding**: Both views pass through the shared encoder (e.g.,
       a VAEEncoder) to produce latent representations.

    3. **Projection**: Latent representations are projected into a
       contrastive embedding space via the projection head MLP.

    4. **NT-Xent Loss**: The loss pushes embeddings of the same
       transaction's two views (positive pairs) closer together while
       pushing embeddings of different transactions (negative pairs)
       further apart, all scaled by a temperature parameter.

    After pre-training, the projection head is discarded and the encoder
    is used directly for anomaly detection (e.g., as part of a VAE).

    Args:
        encoder: The shared encoder network. Must implement ``forward(x)``
            returning ``(mu, log_var)`` (e.g., a :class:`VAEEncoder`).
        input_dim: Dimensionality of the transaction feature vector.
            Default: ``26``.
        projection_dim: Output dimensionality of the projection head.
            Default: ``32``.
        temperature: Temperature parameter for NT-Xent loss scaling.
            Lower values make the loss more sensitive to hard negatives.
            Default: ``0.5``.

    Example::

        from positronic.ai.models.vae import VAEEncoder
        from positronic.ai.engine.optimizers import Adam

        encoder = VAEEncoder(input_dim=26, latent_dim=8)
        learner = ContrastiveLearner(encoder, input_dim=26)
        optimizer = Adam(learner.parameters(), lr=0.001)

        # Pre-training loop
        for batch in data_loader:
            loss = learner.train_step(batch, optimizer)
    """

    def __init__(
        self,
        encoder: Model,
        input_dim: int = 26,
        projection_dim: int = 32,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.temperature = temperature

        # Infer encoder output dimension from the encoder's latent_dim
        # attribute, falling back to 8 (the VAE default).
        encoder_output_dim = getattr(encoder, "latent_dim", 8)
        self.projection = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=64,
            output_dim=projection_dim,
        )

        # Augmentation parameters
        self.mask_ratio: float = 0.15  # Fraction of features to randomly zero
        self.noise_std: float = 0.05   # Standard deviation of additive noise

    def augment(self, x: np.ndarray) -> np.ndarray:
        """Create an augmented view of transaction features.

        Applies two complementary augmentation strategies:

        1. **Random feature masking**: Each feature is independently zeroed
           with probability ``mask_ratio``, forcing the model to learn
           representations that are robust to missing features.

        2. **Gaussian noise**: Additive noise drawn from N(0, noise_std^2)
           teaches invariance to small numerical perturbations.

        Args:
            x: Original features, shape ``(batch_size, input_dim)``.

        Returns:
            Augmented features as a new numpy array, same shape as input.
        """
        augmented = x.copy()

        # Random feature masking: set features to 0 with probability mask_ratio
        mask = (np.random.random(x.shape) >= self.mask_ratio).astype(
            np.float64
        )
        augmented = augmented * mask

        # Additive Gaussian noise
        noise = np.random.randn(*x.shape) * self.noise_std
        augmented = augmented + noise

        return augmented

    def forward(
        self, x1: Tensor, x2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for both augmented views through encoder and projection.

        Args:
            x1: First augmented view, shape ``(batch_size, input_dim)``.
            x2: Second augmented view, shape ``(batch_size, input_dim)``.

        Returns:
            Tuple of ``(z1, z2)`` where each is a projected embedding
            tensor of shape ``(batch_size, projection_dim)``.
        """
        # Encode both views through the shared encoder.
        # The encoder returns (mu, log_var); we use mu as the representation.
        mu1, _ = self.encoder.forward(x1)
        mu2, _ = self.encoder.forward(x2)

        # Project into contrastive embedding space
        z1 = self.projection.forward(mu1)
        z2 = self.projection.forward(mu2)

        return z1, z2

    def nt_xent_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        For a batch of N samples, there are N positive pairs (two views of
        the same sample) and N*(N-1) negative pairs (views of different
        samples). The loss for each positive pair (i, i) is:

            L_i = -log( exp(sim(z1_i, z2_i) / tau) /
                        sum_j( exp(sim(z1_i, z2_j) / tau) ) )

        where sim(a, b) = (a . b) / (||a|| * ||b||) is cosine similarity
        and tau is the temperature parameter.

        Args:
            z1: Projected embeddings from view 1, shape ``(batch_size, dim)``.
            z2: Projected embeddings from view 2, shape ``(batch_size, dim)``.

        Returns:
            Scalar loss tensor with backward support for autograd.
        """
        batch_size = z1.data.shape[0]

        # L2 normalize embeddings for cosine similarity
        z1_norm = z1.data / (
            np.linalg.norm(z1.data, axis=1, keepdims=True) + 1e-8
        )
        z2_norm = z2.data / (
            np.linalg.norm(z2.data, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity matrix scaled by temperature
        # sim[i, j] = cos(z1_i, z2_j) / temperature
        sim = np.dot(z1_norm, z2_norm.T) / self.temperature  # (batch, batch)

        # Positive pair similarities are on the diagonal: sim[i, i]
        pos_sim = np.array([sim[i, i] for i in range(batch_size)])

        # Numerically stable softmax: subtract row max before exp
        sim_max = np.max(sim, axis=1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)  # (batch, batch)
        denom = np.sum(exp_sim, axis=1)  # (batch,)

        # NT-Xent loss: -log(exp(pos_sim) / sum(exp(sim)))
        # = -pos_sim + log(denom) + sim_max (re-adding the subtracted max)
        loss_per_sample = -pos_sim + np.log(denom + 1e-8) + sim_max.ravel()
        loss_val = np.mean(loss_per_sample)

        # Create loss tensor with custom backward for gradient propagation
        loss = Tensor(
            np.array(loss_val), requires_grad=True, _children=(z1, z2)
        )

        def _backward() -> None:
            # Gradient through the softmax-like cross-entropy operation.
            # The softmax probabilities represent the "attention" each sample
            # pays to others; gradients push positive pairs closer and
            # negative pairs apart.
            softmax = exp_sim / (denom[:, np.newaxis] + 1e-8)  # (batch, batch)

            # d_loss/d_z1 = (softmax - I) @ z2_norm / (temperature * batch)
            # d_loss/d_z2 = (softmax^T - I) @ z1_norm / (temperature * batch)
            identity = np.eye(batch_size)
            grad_z1 = (
                (softmax - identity) @ z2_norm
                / (self.temperature * batch_size)
            )
            grad_z2 = (
                (softmax.T - identity) @ z1_norm
                / (self.temperature * batch_size)
            )

            if z1.requires_grad:
                z1.grad = z1.grad + loss.grad * grad_z1
            if z2.requires_grad:
                z2.grad = z2.grad + loss.grad * grad_z2

        loss._backward = _backward
        return loss

    def train_step(self, batch: np.ndarray, optimizer) -> float:
        """Execute one contrastive pre-training step on a mini-batch.

        The training procedure:
        1. Create two independently augmented views of the batch.
        2. Encode and project both views.
        3. Compute NT-Xent contrastive loss.
        4. Backpropagate gradients.
        5. Update parameters via the optimizer.

        Args:
            batch: Mini-batch of transaction features as a numpy array
                of shape ``(batch_size, input_dim)``.
            optimizer: Optimizer instance with ``zero_grad()`` and ``step()``
                methods (e.g., ``positronic.ai.engine.optimizers.Adam``).

        Returns:
            Scalar loss value as a Python float.
        """
        # Create two independently augmented views
        view1 = self.augment(batch)
        view2 = self.augment(batch)

        x1 = Tensor(view1, requires_grad=False)
        x2 = Tensor(view2, requires_grad=False)

        optimizer.zero_grad()

        z1, z2 = self.forward(x1, x2)
        loss = self.nt_xent_loss(z1, z2)

        loss.backward()
        optimizer.step()

        return float(loss.data)

    def pretrain(
        self,
        data: np.ndarray,
        optimizer,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list:
        """Run the full contrastive pre-training loop.

        Convenience method that handles batching, epoch iteration, and
        optional progress reporting.

        Args:
            data: Full dataset of transaction features, shape
                ``(num_samples, input_dim)``.
            optimizer: Optimizer instance for parameter updates.
            epochs: Number of passes over the full dataset. Default: ``10``.
            batch_size: Number of samples per mini-batch. Default: ``32``.
            verbose: If ``True``, print loss at the end of each epoch.
                Default: ``True``.

        Returns:
            List of average loss values per epoch.
        """
        num_samples = data.shape[0]
        epoch_losses = []

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(num_samples)
            total_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Skip very small batches (contrastive loss needs >1 sample)
                if len(batch_indices) < 2:
                    continue

                batch = data[batch_indices]
                loss = self.train_step(batch, optimizer)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            if verbose:
                print(
                    f"[ContrastiveLearner] Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {avg_loss:.6f}"
                )

        return epoch_losses
