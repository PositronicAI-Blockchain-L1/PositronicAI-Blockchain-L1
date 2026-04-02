"""
Positronic Neural Engine - Loss Functions

Provides differentiable loss functions for training neural networks.
All loss functions accept Tensor inputs and return scalar Tensor outputs
with full autograd support through the computation graph.

Loss Functions:
    - MSELoss: Mean Squared Error (regression)
    - BCELoss: Binary Cross-Entropy (binary classification)
    - CrossEntropyLoss: Softmax + Negative Log-Likelihood (multi-class)
    - ContrastiveLoss: NT-Xent for self-supervised contrastive learning
    - VAELoss: Combined reconstruction + KL divergence for VAEs
    - HuberLoss: Smooth L1 loss (robust regression)
"""

from typing import Optional

import numpy as np

from positronic.ai.engine.tensor import Tensor


class MSELoss:
    """Mean Squared Error loss.

    Computes the element-wise squared difference between predictions and
    targets, then takes the mean across all elements::

        loss = mean((pred - target)^2)

    Suitable for regression tasks.

    Args:
        reduction: Reduction mode. ``'mean'`` averages over all elements,
            ``'sum'`` sums all elements. Default: ``'mean'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute MSE loss.

        Args:
            pred: Predicted values of any shape.
            target: Ground truth values with the same shape as ``pred``.

        Returns:
            Scalar Tensor containing the loss value.
        """
        diff = pred - target
        squared = diff * diff
        if self.reduction == "mean":
            return squared.mean()
        return squared.sum()


class BCELoss:
    """Binary Cross-Entropy loss.

    Expects predictions to be probabilities in (0, 1). Computes::

        loss = -mean(target * log(pred) + (1 - target) * log(1 - pred))

    Predictions are clamped to [eps, 1-eps] internally to prevent
    numerical issues with log(0).

    This implementation uses Tensor operations so that the autograd
    computation graph is built automatically.

    Args:
        reduction: Reduction mode. ``'mean'`` or ``'sum'``.
            Default: ``'mean'``.
        eps: Small constant for clamping. Default: ``1e-7``.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute binary cross-entropy loss.

        Args:
            pred: Predicted probabilities in (0, 1).
            target: Binary ground truth labels (0 or 1) with the same
                shape as ``pred``.

        Returns:
            Scalar Tensor containing the loss value.
        """
        # Clamp predictions to avoid log(0). We create a new Tensor with
        # clamped data and wire it into the graph as a child of pred so
        # gradients still flow back through pred.
        clamped_data = np.clip(pred.data, self.eps, 1.0 - self.eps)
        pred_clamped = Tensor(
            clamped_data,
            requires_grad=pred.requires_grad,
            _children=(pred,),
        )

        # Build the backward pass for the clamping operation.
        # d(clamp)/d(pred) = 1 where pred is within (eps, 1-eps), else 0.
        if pred.requires_grad:
            clamp_mask = (
                (pred.data >= self.eps) & (pred.data <= 1.0 - self.eps)
            ).astype(pred.data.dtype)

            def _clamp_backward():
                if pred_clamped.grad is not None:
                    if pred.grad is None:
                        pred.grad = np.zeros_like(pred.data)
                    pred.grad = pred.grad + pred_clamped.grad * clamp_mask

            pred_clamped._backward = _clamp_backward

        # Compute log(p) and log(1-p) using Tensor ops so the rest of the
        # graph is traced automatically.
        log_p = Tensor(
            np.log(pred_clamped.data),
            requires_grad=pred_clamped.requires_grad,
            _children=(pred_clamped,),
        )
        if pred_clamped.requires_grad:
            def _log_p_backward():
                if log_p.grad is not None:
                    if pred_clamped.grad is None:
                        pred_clamped.grad = np.zeros_like(pred_clamped.data)
                    pred_clamped.grad = (
                        pred_clamped.grad
                        + log_p.grad / (pred_clamped.data + 1e-12)
                    )
            log_p._backward = _log_p_backward

        one_minus = Tensor(
            1.0 - pred_clamped.data,
            requires_grad=pred_clamped.requires_grad,
            _children=(pred_clamped,),
        )
        if pred_clamped.requires_grad:
            def _one_minus_backward():
                if one_minus.grad is not None:
                    if pred_clamped.grad is None:
                        pred_clamped.grad = np.zeros_like(pred_clamped.data)
                    pred_clamped.grad = pred_clamped.grad - one_minus.grad
            one_minus._backward = _one_minus_backward

        log_1mp = Tensor(
            np.log(one_minus.data + 1e-12),
            requires_grad=one_minus.requires_grad,
            _children=(one_minus,),
        )
        if one_minus.requires_grad:
            def _log_1mp_backward():
                if log_1mp.grad is not None:
                    if one_minus.grad is None:
                        one_minus.grad = np.zeros_like(one_minus.data)
                    one_minus.grad = (
                        one_minus.grad
                        + log_1mp.grad / (one_minus.data + 1e-12)
                    )
            log_1mp._backward = _log_1mp_backward

        # loss = -(target * log_p + (1 - target) * log_1mp)
        term1 = target * log_p
        one_minus_target = Tensor(
            1.0 - target.data, requires_grad=False
        )
        term2 = one_minus_target * log_1mp
        total = term1 + term2
        neg_total = Tensor(np.zeros(1), requires_grad=False) - total

        if self.reduction == "mean":
            return neg_total.mean()
        return neg_total.sum()


class CrossEntropyLoss:
    """Cross-Entropy loss combining log-softmax and negative log-likelihood.

    Accepts raw logits (unnormalized scores) and integer class indices.
    Internally applies the numerically stable log-softmax transform::

        log_softmax = logits - max(logits) - log(sum(exp(logits - max)))

    Then computes the negative log-likelihood for the target class::

        loss = -mean(log_softmax[range(batch), target])

    Args:
        reduction: Reduction mode. ``'mean'`` or ``'sum'``.
            Default: ``'mean'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Raw unnormalized logits of shape ``(batch, num_classes)``.
            target: Integer class indices of shape ``(batch,)``. Values must
                be in ``[0, num_classes)``. Can be a Tensor wrapping integers.

        Returns:
            Scalar Tensor containing the loss value.
        """
        batch_size = logits.data.shape[0]
        target_idx = target.data.astype(int)

        # Numerically stable log-softmax
        logits_max = np.max(logits.data, axis=1, keepdims=True)
        shifted = logits.data - logits_max
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp  # (batch, classes)

        # Gather the log-probability of the target class for each sample
        nll = -log_probs[np.arange(batch_size), target_idx]  # (batch,)

        if self.reduction == "mean":
            loss_val = np.mean(nll)
        else:
            loss_val = np.sum(nll)

        out = Tensor(
            np.array(loss_val),
            requires_grad=logits.requires_grad,
            _children=(logits,),
        )

        # Build backward: d(loss)/d(logits) = softmax - one_hot(target)
        if logits.requires_grad:
            softmax = np.exp(log_probs)  # (batch, classes)
            one_hot = np.zeros_like(logits.data)
            one_hot[np.arange(batch_size), target_idx] = 1.0

            def _backward():
                grad = softmax - one_hot  # (batch, classes)
                if self.reduction == "mean":
                    grad = grad / batch_size
                if logits.grad is None:
                    logits.grad = np.zeros_like(logits.data)
                logits.grad = logits.grad + grad * (
                    out.grad if out.grad is not None else 1.0
                )

            out._backward = _backward

        return out


class ContrastiveLoss:
    """NT-Xent (Normalized Temperature-Scaled Cross-Entropy) loss.

    Used for self-supervised contrastive learning (SimCLR). Given two sets
    of embeddings from augmented views of the same batch, the loss
    encourages positive pairs (same sample, different augmentation) to be
    similar and negative pairs (different samples) to be dissimilar.

    The similarity matrix is computed as::

        sim = (z1 @ z2.T) / temperature

    Positive pairs lie on the diagonal. The loss is the mean cross-entropy
    over both directions (z1->z2 and z2->z1).

    Args:
        temperature: Temperature scaling factor. Lower values make the
            distribution sharper. Default: ``0.5``.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def __call__(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute NT-Xent contrastive loss.

        Args:
            z1: Embeddings from view 1, shape ``(batch, dim)``.
            z2: Embeddings from view 2, shape ``(batch, dim)``.

        Returns:
            Scalar Tensor containing the loss value.
        """
        batch_size = z1.data.shape[0]

        # L2-normalize embeddings
        z1_norm = z1.data / (np.linalg.norm(z1.data, axis=1, keepdims=True) + 1e-8)
        z2_norm = z2.data / (np.linalg.norm(z2.data, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity matrix: (batch, batch)
        sim_matrix = (z1_norm @ z2_norm.T) / self.temperature

        # Labels are the diagonal indices (positive pairs)
        labels = np.arange(batch_size)

        # Numerically stable cross-entropy for z1 -> z2 direction
        sim_max = np.max(sim_matrix, axis=1, keepdims=True)
        shifted = sim_matrix - sim_max
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs_12 = shifted - log_sum_exp
        loss_12 = -np.mean(log_probs_12[np.arange(batch_size), labels])

        # z2 -> z1 direction
        sim_matrix_t = sim_matrix.T
        sim_max_t = np.max(sim_matrix_t, axis=1, keepdims=True)
        shifted_t = sim_matrix_t - sim_max_t
        log_sum_exp_t = np.log(np.sum(np.exp(shifted_t), axis=1, keepdims=True))
        log_probs_21 = shifted_t - log_sum_exp_t
        loss_21 = -np.mean(log_probs_21[np.arange(batch_size), labels])

        loss_val = (loss_12 + loss_21) / 2.0

        out = Tensor(
            np.array(loss_val),
            requires_grad=(z1.requires_grad or z2.requires_grad),
            _children=(z1, z2),
        )

        # Backward pass
        if z1.requires_grad or z2.requires_grad:
            softmax_12 = np.exp(log_probs_12)  # (batch, batch)
            softmax_21 = np.exp(log_probs_21)  # (batch, batch)

            def _backward():
                scale = (out.grad if out.grad is not None else 1.0) / (
                    2.0 * batch_size * self.temperature
                )

                # Gradient w.r.t. z1 via the z1->z2 similarity
                one_hot = np.eye(batch_size)
                d_sim_12 = (softmax_12 - one_hot) * scale  # (batch, batch)
                d_sim_21 = (softmax_21 - one_hot) * scale  # (batch, batch)

                # d(loss)/d(z1_norm) from z1->z2: d_sim_12 @ z2_norm
                # d(loss)/d(z1_norm) from z2->z1: d_sim_21.T @ z2_norm
                # (simplified; ignoring normalization backward for stability)
                if z1.requires_grad:
                    dz1 = d_sim_12 @ z2_norm + d_sim_21.T @ z2_norm
                    if z1.grad is None:
                        z1.grad = np.zeros_like(z1.data)
                    z1.grad = z1.grad + dz1

                if z2.requires_grad:
                    dz2 = d_sim_12.T @ z1_norm + d_sim_21 @ z1_norm
                    if z2.grad is None:
                        z2.grad = np.zeros_like(z2.data)
                    z2.grad = z2.grad + dz2

            out._backward = _backward

        return out


class VAELoss:
    """Variational Autoencoder loss.

    Combines a reconstruction loss (MSE) with a KL-divergence
    regularization term to form the Evidence Lower Bound (ELBO)::

        loss = MSE(recon, original) + beta * KL(q(z|x) || p(z))

    where the KL divergence for a diagonal Gaussian posterior is::

        KL = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))

    The ``beta`` parameter controls the trade-off between reconstruction
    quality and latent space regularity (beta-VAE).

    Args:
        beta: Weight of the KL divergence term. Default: ``1.0``.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def __call__(
        self,
        recon: Tensor,
        original: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> Tensor:
        """Compute VAE loss.

        Args:
            recon: Reconstructed output from the decoder.
            original: Original input to the encoder.
            mu: Mean of the latent distribution, shape ``(batch, latent_dim)``.
            log_var: Log-variance of the latent distribution, same shape
                as ``mu``.

        Returns:
            Scalar Tensor containing the combined loss value.
        """
        # Reconstruction loss (MSE)
        diff = recon - original
        recon_loss = (diff * diff).mean()

        # KL divergence: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        mu_sq = mu * mu
        var = Tensor(
            np.exp(log_var.data),
            requires_grad=log_var.requires_grad,
            _children=(log_var,),
        )
        if log_var.requires_grad:
            def _var_backward():
                if var.grad is not None:
                    if log_var.grad is None:
                        log_var.grad = np.zeros_like(log_var.data)
                    log_var.grad = log_var.grad + var.grad * np.exp(log_var.data)
            var._backward = _var_backward

        ones_tensor = Tensor(np.ones_like(mu.data), requires_grad=False)
        kl_elements = ones_tensor + log_var - mu_sq - var
        kl_mean = kl_elements.mean()

        # kl_loss = -0.5 * kl_mean
        neg_half = Tensor(np.array(-0.5), requires_grad=False)
        kl_loss = neg_half * kl_mean

        # Total loss
        beta_tensor = Tensor(np.array(self.beta), requires_grad=False)
        total_loss = recon_loss + beta_tensor * kl_loss
        return total_loss


class HuberLoss:
    """Huber loss (Smooth L1 loss).

    Behaves as MSE for small errors and as MAE for large errors,
    controlled by the ``delta`` threshold::

        loss = 0.5 * (pred - target)^2            if |pred - target| <= delta
        loss = delta * (|pred - target| - 0.5 * delta)   otherwise

    This makes it more robust to outliers than pure MSE.

    Args:
        delta: Threshold at which to transition from quadratic to linear.
            Default: ``1.0``.
        reduction: Reduction mode. ``'mean'`` or ``'sum'``.
            Default: ``'mean'``.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.delta = delta
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Huber loss.

        Args:
            pred: Predicted values.
            target: Ground truth values with the same shape as ``pred``.

        Returns:
            Scalar Tensor containing the loss value.
        """
        diff_data = pred.data - target.data
        abs_diff = np.abs(diff_data)

        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic

        loss_data = 0.5 * quadratic ** 2 + self.delta * linear

        if self.reduction == "mean":
            loss_val = np.mean(loss_data)
        else:
            loss_val = np.sum(loss_data)

        out = Tensor(
            np.array(loss_val),
            requires_grad=(pred.requires_grad or target.requires_grad),
            _children=(pred, target),
        )

        if pred.requires_grad or target.requires_grad:
            def _backward():
                # Gradient: clip(diff, -delta, delta) for quadratic region
                grad = np.where(
                    abs_diff <= self.delta,
                    diff_data,
                    self.delta * np.sign(diff_data),
                )
                if self.reduction == "mean":
                    grad = grad / diff_data.size

                upstream = out.grad if out.grad is not None else 1.0

                if pred.requires_grad:
                    if pred.grad is None:
                        pred.grad = np.zeros_like(pred.data)
                    pred.grad = pred.grad + grad * upstream

                if target.requires_grad:
                    if target.grad is None:
                        target.grad = np.zeros_like(target.data)
                    target.grad = target.grad - grad * upstream

            out._backward = _backward

        return out
