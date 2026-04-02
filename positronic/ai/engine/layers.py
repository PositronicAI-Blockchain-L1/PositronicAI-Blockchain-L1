"""
Positronic Neural Engine - Neural Network Layers
=================================================

Layer classes for building neural networks. Each layer operates on Tensor
objects from the Positronic autograd engine and exposes a `parameters()`
method for optimizer integration and a `forward()` method for computation.

All learnable parameters are stored as Tensor objects with
`requires_grad=True`.
"""

import numpy as np
from positronic.ai.engine.tensor import Tensor


class Dense:
    """Fully connected (linear) layer: y = x @ W + b.

    Applies a linear transformation to the input. Weight initialization
    uses Xavier (Glorot) uniform scaling for stable gradient flow.

    Args:
        in_features:  Number of input features.
        out_features: Number of output features.
        bias:         If True, adds a learnable bias vector. Default True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_features), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: x @ weight + bias.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list:
        """Return list of learnable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self) -> str:
        return (
            f"Dense(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )


class LayerNorm:
    """Layer Normalization over the last dimension.

    Normalizes activations to zero mean and unit variance, then applies
    learnable affine parameters (gamma and beta).

    Reference: Ba, Kiros & Hinton, "Layer Normalization", 2016.

    Args:
        normalized_shape: Size of the last dimension to normalize over.
        eps:              Small constant for numerical stability. Default 1e-5.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., normalized_shape).

        Returns:
            Normalized tensor of the same shape.
        """
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x.data - mean) * std_inv

        out_data = x_norm * self.gamma.data + self.beta.data
        out = Tensor(out_data, _children=(x, self.gamma, self.beta))

        def _backward():
            n = self.normalized_shape

            if self.gamma.requires_grad:
                # dL/dgamma = sum over batch of (dL/dout * x_norm)
                gamma_grad = np.sum(
                    out.grad * x_norm,
                    axis=tuple(range(out.grad.ndim - 1)),
                )
                self.gamma.grad = self.gamma.grad + gamma_grad

            if self.beta.requires_grad:
                beta_grad = np.sum(
                    out.grad,
                    axis=tuple(range(out.grad.ndim - 1)),
                )
                self.beta.grad = self.beta.grad + beta_grad

            if x.requires_grad:
                # dL/dx through layer norm
                dout_scaled = out.grad * self.gamma.data
                dx_norm = dout_scaled
                dvar = np.sum(
                    dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5),
                    axis=-1,
                    keepdims=True,
                )
                dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + (
                    dvar * np.mean(-2.0 * (x.data - mean), axis=-1, keepdims=True)
                )
                x.grad = x.grad + dx_norm * std_inv + dvar * 2.0 * (x.data - mean) / n + dmean / n

        out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list:
        """Return list of learnable parameters."""
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


class BatchNorm:
    """Batch Normalization over the first (batch) dimension.

    During training, normalizes using batch statistics and updates running
    estimates. During evaluation, uses the running estimates.

    Reference: Ioffe & Szegedy, "Batch Normalization", 2015.

    Args:
        num_features: Number of features (channels).
        eps:          Small constant for numerical stability. Default 1e-5.
        momentum:     Factor for running mean/var update. Default 0.1.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not learnable, not part of parameters)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features) or
               (batch_size, num_features, ...).

        Returns:
            Normalized tensor of the same shape.
        """
        if self.training:
            # Compute reduction axes: all axes except the feature axis (axis=1)
            if x.data.ndim == 2:
                reduce_axes = (0,)
                keepdims_shape = (1, self.num_features)
            else:
                reduce_axes = tuple(i for i in range(x.data.ndim) if i != 1)
                keepdims_shape = [1] * x.data.ndim
                keepdims_shape[1] = self.num_features
                keepdims_shape = tuple(keepdims_shape)

            batch_mean = np.mean(x.data, axis=reduce_axes)
            batch_var = np.var(x.data, axis=reduce_axes)

            # Update running statistics
            self.running_mean = (
                (1.0 - self.momentum) * self.running_mean
                + self.momentum * batch_mean
            )
            self.running_var = (
                (1.0 - self.momentum) * self.running_var
                + self.momentum * batch_var
            )

            mean = batch_mean.reshape(keepdims_shape)
            var = batch_var.reshape(keepdims_shape)
        else:
            if x.data.ndim == 2:
                keepdims_shape = (1, self.num_features)
            else:
                keepdims_shape = [1] * x.data.ndim
                keepdims_shape[1] = self.num_features
                keepdims_shape = tuple(keepdims_shape)

            mean = self.running_mean.reshape(keepdims_shape)
            var = self.running_var.reshape(keepdims_shape)

        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x.data - mean) * std_inv

        gamma_shape = keepdims_shape
        out_data = x_norm * self.gamma.data.reshape(gamma_shape) + self.beta.data.reshape(gamma_shape)
        out = Tensor(out_data, _children=(x, self.gamma, self.beta))

        if x.data.ndim == 2:
            reduce_axes_bwd = (0,)
        else:
            reduce_axes_bwd = tuple(i for i in range(x.data.ndim) if i != 1)

        n = 1
        for ax in reduce_axes_bwd:
            n *= x.data.shape[ax]

        def _backward():
            if self.gamma.requires_grad:
                gamma_grad = np.sum(out.grad * x_norm, axis=reduce_axes_bwd)
                self.gamma.grad = self.gamma.grad + gamma_grad

            if self.beta.requires_grad:
                beta_grad = np.sum(out.grad, axis=reduce_axes_bwd)
                self.beta.grad = self.beta.grad + beta_grad

            if x.requires_grad:
                dout_scaled = out.grad * self.gamma.data.reshape(gamma_shape)
                dx_norm = dout_scaled

                dvar = np.sum(
                    dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5),
                    axis=reduce_axes_bwd,
                    keepdims=True,
                )
                dmean = np.sum(
                    dx_norm * -std_inv, axis=reduce_axes_bwd, keepdims=True
                ) + dvar * np.mean(
                    -2.0 * (x.data - mean), axis=reduce_axes_bwd, keepdims=True
                )
                x.grad = x.grad + dx_norm * std_inv + dvar * 2.0 * (x.data - mean) / n + dmean / n

        out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list:
        """Return list of learnable parameters."""
        return [self.gamma, self.beta]

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def __repr__(self) -> str:
        return (
            f"BatchNorm(num_features={self.num_features}, "
            f"eps={self.eps}, momentum={self.momentum})"
        )


class Dropout:
    """Dropout regularization layer.

    During training, randomly zeros elements with probability `p` and scales
    the remaining elements by 1/(1-p) to maintain expected values. During
    evaluation, passes input through unchanged.

    Reference: Srivastava et al., "Dropout", JMLR 2014.

    Args:
        p: Probability of dropping an element. Default 0.5.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with dropout applied (training) or unchanged (eval).
        """
        if not self.training or self.p == 0.0:
            return x

        mask = (np.random.rand(*x.data.shape) > self.p).astype(x.data.dtype)
        scale = 1.0 / (1.0 - self.p)
        out_data = x.data * mask * scale
        out = Tensor(out_data, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + out.grad * mask * scale

        out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list:
        """Dropout has no learnable parameters."""
        return []

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Embedding:
    """Embedding lookup table.

    Maps integer indices to dense vectors. This is commonly used to convert
    token IDs into continuous representations for NLP models.

    Args:
        num_embeddings: Size of the vocabulary (number of rows).
        embedding_dim:  Dimension of each embedding vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True,
        )

    def forward(self, indices) -> Tensor:
        """Forward pass: lookup embeddings by index.

        Args:
            indices: Integer array or Tensor of shape (...) containing indices
                     in [0, num_embeddings).

        Returns:
            Tensor of shape (..., embedding_dim).
        """
        if isinstance(indices, Tensor):
            idx = indices.data.astype(int)
        else:
            idx = np.asarray(indices, dtype=int)

        out_data = self.weight.data[idx]
        out = Tensor(out_data, _children=(self.weight,))

        def _backward():
            if self.weight.requires_grad:
                # Scatter-add gradients back to the weight matrix
                grad = np.zeros_like(self.weight.data)
                np.add.at(grad, idx, out.grad)
                self.weight.grad = self.weight.grad + grad

        out._backward = _backward
        return out

    def __call__(self, indices) -> Tensor:
        return self.forward(indices)

    def parameters(self) -> list:
        """Return list of learnable parameters."""
        return [self.weight]

    def __repr__(self) -> str:
        return (
            f"Embedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim})"
        )


class Conv1D:
    """1D Convolution layer for temporal/sequential data.

    Applies a 1D convolution over an input signal composed of several input
    channels. Uses explicit loop-based implementation with full autograd
    support.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size:  Size of the convolving kernel.
        stride:       Stride of the convolution. Default 1.
        padding:      Zero-padding added to both sides of the input. Default 0.
        bias:         If True, adds a learnable bias. Default True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size) * scale,
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_channels), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length).

        Returns:
            Output tensor of shape (batch, out_channels, out_length).
        """
        batch_size, in_ch, length = x.data.shape

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding)),
                mode="constant",
                constant_values=0,
            )
        else:
            x_padded = x.data

        padded_length = x_padded.shape[2]
        out_length = (padded_length - self.kernel_size) // self.stride + 1

        # Build column matrix using stride tricks for efficient convolution
        # Shape: (batch, in_channels * kernel_size, out_length)
        col = np.zeros((batch_size, in_ch * self.kernel_size, out_length))
        for i in range(out_length):
            start = i * self.stride
            end = start + self.kernel_size
            col[:, :, i] = x_padded[:, :, start:end].reshape(batch_size, -1)

        # weight reshaped to (out_channels, in_channels * kernel_size)
        w_col = self.weight.data.reshape(self.out_channels, -1)

        # Convolution as matrix multiply: (batch, out_channels, out_length)
        out_data = np.zeros((batch_size, self.out_channels, out_length))
        for b in range(batch_size):
            out_data[b] = w_col @ col[b]

        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, -1, 1)

        out = Tensor(out_data, _children=(x, self.weight) + ((self.bias,) if self.bias is not None else ()))

        stride = self.stride
        padding = self.padding
        kernel_size = self.kernel_size
        weight_data = self.weight.data
        out_channels = self.out_channels
        in_channels = self.in_channels

        def _backward():
            if self.weight.requires_grad:
                # dL/dW: correlate input with output gradient
                w_grad = np.zeros_like(self.weight.data)
                for b in range(batch_size):
                    # col[b] shape: (in_ch*kernel_size, out_length)
                    # out.grad[b] shape: (out_channels, out_length)
                    w_grad_b = out.grad[b] @ col[b].T  # (out_channels, in_ch*kernel_size)
                    w_grad += w_grad_b.reshape(self.weight.data.shape)
                self.weight.grad = self.weight.grad + w_grad

            if self.bias is not None and self.bias.requires_grad:
                # dL/db: sum over batch and spatial dims
                self.bias.grad = self.bias.grad + np.sum(out.grad, axis=(0, 2))

            if x.requires_grad:
                # dL/dx: full convolution of grad with flipped weights
                w_col_local = weight_data.reshape(out_channels, -1)
                dx_padded = np.zeros_like(x_padded)
                for b in range(batch_size):
                    # dcol: (in_ch*kernel_size, out_length)
                    dcol = w_col_local.T @ out.grad[b]
                    for i in range(out_length):
                        start = i * stride
                        end = start + kernel_size
                        dx_padded[b, :, start:end] += dcol[:, i].reshape(
                            in_channels, kernel_size
                        )

                if padding > 0:
                    x.grad = x.grad + dx_padded[:, :, padding:-padding]
                else:
                    x.grad = x.grad + dx_padded

        out._backward = _backward
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list:
        """Return list of learnable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self) -> str:
        return (
            f"Conv1D(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )


class LSTM:
    """Long Short-Term Memory (single layer).

    Processes sequential data using input, forget, cell, and output gates.
    Supports variable-length sequences and optional initial hidden/cell states.

    Reference: Hochreiter & Schmidhuber, "Long Short-Term Memory", 1997.

    Args:
        input_size:  Number of expected features in the input.
        hidden_size: Number of features in the hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined gate weights for efficiency
        # Gates: input (i), forget (f), cell candidate (g), output (o)
        scale_ih = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))

        self.W_ih = Tensor(
            np.random.randn(4 * hidden_size, input_size) * scale_ih,
            requires_grad=True,
        )
        self.W_hh = Tensor(
            np.random.randn(4 * hidden_size, hidden_size) * scale_hh,
            requires_grad=True,
        )
        self.bias_ih = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
        self.bias_hh = Tensor(np.zeros(4 * hidden_size), requires_grad=True)

        # Initialize forget gate bias to 1 for better gradient flow
        self.bias_ih.data[hidden_size : 2 * hidden_size] = 1.0

    def forward(self, x: Tensor, h0=None, c0=None):
        """Forward pass through the LSTM.

        Args:
            x:  Input tensor of shape (batch, seq_len, input_size).
            h0: Initial hidden state of shape (batch, hidden_size). Default zeros.
            c0: Initial cell state of shape (batch, hidden_size). Default zeros.

        Returns:
            Tuple of (output, (h_n, c_n)):
                output: Tensor of shape (batch, seq_len, hidden_size) - all hidden states.
                h_n:    Tensor of shape (batch, hidden_size) - final hidden state.
                c_n:    Tensor of shape (batch, hidden_size) - final cell state.
        """
        batch_size, seq_len, _ = x.data.shape
        H = self.hidden_size

        if h0 is not None:
            h_prev = h0.data.copy()
        else:
            h_prev = np.zeros((batch_size, H))

        if c0 is not None:
            c_prev = c0.data.copy()
        else:
            c_prev = np.zeros((batch_size, H))

        # Cache for backward pass
        cache_gates_i = []
        cache_gates_f = []
        cache_gates_g = []
        cache_gates_o = []
        cache_c = [c_prev.copy()]
        cache_h = [h_prev.copy()]
        cache_c_tanh = []

        outputs = np.zeros((batch_size, seq_len, H))

        W_ih = self.W_ih.data
        W_hh = self.W_hh.data
        b_ih = self.bias_ih.data
        b_hh = self.bias_hh.data

        for t in range(seq_len):
            x_t = x.data[:, t, :]  # (batch, input_size)

            # Combined gate computation
            gates = x_t @ W_ih.T + h_prev @ W_hh.T + b_ih + b_hh  # (batch, 4*H)

            # Split into four gates
            i_gate = self._sigmoid(gates[:, 0:H])
            f_gate = self._sigmoid(gates[:, H : 2 * H])
            g_gate = np.tanh(gates[:, 2 * H : 3 * H])
            o_gate = self._sigmoid(gates[:, 3 * H : 4 * H])

            c_new = f_gate * c_prev + i_gate * g_gate
            c_tanh = np.tanh(c_new)
            h_new = o_gate * c_tanh

            cache_gates_i.append(i_gate)
            cache_gates_f.append(f_gate)
            cache_gates_g.append(g_gate)
            cache_gates_o.append(o_gate)
            cache_c.append(c_new.copy())
            cache_h.append(h_new.copy())
            cache_c_tanh.append(c_tanh)

            outputs[:, t, :] = h_new
            h_prev = h_new
            c_prev = c_new

        out_tensor = Tensor(
            outputs,
            _children=(x, self.W_ih, self.W_hh, self.bias_ih, self.bias_hh),
        )
        h_n = Tensor(h_prev, _children=())
        c_n = Tensor(c_prev, _children=())

        def _backward():
            dh_next = np.zeros((batch_size, H))
            dc_next = np.zeros((batch_size, H))

            dW_ih = np.zeros_like(W_ih)
            dW_hh = np.zeros_like(W_hh)
            db = np.zeros(4 * H)

            dx_data = np.zeros_like(x.data)

            for t in reversed(range(seq_len)):
                dh = out_tensor.grad[:, t, :] + dh_next
                x_t = x.data[:, t, :]
                h_prev_t = cache_h[t]

                i_gate = cache_gates_i[t]
                f_gate = cache_gates_f[t]
                g_gate = cache_gates_g[t]
                o_gate = cache_gates_o[t]
                c_tanh = cache_c_tanh[t]
                c_prev_t = cache_c[t]

                # Gradients through output gate
                dc = dh * o_gate * (1.0 - c_tanh ** 2) + dc_next
                do = dh * c_tanh

                # Gate gradients (pre-activation)
                di = dc * g_gate
                df = dc * c_prev_t
                dg = dc * i_gate

                # Sigmoid / tanh derivatives
                di_raw = di * i_gate * (1.0 - i_gate)
                df_raw = df * f_gate * (1.0 - f_gate)
                dg_raw = dg * (1.0 - g_gate ** 2)
                do_raw = do * o_gate * (1.0 - o_gate)

                dgates = np.concatenate(
                    [di_raw, df_raw, dg_raw, do_raw], axis=1
                )  # (batch, 4*H)

                # Parameter gradients
                dW_ih += dgates.T @ x_t
                dW_hh += dgates.T @ h_prev_t
                db += np.sum(dgates, axis=0)

                # Input gradient
                if x.requires_grad:
                    dx_data[:, t, :] = dgates @ W_ih

                # Gradients to pass to previous timestep
                dh_next = dgates @ W_hh
                dc_next = dc * f_gate

            if x.requires_grad:
                x.grad = x.grad + dx_data

            if self.W_ih.requires_grad:
                self.W_ih.grad = self.W_ih.grad + dW_ih
            if self.W_hh.requires_grad:
                self.W_hh.grad = self.W_hh.grad + dW_hh
            if self.bias_ih.requires_grad:
                self.bias_ih.grad = self.bias_ih.grad + db
            if self.bias_hh.requires_grad:
                self.bias_hh.grad = self.bias_hh.grad + db

        out_tensor._backward = _backward
        return out_tensor, (h_n, c_n)

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def __call__(self, x: Tensor, h0=None, c0=None):
        return self.forward(x, h0, c0)

    def parameters(self) -> list:
        """Return list of learnable parameters."""
        return [self.W_ih, self.W_hh, self.bias_ih, self.bias_hh]

    def __repr__(self) -> str:
        return (
            f"LSTM(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size})"
        )


class MultiHeadAttention:
    """Multi-Head Scaled Dot-Product Attention.

    Computes attention over queries, keys, and values split into multiple
    heads for parallel processing. Supports optional masking for causal
    (autoregressive) decoding.

    Reference: Vaswani et al., "Attention Is All You Need", 2017.

    Args:
        d_model:   Total model dimension.
        num_heads: Number of attention heads. Must divide d_model evenly.
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Dense(d_model, d_model)
        self.W_k = Dense(d_model, d_model)
        self.W_v = Dense(d_model, d_model)
        self.W_o = Dense(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        """Forward pass.

        Args:
            query: Tensor of shape (batch, seq_q, d_model).
            key:   Tensor of shape (batch, seq_k, d_model).
            value: Tensor of shape (batch, seq_k, d_model).
            mask:  Optional mask array broadcastable to (batch, num_heads, seq_q, seq_k).
                   Positions with True or 1 will be masked (set to -inf before softmax).

        Returns:
            Output tensor of shape (batch, seq_q, d_model).
        """
        batch_size = query.data.shape[0]
        seq_q = query.data.shape[1]
        seq_k = key.data.shape[1]
        H = self.num_heads
        d_k = self.d_k

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)    # (batch, seq_k, d_model)
        V = self.W_v(value)  # (batch, seq_k, d_model)

        # Reshape and transpose to (batch, num_heads, seq, d_k)
        Q_data = Q.data.reshape(batch_size, seq_q, H, d_k).transpose(0, 2, 1, 3)
        K_data = K.data.reshape(batch_size, seq_k, H, d_k).transpose(0, 2, 1, 3)
        V_data = V.data.reshape(batch_size, seq_k, H, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = np.sqrt(d_k)
        scores = np.matmul(Q_data, K_data.transpose(0, 1, 3, 2)) / scale
        # scores shape: (batch, num_heads, seq_q, seq_k)

        if mask is not None:
            mask_arr = np.asarray(mask)
            scores = np.where(mask_arr, -1e9, scores)

        # Softmax over last axis
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Weighted sum of values
        attn_output = np.matmul(attn_weights, V_data)
        # attn_output shape: (batch, num_heads, seq_q, d_k)

        # Concatenate heads: (batch, seq_q, d_model)
        concat = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.d_model)

        # Create Tensor for concat so we can feed it to W_o
        concat_tensor = Tensor(concat, _children=(Q, K, V))

        # Final linear projection
        output = self.W_o(concat_tensor)

        # Attach backward for the attention mechanism itself
        # The Dense layers handle their own backward via the Tensor autograd.
        # We need to propagate gradients from concat_tensor back through
        # the attention computation to Q, K, V.
        original_concat_backward = concat_tensor._backward

        def _backward_attention():
            # concat_tensor.grad has been set by W_o's backward
            # Reshape back to multi-head form
            dconcat = concat_tensor.grad  # (batch, seq_q, d_model)
            dattn_output = dconcat.reshape(batch_size, seq_q, H, d_k).transpose(0, 2, 1, 3)

            # Gradient through attn_output = attn_weights @ V_data
            dattn_weights = np.matmul(dattn_output, V_data.transpose(0, 1, 3, 2))
            dV_data = np.matmul(attn_weights.transpose(0, 1, 3, 2), dattn_output)

            # Gradient through softmax
            # d(softmax)/d(scores) Jacobian-vector product
            ds = attn_weights * (dattn_weights - np.sum(dattn_weights * attn_weights, axis=-1, keepdims=True))

            if mask is not None:
                ds = np.where(np.asarray(mask), 0.0, ds)

            ds = ds / scale

            # Gradient through scores = Q @ K^T
            dQ_data = np.matmul(ds, K_data)  # (batch, H, seq_q, d_k)
            dK_data = np.matmul(ds.transpose(0, 1, 3, 2), Q_data)  # (batch, H, seq_k, d_k)

            # Reshape back to (batch, seq, d_model)
            dQ = dQ_data.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.d_model)
            dK = dK_data.transpose(0, 2, 1, 3).reshape(batch_size, seq_k, self.d_model)
            dV = dV_data.transpose(0, 2, 1, 3).reshape(batch_size, seq_k, self.d_model)

            if Q.requires_grad:
                Q.grad = Q.grad + dQ
            if K.requires_grad:
                K.grad = K.grad + dK
            if V.requires_grad:
                V.grad = V.grad + dV

        concat_tensor._backward = _backward_attention
        return output

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        return self.forward(query, key, value, mask)

    def parameters(self) -> list:
        """Return list of learnable parameters from all projection layers."""
        return (
            self.W_q.parameters()
            + self.W_k.parameters()
            + self.W_v.parameters()
            + self.W_o.parameters()
        )

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(d_model={self.d_model}, "
            f"num_heads={self.num_heads})"
        )
