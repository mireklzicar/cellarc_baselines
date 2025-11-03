# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transformer model."""

import dataclasses
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import positional_encodings as pos_encs_lib


class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""
  def __init__(self,
               output_size: int,
               embedding_dim: int = 64,
               num_layers: int = 5,
               num_heads: int = 8,
               num_hiddens_per_head: Optional[int] = None,
               dropout_prob: float = 0.1,
               emb_init_scale: float = 0.02,
               use_embeddings: bool = True,
               share_embeddings: bool = False,
               attention_window: Optional[int] = None,
               positional_encodings: pos_encs_lib.PositionalEncodings = pos_encs_lib.PositionalEncodings.SIN_COS,
               max_time: int = 10_000,
               positional_encodings_params: Optional[pos_encs_lib.PositionalEncodingsParams] = None,
               widening_factor: int = 4,
               causal_masking: bool = False):
    # The size of the model output (i.e., the output vocabulary size).
    self.output_size = output_size
    # The dimension of the first embedding.
    self.embedding_dim = embedding_dim
    # The number of multi-head attention layers.
    self.num_layers = num_layers
    # The number of heads per layer.
    self.num_heads = num_heads
    # The number of hidden neurons per head. If None, it is set to be equal to
    # `embedding_dim // num_heads`.
    self.num_hiddens_per_head = num_hiddens_per_head
    # The probability that each element is discarded by the dropout modules.
    self.dropout_prob = dropout_prob
    # The parameter initialization scale for the embeddings.
    self.emb_init_scale = emb_init_scale
    # Whether to use the embeddings rather than raw inputs.
    self.use_embeddings = use_embeddings
    # Whether to share embeddings between the Encoder and the Decoder.
    self.share_embeddings = share_embeddings
    # The size of the sliding attention window. See MultiHeadDotProductAttention.
    self.attention_window = attention_window
    # The positional encoding used with default sin/cos (Vaswani et al., 2017).
    self.positional_encodings = positional_encodings
    # The maximum size of the context (used by the posiitonal encodings).
    self.max_time = max_time
    # The parameters for the positional encodings, default sin/cos.
    if positional_encodings_params is None:
      self.positional_encodings_params = pos_encs_lib.SinCosParams()
    else:
      self.positional_encodings_params = positional_encodings_params
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    self.widening_factor = widening_factor
    # Add mask to make causal predictions.
    self.causal_masking = causal_masking

    # Set `num_hiddens_per_head` if it is `None`.
    if self.num_hiddens_per_head is None:
      self.num_hiddens_per_head = self.embedding_dim // self.num_heads


def layer_norm(x: torch.Tensor, num_features: int) -> torch.Tensor:
  layer_norm_fn = nn.LayerNorm(num_features)
  return layer_norm_fn(x)


def shift_right(x: torch.Tensor, output_size: int) -> torch.Tensor:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  x = torch.argmax(x, dim=-1)

  # Add a time dimension for the single-output case (i.e., `ndim == 1`).
  if x.ndim == 1:
    x = torch.unsqueeze(x, dim=1)

  padded = F.pad(
      x, (1, 0), mode='constant', value=output_size)

  return F.one_hot(padded[:, :-1], num_classes=output_size + 1).float()


def compute_sliding_window_mask(sequence_length: int,
                                attention_window: int) -> torch.Tensor:
  """Returns a k-diagonal mask for a sliding window.

  Args:
    sequence_length: The length of the sequence, which will determine the shape
      of the output.
    attention_window: The size of the sliding window.

  Returns:
    A symmetric matrix of shape (sequence_length, sequence_length),
    attention_window-diagonal, with ones on the diagonal and on all the
    upper/lower diagonals up to attention_window // 2.

  Raises:
    ValueError if attention_window is <= 0.
  """
  if attention_window <= 0:
    raise ValueError(
        f'The attention window should be > 0. Got {attention_window}.')

  if attention_window == 1:
    return torch.eye(sequence_length, sequence_length)

  attention_mask = torch.sum(
      torch.stack([
          torch.eye(sequence_length, sequence_length).roll(k, dims=1).int()
          for k in range(1, attention_window // 2 + 1)
      ]),
      dim=0)
  attention_mask = attention_mask + attention_mask.transpose(0, 1)
  attention_mask += torch.eye(sequence_length, sequence_length).int()
  return attention_mask


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      positional_encodings: pos_encs_lib.PositionalEncodings,
      positional_encodings_params: pos_encs_lib.PositionalEncodingsParams,
      attention_window: Optional[int] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      positional_encodings: Which positional encodings to use in the attention.
      positional_encodings_params: Parameters for the positional encodings.
      attention_window: Size of the attention sliding window. None means no
        sliding window is used (or equivalently, window=full_attention_length).
        We attend only on attention_window tokens around a given query token. We
        attend to tokens before AND after the query token. If attention_window
        is even, we use the value +1.
      name: Name of the module.
    """
    super().__init__()
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._positional_encodings = positional_encodings
    self._attention_window = attention_window
    self._positional_encodings_params = positional_encodings_params
    
    # Initialize linear layers for q, k, v projections
    num_hiddens = self._num_hiddens_per_head * self._num_heads
    self.q_proj = nn.Linear(num_hiddens, num_hiddens, bias=False)
    self.k_proj = nn.Linear(num_hiddens, num_hiddens, bias=False)
    self.v_proj = nn.Linear(num_hiddens, num_hiddens, bias=False)
    self.output_proj = nn.Linear(num_hiddens, num_hiddens, bias=False)

  def forward(
      self,
      inputs_q: torch.Tensor,
      inputs_kv: torch.Tensor,
      mask: Optional[torch.Tensor] = None,
      causal: bool = False,
  ) -> torch.Tensor:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = self.q_proj(inputs_q)
    k = self.k_proj(inputs_kv)
    v = self.v_proj(inputs_kv)
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding.
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = q.reshape(new_shape)
    k = k.reshape(new_shape)
    v = v.reshape(new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    if self._positional_encodings == pos_encs_lib.PositionalEncodings.RELATIVE:
      # We type hint the params to match the if statement, for pytype.
      self._positional_encodings_params: pos_encs_lib.RelativeParams
      attention = pos_encs_lib.compute_attention_with_relative_encodings(
          q, k, self._positional_encodings_params.max_time, causal=causal
      )
    else:
      if self._positional_encodings == pos_encs_lib.PositionalEncodings.ROTARY:
        q = pos_encs_lib.apply_rotary_encoding(
            q, position=torch.arange(q.shape[1])[None, :]
        )
        k = pos_encs_lib.apply_rotary_encoding(
            k, position=torch.arange(k.shape[1])[None, :]
        )
      attention = torch.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / np.sqrt(self._num_hiddens_per_head)

    # ALiBi encodings are not scaled with the 1 / sqrt(d_k) factor.
    if self._positional_encodings == pos_encs_lib.PositionalEncodings.ALIBI:
      attention += pos_encs_lib.compute_alibi_encodings_biases(
          attention.shape[1:]
      )

    if self._attention_window is not None:
      # We compute the sliding attention by just applying a mask on the values
      # that are outside our window.
      attention_mask = compute_sliding_window_mask(
          sequence_length, self._attention_window
      ).to(torch.bool)
      attention = torch.where(
          attention_mask, attention, torch.finfo(torch.float32).min
      )

    if mask is not None:
      if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
      attention = torch.where(mask, attention, torch.finfo(torch.float32).min)

    normalized_attention = F.softmax(attention, dim=-1)

    output = torch.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = output.reshape(batch_size, sequence_length, num_hiddens)
    return self.output_proj(output)


class TransformerEncoder(nn.Module):
  """Transformer Encoder (Vaswani et al., 2017)."""

  def __init__(
      self,
      config: TransformerConfig,
      shared_embeddings_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the transformer encoder.

    Args:
      config: The hyperparameters used in Transformer architectures.
      shared_embeddings_fn: Embedding function that is shared with the decoder.
      name: The name of the module.
    """
    super().__init__()
    self._config = config
    self._shared_embeddings_fn = shared_embeddings_fn
    
    # Initialize layers
    if config.use_embeddings and shared_embeddings_fn is None:
      # Use input_size for embeddings if available, otherwise fall back to output_size
      input_size = getattr(config, 'input_size', config.output_size)
      self.embeddings = nn.Linear(input_size, config.embedding_dim, bias=False)
      nn.init.trunc_normal_(self.embeddings.weight, std=config.emb_init_scale)
    
    self.layer_norm = nn.LayerNorm(config.embedding_dim)
    self.dropout = nn.Dropout(config.dropout_prob)
    
    # Build transformer layers
    self.layers = nn.ModuleList()
    for _ in range(config.num_layers):
      layer_dict = nn.ModuleDict({
        'attention': MultiHeadDotProductAttention(
          num_heads=config.num_heads,
          num_hiddens_per_head=config.num_hiddens_per_head,
          positional_encodings=config.positional_encodings,
          positional_encodings_params=config.positional_encodings_params,
          attention_window=config.attention_window,
        ),
        'attention_layer_norm': nn.LayerNorm(config.embedding_dim),
        'ffn': nn.Sequential(
          nn.Linear(config.embedding_dim, config.embedding_dim * config.widening_factor),
          nn.ReLU(),
          nn.Linear(config.embedding_dim * config.widening_factor, config.embedding_dim)
        ),
        'ffn_layer_norm': nn.LayerNorm(config.embedding_dim)
      })
      self.layers.append(layer_dict)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Returns the transformer encoder output, shape [B, T, E]."""
    if self._config.use_embeddings:
      if self._shared_embeddings_fn is not None:
        embeddings = self._shared_embeddings_fn(x)
      else:
        # Since `x` is one-hot encoded, using nn.Linear is equivalent to embedding lookup
        embeddings = self.embeddings(x)

      embeddings *= np.sqrt(self._config.embedding_dim)

    else:
      embeddings = x

    batch_size, sequence_length, embedding_size = embeddings.shape

    pos_enc_params = self._config.positional_encodings_params
    if (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.SIN_COS
    ):
      pos_encodings = pos_encs_lib.sinusoid_position_encoding(
          sequence_length=sequence_length,
          hidden_size=embedding_size,
          memory_length=0,
          max_timescale=pos_enc_params.max_time,
          min_timescale=2,
          clamp_length=0,
          causal=True,
      )
      # Convert to tensor if it's a numpy array, otherwise use as-is
      if isinstance(pos_encodings, np.ndarray):
        h = embeddings + torch.from_numpy(pos_encodings).to(embeddings.device)
      else:
        h = embeddings + pos_encodings.to(embeddings.device)
      h = self.dropout(h)
    else:
      h = embeddings

    # The causal mask is shared across heads.
    if self._config.causal_masking:
      causal_mask = torch.tril(
          torch.ones((batch_size, 1, sequence_length, sequence_length))
      )
    else:
      causal_mask = None

    for layer in self.layers:
      attention = layer['attention'](
          inputs_q=h,
          inputs_kv=h,
          mask=causal_mask,
          causal=self._config.causal_masking,
      )
      attention = self.dropout(attention)
      attention = layer['attention_layer_norm'](h + attention)

      # Position-wise feedforward network.
      ffn_output = layer['ffn'](attention)
      ffn_output = self.dropout(ffn_output)
      h = layer['ffn_layer_norm'](ffn_output + attention)
    return h


class TransformerDecoder(nn.Module):
  """Transformer Decoder (Vaswani et al., 2017)."""

  def __init__(
      self,
      config: TransformerConfig,
      shared_embeddings_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the Transformer decoder.

    Args:
      config: The hyperparameters used in Transformer architectures.
      shared_embeddings_fn: Embedding function that is shared with the encoder.
      name: The name of the module.
    """
    super().__init__()
    self._config = config
    self._shared_embeddings_fn = shared_embeddings_fn
    
    if self._config.use_embeddings and self._shared_embeddings_fn is None:
        self.embeddings = nn.Linear(self._config.output_size + 1, self._config.embedding_dim, bias=False)
        nn.init.trunc_normal_(self.embeddings.weight, std=self._config.emb_init_scale)


    self.dropout = nn.Dropout(self._config.dropout_prob)
    self.layers = nn.ModuleList()
    for _ in range(self._config.num_layers):
        self.layers.append(nn.ModuleDict({
            'self_attention': MultiHeadDotProductAttention(
                num_heads=self._config.num_heads,
                num_hiddens_per_head=self._config.num_hiddens_per_head,
                positional_encodings=self._config.positional_encodings,
                positional_encodings_params=self._config.positional_encodings_params,
                attention_window=self._config.attention_window,
            ),
            'self_attention_layer_norm': nn.LayerNorm(self._config.embedding_dim),
            'cross_attention': MultiHeadDotProductAttention(
                num_heads=self._config.num_heads,
                num_hiddens_per_head=self._config.num_hiddens_per_head,
                positional_encodings=self._config.positional_encodings,
                positional_encodings_params=self._config.positional_encodings_params,
                attention_window=self._config.attention_window,
            ),
            'cross_attention_layer_norm': nn.LayerNorm(self._config.embedding_dim),
            'ffn': nn.Sequential(
                nn.Linear(self._config.embedding_dim, self._config.embedding_dim * self._config.widening_factor),
                nn.ReLU(),
                nn.Linear(self._config.embedding_dim * self._config.widening_factor, self._config.embedding_dim),
            ),
            'ffn_layer_norm': nn.LayerNorm(self._config.embedding_dim),
        }))


  def forward(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Returns the transformer decoder output, shape [B, T_O, E].

    Args:
      encoded: The output of the encoder, shape [B, T_I, E].
      targets: The one-hot encoded target values, shape [B, T_O, 2].
    """
    targets = shift_right(targets, self._config.output_size)

    if self._config.use_embeddings:
      if self._shared_embeddings_fn is not None:
        output_embeddings = self._shared_embeddings_fn(targets)
      else:
        # Since `x` is one-hot encoded, using nn.Linear is equivalent to embedding lookup
        output_embeddings = self.embeddings(targets)

      output_embeddings *= np.sqrt(self._config.embedding_dim)

    else:
      output_embeddings = targets

    batch_size, output_sequence_length, embedding_size = output_embeddings.shape

    if (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.SIN_COS
    ):
      pos_encodings = pos_encs_lib.sinusoid_position_encoding(
          sequence_length=output_sequence_length,
          hidden_size=embedding_size,
          memory_length=0,
          max_timescale=self._config.positional_encodings_params.max_time,
          min_timescale=2,
          clamp_length=0,
          causal=True,
      )
      # Convert to tensor if it's a numpy array, otherwise use as-is
      if isinstance(pos_encodings, np.ndarray):
        h = output_embeddings + torch.from_numpy(pos_encodings).to(output_embeddings.device)
      else:
        h = output_embeddings + pos_encodings.to(output_embeddings.device)
      h = self.dropout(h)
    else:
      h = output_embeddings

    # The causal mask is shared across heads.
    causal_mask = torch.tril(
        torch.ones(
            (batch_size, 1, output_sequence_length, output_sequence_length))).to(h.device)

    for layer in self.layers:
      self_attention = layer['self_attention'](inputs_q=h, inputs_kv=h, mask=causal_mask, causal=True)
      self_attention = self.dropout(self_attention)
      self_attention = layer['self_attention_layer_norm'](h + self_attention)

      cross_attention = layer['cross_attention'](inputs_q=self_attention, inputs_kv=encoded, causal=True)
      cross_attention = self.dropout(cross_attention)
      cross_attention = layer['cross_attention_layer_norm'](self_attention + cross_attention)

      # Position-wise feedforward network.
      ffn_output = layer['ffn'](cross_attention)
      ffn_output = self.dropout(ffn_output)
      h = layer['ffn_layer_norm'](ffn_output + cross_attention)

    return h


class Transformer(nn.Module):
  """Transformer (Vaswani et al., 2017)."""

  def __init__(self, config: TransformerConfig, name: Optional[str] = None):
    """Initializes the Transformer.

    Args:
      config: The hyperparameters used in Transformer architectures.
      name: The name of the module.
    """
    super().__init__()
    shared_embeddings_fn = None

    if config.share_embeddings:
      shared_embeddings_fn = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
      nn.init.trunc_normal_(shared_embeddings_fn.weight, std=config.emb_init_scale)


    self._encoder = TransformerEncoder(config, shared_embeddings_fn)
    self._decoder = TransformerDecoder(config, shared_embeddings_fn)

  def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return self._decoder(self._encoder(inputs), targets)


def make_transformer_encoder(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    attention_window: Optional[int] = None,
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    return_all_outputs: bool = False,
    causal_masking: bool = False,
    input_size: Optional[int] = None,
) -> nn.Module:
  """Returns a transformer encoder model."""
  if positional_encodings is None:
    positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
    positional_encodings_params = pos_encs_lib.SinCosParams()
  elif positional_encodings_params is None:
    raise ValueError('No parameters for positional encodings are passed.')
  config = TransformerConfig(
      output_size=output_size,
      embedding_dim=embedding_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      num_hiddens_per_head=num_hiddens_per_head,
      dropout_prob=dropout_prob,
      emb_init_scale=emb_init_scale,
      use_embeddings=use_embeddings,
      share_embeddings=share_embeddings,
      attention_window=attention_window,
      positional_encodings=positional_encodings,
      positional_encodings_params=positional_encodings_params,
      widening_factor=widening_factor,
      causal_masking=causal_masking,
  )
  
  # Add input_size to config if provided
  if input_size is not None:
    config.input_size = input_size
  
  class TransformerEncoderWrapper(nn.Module):
      def __init__(self):
          super().__init__()
          self.transformer_encoder = TransformerEncoder(config)
          self.linear = nn.Linear(embedding_dim, output_size)

      def forward(self, inputs):
          output = self.transformer_encoder(inputs)
          if not return_all_outputs:
              output = output[:, -1, :]
          return self.linear(output)

  return TransformerEncoderWrapper()


def make_transformer(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    attention_window: Optional[int] = None,
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    return_all_outputs: bool = False,
    input_size: Optional[int] = None,
) -> nn.Module:
  """Returns a transformer model."""
  if positional_encodings is None:
    positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
    positional_encodings_params = pos_encs_lib.SinCosParams()
  elif positional_encodings_params is None:
    raise ValueError('No parameters for positional encodings are passed.')
  config = TransformerConfig(
      output_size=output_size,
      embedding_dim=embedding_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      num_hiddens_per_head=num_hiddens_per_head,
      dropout_prob=dropout_prob,
      emb_init_scale=emb_init_scale,
      use_embeddings=use_embeddings,
      share_embeddings=share_embeddings,
      attention_window=attention_window,
      positional_encodings=positional_encodings,
      positional_encodings_params=positional_encodings_params,
      widening_factor=widening_factor,
  )

  if input_size is not None:
    config.input_size = input_size

  class TransformerWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer(config)
            self.linear = nn.Linear(embedding_dim, output_size)

        def forward(self, inputs, targets):
            output = self.transformer(inputs, targets)
            if not return_all_outputs:
                output = output[:, -1, :]
            return self.linear(output)

  return TransformerWrapper()
