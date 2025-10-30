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

"""Implements the Tape RNN."""

import abc
import functools
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# The first element is the memory, the second is the hidden internal state, and
# the third is the input length, necessary for adaptive actions.
_TapeRNNState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TapeRNNCore(nn.Module, abc.ABC):
  """Core for the tape RNN."""

  def __init__(
      self,
      memory_cell_size: int,
      memory_size: int = 30,
      n_tapes: int = 1,
      mlp_layers_size: Sequence[int] = (64, 64),
      inner_core: str = "vanilla",
      hidden_size: int = 128,
      input_size: int = 10,
      name: Optional[str] = None,
      **inner_core_kwargs: Any
  ):
    """Initializes.
    Args:
      memory_cell_size: The dimension of the vectors we put in memory.
      memory_size: The size of the tape, fixed value along the episode.
      n_tapes: Number of tapes to use. Default is 1.
      mlp_layers_size: Sizes for the inner MLP layers. Can be empty, in which
        case the MLP is a linear layer.
      inner_core: The inner RNN core builder.
      name: See base class.
      **inner_core_kwargs: The arguments to be passed to the inner RNN core
        builder.
    """
    super().__init__()
    rnn_input_size = input_size + n_tapes * memory_cell_size
    if inner_core == 'lstm':
        self._rnn_core = nn.LSTM(rnn_input_size, hidden_size, batch_first=True)
    else:
        self._rnn_core = nn.RNN(rnn_input_size, hidden_size, batch_first=True)
    
    self._mlp_layers_size = mlp_layers_size
    self._memory_cell_size = memory_cell_size
    self._memory_size = memory_size
    self._n_tapes = n_tapes
    self._hidden_size = hidden_size

    mlp = []
    current_size = hidden_size
    for size in mlp_layers_size:
        mlp.append(nn.Linear(current_size, size))
        mlp.append(nn.ReLU())
        current_size = size
    mlp.append(nn.Linear(current_size, n_tapes * memory_cell_size))
    self._readout_mlp = nn.Sequential(*mlp)
    self._action_linears = nn.ModuleList([
        nn.Linear(hidden_size, self.num_actions) for _ in range(n_tapes)
    ])


  @abc.abstractmethod
  def _tape_operations(
      self, eye_memory: torch.Tensor, input_length: int
  ) -> list[torch.Tensor]:
    """Returns a set of updated memory slots.
    An eye matrix is passed and corresponds to the positions of the memory
    slots. This method returns a matrix with the new positions associated with
    the actions. For instance, for a 'left' action, the new matrix will just be
    a roll(eye_memory, shift=-1). This is general enough to allow any
    permutation on the indexes.
    Args:
      eye_memory: An eye matrix of shape [memory_size, memory_size].
      input_length: The length of the input sequence. Can be useful for some
        operations.
    """

  @property
  @abc.abstractmethod
  def num_actions(self) -> int:
    """Returns the number of actions which can be taken on the tape."""

  def forward(
      self, inputs: torch.Tensor, prev_state: _TapeRNNState
  ) -> tuple[torch.Tensor, _TapeRNNState]:
    """Steps the tape RNN core."""
    memories, old_core_state, input_length = prev_state

    # The network can always read the value of the current cell.
    batch_size = memories.shape[0]
    current_memories = memories[:, :, 0, :]
    current_memories = current_memories.reshape(
        (batch_size, self._n_tapes * self._memory_cell_size))
    inputs = torch.cat([inputs, current_memories], dim=-1)
    
    inputs = inputs.unsqueeze(1)
    new_core_output, new_core_state = self._rnn_core(inputs, old_core_state)
    new_core_output = new_core_output.squeeze(1)

    write_values = self._readout_mlp(new_core_output)
    write_values = write_values.reshape(
        (batch_size, self._n_tapes, self._memory_cell_size))

    # Shape (batch_size, num_actions).
    actions = []
    for i in range(self._n_tapes):
      actions.append(
          F.softmax(self._action_linears[i](new_core_output), dim=-1))
    actions = torch.stack(actions, dim=1)

    new_memories = []
    for i in range(self._n_tapes):
        new_memories.append(self._update_memory(memories[:, i], actions[:, i], write_values[:, i], input_length[0].item()))
    new_memories = torch.stack(new_memories, dim=1)

    return new_core_output, (new_memories, new_core_state, input_length)

  def initial_state(self, batch_size: Optional[int],
                    input_length: int) -> _TapeRNNState:
    """Returns the initial state of the core."""
    # Get device from model parameters to ensure consistency
    device = next(self.parameters()).device
    
    if isinstance(self._rnn_core, nn.LSTM):
        h0 = torch.zeros(1, batch_size, self._hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self._hidden_size, device=device)
        core_state = (h0, c0)
    else:
        core_state = torch.zeros(1, batch_size, self._hidden_size, device=device)
    memories = torch.zeros(
        (batch_size, self._n_tapes, self._memory_size, self._memory_cell_size), device=device)
    return memories, core_state, torch.tensor([input_length], device=device)

  def _update_memory(self, memory: torch.Tensor, actions: torch.Tensor,
                     write_values: torch.Tensor, input_length: int) -> torch.Tensor:
    """Computes the new memory based on the `actions` and `write_values`.
    Args:
      memory: The current memory with shape `[batch_size, memory_size,
        memory_cell_size]`.
      actions: The action probabilities with shape `[batch_size, num_actions]`.
      write_values: The values added to the first memory entry with shape
        `[batch_size, memory_cell_size]`.
      input_length: The length of the input.
    Returns:
      The new memory with shape `[batch_size, memory_size]`.
    """
    _, memory_size, _ = memory.shape

    memory_with_write = torch.cat(
        [write_values.unsqueeze(1), memory[:, 1:]], dim=1)

    # Create eye_memory on the same device as memory
    eye_memory = torch.eye(memory_size, device=memory.device)
    operations = self._tape_operations(eye_memory, input_length)
    
    op_tensors = []
    for op in operations:
        op_tensors.append(torch.einsum('mM,bMc->bmc', op, memory_with_write))

    memory_operations = torch.stack(op_tensors)
    return torch.einsum('Abmc,bA->bmc', memory_operations, actions)


class TapeInputLengthJumpCore(TapeRNNCore):
  """A tape-RNN with extra jumps of the length of the input.
  5 possible actions:
    - write and stay
    - write and move one cell left
    - write and move one cell right
    - write and move input_length cells left
    - write and move input_length cells right
  """

  @property
  def num_actions(self) -> int:
    """Returns the number of actions of the tape."""
    return 5

  def _tape_operations(
      self, eye_memory: torch.Tensor, input_length: int
  ) -> list[torch.Tensor]:
    write_stay = eye_memory
    write_left = torch.roll(eye_memory, shifts=-1, dims=0)
    write_right = torch.roll(eye_memory, shifts=1, dims=0)
    write_jump_left = torch.roll(eye_memory, shifts=-input_length, dims=0)
    write_jump_right = torch.roll(eye_memory, shifts=input_length, dims=0)
    return [
        write_stay, write_left, write_right, write_jump_left, write_jump_right
    ]
