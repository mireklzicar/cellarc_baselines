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

"""Stack RNN core.

Following the paper from Joulin et al (2015):
https://arxiv.org/abs/1503.01007

The idea is to add a stack extension to a recurrent neural network to be able to
simulate a machine accepting context-free languages.
The stack is completely differentiable. The actions taken are probabilities
only and therefore no RL is required. The stack and state update are just linear
combinations of the last states and these probabilities.
"""

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# First element is the stacks, second is the hidden internal state.
_StackRnnState = tuple[torch.Tensor, torch.Tensor]

# Number of actions the stack-RNN can take, namely POP, PUSH and NO_OP.
_NUM_ACTIONS = 3


def _update_stack(stack: torch.Tensor, actions: torch.Tensor,
                  push_value: torch.Tensor) -> torch.Tensor:
  """Updates the stack values.

  We update the stack in  two steps.
  In the first step, we update the top of the stack, and essentially do:
    stack[0] = push_action * push_value
               + pop_action * stack[1]
               + noop_action * stack[0]

  Then, in the second step, we update the rest of the stack and we move the
  elements up and down, depending on the action executed:
  * If push_action were 1, then we'd be purely pushing a new element
     to the top of the stack, so we'd move all elements down by one.
  * Likewise, if pop_action were 1, we'd be purely taking an element
     off the top of the stack, so we'd move all elements up by one.
  * Finally, if noop_action were 1, we'd leave elements where they were.
  The update is therefore essentially:
    stack[i] = push_action * stack[i-1]
               + pop_action * stack[i+1]
               + noop_action * stack[i]

  Args:
    stack: The current stack, shape (batch_size, stack_size, stack_cell_size).
    actions: The array of probabilities of the actions, shape (batch_size, 3).
    push_value: The vector to push on the stack, if the push action probability
      is positive, shape (batch_size, stack_cell_size).

  Returns:
    The new stack, same shape as the input stack.
  """
  batch_size, stack_size, stack_cell_size = stack.shape

  # Tiling the actions to match the top of the stack.
  # Shape (batch_size, stack_cell_size, _NUM_ACTIONS)
  cell_tiled_stack_actions = actions.unsqueeze(1).expand(
      -1, stack_cell_size, -1)
  push_action = cell_tiled_stack_actions[..., 0]
  pop_action = cell_tiled_stack_actions[..., 1]
  pop_value = stack[..., 1, :]
  no_op_action = cell_tiled_stack_actions[..., 2]
  no_op_value = stack[..., 0, :]

  # Shape (batch_size, 1, stack_cell_size)
  top_new_stack = (
      push_action * push_value + pop_action * pop_value +
      no_op_action * no_op_value)
  top_new_stack = top_new_stack.unsqueeze(1)

  # Tiling the actions to match all of the stack except the top.
  # Shape (batch_size, stack_size,  stack_cell_size, _NUM_ACTIONS)
  stack_tiled_stack_actions = actions.unsqueeze(1).unsqueeze(2).expand(
      -1, stack_size - 1, stack_cell_size, -1)
  push_action = stack_tiled_stack_actions[..., 0]
  push_value = stack[..., :-1, :]
  pop_action = stack_tiled_stack_actions[..., 1]
  pop_extra_zeros = torch.zeros((batch_size, 1, stack_cell_size), device=stack.device)
  pop_value = torch.cat([stack[..., 2:, :], pop_extra_zeros], dim=1)
  no_op_action = stack_tiled_stack_actions[..., 2]
  no_op_value = stack[..., 1:, :]

  # Shape (batch_size, stack_size-1, stack_cell_size)
  rest_new_stack = (
      push_action * push_value + pop_action * pop_value +
      no_op_action * no_op_value)

  # Finally concatenate the new top with the new rest of the stack.
  return torch.cat([top_new_stack, rest_new_stack], dim=1)


class StackRNNCore(nn.Module):
  """Core for the stack RNN."""

  def __init__(
      self,
      stack_cell_size: int,
      stack_size: int = 30,
      n_stacks: int = 1,
      inner_core: str = "vanilla",
      hidden_size: int = 128,
      input_size: int = 10,
      name: Optional[str] = None,
      **inner_core_kwargs: Mapping[str, Any]
  ):
    """Initializes.

    Args:
      stack_cell_size: The dimension of the vectors we put in the stack.
      stack_size: The total number of vectors we can stack.
      n_stacks: Number of stacks to use in the network.
      inner_core: The inner RNN core builder.
      name: See base class.
      **inner_core_kwargs: The arguments to be passed to the inner RNN core
        builder.
    """
    super().__init__()
    rnn_input_size = input_size + n_stacks * stack_cell_size
    if inner_core == 'lstm':
        self._rnn_core = nn.LSTM(rnn_input_size, hidden_size, batch_first=True)
    else:
        self._rnn_core = nn.RNN(rnn_input_size, hidden_size, batch_first=True)
    self._stack_cell_size = stack_cell_size
    self._stack_size = stack_size
    self._n_stacks = n_stacks
    self._hidden_size = hidden_size
    self.push_proj = nn.Linear(hidden_size, n_stacks * stack_cell_size)
    self.action_proj = nn.Linear(hidden_size, n_stacks * _NUM_ACTIONS)

  def forward(
      self, inputs: torch.Tensor, prev_state: _StackRnnState
  ) -> tuple[torch.Tensor, _StackRnnState]:
    """Steps the stack RNN core.

    See base class docstring.

    Args:
      inputs: An input array of shape (batch_size, input_size). The time
        dimension is not included since it is an RNNCore, which is unrolled over
        the time dimension.
      prev_state: A _StackRnnState tuple, consisting of the previous stacks and
        the previous state of the inner core. Each stack has shape (batch_size,
        stack_size, stack_cell_size), such that `stack[n][0]` represents the top
        of the stack for the nth batch item, and `stack[n][-1]` the bottom of
        the stack. The stacks are just the concatenation of all these tensors.

    Returns:
      - output: An output array of shape (batch_size, output_size).
      - next_state: Same format as prev_state.
    """
    stacks, old_core_state = prev_state

    # The network can always read the top of the stack.
    batch_size = stacks.shape[0]
    top_stacks = stacks[:, :, 0, :]
    top_stacks = top_stacks.reshape(
        (batch_size, self._n_stacks * self._stack_cell_size))
    inputs = torch.cat([inputs, top_stacks], dim=-1)
    
    # Add sequence dimension for RNN
    inputs = inputs.unsqueeze(1)  # (batch_size, 1, input_size)
    
    new_core_output, new_core_state = self._rnn_core(inputs, old_core_state)
    new_core_output = new_core_output.squeeze(1)  # Remove sequence dimension
    
    push_values = self.push_proj(new_core_output)
    push_values = push_values.reshape(
        (batch_size, self._n_stacks, self._stack_cell_size))

    # Shape (batch_size, _NUM_ACTIONS)
    stack_actions = F.softmax(self.action_proj(new_core_output), dim=-1)
    stack_actions = stack_actions.reshape(
        (batch_size, self._n_stacks, _NUM_ACTIONS))

    # Update stacks for each stack index
    new_stacks = []
    for i in range(self._n_stacks):
        updated_stack = _update_stack(stacks[:, i], stack_actions[:, i], push_values[:, i])
        new_stacks.append(updated_stack)
    new_stacks = torch.stack(new_stacks, dim=1)
    
    return new_core_output, (new_stacks, new_core_state)

  def initial_state(self, batch_size: Optional[int]) -> _StackRnnState:
    """Returns the initial state of the core, a hidden state and an empty stack."""
    device = next(self.parameters()).device
    if isinstance(self._rnn_core, nn.LSTM):
      h0 = torch.zeros(1, batch_size, self._hidden_size, device=device)
      c0 = torch.zeros(1, batch_size, self._hidden_size, device=device)
      core_state = (h0, c0)
    else:  # RNN
      core_state = torch.zeros(1, batch_size, self._hidden_size, device=device)
    
    stacks = torch.zeros(
        (batch_size, self._n_stacks, self._stack_size, self._stack_cell_size),
        device=device,
    )
    return stacks, core_state
