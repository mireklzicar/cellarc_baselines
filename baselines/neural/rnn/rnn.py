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

"""Builders for RNN/LSTM cores."""

from typing import Any, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """PyTorch RNN model wrapper."""
    
    def __init__(self, output_size: int, rnn_core: Union[str, nn.Module],
                 return_all_outputs: bool = False, input_window: int = 1,
                 **rnn_kwargs: Any):
        super().__init__()
        self.output_size = output_size
        self.return_all_outputs = return_all_outputs
        self.input_window = input_window
        
        # Handle different RNN core types
        if isinstance(rnn_core, str):
            if rnn_core == "vanilla":
                hidden_size = rnn_kwargs.get('hidden_size', 256)
                self.rnn_core = nn.RNN(input_size=rnn_kwargs.get('input_size', 1),
                                      hidden_size=hidden_size, batch_first=True)
            elif rnn_core == "lstm":
                hidden_size = rnn_kwargs.get('hidden_size', 256)
                self.rnn_core = nn.LSTM(input_size=rnn_kwargs.get('input_size', 1),
                                       hidden_size=hidden_size, batch_first=True)
            else:
                raise ValueError(f"Unknown RNN core type: {rnn_core}")
        else:
            self.rnn_core = rnn_core
            
        # Output linear layer
        if hasattr(self.rnn_core, 'hidden_size'):
            self.output_layer = nn.Linear(self.rnn_core.hidden_size, output_size)
        else:
            # For custom cores, assume they provide hidden_size
            hidden_size = rnn_kwargs.get('hidden_size', 256)
            self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, input_length: int = 1) -> torch.Tensor:
        batch_size, seq_length, embed_size = x.shape
        
        # Handle input windowing
        if seq_length % self.input_window != 0:
            padding = self.input_window - seq_length % self.input_window
            x = F.pad(x, (0, 0, 0, padding))
        new_seq_length = x.shape[1]
        
        if self.input_window > 1:
            x = x.view(batch_size, new_seq_length // self.input_window,
                      self.input_window * embed_size)

        # Forward pass through RNN
        if hasattr(self.rnn_core, "initial_state"):
            # Handle custom RNN cores that need timestep-by-timestep processing
            try:
                initial_state = self.rnn_core.initial_state(batch_size, input_length)
            except TypeError:
                initial_state = self.rnn_core.initial_state(batch_size)
            
            # Manual unroll over time dimension (similar to hk.dynamic_unroll)
            outputs = []
            state = initial_state
            seq_len = x.shape[1]
            
            for t in range(seq_len):
                timestep_input = x[:, t, :]  # Shape: (batch, input_size)
                timestep_output, state = self.rnn_core(timestep_input, state)
                outputs.append(timestep_output)
            
            # Stack outputs to get (batch, seq_len, hidden_size)
            output = torch.stack(outputs, dim=1)
        else:
            # Handle standard PyTorch RNN cores
            output, _ = self.rnn_core(x)
            
        if self.input_window > 1:
            output = output.view(batch_size, new_seq_length, -1)

        if not self.return_all_outputs:
            output = output[:, -1, :]  # (batch, hidden_dim)
        
        # Apply ReLU and output layer
        output = F.relu(output)
        output = self.output_layer(output)
        return output


def make_rnn(
    output_size: int,
    rnn_core: Union[str, nn.Module] = "lstm",
    return_all_outputs: bool = False,
    input_window: int = 1,
    **rnn_kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Returns an RNN model function.

  Args:
    output_size: The output size of the model.
    rnn_core: The RNN core to use. Can be "vanilla", "lstm", or a custom module.
    return_all_outputs: Whether to return the whole sequence of outputs of the
      RNN, or just the last one.
    input_window: The number of tokens that are fed at once to the RNN.
    **rnn_kwargs: Kwargs to be passed to the RNN core.
  """
  
  model = RNNModel(output_size, rnn_core, return_all_outputs, input_window, **rnn_kwargs)
  
  def rnn_model_fn(x: torch.Tensor, input_length: int = 1) -> torch.Tensor:
      return model(x, input_length)
      
  # Attach the model instance for parameter access
  rnn_model_fn.model = model
  return rnn_model_fn
