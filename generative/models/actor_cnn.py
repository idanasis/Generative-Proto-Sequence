import itertools
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_utils import get_activation_fn, create_linear_layer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class Actor(nn.Module):

    def __init__(self, n_input_channels: int,
                 n_output_channels: int,
                 observation_shape: tuple[int, int] = (8, 8),
                 noise: Optional[float] = None,
                 linear_layers: Optional[list[int]] = None,
                 linear_layers_activation: str = "leaky_relu",
                 use_batchnorm_linear_layers: bool = True,
                 num_heads: int = 5,
                 pe_embedding_dim: int = 128):
        if linear_layers is None:
            linear_layers = [512, 128, 32]

        super().__init__()

        self.observation_shape = observation_shape
        self.noise = noise
        self.activation_fn = get_activation_fn(linear_layers_activation)
        self.num_heads = num_heads

        self.embedding_dim = pe_embedding_dim
        self.use_position_encoding = pe_embedding_dim != -1
        
        if self.use_position_encoding:
            self.register_buffer("position_encoding", self._create_encoding())
        else:
            self.register_buffer("position_encoding", None)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            cnn_output_size = self.cnn(torch.as_tensor(torch.randn(1, n_input_channels, observation_shape[0], observation_shape[1])).float()).shape[1]
            pe_features_size = (2 * self.embedding_dim) if self.use_position_encoding else 0
            n_flatten = cnn_output_size + pe_features_size

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=use_batchnorm_linear_layers,
                                    activation_fn=self.activation_fn)

        linear_layers = [n_flatten] + linear_layers + [n_output_channels]

        self.linear_heads = nn.ModuleList([nn.Sequential(*list(itertools.chain.from_iterable(
            [self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
             range(len(linear_layers) - 1)]))) for _ in range(self.num_heads)])

        self.apply(self._init_weights)

    def _create_encoding(self):
        # Return None if position encoding is disabled
        if not self.use_position_encoding:
            return None
            
        height = self.observation_shape[0]
        width = self.observation_shape[1]
        embedding_dim = self.embedding_dim

        encoding = torch.zeros(height, width, embedding_dim)

        # Calculate the positional encodings for rows and columns
        position_row = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        position_col = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim // 2, 2).float() * (-math.log(10000.0) / (embedding_dim // 2))
        )

        # Positional encoding for rows
        pe_row = torch.zeros(height, embedding_dim // 2)
        pe_row[:, 0::2] = torch.sin(position_row * div_term)
        pe_row[:, 1::2] = torch.cos(position_row * div_term)

        # Positional encoding for columns
        pe_col = torch.zeros(width, embedding_dim // 2)
        pe_col[:, 0::2] = torch.sin(position_col * div_term)
        pe_col[:, 1::2] = torch.cos(position_col * div_term)

        # Combine row and column positional encodings
        for x in range(height):
            for y in range(width):
                encoding[x, y, :embedding_dim // 2] = pe_row[x]
                encoding[x, y, embedding_dim // 2:] = pe_col[y]

        return encoding

    def _create_agent_goal_pe_features(self, state, state_embedding):
        batch_size = state.shape[0]
        
        # If position encoding is disabled, return only the state embedding
        if not self.use_position_encoding:
            return state_embedding.view(batch_size, -1)
        
        # Ensure state is in the correct shape
        assert state.shape[1:] == (3, self.observation_shape[0], self.observation_shape[1]), "State should be of shape (batch_size, 3, 8, 8)"

        with torch.no_grad():
            # Reshape to (batch_size, H*W) for easier processing
            agent_channel = state[:, 0].reshape(batch_size, -1)  # Flatten spatial dimensions

            # Find positions using argmax
            agent_indices = (agent_channel == 10).float().argmax(dim=1)
            goal_mask = (agent_channel == 8).float()
            goal_exists = goal_mask.sum(dim=1) > 0
            goal_indices = goal_mask.argmax(dim=1)

            # Convert linear indices back to 2D coordinates
            H, W = self.observation_shape
            agent_row = agent_indices // W
            agent_col = agent_indices % W
            goal_row = goal_indices // W
            goal_col = goal_indices % W

            # Where no goal exists, use agent position
            goal_row = torch.where(goal_exists, goal_row, agent_row)
            goal_col = torch.where(goal_exists, goal_col, agent_col)

        # Retrieve position encodings for agent and goal - maintaining gradients
        agent_pe = self.position_encoding[agent_row, agent_col]
        goal_pe = self.position_encoding[goal_row, goal_col]

        # These operations need to maintain gradients for the network
        # Stack agent and goal position encodings as new features
        new_features = torch.cat([agent_pe, goal_pe], dim=1).float()

        # Return concatenated state embedding and new position encodings
        return torch.cat([state_embedding.view(batch_size, -1), new_features], dim=1)

    def forward(self, obs: torch.tensor):
        cnn_output = self.cnn(obs)
        x = self._create_agent_goal_pe_features(obs, cnn_output)

        head_outputs = torch.stack([head(x) for head in self.linear_heads], dim=1)

        return head_outputs

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight,
                                     a=0.1,
                                     mode='fan_in',
                                     nonlinearity='leaky_relu')
            if module.bias is not None:
                # Consistent with Conv2d, simply initialize to zero
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        return
