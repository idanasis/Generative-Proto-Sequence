import itertools
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from utils.torch_utils import get_activation_fn, create_linear_layer


class Critic(nn.Module):

    def __init__(self, n_input_channels: int,
                 observation_shape: tuple[int, int] = (8, 8),
                 action_seq_dim: int = 50,
                 linear_layers: Optional[list[int]] = None,
                 linear_layers_activation: str = "leaky_relu",
                 use_batchnorm_linear_layers: bool = True,
                 include_actor_embedding_in_critic_input: bool = False):
        if linear_layers is None:
            linear_layers = [512]

        super().__init__()

        self.include_actor_embedding_in_critic_input = include_actor_embedding_in_critic_input

        self.activation_fn = get_activation_fn(linear_layers_activation)

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
            n_flatten = self.cnn(torch.as_tensor(torch.randn(1, n_input_channels, observation_shape[0], observation_shape[1])).float()).shape[1]

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=use_batchnorm_linear_layers,
                                    activation_fn=self.activation_fn)

        linear_layers = [n_flatten + action_seq_dim] + linear_layers + [1]
        layers = list(itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                  range(len(linear_layers) - 1)]))

        self.linear = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def forward(self, obs: torch.tensor, actor_emb: torch.tensor, action_sequence: torch.tensor):
        if self.include_actor_embedding_in_critic_input:
            x = torch.cat((self.cnn(obs), actor_emb.view(-1, actor_emb.size(2)), action_sequence), 1)
        else:
            x = torch.cat((self.cnn(obs), action_sequence), 1)

        return self.linear(x)

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
