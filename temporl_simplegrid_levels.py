import itertools
import os
import random
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from wandb.sdk.wandb_run import Run

from gps_simplegrid_levels import set_seed, env_setup, get_device, dataset_setup, SaveModelStrategy
from gym_simplegrid.envs.simple_grid_levels import (
    RewardStrategy,
    ObservationEncodingStrategy,
    PartialObservabilityStrategy,
)
from simplegrid_dataset import PositionXY
from utils.torch_utils import get_activation_fn, create_linear_layer


# Helper functions
def hard_update(target_network, source_network):
    """Hard update target network parameters from source network."""
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update(target_network, source_network, tau):
    """Soft update target network parameters from source network."""
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def tt(np_array):
    """Convert numpy array to PyTorch tensor on the correct device."""
    if not isinstance(np_array, torch.Tensor):
        return torch.tensor(np_array, dtype=torch.float32, device=device)
    else:
        return np_array.to(device)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)


# Custom replay buffers for TempoRL
class NoneConcatSkipReplayBuffer:
    """
    Replay buffer for the skip actions in TempoRL.
    Stores experiences with skip actions and behaviors.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def add_transition(self, state, action, next_state, reward, done, length, behavior):
        """Add a new experience to memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done, length, behavior)
        self.position = (self.position + 1) % self.capacity

    def random_next_batch(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, next_states, rewards, dones, lengths, behaviors = map(np.array, zip(*batch))

        return (
            tt(states),
            tt(actions),
            tt(next_states),
            tt(rewards),
            tt(dones),
            tt(lengths),
            tt(behaviors)
        )


# Network Architectures for TempoRL
class Q(nn.Module):
    """Q-Network for action selection."""

    def __init__(self, state_dim, action_dim,
                 linear_layers, linear_layers_activation):
        if linear_layers is None:
            linear_layers = [512, 128, 32]
        self.activation_fn = get_activation_fn(linear_layers_activation)

        super(Q, self).__init__()

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.activation_fn)

        self.layers = [state_dim] + linear_layers + [action_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        self.model = nn.Sequential(*layers)
        # input_dim = state_dim

        # for hidden_dim in hidden_layers:
        #     self.layers.append(nn.Linear(input_dim, hidden_dim))
        #     self.layers.append(nn.LeakyReLU())
        #     input_dim = hidden_dim
        #
        # self.layers.append(nn.Linear(input_dim, action_dim))
        # self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class TQ(nn.Module):
    """Q-Network for skip selection."""

    def __init__(self, state_dim, skip_dim, linear_layers, linear_layers_activation):
        if linear_layers is None:
            linear_layers = [512, 128, 32]
        self.activation_fn = get_activation_fn(linear_layers_activation)

        super(TQ, self).__init__()

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.activation_fn)

        # +1 for behavior action
        self.skip_action_embed = nn.Linear(1, 10)

        self.layers = [state_dim + 10] + linear_layers + [skip_dim]
        layers = list(
            itertools.chain.from_iterable(
                [self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                 range(len(linear_layers) - 1)]))
        self.model = nn.Sequential(*layers)
        # self.layers = []
        # # +1 for behavior action
        # input_dim = state_dim + 1
        #
        # for hidden_dim in hidden_layers:
        #     self.layers.append(nn.Linear(input_dim, hidden_dim))
        #     self.layers.append(nn.LeakyReLU())
        #     input_dim = hidden_dim
        #
        # self.layers.append(nn.Linear(input_dim, skip_dim))
        # self.model = nn.Sequential(*self.layers)

    def forward(self, x, behavior):
        # Concatenate state and behavior
        x_combined = torch.cat([x, self.skip_action_embed(behavior)], dim=1)
        return self.model(x_combined)


class WeightSharingTQ(nn.Module):
    """
    Q-Network with weight sharing between action selection and skip selection.
    """

    def __init__(self, state_dim, action_dim, skip_dim,
                 action_linear_layers, action_linear_layers_activation,
                 skip_linear_layers, skip_linear_layers_activation
                 ):
        if action_linear_layers is None:
            action_linear_layers = [512, 128, 32]
        self.action_activation_fn = get_activation_fn(action_linear_layers_activation)
        if skip_linear_layers is None:
            skip_linear_layers = [512, 128, 32]
        self.skip_activation_fn = get_activation_fn(skip_linear_layers_activation)

        super(WeightSharingTQ, self).__init__()

        # Shared feature extractor
        self.feature_layers = []
        input_dim = state_dim

        hidden_layers = [256, 128]
        for hidden_dim in hidden_layers:
            self.feature_layers.append(nn.Linear(input_dim, hidden_dim))
            self.feature_layers.append(nn.LeakyReLU())
            input_dim = hidden_dim

        self.features = nn.Sequential(*self.feature_layers)

        # Separate heads for action and skip
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.action_activation_fn)
        linear_layers = [input_dim] + action_linear_layers + [action_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        # self.action_head = nn.Linear(input_dim, action_dim)
        self.action_head = nn.Sequential(*layers)

        # Skip head takes features + behavior action
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.skip_activation_fn)

        self.skip_action_embed = nn.Linear(1, 10)

        # self.skip_head_input = nn.Linear(input_dim + 10, input_dim)
        linear_layers = [input_dim + 10] + skip_linear_layers + [skip_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        self.skip_head = nn.Sequential(*layers)
        # self.skip_head = nn.Linear(input_dim, skip_dim)

    def forward(self, x):
        features = self.features(x)
        return self.action_head(features)

    def forward_skip(self, x, behavior):
        features = self.features(x)
        # Concatenate features and behavior
        skip_input = torch.cat([features, self.skip_action_embed(behavior)], dim=1)
        # skip_features = F.leaky_relu(self.skip_head_input(skip_input))
        return self.skip_head(skip_input)


# Vision networks for handling image observations
class NatureDQN(nn.Module):
    """DQN with convolutional layers for image observations."""

    def __init__(self, state_dim, action_dim,
                 linear_layers, linear_layers_activation):
        if linear_layers is None:
            linear_layers = [512, 128, 32]
        self.activation_fn = get_activation_fn(linear_layers_activation)

        super(NatureDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size of conv layers
        conv_output_size = self._get_conv_output_size(state_dim)

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.activation_fn)

        # Separate heads for action and skip
        linear_layers = [conv_output_size] + linear_layers + [action_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        self.fc = nn.Sequential(*layers)

    def _get_conv_output_size(self, shape):
        # Forward pass with dummy input to determine conv output size
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        conv_out = self.conv(x)
        return self.fc(conv_out)


class NatureTQN(nn.Module):
    """TQN with convolutional layers for image observations."""

    def __init__(self, state_dim, skip_dim,
                 linear_layers, linear_layers_activation):
        if linear_layers is None:
            linear_layers = [512, 128, 32]
        self.activation_fn = get_activation_fn(linear_layers_activation)

        super(NatureTQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size of conv layers
        conv_output_size = self._get_conv_output_size(state_dim)

        self.skip_action_embed = nn.Linear(1, 10)

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.activation_fn)

        linear_layers = [conv_output_size + 10] + linear_layers + [skip_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        self.fc = nn.Sequential(*layers)

    def _get_conv_output_size(self, shape):
        # Forward pass with dummy input to determine conv output size
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, behavior):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        conv_out = self.conv(x)
        # Concatenate conv features with behavior
        combined = torch.cat([conv_out, self.skip_action_embed(behavior)], dim=1)
        return self.fc(combined)


class NatureWeightsharingTQN(nn.Module):
    """Weight sharing TQN with convolutional layers for image observations."""

    def __init__(self, state_dim, action_dim, skip_dim,
                 action_linear_layers, action_linear_layers_activation,
                 skip_linear_layers, skip_linear_layers_activation):
        if action_linear_layers is None:
            action_linear_layers = [512, 128, 32]
        self.action_activation_fn = get_activation_fn(action_linear_layers_activation)
        if skip_linear_layers is None:
            skip_linear_layers = [512, 128, 32]
        self.skip_activation_fn = get_activation_fn(skip_linear_layers_activation)

        super(NatureWeightsharingTQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size of conv layers
        self.conv_output_size = self._get_conv_output_size(state_dim)

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU()
        )

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.action_activation_fn)

        # Separate heads for action and skip
        linear_layers = [512] + action_linear_layers + [action_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        self.action_head = nn.Sequential(*layers)
        # self.action_head = nn.Linear(512, action_dim)

        # Skip head takes features + behavior action
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.skip_activation_fn)

        self.skip_action_embed = nn.Linear(1, 10)

        linear_layers = [512 + 10] + skip_linear_layers + [skip_dim]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))
        self.skip_head = nn.Sequential(*layers)

    def _get_conv_output_size(self, shape):
        # Forward pass with dummy input to determine conv output size
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        conv_out = self.conv(x)
        features = self.features(conv_out)
        return self.action_head(features)

    def forward_skip(self, x, behavior):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        conv_out = self.conv(x)
        features = self.features(conv_out)
        # Concatenate features and behavior
        skip_input = torch.cat([features, self.skip_action_embed(behavior)], dim=1)
        # skip_features = F.relu(self.skip_head_input(skip_input))
        # return self.skip_head(skip_features)
        return self.skip_head(skip_input)


@dataclass
class Config(object):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 123
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    mps: bool = True
    """if toggled, mps will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "temporl-grid-experiments"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    save_model_strategy: SaveModelStrategy = SaveModelStrategy.SUCCESS_RATE
    """Specifies the criteria for saving the best model"""
    val_eval_freq: int = 5000
    """the frequency of evaluation during training on validation environments"""
    train_eval_freq: int = 5000
    """the frequency of evaluation during training on training environments"""
    eval_test_dataset_during_training_freq: int = 100000
    """the frequency of evaluation during training on test environments"""
    train_dataset_size: Optional[int] = 100
    """the size of the training dataset"""
    train_eval_dataset_size: Optional[int] = 100
    """the size of the training dataset for evaluation"""
    val_dataset_size: Optional[int] = 100
    """the size of the val dataset"""
    test_dataset_size: Optional[int] = 1000
    """the size of the test dataset"""
    slurm_job_id: Optional[int] = None
    """the experiment's slurm job id"""

    # Env specific arguments
    env_id: str = "SimpleGrid-v0"
    """the id of the environment"""
    max_episode_steps: int = 75
    """the environment episode maximum number steps"""
    reward_strategy: RewardStrategy = RewardStrategy.NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES
    """specifies the reward strategy for the environment"""
    observation_encoding_strategy: ObservationEncodingStrategy = ObservationEncodingStrategy.DEFAULT
    """specifies the observation encoding strategy for the environment"""
    partial_observability_strategy: PartialObservabilityStrategy = PartialObservabilityStrategy.FULL
    """specifies the partial observability strategy for the environment"""
    view_radius: int = 3
    """radius of the agent's view when using partial observability (LOCAL_VIEW strategy)"""
    is_slippery: bool = False
    """if True, actions have stochastic outcomes with perpendicular movement probabilities"""
    slippery_prob: float = 1/3
    """probability of executing intended action when is_slippery=True (remaining probability split equally between perpendicular directions)"""
    sticky_action_prob: float = 0.0
    """probability of repeating the previous action instead of executing the intended action"""
    random_action_prob: float = 0.0
    """probability of selecting a uniformly random action instead of executing the intended action"""
    obstacle_map: str = "8x8_empty"
    """the obstacle map of the environment"""
    max_level: int = 14
    """the environment level of the dataset's difficulty"""
    start_level: int = 1
    """the environment start level of the dataset's difficulty"""

    # TempoRL specific arguments
    agent_type: str = "tdqn"
    """which agent to train: dqn, dar, tdqn, t-dqn"""
    weight_sharing: bool = True
    """whether to use weight sharing between action and skip networks"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    skip_dim: int = 10
    """maximum skip size (number of times to repeat action)"""
    buffer_size: int = 50000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.01
    """the target network update rate"""
    target_network_frequency: int = 100
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 2
    """the frequency of training"""
    grad_clip_val: float = 40.0
    """gradient clipping value"""
    action_linear_layers: list[int] = field(default_factory=lambda: [128, 32])
    """the hidden layer sizes for the network's linear layers"""
    action_linear_layers_activation_function: str = "leaky_relu"
    """activation function to use in the linear layers"""
    skip_linear_layers: list[int] = field(default_factory=lambda: [128, 32])
    """the hidden layer sizes for the network's linear layers"""
    skip_linear_layers_activation_function: str = "leaky_relu"
    """activation function to use in the linear layers"""

class TempoRLDQN:
    """
    TempoRL DQN agent capable of handling more complex state inputs through use of contextualized behavior actions.
    """

    def __init__(self, cfg, state_dim, action_dim, device):
        """
        Initialize the TempoRL DQN Agent
        """
        self.cfg = cfg
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skip_dim = cfg.skip_dim
        self._gamma = cfg.gamma
        # self._env = env
        # self._eval_env = eval_env

        # Determine if using vision-based (image) observations
        self.vision = isinstance(state_dim, tuple) and len(state_dim) == 3

        # Initialize networks based on agent type and vision flag
        if self.vision:
            if cfg.weight_sharing:
                self._q = NatureWeightsharingTQN(state_dim, action_dim, cfg.skip_dim,
                                                 cfg.action_linear_layers, cfg.action_linear_layers_activation_function,
                                                 cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
                self._q_target = NatureWeightsharingTQN(state_dim, action_dim, cfg.skip_dim,
                                                        cfg.action_linear_layers, cfg.action_linear_layers_activation_function,
                                                        cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
            else:
                self._q = NatureDQN(state_dim, action_dim, cfg.action_linear_layers, cfg.action_linear_layers_activation_function).to(device)
                self._q_target = NatureDQN(state_dim, action_dim, cfg.action_linear_layers, cfg.action_linear_layers_activation_function).to(device)
        else:
            if cfg.weight_sharing:
                self._q = WeightSharingTQ(state_dim, action_dim, cfg.skip_dim,
                                          cfg.action_linear_layers, cfg.action_linear_layers_activation_function,
                                          cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
                self._q_target = WeightSharingTQ(state_dim, action_dim, cfg.skip_dim,
                                                 cfg.action_linear_layers, cfg.action_linear_layers_activation_function,
                                                 cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
            else:
                self._q = Q(state_dim, action_dim, cfg.action_linear_layers, cfg.action_linear_layers_activation_function).to(device)
                self._q_target = Q(state_dim, action_dim, cfg.action_linear_layers, cfg.action_linear_layers_activation_function).to(device)

        # Initialize skip networks if not using weight sharing
        if cfg.weight_sharing:
            self._skip_q = self._q
            self._skip_q_target = self._q_target
        else:
            if not self.vision:
                self._skip_q = TQ(state_dim, cfg.skip_dim, cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
                self._skip_q_target = TQ(state_dim, cfg.skip_dim, cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
            else:
                self._skip_q = NatureTQN(state_dim, cfg.skip_dim, cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)
                self._skip_q_target = NatureTQN(state_dim, cfg.skip_dim, cfg.skip_linear_layers, cfg.skip_linear_layers_activation_function).to(device)

        print(f'Using {str(self._q)} as Q')
        print(f'Using {str(self._skip_q)} as skip-Q\n{"#" * 80}')

        # Copy weights to target networks
        hard_update(self._q_target, self._q)
        if not cfg.weight_sharing:
            hard_update(self._skip_q_target, self._skip_q)

        # Initialize optimizers
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)

        if cfg.weight_sharing:
            self._skip_q_optimizer = self._q_optimizer
        else:
            self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999),
                                                eps=1e-08)

        self._replay_buffer = ReplayBuffer(cfg.buffer_size)
        self._skip_replay_buffer = NoneConcatSkipReplayBuffer(cfg.buffer_size)

        # Initialize loss functions
        self._loss_function = nn.SmoothL1Loss()  # Huber loss
        self._skip_loss_function = nn.SmoothL1Loss()

    def get_action(self, x, epsilon):
        """Get action epsilon-greedy based on observation x"""
        if isinstance(x, np.ndarray):
            x = tt(x if len(x.shape) > 1 else x[None, :])

        if random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self._q(x)
                return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def get_skip(self, x, a, epsilon):
        """Get skip value epsilon-greedy based on observation x and behavior action a"""
        if isinstance(x, np.ndarray):
            x = tt(x if len(x.shape) > 1 else x[None, :])

        if isinstance(a, np.ndarray):
            a = tt(a if len(a.shape) > 1 else a[None, :])

        if random.random() < epsilon:
            return np.random.randint(self.skip_dim)
        else:
            with torch.no_grad():
                if self.cfg.weight_sharing:
                    skip_values = self._skip_q.forward_skip(x, a)
                else:
                    skip_values = self._skip_q(x, a)
                return torch.argmax(skip_values, dim=1).cpu().numpy()[0]

    def update_networks(self, global_step):
        """Update networks with samples from replay buffers"""
        batch_size = self.cfg.batch_size

        # Skip Q update based on double DQN where target is behavior Q
        skip_loss = torch.tensor(0.0)
        if len(self._skip_replay_buffer.memory) >= batch_size:
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags, batch_lengths, batch_behaviors = \
                self._skip_replay_buffer.random_next_batch(batch_size)

            # Compute target skip values
            with torch.no_grad():
                if self.cfg.weight_sharing:
                    next_q_values = self._q_target(batch_next_states)
                    next_action_indices = torch.argmax(self._q(batch_next_states), dim=1)
                    next_q_values = next_q_values.gather(1, next_action_indices.unsqueeze(1)).squeeze(1)
                else:
                    next_q_values = self._q_target(batch_next_states)
                    next_action_indices = torch.argmax(self._q(batch_next_states), dim=1)
                    next_q_values = next_q_values.gather(1, next_action_indices.unsqueeze(1)).squeeze(1)

                target = batch_rewards + (1 - batch_terminal_flags) * torch.pow(self._gamma,
                                                                                batch_lengths) * next_q_values

            # Compute current prediction
            if self.cfg.weight_sharing:
                current_prediction = self._skip_q.forward_skip(batch_states, batch_behaviors)
            else:
                current_prediction = self._skip_q(batch_states, batch_behaviors)

            current_prediction = current_prediction.gather(1, batch_actions.long().unsqueeze(1)).squeeze(1)
            skip_loss = self._skip_loss_function(current_prediction, target)

            # Update skip Q network
            self._skip_q_optimizer.zero_grad()
            skip_loss.backward()

            # Clip gradients
            for param in self._skip_q.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-self.cfg.grad_clip_val, self.cfg.grad_clip_val)

            self._skip_q_optimizer.step()

        # Action Q update based on double DQN
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = self._replay_buffer.random_next_batch(batch_size)

        # Compute target action values
        with torch.no_grad():
            next_q_values = self._q_target(batch_next_states)
            next_action_indices = torch.argmax(self._q(batch_next_states), dim=1)
            next_q_values = next_q_values.gather(1, next_action_indices.unsqueeze(1)).squeeze(1)

            target = batch_rewards.flatten() + self._gamma * next_q_values * (1 - batch_terminal_flags.flatten())

        # Compute current prediction
        current_q_values = self._q(batch_states)
        current_prediction = current_q_values.gather(1, batch_actions.to(torch.int64)).squeeze()

        action_loss = self._loss_function(current_prediction, target)

        # Update action Q network
        self._q_optimizer.zero_grad()
        action_loss.backward()

        # Clip gradients
        for param in self._q.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.cfg.grad_clip_val, self.cfg.grad_clip_val)

        self._q_optimizer.step()

        # Update target networks
        if global_step % self.cfg.target_network_frequency == 0:
            if self.cfg.tau >= 1.0:
                # Hard update
                hard_update(self._q_target, self._q)
                if not self.cfg.weight_sharing:
                    hard_update(self._skip_q_target, self._skip_q)
            else:
                # Soft update
                soft_update(self._q_target, self._q, self.cfg.tau)
                if not self.cfg.weight_sharing:
                    soft_update(self._skip_q_target, self._skip_q, self.cfg.tau)

        return action_loss.item(), skip_loss.item()

    def save_model(self, path):
        """Save models to the given path"""
        os.makedirs(path, exist_ok=True)
        torch.save(self._q.state_dict(), os.path.join(path, 'Q.pt'))
        if not self.cfg.weight_sharing:
            torch.save(self._skip_q.state_dict(), os.path.join(path, 'TQ.pt'))
        print(f"Models saved to {path}")

    def load_model(self, path):
        """Load models from the given path"""
        self._q.load_state_dict(torch.load(os.path.join(path, 'Q.pt'), map_location=self.device))
        hard_update(self._q_target, self._q)

        if not self.cfg.weight_sharing:
            self._skip_q.load_state_dict(torch.load(os.path.join(path, 'TQ.pt'), map_location=self.device))
            hard_update(self._skip_q_target, self._skip_q)

        print(f"Models loaded from {path}")


@torch.no_grad()
def eval_temporl_model(
        cfg: Config,
        dataset: list[tuple[PositionXY, PositionXY, int]],
        run_name: str,
        agent: TempoRLDQN,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.05,
        capture_video: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[float]], list[float]]:
    """Evaluate TempoRL agent on the given dataset"""
    envs = env_setup(cfg, run_name, capture_video=capture_video)
    envs.envs[0].unwrapped.start_goal_dataset = dataset

    eval_episodes = len(dataset)

    # Set agent to evaluation mode
    agent._q.eval()
    agent._skip_q.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_successes = []
    episodic_per = []
    episodic_sgf = []
    episodic_asl = []
    current_episode_inference_times = []  # Track inference times for current episode
    successful_episode_inference_times = []  # Track inference times per successful episode
    successful_episode_wall_clock_times_per_optimal_step = []  # Track wall-clock time per optimal step
    
    # Track episode timing
    episode_start_time = time.perf_counter()

    episode_count = 0
    done = False
    episode_reward = 0
    steps_taken = 0
    sgf_count = 0
    asl = 0

    while episode_count < eval_episodes:
        optimal_episode_length = envs.envs[0].unwrapped.level

        # Get action and skip using epsilon-greedy
        # Measure inference time
        start_time = time.perf_counter()
        action = agent.get_action(obs, epsilon)
        skip = agent.get_skip(obs, np.array([action]), epsilon)
        end_time = time.perf_counter()
        current_episode_inference_times.append(end_time - start_time)
        
        sgf_count += 1
        asl += skip

        # Execute the action for (skip + 1) steps
        for _ in range(skip + 1):
            next_obs, reward, termination, truncation, info = envs.step(np.array([action]))
            episode_reward += reward[0]
            steps_taken += 1

            done = termination[0] or truncation[0]
            if done:
                break

        if done:
            # Record episode statistics
            avg_per = np.round(optimal_episode_length / steps_taken, 2)
            is_successful = termination[0]
            
            # Calculate episode wall-clock time
            episode_end_time = time.perf_counter()
            episode_wall_clock_time = episode_end_time - episode_start_time

            print(f"eval_episode={episode_count}, "
                  f"episodic_length={steps_taken}, "
                  f"episodic_return={episode_reward}, "
                  f"episodic_sfg={sgf_count}, "
                  f"episodic_per={avg_per}, "
                  f"successful={is_successful}")

            episodic_returns.append(episode_reward)
            episodic_successes.append(1 if is_successful else 0)
            
            # Only track inference times and wall-clock times for successful episodes
            if is_successful and len(current_episode_inference_times) > 0:
                successful_episode_inference_times.append(current_episode_inference_times.copy())
                # Calculate wall-clock time per optimal step
                wall_clock_time_per_optimal_step = episode_wall_clock_time / optimal_episode_length
                successful_episode_wall_clock_times_per_optimal_step.append(wall_clock_time_per_optimal_step)
                
            if not truncation[0]:
                episodic_per.append(avg_per)
                episodic_sgf.append(sgf_count)
                episodic_asl.append(np.round(asl / sgf_count, 2))

            episode_count += 1
            obs, _ = envs.reset()
            done = False
            episode_reward = 0
            steps_taken = 0
            sgf_count = 0
            asl = 0
            # Reset for next episode
            current_episode_inference_times = []
            episode_start_time = time.perf_counter()  # Start timing next episode

        obs = next_obs

    envs.close()

    # Set agent back to training mode
    agent._q.train()
    agent._skip_q.train()

    return (np.asarray(episodic_returns),
            np.asarray(episodic_successes),
            np.asarray(episodic_per),
            np.asarray(episodic_sgf),
            np.asarray(episodic_asl),
            successful_episode_inference_times,
            successful_episode_wall_clock_times_per_optimal_step)


def perform_temporl_evaluation(
        cfg: Config,
        dataset: list[tuple[PositionXY, PositionXY, int]],
        run_name: str,
        agent: TempoRLDQN,
        device: torch.device,
        global_step: int,
        writer: SummaryWriter,
        eval_on_train_dataset: bool,
        eval_freq: int,
        model_path: str,
        best_mean_reward: float = float('-inf'),
        best_mean_success: float = float('-inf'),
        best_mean_value_global_step: float = float('-inf'),
        best_mean_per: float = float('-inf'),
        best_mean_sgf: float = float('-inf'),
        best_mean_asl: float = float('-inf'),
        save_prefix: str = "train-eval",
) -> tuple[float, float, float, float, float, float, bool]:
    """Perform evaluation during training and save best model"""
    found_new_best_model = False

    if eval_freq > 0 and global_step % eval_freq == 0 or global_step == cfg.total_timesteps - 1:
        with torch.no_grad():
            train_or_eval_message = f"{'train' if eval_on_train_dataset else 'val'}"
            save_prefix = f"{save_prefix if eval_on_train_dataset else 'val-eval'}"

            message = f"Evaluation on {train_or_eval_message} dataset"
            border = "*" * 20
            print(border)
            print("* " + message + " *")

            episode_rewards, episodic_successes, episodic_per, episodic_sgf, episodic_asl, _, _ = eval_temporl_model(
                cfg,
                dataset=dataset,
                run_name=run_name,
                agent=agent,
                device=device,
                epsilon=0.05  # Small epsilon for some exploration during evaluation
            )

            mean_reward = np.round(np.mean(episode_rewards), 3)
            mean_success = np.round(np.mean(episodic_successes), 3)
            mean_per = np.round(np.mean(episodic_per), 3)
            mean_sgf = np.round(np.mean(episodic_sgf), 3)
            mean_asl = np.round(np.mean(episodic_asl), 3)

            print_msg_prefix = f"{train_or_eval_message} dataset"
            print(f"{print_msg_prefix} mean return: {mean_reward}")
            print(f"{print_msg_prefix} mean success: {mean_success}")
            print(f"{print_msg_prefix} mean per: {mean_per}")
            print(f"{print_msg_prefix} mean sgf: {mean_sgf}")
            print(f"{print_msg_prefix} mean asl: {mean_asl}")

            if cfg.save_model_strategy == SaveModelStrategy.SUCCESS_RATE:
                if mean_success > best_mean_success or (
                        mean_success == best_mean_success and mean_reward > best_mean_reward):
                    best_mean_reward = mean_reward
                    best_mean_success = mean_success
                    best_mean_per = mean_per
                    best_mean_sgf = mean_sgf
                    best_mean_asl = mean_asl
                    best_mean_value_global_step = global_step
                    found_new_best_model = True
            else:
                if mean_reward > best_mean_reward or (
                        mean_reward == best_mean_reward and mean_success > best_mean_success):
                    best_mean_reward = mean_reward
                    best_mean_success = mean_success
                    best_mean_per = mean_per
                    best_mean_sgf = mean_sgf
                    best_mean_asl = mean_asl
                    best_mean_value_global_step = global_step
                    found_new_best_model = True

            if found_new_best_model:
                msg = f"New best {train_or_eval_message} model mean values, "
                msg += f"episodic-return: {best_mean_reward}, "
                msg += f"success: {best_mean_success}, "
                msg += f", episodic-per={best_mean_per}"
                msg += f", episodic-sgf={best_mean_sgf}"
                msg += f", episodic-asl={best_mean_asl}"
                print(msg)

                writer.add_scalar(
                    f"{save_prefix}/best_episodic_return",
                    best_mean_reward,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_success",
                    best_mean_success,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_mean_value_global_step",
                    best_mean_value_global_step,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_per",
                    best_mean_per,
                    global_step
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_sgf",
                    best_mean_sgf,
                    global_step
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_asl",
                    best_mean_asl,
                    global_step
                )

            if not eval_on_train_dataset and cfg.save_model and found_new_best_model:
                agent.save_model(model_path)
                print(f"{train_or_eval_message}: model saved to {model_path}")

            print(border)

            # Log current evaluation metrics
            writer.add_scalar(
                f"{save_prefix}/episodic_return", mean_reward, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_success", mean_success, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_per", mean_per, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_sgf", mean_sgf, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_asl", mean_asl, global_step
            )

    return best_mean_reward, best_mean_success, best_mean_value_global_step, best_mean_per, best_mean_sgf, best_mean_asl, found_new_best_model


@torch.no_grad()
def eval_with_pretrained_model(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    network: Type[TempoRLDQN],
    model_path: str,
    state_dim: tuple,
    action_dim: int,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    message = f"Evaluation with pretrained model {model_path}"

    border = "*" * 20
    print(border)
    print("* " + message + " *")

    agent = network(cfg=cfg,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    device=device)
    agent.load_model(model_path)

    episode_rewards, episodic_successes, episodic_per, episodic_sgf, episodic_asl, successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step = eval_temporl_model(cfg,
                                                                                                       dataset,
                                                                                                       run_name,
                                                                                                       agent,
                                                                                                       device,
                                                                                                       epsilon=0.0,
                                                                                                       capture_video=capture_video)

    # Calculate inference timing statistics for successful episodes only
    total_episodes = len(episode_rewards)
    successful_episodes = len(successful_episode_inference_times)
    
    if successful_episodes > 0:
        # Calculate wall-clock time per optimal step (fair comparison metric)
        avg_wall_clock_time_per_optimal_step = np.mean(successful_episode_wall_clock_times_per_optimal_step)
        std_wall_clock_time_per_optimal_step = np.std(successful_episode_wall_clock_times_per_optimal_step)
        
        # Also calculate overall stats for all decisions in successful episodes
        all_successful_decisions = [time for episode_times in successful_episode_inference_times for time in episode_times]
        avg_inference_time_per_decision_successful = np.mean(all_successful_decisions)
        std_inference_time_per_decision_successful = np.std(all_successful_decisions)
        total_decisions_successful = len(all_successful_decisions)
    else:
        avg_wall_clock_time_per_optimal_step = 0.0
        std_wall_clock_time_per_optimal_step = 0.0
        avg_inference_time_per_decision_successful = 0.0
        std_inference_time_per_decision_successful = 0.0
        total_decisions_successful = 0
    
    timing_stats = {
        'avg_wall_clock_time_per_optimal_step': avg_wall_clock_time_per_optimal_step,
        'std_wall_clock_time_per_optimal_step': std_wall_clock_time_per_optimal_step,
        'avg_inference_time_per_decision_successful': avg_inference_time_per_decision_successful,
        'std_inference_time_per_decision_successful': std_inference_time_per_decision_successful,
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes,
        'total_decisions_successful': total_decisions_successful
    }

    print(f"test dataset mean return: {np.round(np.mean(episode_rewards), 3)}")
    print(f"test dataset mean success: {np.round(np.mean(episodic_successes), 3)}")
    print(f"test dataset mean per: {np.round(np.mean(episodic_per), 3)}")
    print(f"test dataset mean sgf: {np.round(np.mean(episodic_sgf), 3)}")
    print(f"test dataset mean asl: {np.round(np.mean(episodic_asl), 3)}")
    print(f"successful episodes: {successful_episodes}/{total_episodes}")
    print(f"average wall-clock time per optimal step: {avg_wall_clock_time_per_optimal_step:.6f} seconds")
    print(f"std wall-clock time per optimal step: {std_wall_clock_time_per_optimal_step:.6f} seconds")
    print(f"average inference time per decision (successful episodes only): {avg_inference_time_per_decision_successful:.6f} seconds")
    print(f"std inference time per decision (successful episodes only): {std_inference_time_per_decision_successful:.6f} seconds")
    print(f"total decisions in successful episodes: {total_decisions_successful}")

    print(border)

    return episode_rewards, episodic_successes, episodic_per, episodic_sgf, episodic_asl, timing_stats


def eval_test_dataset(test_dataset: list[tuple[PositionXY, PositionXY, int]],
                      cfg: Config,
                      writer: SummaryWriter,
                      run_name: str,
                      state_dim: tuple,
                      action_dim: int,
                      device: torch.device,
                      model_path: str,
                      global_step: Optional[int] = None):
    set_seed(seed=cfg.seed, deterministic_torch=cfg.torch_deterministic)

    message = "Evaluation on test dataset"

    border = "*" * 20
    print(border)
    print("* " + message + " *")

    tag_name = "test-eval"

    if global_step is not None:
        tag_name += f"-{global_step}"

    episode_rewards, episodic_successes, episodic_per, episodic_sgf, episodic_asl, timing_stats = eval_with_pretrained_model(
        cfg=cfg,
        dataset=test_dataset,
        run_name=f"{run_name}-{tag_name}",
        network=TempoRLDQN,
        model_path=model_path,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        capture_video=True if global_step is None else False,
    )

    if global_step is None:
        for idx, episodic_return in enumerate(episode_rewards):
            writer.add_scalar(f"{tag_name}/episodic_return", episodic_return, idx)
        for idx, agent_step_ratio in enumerate(episodic_per):
            writer.add_scalar(f"{tag_name}/episodic_per", agent_step_ratio, idx)
        for idx, n_actor_activations in enumerate(episodic_sgf):
            writer.add_scalar(f"{tag_name}/episodic_sgf", n_actor_activations, idx)
        for idx, n_actor_activations in enumerate(episodic_asl):
            writer.add_scalar(f"{tag_name}/episodic_asl", n_actor_activations, idx)

    writer.add_scalar(f"{tag_name}/mean_episodic_return", np.round(np.mean(episode_rewards), 3))
    writer.add_scalar(f"{tag_name}/mean_episodic_success", np.round(np.mean(episodic_successes), 3))
    writer.add_scalar(f"{tag_name}/mean_episodic_per", np.round(np.mean(episodic_per), 3))
    writer.add_scalar(f"{tag_name}/mean_episodic_sgf", np.round(np.mean(episodic_sgf), 3))
    writer.add_scalar(f"{tag_name}/mean_episodic_asl", np.round(np.mean(episodic_asl), 3))
    
    # Add timing statistics to logs (successful episodes only)
    writer.add_scalar(f"{tag_name}/avg_wall_clock_time_per_optimal_step", timing_stats['avg_wall_clock_time_per_optimal_step'])
    writer.add_scalar(f"{tag_name}/std_wall_clock_time_per_optimal_step", timing_stats['std_wall_clock_time_per_optimal_step'])
    writer.add_scalar(f"{tag_name}/avg_inference_time_per_decision_successful", timing_stats['avg_inference_time_per_decision_successful'])
    writer.add_scalar(f"{tag_name}/std_inference_time_per_decision_successful", timing_stats['std_inference_time_per_decision_successful'])
    writer.add_scalar(f"{tag_name}/total_episodes", timing_stats['total_episodes'])
    writer.add_scalar(f"{tag_name}/successful_episodes", timing_stats['successful_episodes'])
    writer.add_scalar(f"{tag_name}/total_decisions_successful", timing_stats['total_decisions_successful'])

    print(border)


def train(cfg: Config, run_name: str, writer: SummaryWriter):
    global device  # Make device global for tt function

    device = get_device(cfg)
    print(f"Using device: {device}")

    # Environment setup
    envs = env_setup(cfg, run_name)
    train_dataset, val_dataset, test_dataset, sub_training_envs = dataset_setup(
        cfg, run_name, max_level=cfg.max_level, start_level=cfg.start_level
    )
    envs.envs[0].unwrapped.start_goal_dataset = train_dataset

    # Get state and action dimensions
    observation_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    # Initialize TempoRL agent
    agent = TempoRLDQN(
        cfg=cfg,
        state_dim=observation_shape,
        action_dim=action_dim,
        # env=envs,
        # eval_env=envs,  # Using same env for training and evaluation
        device=device
    )

    # Initialize tracking variables
    start_time = time.time()
    model_path = f"runs/{run_name}/{cfg.exp_name}.model"

    best_val_mean_reward = float("-inf")
    best_train_mean_reward = float("-inf")
    best_val_mean_success = float("-inf")
    best_train_mean_success = float("-inf")
    best_val_mean_per = float("-inf")
    best_train_mean_per = float("-inf")
    best_val_mean_sgf = float("-inf")
    best_train_mean_sgf = float("-inf")
    best_val_mean_asl = float("-inf")
    best_train_mean_asl = float("-inf")
    best_val_mean_value_global_step = 0
    best_train_mean_value_global_step = 0

    # Start training
    obs, _ = envs.reset(seed=cfg.seed)
    episode_reward = 0
    episode_steps = 0
    episode_sgf = 0
    episode_asl = 0

    global_step = 0

    while global_step < cfg.total_timesteps:
        # Calculate epsilon based on linear schedule
        if global_step > cfg.exploration_fraction * cfg.total_timesteps:
            epsilon = cfg.end_e
        else:
            epsilon = cfg.start_e - (cfg.start_e - cfg.end_e) * (
                    global_step / (cfg.exploration_fraction * cfg.total_timesteps))

        # Get action and skip using epsilon-greedy
        action = agent.get_action(obs, epsilon)
        skip = agent.get_skip(obs, np.array([action]), epsilon)
        episode_sgf += 1
        episode_asl += skip

        # Execute the action for (skip + 1) steps
        skip_states = []
        skip_rewards = []

        for curr_skip in range(skip + 1):
            skip_states.append(obs.copy())
            next_obs, reward, termination, truncation, infos = envs.step(np.array([action]))
            global_step += 1
            reward = reward[0]  # Extract scalar from array
            skip_rewards.append(reward)

            episode_reward += reward
            episode_steps += 1

            done = termination[0] or truncation[0]

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncation):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            # Store transition in replay buffer
            agent._replay_buffer.add_transition(
                obs.squeeze(0),
                np.array([action]),
                real_next_obs.squeeze(0),
                np.array([reward]),
                np.array([termination[0]]),
            )

            # Update skip replay buffer
            skip_id = 0
            for start_state in skip_states:
                skip_reward = 0
                for exp, r in enumerate(skip_rewards[skip_id:]):
                    skip_reward += np.power(agent._gamma, exp) * r

                agent._skip_replay_buffer.add_transition(
                    start_state.squeeze(0),
                    curr_skip - skip_id,
                    real_next_obs.copy().squeeze(0),
                    skip_reward,
                    termination[0],
                    curr_skip - skip_id + 1,
                    np.array([action])
                )
                skip_id += 1

            obs = next_obs

            if done:
                # Log episode stats
                writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                writer.add_scalar("charts/episodic_length", episode_steps, global_step)
                writer.add_scalar("charts/episodic_sgf", episode_sgf, global_step)
                writer.add_scalar("charts/episodic_asl", np.round(episode_asl / episode_sgf, 2), global_step)

                print(f"global_step={global_step}, episodic_length={episode_steps}, "
                      f"episodic_return={episode_reward}, episodic_sgf={episode_sgf}, episodic_asl={np.round(episode_asl / episode_sgf, 2)}")

                episode_reward = 0
                episode_steps = 0
                episode_sgf = 0
                episode_asl = 0

                break

            # Training step
            if global_step > cfg.learning_starts and global_step % cfg.train_frequency == 0:
                action_loss, skip_loss = agent.update_networks(global_step)
                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", action_loss, global_step)
                    writer.add_scalar("losses/skip_loss", skip_loss, global_step)

                    # Log training progress
            if global_step % 1000 == 0:
                writer.add_scalar(
                    "charts/epsilon", epsilon, global_step
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            # Evaluation
            best_val_mean_reward, best_val_mean_success, best_val_mean_value_global_step, \
                best_val_mean_per, best_val_mean_sgf, best_val_mean_asl, _ = perform_temporl_evaluation(
                cfg,
                dataset=val_dataset,
                run_name=run_name,
                agent=agent,
                device=device,
                global_step=global_step,
                writer=writer,
                eval_on_train_dataset=False,
                eval_freq=cfg.val_eval_freq,
                model_path=model_path,
                best_mean_reward=best_val_mean_reward,
                best_mean_success=best_val_mean_success,
                best_mean_value_global_step=best_val_mean_value_global_step,
                best_mean_per=best_val_mean_per,
                best_mean_sgf=best_val_mean_sgf,
                best_mean_asl=best_val_mean_asl
            )

            best_train_mean_reward, best_train_mean_success, best_train_mean_value_global_step, \
                best_train_mean_per, best_train_mean_sgf, best_train_mean_asl, _ = perform_temporl_evaluation(
                cfg,
                dataset=sub_training_envs,
                run_name=run_name,
                agent=agent,
                device=device,
                global_step=global_step,
                writer=writer,
                eval_on_train_dataset=True,
                eval_freq=cfg.train_eval_freq,
                model_path=model_path,
                best_mean_reward=best_train_mean_reward,
                best_mean_success=best_train_mean_success,
                best_mean_value_global_step=best_train_mean_value_global_step,
                best_mean_per=best_train_mean_per,
                best_mean_sgf=best_train_mean_sgf,
                best_mean_asl=best_train_mean_asl
            )

            if global_step % min(cfg.train_eval_freq, cfg.total_timesteps) == 0:
                print("######################")
                print(f"best val mean reward={best_val_mean_reward}")
                print(f"best val mean success={best_val_mean_success}")
                print()
                print(f"best train mean reward={best_train_mean_reward}")
                print(f"best train mean success={best_train_mean_success}")
                print()
                print(f"best eval mean global step={best_val_mean_value_global_step}")
                print(f"best train mean global step={best_train_mean_value_global_step}")
                print("######################")

            if cfg.eval_test_dataset_during_training_freq > 1 and global_step % cfg.eval_test_dataset_during_training_freq == 0:
                eval_test_dataset(test_dataset,
                                  cfg,
                                  writer,
                                  run_name,
                                  state_dim=observation_shape,
                                  action_dim=action_dim,
                                  device=device,
                                  model_path=model_path,
                                  global_step=global_step)

            if global_step >= cfg.total_timesteps:
                break


    # Save final model if not saved during training
    if not cfg.save_model or (cfg.val_eval_freq < 0 and cfg.save_model):
        agent.save_model(model_path)
        print(f"Final model saved to {model_path}")

    envs.close()

    # eval on test set
    eval_test_dataset(test_dataset,
                      cfg,
                      writer,
                      run_name,
                      state_dim=observation_shape,
                      action_dim=action_dim,
                      device=device,
                      model_path=model_path)


def wandb_init(cfg: Config, run_name: str) -> Run:
    """Initialize WandB run"""
    import wandb

    run = wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        sync_tensorboard=True,
        config=vars(cfg),
        monitor_gym=True,
        save_code=True,
    )
    run.name = f"{run_name}__{run.name}"

    return run


if __name__ == "__main__":
    args = tyro.cli(Config)

    # Validate arguments
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    assert (not args.mps and not args.cuda) or (args.mps and not args.cuda) or (args.cuda and not args.mps)

    # Create run name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Initialize wandb if tracking is enabled
    if args.track:
        run = wandb_init(args, run_name)
        run_name = run.name

    # Create tensorboard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set random seeds
    set_seed(seed=args.seed, deterministic_torch=args.torch_deterministic)

    # Start training
    train(args, run_name, writer)

    writer.close()