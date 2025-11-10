# GPS: Generative Proto-Sequence: Sequence-Level Decision Making for Long-Horizon Reinforcement Learning

This repository contains implementations of various reinforcement learning algorithms for solving navigation tasks in a grid-based environment. The project includes several state-of-the-art methods:

- **DQN (Deep Q-Network)**: A standard Q-learning approach with neural networks
- **DAR (Deep Action Repetition)**: DQN with the ability to repeat actions multiple times
- **TempoRL**: Temporal abstraction method that learns when to apply temporal jumps/skips
- **GPS (Our method)**: An actor-critic framework that generates complete action sequences

## About GPS

Deep reinforcement learning (DRL) methods often face challenges in environments characterized by large state spaces, long action horizons, and sparse rewards, where effective exploration and credit assignment are critical. We introduce Generative Proto-Sequence (GPS), a novel generative DRL approach that produces variable-length discrete action sequences. By generating entire action sequences in a single decision rather than selecting individual actions at each timestep, GPS reduces the temporal decision bottleneck that impedes learning in long-horizon tasks. This sequence-level abstraction provides three key advantages: (1) it facilitates more effective credit assignment by directly connecting state observations with the outcomes of complete behavioral patterns; (2) by committing to coherent multi-step strategies, our approach facilitates better exploration of the state space; and (3) it promotes better generalization by learning macro-behaviors that transfer across similar situations rather than memorizing state-specific responses. Evaluations on maze navigation tasks of varying sizes and complexities demonstrate that GPS outperforms leading action repetition and temporal methods in most tested configurations, where it converges faster and achieves higher success rates in the majority of environments.

## Installation

This project uses Poetry for dependency management. Follow these steps to install and set up the environment:

### Prerequisites

- Python 3.10+
- Poetry

### Setting up Python 3.10 with Poetry

Poetry allows you to specify which Python version to use for your project. Here's how to set up Python 3.10:

1. Install Python 3.10 on your system if you haven't already.

2. To create a new Poetry environment with Python 3.10:
   ```bash
   poetry env use python3.10
   ```
   
3. If you're on macOS and have installed Python via Homebrew:
   ```bash
   poetry env use $(brew --prefix python@3.10)/bin/python3.10
   ```
   
4. If you're using pyenv:
   ```bash
   pyenv install 3.10.x  # Replace x with the latest minor version
   pyenv local 3.10.x
   poetry env use $(pyenv which python)
   ```

5. To verify the Python version Poetry is using:
   ```bash
   poetry run python --version
   ```

### Setup

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Activate the Poetry environment:
   ```bash
   poetry shell
   ```

## Running the Algorithms

### DQN (Deep Q-Network)

To run the standard DQN implementation:

```bash
python dqn_simplegrid_levels.py --cuda --track --train_dataset_size 100 --val_dataset_size 100 --test_dataset_size 1000
```

#### Additional DQN Options Examples:
- `--learning_rate`: Learning rate (default: 1e-3)
- `--total_timesteps`: Total training timesteps (default: 80000)
- `--batch_size`: Batch size for training (default: 256)
- `--max_level`: Maximum difficulty level (default: 14)
- `--obstacle_map`: Map to use (default: "8x8_empty")

### DAR (Deep Action Repetition)

To run the DAR implementation:

```bash
python dar_simplegrid_levels.py --cuda --track --train_dataset_size 100 --val_dataset_size 100 --test_dataset_size 1000
```

#### Additional DAR Options Examples:
- `--dar_r_l`: List of repetition lengths (default: [1, 5, 10])
- `--dar_skip_net_max_skips`: Maximum skip size if dar_r_l is not specified (default: 2)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--total_timesteps`: Total training timesteps (default: 1000000)

### TempoRL

To run the TempoRL implementation:

```bash
python temporl_simplegrid_levels.py --cuda --track --train_dataset_size 100 --val_dataset_size 100 --test_dataset_size 1000
```

#### Additional TempoRL Options Examples:
- `--agent_type`: Agent type to train (options: "dqn", "dar", "tdqn", "t-dqn"; default: "tdqn")
- `--weight_sharing`: Whether to use weight sharing between action and skip networks (default: True)
- `--skip_dim`: Maximum skip size (default: 10)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--total_timesteps`: Total training timesteps (default: 100000)

### GPS (Our Method)

To run our actor-critic framework with action sequence generation:

```bash
python gps_simplegrid_levels.py --cuda --track --train_dataset_size 100 --val_dataset_size 100 --test_dataset_size 1000
```

#### GPS vs GPS-E2E Modes

GPS supports two training modes:

- **GPS (default)**: Uses a frozen pretrained decoder (`--train_decoder_end_to_end False`)
- **GPS-E2E**: Trains the decoder end-to-end with the actor and critic (`--train_decoder_end_to_end True`)

#### Additional GPS Options Examples:

**Learning Rates:**
- `--actor_learning_rate`: Actor network learning rate (default: 1e-4)
- `--critic_learning_rate`: Critic network learning rate (default: 1e-4)
- `--decoder_learning_rate`: Decoder learning rate for GPS-E2E mode (default: 1e-5)
- `--total_timesteps`: Total training timesteps (default: 1000000)

**Decoder Configuration:**
- `--train_decoder_end_to_end`: Train decoder end-to-end (GPS-E2E mode) or use frozen pretrained decoder (GPS mode) (default: True)
- `--initialize_decoder_from_pretrained`: Initialize decoder from pretrained checkpoint in E2E mode (default: False)
- `--decoder_model_path`: Path to pretrained decoder model
- `--n_actions_in_seq`: Maximum number of actions in generated sequences (default: 10)
- `--use_gumble_in_decoder`: Use Gumbel-Softmax for differentiable sampling (default: True)

**Actor Architecture:**
- `--actor_n_output_channels`: Size of the proto-action-plan embedding (default: 16)
- `--n_proto_plan_candidates`: Number of proto-plan candidate heads (default: 1)
- `--actor_linear_layers`: Hidden layer sizes for actor's linear layers (default: [512, 128, 32])
- `--pe_embedding_dim`: Position encoding embedding dimension (default: 128, set to -1 to disable)

**Critic Configuration:**
- `--critic_linear_layers`: Hidden layer sizes for critic's linear layers (default: [512, 128, 32])
- `--action_seq_representation`: How critic evaluates action sequences (options: "action_seq_as_int", "action_seq_as_one_hot", "action_seq_as_prob"; default: "action_seq_as_one_hot")

**Exploration:**
- `--start_e`: Starting epsilon for exploration (default: 1.0)
- `--end_e`: Ending epsilon for exploration (default: 0.1)
- `--total_steps_e`: Total steps for epsilon decay (default: 15000)

**Replay Buffer:**
- `--buffer_size`: Replay memory buffer size (default: 50000)
- `--push_every_one_step_transition_to_buffer`: Store every single-step transition (default: True)
- `--push_sub_sequences_to_buffer_move_start_point`: Store sub-sequences with moving start point (default: True)
- `--push_sub_sequences_to_buffer_move_end_point`: Store sub-sequences with moving end point (default: True)
- `--sub_sequences_min_jump_move_start_point`: Minimum jump between elements for start point sub-sequences (default: 1)
- `--sub_sequences_min_jump_move_end_point`: Minimum jump between elements for end point sub-sequences (default: 1)

**Network Updates:**
- `--actor_policy_frequency`: Frequency of actor training (delayed updates) (default: 4)
- `--actor_target_network_frequency`: Actor target network update frequency (default: 100)
- `--critic_target_network_frequency`: Critic target network update frequency (default: 10)
- `--tau`: Target network soft update rate (default: 0.005)

## Configuration Options

All algorithms support a wide range of configuration options. You can see all available options by running:

```bash
python [algorithm_file].py --help
```

Common configuration options across all algorithms:

**System:**
- `--cuda`: Use CUDA for GPU acceleration
- `--mps`: Use MPS for Apple Silicon GPU acceleration
- `--track`: Track experiments with Weights & Biases
- `--capture_video`: Capture videos of agent performance
- `--seed`: Random seed (default: 123)

**Dataset:**
- `--train_dataset_size`: Size of training dataset
- `--val_dataset_size`: Size of validation dataset
- `--test_dataset_size`: Size of test dataset
- `--max_level`: Maximum difficulty level (default: 14)
- `--start_level`: Starting difficulty level (default: 1)
- `--obstacle_map`: The obstacle map of the environment (default: '8x8_empty')

**Environment Observability:**
- `--partial_observability_strategy`: Observability mode (options: "FULL", "LOCAL_VIEW"; default: "FULL")
  - `FULL`: Agent observes the entire grid
  - `LOCAL_VIEW`: Agent observes only a local window around its position
- `--view_radius`: Radius of agent's local view when using LOCAL_VIEW strategy (default: 3)
  - Creates a (2*view_radius+1) × (2*view_radius+1) observation window
  - Example: view_radius=3 gives a 7×7 observation

**Environment Stochasticity:**
- `--is_slippery`: Enable stochastic action outcomes (default: False)
  - When enabled, actions may result in perpendicular movement
- `--slippery_prob`: Probability of executing intended action when `is_slippery=True` (default: 1/3)
  - Remaining probability is split equally between perpendicular directions
  - Example: slippery_prob=0.33 means 33% intended, 33% left perpendicular, 33% right perpendicular
- `--sticky_action_prob`: Probability of repeating the previous action instead of the intended action (default: 0.0)
  - Simulates action lag or momentum in the environment
  - Value between 0.0 (no stickiness) and 1.0 (always repeat previous action)
- `--random_action_prob`: Probability of executing a uniformly random action instead of the intended action (default: 0.0)
  - Adds uniform noise to action execution
  - Value between 0.0 (no noise) and 1.0 (always random)

## Available Maps:
8x8_empty, 16x16_empty, 24x24_empty,
16x16_obstacles_v1_15p, 16x16_obstacles_v1_25p, 24x24_obstacles_v1_15p, 24x24_obstacles_v1_25p
16x16_rooms_v1
16x16_corridors

## Hyperparameter Search

The repository includes hyperparameter search grid scripts that can be used to explore a wide range of parameter combinations for each algorithm. These scripts provide configuration for various additional parameters not mentioned above. For detailed hyperparameter information, please refer to these scripts in the repository.

For example, for our GPS method, you can explore parameters such as:
- Network architecture configurations
- Learning rates and optimization parameters
- Regularization weights
- Action sequence generation parameters

## Environment

The algorithms are tested in SimpleGrid, a customized grid-based navigation environment based on MiniGrid. The environment supports various levels of difficulty and different reward strategies.

## Visualization and Tracking

Results can be tracked using Weights & Biases:

```bash
# For W&B:
# Just add --track to your command line arguments
```

## Citation

If you use this code in your research, please cite. 