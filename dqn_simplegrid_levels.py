import itertools
import os
import random
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
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


@dataclass
class Config:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 123
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    mps: bool = False
    """if toggled, mps will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "dqn-grid-experiments"
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

    # Algorithm specific arguments
    total_timesteps: int = 80000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.01
    """the target network update rate"""
    target_network_frequency: int = 100
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.1
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 2
    """the frequency of training"""
    dqn_linear_layers: list[int] = field(default_factory=lambda: [512, 32])
    """the hidden layer sizes for the dqn's linear layers"""
    dqn_linear_layers_activation_function: str = "leaky_relu"
    """activation function to use in the dqn's linear layers"""



# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self,
                 n_input_channels: int,
                 n_output_channels: int,
                 observation_shape: tuple[int, int] = (8, 8),
                 linear_layers: Optional[list[int]] = None,
                 linear_layers_activation: str = "Relu"):
        if linear_layers is None:
            linear_layers = [512, 128, 32]
        self.activation_fn = get_activation_fn(linear_layers_activation)

        super().__init__()

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
            n_flatten = self.cnn(
                torch.as_tensor(observation_shape).float()
            ).shape[1]

        # Define a partial function for creating a linear layer with optional batchnorm and activation
        self.create_layer = partial(create_linear_layer, use_batchnorm=False, activation_fn=self.activation_fn)

        linear_layers = [n_flatten] + linear_layers + [n_output_channels]
        layers = list(
            itertools.chain.from_iterable([self.create_layer(linear_layers[i], linear_layers[i + 1], i < len(linear_layers) - 2) for i in
                                           range(len(linear_layers) - 1)]))

        self.linear = nn.Sequential(*layers)

        self.network = nn.Sequential(self.cnn, self.linear)

    def forward(self, obs: torch.tensor):
        return self.network(obs)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@torch.no_grad()
def eval_model(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]], list[float]]:
    envs = env_setup(cfg, run_name, capture_video=capture_video)
    envs.envs[0].unwrapped.start_goal_dataset = dataset

    eval_episodes = len(dataset)

    model.eval()
    obs, _ = envs.reset()
    episodic_returns = []
    episodic_successes = []
    avg_agent_step_ratio_in_sequences_list = []
    current_episode_inference_times = []  # Track inference times for current episode
    successful_episode_inference_times = []  # Track inference times per successful episode
    successful_episode_wall_clock_times_per_optimal_step = []  # Track wall-clock time per optimal step
    
    # Track episode timing
    episode_start_time = time.perf_counter()

    while len(episodic_returns) < eval_episodes:
        optimal_episode_length = envs.envs[0].unwrapped.level

        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            # Measure inference time
            start_time = time.perf_counter()
            q_values = model(torch.tensor(obs, dtype=torch.float32, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            end_time = time.perf_counter()
            current_episode_inference_times.append(end_time - start_time)
            
        next_obs, _, termination, truncation, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue

                episode_length = info['episode']['l'][0]
                avg_agent_step_ratio_per_episode = np.round(optimal_episode_length / episode_length, 2)
                is_successful = termination[0]
                
                # Calculate episode wall-clock time
                episode_end_time = time.perf_counter()
                episode_wall_clock_time = episode_end_time - episode_start_time

                msg = f"eval_episode={len(episodic_returns)}, episodic_length={info['episode']['l']}, episodic_return={info['episode']['r']}"
                msg += f", avg_agent_step_ratio_in_episode={avg_agent_step_ratio_per_episode}"
                msg += f", successful={is_successful}"
                print(msg)

                episodic_returns += [info["episode"]["r"]]
                episodic_successes += [1 if is_successful else 0]

                # Only track inference times and wall-clock times for successful episodes
                if is_successful and len(current_episode_inference_times) > 0:
                    successful_episode_inference_times.append(current_episode_inference_times.copy())
                    # Calculate wall-clock time per optimal step
                    wall_clock_time_per_optimal_step = episode_wall_clock_time / optimal_episode_length
                    successful_episode_wall_clock_times_per_optimal_step.append(wall_clock_time_per_optimal_step)

                # Reset for next episode
                current_episode_inference_times = []
                episode_start_time = time.perf_counter()  # Start timing next episode

                if not truncation[0]:
                    avg_agent_step_ratio_in_sequences_list.append(avg_agent_step_ratio_per_episode)
        obs = next_obs

    envs.close()

    return np.asarray(episodic_returns), np.asarray(episodic_successes), np.asarray(avg_agent_step_ratio_in_sequences_list), successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step


@torch.no_grad()
def eval_during_training(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    network: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]], list[float]]:
    network_in_training = network.training

    episodic_returns, episodic_successes, episodic_avg_agent_step_ratio_in_sequence, successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step = eval_model(cfg, dataset, run_name, network, device, capture_video=capture_video)

    if network_in_training:
        network.train()

    return episodic_returns, episodic_successes, episodic_avg_agent_step_ratio_in_sequence, successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step


def perform_evaluation(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    model: torch.nn.Module,
    device: torch.device,
    global_step: int,
    writer,
    eval_on_train_dataset: bool,
    eval_freq: int,
    model_path: str,
    best_mean_reward: float = float('-inf'),
    best_mean_success: float = float('-inf'),
    best_mean_value_global_step: float = float('-inf'),
    best_mean_agent_step_ratio_in_sequences: float = float('-inf'),
    save_prefix: str = "train-eval",
) -> tuple[float, float, float, float, bool]:
    found_new_best_model = False

    if eval_freq > 0 and global_step % eval_freq == 0 or global_step == cfg.total_timesteps - 1:
        with torch.no_grad():
            train_or_eval_message = f"{'train' if eval_on_train_dataset else 'val'}"
            save_prefix = f"{save_prefix if eval_on_train_dataset else 'val-eval'}"

            message = f"Evaluation on {train_or_eval_message} dataset"
            border = "*" * 20
            print(border)
            print("* " + message + " *")

            episode_rewards, episodic_successes, episodic_avg_agent_step_ratio_in_sequences, _, _ = eval_during_training(
                cfg,
                dataset=dataset,
                run_name=run_name,
                network=model,
                device=device,
            )

            mean_reward = np.round(np.mean(episode_rewards), 3)
            mean_success = np.round(np.mean(episodic_successes), 3)
            mean_agent_step_ratio_in_sequences_per_episode = np.round(np.mean(episodic_avg_agent_step_ratio_in_sequences), 3)
            print_msg_prefix = f"{train_or_eval_message} dataset"
            print(f"{print_msg_prefix} mean reward: {mean_reward}")
            print(f"{print_msg_prefix} mean success: {mean_success}")
            print(f"{print_msg_prefix} mean agent step ratio in sequences per episode: {mean_agent_step_ratio_in_sequences_per_episode}")

            if cfg.save_model_strategy == SaveModelStrategy.SUCCESS_RATE:
                if mean_success > best_mean_success or (
                        mean_success == best_mean_success and mean_reward > best_mean_reward):
                    best_mean_reward = mean_reward
                    best_mean_success = mean_success
                    best_mean_agent_step_ratio_in_sequences = mean_agent_step_ratio_in_sequences_per_episode
                    best_mean_value_global_step = global_step
                    found_new_best_model = True
            else:
                if mean_reward > best_mean_reward or (
                        mean_reward == best_mean_reward and mean_success > best_mean_success):
                    best_mean_reward = mean_reward
                    best_mean_success = mean_success
                    best_mean_agent_step_ratio_in_sequences = mean_agent_step_ratio_in_sequences_per_episode
                    best_mean_value_global_step = global_step
                    found_new_best_model = True

            if found_new_best_model:
                msg = f"New best {train_or_eval_message} model mean values, episodic-return: {best_mean_reward}"
                msg += f", success-rate: {best_mean_success}"
                msg += f", avg_agent_step_ratio_per_sequence_in_episode={best_mean_agent_step_ratio_in_sequences}"
                print(msg)
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_reward",
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
                    f"{save_prefix}/best_mean_avg_agent_step_ratio_per_sequence",
                    best_mean_agent_step_ratio_in_sequences,
                    global_step
                )

            if not eval_on_train_dataset and cfg.save_model and found_new_best_model:
                torch.save(model.state_dict(), model_path)
                print(
                    f"{train_or_eval_message}: model saved to {model_path}")

            print(border)
            writer.add_scalar(
                f"{save_prefix}/episodic_return", mean_reward, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_success_rate", mean_success, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_avg_agent_step_raio_per_sequence", mean_agent_step_ratio_in_sequences_per_episode, global_step
            )

    return best_mean_reward, best_mean_success, best_mean_value_global_step, best_mean_agent_step_ratio_in_sequences, found_new_best_model


@torch.no_grad()
def eval_with_pretrained_model(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    network: Type[QNetwork],
    model_path: str,
    n_input_channels: int,
    n_output_channels: int,
    observation_shape: tuple[int, int],
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    message = f"Evaluation with pretrained model {model_path}"

    border = "*" * 20
    print(border)
    print("* " + message + " *")

    q_network = network(n_input_channels=n_input_channels,
                        n_output_channels=n_output_channels,
                        observation_shape=observation_shape,
                        linear_layers=cfg.dqn_linear_layers,
                        linear_layers_activation=cfg.dqn_linear_layers_activation_function).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))

    episodic_returns, episodic_successes, episodic_avg_agent_step_ratio_in_sequences, successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step = eval_model(cfg, dataset, run_name, q_network, device, epsilon=0.0, capture_video=capture_video)

    # Calculate inference timing statistics for successful episodes only
    total_episodes = len(episodic_returns)
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

    print(f"test dataset mean reward: {np.round(np.mean(episodic_returns), 3)}")
    print(f"test dataset mean success: {np.round(np.mean(episodic_successes), 3)}")
    print(f"test dataset mean agent step ratio in sequences per episode: {np.round(np.mean(episodic_avg_agent_step_ratio_in_sequences), 3)}")
    print(f"successful episodes: {successful_episodes}/{total_episodes}")
    print(f"average wall-clock time per optimal step: {avg_wall_clock_time_per_optimal_step:.6f} seconds")
    print(f"std wall-clock time per optimal step: {std_wall_clock_time_per_optimal_step:.6f} seconds")
    print(f"average inference time per decision (successful episodes only): {avg_inference_time_per_decision_successful:.6f} seconds")
    print(f"std inference time per decision (successful episodes only): {std_inference_time_per_decision_successful:.6f} seconds")
    print(f"total decisions in successful episodes: {total_decisions_successful}")
    print(border)

    return episodic_returns, episodic_successes, episodic_avg_agent_step_ratio_in_sequences, timing_stats


def eval_test_dataset(test_dataset: list[tuple[PositionXY, PositionXY, int]],
                      cfg: Config,
                      writer: SummaryWriter,
                      n_input_channels: int,
                      n_output_channels: int,
                      observation_shape: tuple[int, int],
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

    episodic_returns, episodic_successes, episodic_avg_agent_step_ratio_in_sequences, timing_stats = eval_with_pretrained_model(
        cfg=cfg,
        dataset=test_dataset,
        run_name=f"{run_name}-{tag_name}",
        network=QNetwork,
        model_path=model_path,
        n_input_channels=n_input_channels,
        n_output_channels=n_output_channels,
        observation_shape=observation_shape,
        device=device,
        capture_video=True if global_step is None else False,
    )

    if global_step is None:
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar(f"{tag_name}/episodic_return", episodic_return, idx)
        for idx, agent_step_ratio in enumerate(episodic_avg_agent_step_ratio_in_sequences):
            writer.add_scalar(f"{tag_name}/episodic_agent_step_ratio", agent_step_ratio, idx)

    writer.add_scalar(f"{tag_name}/mean_episodic_return", np.round(np.mean(episodic_returns), 3))
    writer.add_scalar(f"{tag_name}/mean_episodic_success", np.round(np.mean(episodic_successes), 3))
    writer.add_scalar(f"{tag_name}/mean_agent_step_ratio_per_sequence_in_episode", np.round(np.mean(episodic_avg_agent_step_ratio_in_sequences), 3))
    
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
    device = get_device(cfg)
    print(f"using device: {device}")

    # env setup
    envs = env_setup(cfg, run_name)

    train_dataset, val_dataset, test_dataset, sub_training_envs = dataset_setup(
        cfg, run_name, max_level=cfg.max_level, start_level=cfg.start_level
    )
    envs.envs[0].unwrapped.start_goal_dataset = train_dataset

    observation_shape = envs.single_observation_space.sample()[None]
    n_output_channels = envs.single_action_space.n
    n_input_channels = envs.single_observation_space.shape[0]  # obs shape is (3, x, x)

    q_network = QNetwork(n_input_channels=n_input_channels,
                         n_output_channels=n_output_channels,
                         observation_shape=observation_shape,
                         linear_layers=cfg.dqn_linear_layers,
                         linear_layers_activation=cfg.dqn_linear_layers_activation_function).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    target_network = QNetwork(n_input_channels=n_input_channels,
                              n_output_channels=n_output_channels,
                              observation_shape=observation_shape,
                              linear_layers=cfg.dqn_linear_layers,
                              linear_layers_activation=cfg.dqn_linear_layers_activation_function).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    continue_training = True
    model_path = f"runs/{run_name}/{cfg.exp_name}.model"
    best_val_mean_reward = float("-inf")
    best_train_mean_reward = float("-inf")
    best_val_mean_success = float("-inf")
    best_train_mean_success = float("-inf")
    best_val_mean_agent_step_ratio_in_sequences_per_episode = float("-inf")
    best_train_mean_agent_step_ratio_in_sequences_per_episode = float("-inf")

    best_val_mean_value_global_step = 0
    best_train_mean_value_global_step = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            cfg.start_e,
            cfg.end_e,
            int(cfg.exploration_fraction * cfg.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.tensor(obs, dtype=torch.float32, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_length={info['episode']['l']}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(
                        data.next_observations.to(torch.float32)
                    ).max(dim=1)
                    td_target = data.rewards.flatten() + cfg.gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = (
                    q_network(data.observations.to(torch.float32))
                    .gather(1, data.actions)
                    .squeeze()
                )
                loss = F.mse_loss(td_target, old_val)

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data
                        + (1.0 - cfg.tau) * target_network_param.data
                    )

            # evaluation on validation dataset
            best_val_mean_reward, best_val_mean_success, best_val_mean_value_global_step, best_val_mean_agent_step_ratio_in_sequences_per_episode, _ = perform_evaluation(
                cfg,
                dataset=val_dataset,
                run_name=run_name,
                model=target_network,
                device=device,
                global_step=global_step,
                writer=writer,
                eval_on_train_dataset=False,
                eval_freq=cfg.val_eval_freq,
                model_path=model_path,
                best_mean_reward=best_val_mean_reward,
                best_mean_success=best_val_mean_success,
                best_mean_value_global_step=best_val_mean_value_global_step,
                best_mean_agent_step_ratio_in_sequences=best_val_mean_agent_step_ratio_in_sequences_per_episode,
            )

            # evaluation on train dataset
            best_train_mean_reward, best_train_mean_success, best_train_mean_value_global_step, best_train_mean_agent_step_ratio_in_sequences_per_episode, _ = perform_evaluation(
                cfg,
                dataset=sub_training_envs,
                run_name=run_name,
                model=target_network,
                device=device,
                global_step=global_step,
                writer=writer,
                eval_on_train_dataset=True,
                eval_freq=cfg.train_eval_freq,
                model_path=model_path,
                best_mean_reward=best_train_mean_reward,
                best_mean_success=best_train_mean_success,
                best_mean_value_global_step=best_train_mean_value_global_step,
                best_mean_agent_step_ratio_in_sequences=best_train_mean_agent_step_ratio_in_sequences_per_episode,
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
                                  n_input_channels,
                                  n_output_channels,
                                  observation_shape,
                                  device,
                                  model_path=model_path,
                                  global_step=global_step)

            if not continue_training:
                break

    if not cfg.save_model or (cfg.val_eval_freq < 0 and cfg.save_model):
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()

    # eval on test set
    eval_test_dataset(test_dataset,
                      cfg,
                      writer,
                      n_input_channels,
                      n_output_channels,
                      observation_shape,
                      device,
                      model_path=model_path)


def wandb_init(cfg: Config, run_name: str) -> Run:
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

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    assert (not args.mps and not args.cuda) or (args.mps and not args.cuda) or (args.cuda and not args.mps)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        run = wandb_init(args, run_name)
        run_name = run.name

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    set_seed(seed=args.seed, deterministic_torch=args.torch_deterministic)

    train(args, run_name, writer)

    writer.close()
