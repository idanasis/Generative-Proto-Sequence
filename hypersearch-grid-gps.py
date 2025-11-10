import os
import subprocess
import time
from dataclasses import dataclass, field, asdict, fields
from itertools import product
from typing import Any, Optional

import tyro
from wandb.sdk.wandb_run import Run

import wandb


@dataclass
class Config:
    file_to_replace: str = "sbatch_gps"
    original_command_pattern: str = "poetry run python gps_simplegrid_levels.py"
    source_wandb_project_name: str = ""
    target_wandb_project_name: str = ""
    top_k: int = -1
    trigger_slurm: bool = False
    override_based_on_target_config: bool = True

    param_grid: dict[str, Any] = field(default_factory=lambda: {
        'seed': [123],
        'save_model_strategy': ['SUCCESS_RATE'],
        'action_seq_representation': ['ACTION_SEQ_AS_ONE_HOT'],
        'val_eval_freq': [5000],
        'train_eval_freq': [5000],
        'reward_strategy': ['NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES'],
        'observation_encoding_strategy': ['DEFAULT'],
        'partial_observability_strategy': ['FULL'],
        'view_radius': [3],
        'is_slippery': [False],
        'slippery_prob': [1 / 3],
        'sticky_action_prob': [0.0],
        'random_action_prob': [0.0],
        'actor_learning_rate': [1e-4, 1e-3, 1e-5],
        'critic_learning_rate': [1e-4],
        'buffer_size': [50000, 10000],
        'tau': [0.005, 0.01],
        'batch_size': [256],
        'learning_starts': [1000],
        'actor_policy_frequency': [2],
        'push_sub_sequences_to_buffer_move_start_point': [True],
        'push_sub_sequences_to_buffer_move_end_point': [True],
        'sub_sequences_min_jump_move_start_point': [1],
        'sub_sequences_min_jump_move_end_point': [1],
        'push_every_one_step_transition_to_buffer': [True],
        'actor_target_network_frequency': [10, 100],
        'critic_target_network_frequency': [10],
        'end_e': [0.1],
        'total_steps_e': [15000],
        'n_proto_plan_candidates': [1],
        'min_qf_value': [None],
        'actor_weight_decay': [1e-4],
        'critic_weight_decay': [1e-4],
        'sample_from_candidates': [0.1],
        'explore_without_critic': [True],
        'exploration_sequence_length_constraint': [-1],
        'diversity_weight': [-1],
        'penalize_cyclic_position_revisits': [False],
        'include_actor_embedding_in_critic_input': [False],
        'deterministic_inference': [True],

        'total_timesteps': [1500000],
        'max_level': [30],
        'start_level': [20],
        'max_episode_steps': [75],
        'obstacle_map': ["16x16_obstacles_v1_25p"],

        'train_dataset_size': [100],
        'train_eval_dataset_size': [100],
        'val_dataset_size': [100],
        'test_dataset_size': [1000],

        'actor_linear_layers': ["512 128 32"],
        'actor_linear_layers_activation_function': ["leaky_relu"],
        'actor_use_batchnorm_linear_layers': [False],
        'pe_embedding_dim': [128],
        'exclude_decoder_from_computation_graph': [False],
        'upload_model_to_wandb': [False],

        'critic_linear_layers': ["512 128 32"],
        'critic_linear_layers_activation_function': ["leaky_relu"],
        'critic_use_batchnorm_linear_layers': [False],

        'decoder_model_path': [
            "generative/models/dense_autoencoder/denseAE_generic_2act_seq_VAE_v7_bs=32_epochs=19999-20000_lr=0.0001_end_size_16_var_1_leakyrelu_normalized_by_seq_len_without_aug_instance_norm.pt"],
        
        GPS-E2E mode parameters
        'train_decoder_end_to_end': [False],
        'decoder_learning_rate': [1e-5],
        'decoder_weight_decay': [1e-3],
        'initialize_decoder_from_pretrained': [True],
    })

    target_config: dict[str, Any] = field(default_factory=lambda: {
        "reward_strategy": 'NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES',
        "action_seq_representation": None,
        "observation_encoding_strategy": None,
        "partial_observability_strategy": None,
        "view_radius": None,
        "is_slippery": None,
        "slippery_prob": None,
        "sticky_action_prob": None,
        "random_action_prob": None,
        "push_sub_sequences_to_buffer_move_start_point": None,
        "push_sub_sequences_to_buffer_move_end_point": None,
        "sub_sequences_min_jump_move_start_point": None,
        "sub_sequences_min_jump_move_end_point": None,
        'push_every_one_step_transition_to_buffer': None,
        "sample_from_candidates": None,
        "n_proto_plan_candidates": None,
        "actor_policy_frequency": None,
        "actor_target_network_frequency": None,
        "critic_target_network_frequency": None,
        "explore_without_critic": None,
        "exploration_sequence_length_constraint": None,
        "train_dataset_size": None,
        "test_dataset_size": None,
        "total_timesteps": None,
        "obstacle_map": None,
        "max_level": None,
        "start_level": None,
        "max_episode_steps": None,
        "val_eval_freq": None,
        "train_eval_freq": None,
        "actor_linear_layers": None,
        "critic_linear_layers": None,
        "actor_linear_layers_activation_function": None,
        "decoder_model_path": None,
        "upload_model_to_wandb": False,
        "deterministic_inference": False,
        "train_decoder_end_to_end": None,
        "decoder_learning_rate": None,
        "decoder_weight_decay": None,
        "initialize_decoder_from_pretrained": None,
    })


@dataclass(frozen=True, eq=True)
class ExperimentConfig:
    seed: int
    save_model_strategy: str
    action_seq_representation: str
    val_eval_freq: int
    train_eval_freq: int
    reward_strategy: str
    observation_encoding_strategy: str
    partial_observability_strategy: str
    view_radius: int
    is_slippery: bool
    slippery_prob: float
    sticky_action_prob: float
    random_action_prob: float
    actor_learning_rate: float
    critic_learning_rate: float
    buffer_size: int
    tau: float
    batch_size: int
    learning_starts: int
    actor_policy_frequency: int
    push_sub_sequences_to_buffer_move_start_point: bool
    push_sub_sequences_to_buffer_move_end_point: bool
    sub_sequences_min_jump_move_start_point: int
    sub_sequences_min_jump_move_end_point: int
    push_every_one_step_transition_to_buffer: bool
    actor_target_network_frequency: int
    critic_target_network_frequency: int
    end_e: float
    total_steps_e: int
    n_proto_plan_candidates: int
    min_qf_value: Optional[float]
    actor_weight_decay: float
    critic_weight_decay: float
    sample_from_candidates: float
    total_timesteps: int
    max_level: int
    start_level: int
    max_episode_steps: int
    obstacle_map: str
    train_dataset_size: int
    train_eval_dataset_size: int
    val_dataset_size: int
    test_dataset_size: int
    actor_linear_layers: str
    actor_linear_layers_activation_function: str
    critic_linear_layers: str
    critic_linear_layers_activation_function: str
    decoder_model_path: str
    explore_without_critic: bool
    exploration_sequence_length_constraint: int
    diversity_weight: float
    pe_embedding_dim: int
    exclude_decoder_from_computation_graph: bool
    actor_use_batchnorm_linear_layers: bool
    critic_use_batchnorm_linear_layers: bool
    penalize_cyclic_position_revisits: bool
    include_actor_embedding_in_critic_input: bool
    deterministic_inference: bool
    upload_model_to_wandb: bool
    train_decoder_end_to_end: bool
    decoder_learning_rate: float
    decoder_weight_decay: float
    initialize_decoder_from_pretrained: bool


@dataclass
class RunData:
    config: ExperimentConfig
    val_eval: float
    train_eval: float
    test_eval: float


def transform_config(config: dict[str, Any]) -> dict[str, Any]:
    transformed = config.copy()  # Start with a copy of all config parameters

    # Handle specific fields
    prefix_removals = {
        'save_model_strategy': 'SaveModelStrategy.',
        'action_seq_representation': 'ActionSeqRepresentation.',
        'reward_strategy': 'RewardStrategy.',
        'observation_encoding_strategy': 'ObservationEncodingStrategy.',
        'partial_observability_strategy': 'PartialObservabilityStrategy.'
    }
    for key, prefix in prefix_removals.items():
        if key in transformed:
            transformed[key] = transformed[key].removeprefix(prefix)

    # None to string conversion
    none_to_string_fields = ['min_qf_value', 'sample_from_candidates', 'train_dataset_size',
                             'train_eval_dataset_size', 'val_dataset_size', 'test_dataset_size',
                             'decoder_model_path']
    for field in none_to_string_fields:
        if field in transformed:
            transformed[field] = None if transformed[field] is None else transformed[field]

    # Special handling for specific fields
    if 'exploration_sequence_length_constraint' in transformed:
        transformed['exploration_sequence_length_constraint'] = -1 if transformed[
                                                                          'exploration_sequence_length_constraint'] is None else \
        transformed['exploration_sequence_length_constraint']
    else:
        transformed['exploration_sequence_length_constraint'] = -1

    # Join list fields
    list_fields = ['actor_linear_layers', 'critic_linear_layers']
    for field in list_fields:
        if field in transformed:
            transformed[field] = " ".join(str(x) for x in transformed[field])

    return transformed


def get_finished_runs(api: wandb.Api, wandb_project_name: str) -> tuple[list[RunData], list[Run]]:
    required_metrics = [
        "test-eval-with-critic/mean_episodic_success",
        "train-eval-with-critic/best_episodic_success",
        "val-eval-with-critic/best_episodic_success"
    ]

    finished_runs, invalid_runs = [], []
    for run in api.runs(wandb_project_name):
        if run.state == "finished" or run.state == "running":
            config = run.config
            metrics = run.summary

            missing_metrics = [m for m in required_metrics if metrics.get(m) is None and run.state == "finished"]

            if missing_metrics:
                invalid_runs.append(run)
                print(f"Run {run.id} is missing metrics: {', '.join(missing_metrics)}")
                continue

            transformed_config = transform_config(config)
            # Filter the dictionary to only include valid dataclass fields
            filtered_config = {k: v for k, v in transformed_config.items() if
                               k in {f.name for f in fields(ExperimentConfig)}}
 
            # Provide default values for missing required fields (only for new observability params)
            field_defaults = {
                'partial_observability_strategy': 'FULL',
                'view_radius': 3,
                'is_slippery': False,
                'slippery_prob': 1 / 3,
                'sticky_action_prob': 0.0,
                'random_action_prob': 0.0,
                'train_decoder_end_to_end': False,
                'decoder_learning_rate': 1e-5,
                'decoder_weight_decay': 1e-3,
                'initialize_decoder_from_pretrained': True,
            }

            for field_name, default_value in field_defaults.items():
                if field_name not in filtered_config:
                    filtered_config[field_name] = default_value

            run_data = RunData(
                config=ExperimentConfig(**filtered_config),
                val_eval=metrics.get('val-eval-with-critic/best_episodic_success'),
                train_eval=metrics.get('train-eval-with-critic/best_episodic_success'),
                test_eval=metrics.get('test-eval-with-critic/mean_episodic_success')
            )
            finished_runs.append(run_data)

    return finished_runs, invalid_runs


def get_top_experiments(all_runs: list[RunData], top_k: int = 100) -> list[ExperimentConfig]:
    metrics = ['val_eval', 'train_eval', 'test_eval']
    top_runs = []
    for metric in metrics:
        top_runs.extend(sorted(all_runs, key=lambda x: getattr(x, metric), reverse=True)[:top_k])

    top_runs_without_duplication = []
    for top_run in top_runs:
        assert top_run.val_eval > 0
        assert top_run.train_eval > 0
        assert top_run.test_eval > 0

        if top_run.config not in top_runs_without_duplication:
            top_runs_without_duplication.append(top_run.config)

    return top_runs_without_duplication


def generate_command(cfg: Config, experiment: ExperimentConfig, wandb_project_name: str) -> str:
    command = f"{cfg.original_command_pattern} "
    for key, value in asdict(experiment).items():
        # Check if it's a boolean
        if isinstance(value, bool):
            # Convert the variable name to kebab-case
            bool_flag_kebab_name = key.replace("_", "-")

            # Add 'no-' prefix if the value is False
            if not value:
                bool_flag_kebab_name = "no-" + bool_flag_kebab_name
            command += f"--{bool_flag_kebab_name} "
        else:
            command += f"--{key} {value} "

    command += f"--slurm_job_id $SLURM_JOBID "
    command += f"--wandb_project_name {wandb_project_name}"

    return command


def hypersearch(cfg: Config):
    api = wandb.Api()
    all_experiments, invalid_experiments = get_finished_runs(api, cfg.source_wandb_project_name)

    if invalid_experiments:
        print(f"Deleting invalid runs, project {cfg.source_wandb_project_name}")
        for run in invalid_experiments:
            run.delete()
        return

    if cfg.top_k > 0:
        ATTR_MATCHING_L: list[str] = ['train_dataset_size', 'total_timesteps', 'obstacle_map', 'actor_linear_layers',
                                      'buffer_size',
                                      'max_level', 'start_level', 'deterministic_inference', 'reward_strategy', 'seed']

        def is_matching_conditions(exp, conf, attr_l) -> bool:
            for attr in attr_l:
                if exp.config.__getattribute__(attr) not in conf.param_grid[attr]:
                    return False
            return True

        all_experiments_filtered = [e for e in all_experiments if is_matching_conditions(e, cfg, ATTR_MATCHING_L)]
        top_experiments = get_top_experiments(all_experiments_filtered, cfg.top_k)
        assert len(top_experiments) > 0

        finished_experiments, invalid_experiments = get_finished_runs(api, cfg.target_wandb_project_name)

        if invalid_experiments:
            print(f"Deleting invalid runs, project {cfg.target_wandb_project_name}")
            for run in invalid_experiments:
                run.delete()
            return

        finished_configs = [run.config for run in finished_experiments]

        combinations_based_on_topk = []
        target_config = {k: v for k, v in cfg.target_config.items() if v is not None} if cfg.override_based_on_target_config else {}
        for exp in top_experiments:
            exp_config = ExperimentConfig(**{**asdict(exp), **target_config})
            if exp_config not in combinations_based_on_topk:
                combinations_based_on_topk.append(exp_config)

        combinations_to_run = [
            exp
            for exp in combinations_based_on_topk
            if exp not in finished_configs
        ]
    else:
        all_combinations = [dict(zip(cfg.param_grid.keys(), values)) for values in product(*cfg.param_grid.values())]

        all_configs = [run.config for run in all_experiments]

        combinations_to_run = []
        for comb in all_combinations:
            experiment = ExperimentConfig(**comb)
            if experiment not in all_configs and experiment.total_steps_e >= experiment.learning_starts:
                combinations_to_run.append(experiment)

    for counter, experiment in enumerate(combinations_to_run, 1):
        print(counter)

        wandb_project_name = cfg.source_wandb_project_name
        if cfg.top_k > 0:
            wandb_project_name = cfg.target_wandb_project_name

        new_command = generate_command(cfg, experiment, wandb_project_name)

        local_file_path = 'temp_slurm_file'
        with open('temp_slurm_file', 'w') as local_file, open(cfg.file_to_replace, 'r') as original_file:
            for line in original_file:
                if cfg.original_command_pattern not in line:
                    local_file.write(line)
            local_file.write(new_command + "\n")

        os.replace(local_file_path, cfg.file_to_replace)
        print(new_command)

        if cfg.trigger_slurm:
            hours_counter = int(counter / 25)
            #sbatch_command = ["sbatch", "--qos=giladkz"]
            sbatch_command = ["sbatch"]
            if hours_counter > 0:
                sbatch_command.extend([f"--begin=now+{hours_counter}hours"])
            sbatch_command.append(cfg.file_to_replace)
            subprocess.run(sbatch_command)

        # Delete the local file
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

        time.sleep(0.01)


if __name__ == '__main__':
    args = tyro.cli(Config)
    hypersearch(args)
