import os
import subprocess
import time
from dataclasses import dataclass, field, asdict, fields
from itertools import product
from typing import Any

import tyro
from wandb.sdk.wandb_run import Run

import wandb


@dataclass
class Config:
    file_to_replace: str = "sbatch_temporl"
    original_command_pattern: str = "poetry run python temporl_simplegrid_levels.py"
    source_wandb_project_name: str = ""
    target_wandb_project_name: str = ""
    top_k: int = -1
    trigger_slurm: bool = True
    override_based_on_target_config: bool = True

    param_grid: dict[str, Any] = field(default_factory=lambda: {
        'seed': [123],
        'save_model_strategy': ['SUCCESS_RATE'],
        'val_eval_freq': [5000],
        'train_eval_freq': [5000],
        'reward_strategy': ['NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES'],
        'observation_encoding_strategy': ['DEFAULT'],
        'partial_observability_strategy': ['FULL'],
        'view_radius': [3],
        'is_slippery': [False],
        'slippery_prob': [1/3],
        'sticky_action_prob': [0.0],
        'random_action_prob': [0.0],
        'learning_rate': [1e-3, 1e-4],
        'buffer_size': [50000, 10000],
        'tau': [0.005, 0.01],
        'batch_size': [256],
        'learning_starts': [1000],
        'train_frequency': [2],
        'target_network_frequency': [10, 100],
        'end_e': [0.1],
        'exploration_fraction': [0.1, 0.3, 0.5],
        'total_timesteps': [1500000],
        'max_level': [30],
        'start_level': [20],
        'max_episode_steps': [75],
        'obstacle_map': ["24x24_obstacles_v1_15p"],
        'train_dataset_size': [100],
        'train_eval_dataset_size': [100],
        'val_dataset_size': [100],
        'test_dataset_size': [1000],
        'action_linear_layers': ["128 32"],
        'action_linear_layers_activation_function': ["leaky_relu"],
        'skip_linear_layers': ["128 32"],
        'skip_linear_layers_activation_function': ["leaky_relu"],
        'weight_sharing': [True],
        'skip_dim': [10]
    })

    target_config: dict[str, Any] = field(default_factory=lambda: {
        'total_timesteps': None,
        'obstacle_map': None,
        'buffer_size': None,
        'reward_strategy': None,
        'start_level': None,
        'max_level': None,
        'train_dataset_size': None,
        'partial_observability_strategy': None,
        'view_radius': None,
        'is_slippery': None,
        'slippery_prob': None,
        'sticky_action_prob': None,
        'random_action_prob': None
    })


@dataclass(frozen=True, eq=True)
class ExperimentConfig:
    seed: int
    save_model_strategy: str
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
    learning_rate: float
    buffer_size: int
    tau: float
    batch_size: int
    learning_starts: int
    train_frequency: int
    target_network_frequency: int
    end_e: float
    exploration_fraction: int
    total_timesteps: int
    max_level: int
    start_level: int
    max_episode_steps: int
    obstacle_map: str
    train_dataset_size: int
    train_eval_dataset_size: int
    val_dataset_size: int
    test_dataset_size: int
    action_linear_layers: str
    action_linear_layers_activation_function: str
    skip_linear_layers: str
    skip_linear_layers_activation_function: str
    weight_sharing: bool
    skip_dim: int


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
        'reward_strategy': 'RewardStrategy.',
        'observation_encoding_strategy': 'ObservationEncodingStrategy.',
        'partial_observability_strategy': 'PartialObservabilityStrategy.'
    }
    for key, prefix in prefix_removals.items():
        if key in transformed:
            transformed[key] = transformed[key].removeprefix(prefix)

    # None to string conversion
    none_to_string_fields = ['train_dataset_size',
                             'train_eval_dataset_size', 'val_dataset_size', 'test_dataset_size']
    for field in none_to_string_fields:
        if field in transformed:
            transformed[field] = None if transformed[field] is None else transformed[field]

    # Join list fields
    list_fields = ['action_linear_layers', 'skip_linear_layers']
    for field in list_fields:
        if field in transformed:
            transformed[field] = " ".join(str(x) for x in transformed[field])

    return transformed


def get_finished_runs(api: wandb.Api, wandb_project_name: str) -> tuple[list[RunData], list[Run]]:
    required_metrics = [
        "test-eval/mean_episodic_success",
        "train-eval/best_episodic_success",
        "val-eval/best_episodic_success"
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
            
            # Provide default values for missing required fields
            field_defaults = {
                'partial_observability_strategy': 'FULL',
                'view_radius': 3,
                'is_slippery': False,
                'slippery_prob': 1/3,
                'sticky_action_prob': 0.0,
                'random_action_prob': 0.0
            }
            
            for field_name, default_value in field_defaults.items():
                if field_name not in filtered_config:
                    filtered_config[field_name] = default_value

            run_data = RunData(
                config=ExperimentConfig(**filtered_config),
                val_eval=metrics.get('val-eval/best_episodic_success'),
                train_eval=metrics.get('train-eval/best_episodic_success'),
                test_eval=metrics.get('test-eval/mean_episodic_success')
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
        ATTR_MATCHING_L: list[str] = ['train_dataset_size', 'total_timesteps', 'obstacle_map', 'action_linear_layers',
                                      'skip_linear_layers', 'buffer_size',
                                      'max_level', 'start_level', 'reward_strategy', 'weight_sharing', 'skip_dim', 'seed']

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
            if experiment not in all_configs:
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
            sbatch_command = ["sbatch", "--qos=giladkz"]
            #sbatch_command = ["sbatch"]
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
