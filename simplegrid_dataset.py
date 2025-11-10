import itertools
import random
from collections import defaultdict
from dataclasses import dataclass

from torch.utils.data import Dataset

from gym_simplegrid.envs import SimpleGridEnv


# Define a named tuple type for xy position
PositionXY = tuple[int, int]


@dataclass
class StartGoalPositions:
    start_xy: PositionXY
    goal_xy: PositionXY


@dataclass
class SimpleGridLevelDataset:
    level: int
    start_goal_pairs: list[StartGoalPositions]


@dataclass
class SimpleGridDataset:
    max_level: int
    levels: list[SimpleGridLevelDataset]


def precompute_path_lengths(env: SimpleGridEnv) -> dict[int, list[StartGoalPositions]]:
    """
    Precompute all path lengths for every possible start-goal pair in the grid.
    Returns a dictionary where keys are path lengths and values are lists of start-goal pairs.
    """
    path_length_map = defaultdict(list)

    for start_xy in itertools.product(range(env.unwrapped.nrow), range(env.unwrapped.ncol)):
        for goal_xy in itertools.product(range(env.unwrapped.nrow), range(env.unwrapped.ncol)):
            if start_xy != goal_xy:
                env.unwrapped.agent_xy = start_xy
                env.unwrapped.start_xy = start_xy
                env.unwrapped.goal_xy = goal_xy

                path_length = env.get_path_length()
                if path_length > 0:  # Ignore invalid or unreachable paths
                    path_length_map[path_length].append(StartGoalPositions(start_xy, goal_xy))

    return path_length_map


def find_start_goal_pairs(path_length_map: dict[int, list[StartGoalPositions]], target_path_length: int) -> SimpleGridLevelDataset:
    """
    Retrieve all start-goal pairs for the specified path length from the precomputed map.
    """
    return SimpleGridLevelDataset(
        level=target_path_length,
        start_goal_pairs=path_length_map.get(target_path_length, []),
    )


def create_levels_dataset(env: SimpleGridEnv, max_level: int = -1, start_level: int = 1) -> SimpleGridDataset:
    """
    Create a dataset of levels by precomputing all path lengths once and filtering for each level.
    """
    if max_level == -1:
        max_level = 14 if env.nrow == 8 and env.ncol == 8 else 30

    # Precompute path lengths
    path_length_map = precompute_path_lengths(env)

    # Create levels dataset
    levels_dataset = SimpleGridDataset(max_level=max_level, levels=[])

    for level in range(start_level, max_level + 1):
        levels_dataset.levels.append(find_start_goal_pairs(path_length_map, level))

    return levels_dataset


class GridDataset(Dataset):
    def __init__(
        self, level_dataset: SimpleGridDataset, train_ratio: float = 0.5, eval_ratio: float = 0.2, test_ratio: float = 0.3
    ):
        assert train_ratio + eval_ratio + test_ratio == 1.0

        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio

        self.train_data = []
        self.eval_data = []
        self.test_data = []

        self.generate_train_eval_test_datasets(level_dataset)

        # Group the data by level
        self.train_data_by_level = defaultdict(list)
        for start_xy, goal_xy, level in self.train_data:
            self.train_data_by_level[level].append((start_xy, goal_xy))

    def generate_train_eval_test_datasets(self, level_dataset: SimpleGridDataset):
        # Split the data by level
        train_data = []
        eval_data = []
        test_data = []

        for level_dataset in level_dataset.levels:
            start_goal_pairs = level_dataset.start_goal_pairs
            random.shuffle(start_goal_pairs)
            train_len = int(len(start_goal_pairs) * self.train_ratio)
            eval_len = int(len(start_goal_pairs) * self.eval_ratio)
            train_data.extend(
                [
                    (start_goal_positions.start_xy, start_goal_positions.goal_xy, level_dataset.level)
                    for start_goal_positions in start_goal_pairs[:train_len]
                ]
            )
            eval_data.extend(
                [
                    (start_goal_positions.start_xy, start_goal_positions.goal_xy, level_dataset.level)
                    for start_goal_positions in start_goal_pairs[
                                             train_len: train_len + eval_len
                                             ]
                ]
            )
            test_data.extend(
                [
                    (start_goal_positions.start_xy, start_goal_positions.goal_xy, level_dataset.level)
                    for start_goal_positions in start_goal_pairs[train_len + eval_len:]
                ]
            )

        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        random.shuffle(self.train_data)
        random.shuffle(self.eval_data)
        random.shuffle(self.test_data)

    @property
    def train_dataset(self) -> list[tuple[PositionXY, PositionXY, int]]:
        return self.train_data

    @property
    def eval_dataset(self) -> list[tuple[PositionXY, PositionXY, int]]:
        return self.eval_data

    @property
    def test_dataset(self) -> list[tuple[PositionXY, PositionXY, int]]:
        return self.test_data


def simplegrid_dataset_setup(
        env: SimpleGridEnv, max_level: int = -1, start_level: int = 1
) -> tuple[list[tuple[PositionXY, PositionXY, int]], list[tuple[PositionXY, PositionXY, int]], list[
    tuple[PositionXY, PositionXY, int]]]:
    levels_dataset: SimpleGridDataset = create_levels_dataset(env, max_level=max_level, start_level=start_level)
    env.close()

    grid_dataset = GridDataset(levels_dataset)

    return grid_dataset.train_dataset, grid_dataset.eval_dataset, grid_dataset.test_dataset
