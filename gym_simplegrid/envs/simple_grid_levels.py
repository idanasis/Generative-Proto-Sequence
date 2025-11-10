from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
from gymnasium import spaces, Env

import gym_simplegrid.rendering as r
from gym_simplegrid.window import Window

MAPS = {
    "8x8_empty": [
        "00000000",
        "00000000",
        "00000000",
        "00000000",
        "00000000",
        "00000000",
        "00000000",
        "00000000",
    ],
    "16x16_empty": [
        "1111111111111111",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1000000000000001",
        "1111111111111111"
    ],
    "16x16_room_v1": [
        '1111111111111111',
        '1000000100000001',
        '1100000101001101',
        '1000000100000101',
        '1001000000000001',
        '1100000100001001',
        '1000010100000001',
        '1101111111111011',
        '1000000110000001',
        '1000000100000001',
        '1000001100000001',
        '1000000100000001',
        '1000000000000001',
        '1000000100010001',
        '1000100100000001',
        '1111111111111111'
    ],
    "16x16_corridors_v1": [
        "1111111111111111",
        "1101111111110111",
        "1101111111110111",
        "1101111111110111",
        "1101111111110111",
        "1000000000000001",
        "1101111111110111",
        "1000000000000001",
        "1101111111110111",
        "1000000000000001",
        "1101111111110111",
        "1101111111110111",
        "1101111111110111",
        "1101111111110111",
        "1101111111110111",
        "1111111111111111",
    ],
    "16x16_obstacles_v1_15p": [
        "1111111111111111",
        "1000100000000001",
        "1001000000000001",
        "1000001000010001",
        "1000000010000001",
        "1000110000010001",
        "1000000000000001",
        "1000000000010001",
        "1010100000000001",
        "1000000001100001",
        "1000000010000111",
        "1110000001001001",
        "1110000101100001",
        "1000010000000101",
        "1010000010000001",
        "1111111111111111",
    ],
    "16x16_obstacles_v1_25p": [
        "1111111111111111",
        "1000000100001001",
        "1000001011000011",
        "1001111000000001",
        "1000010000100011",
        "1111000011000001",
        "1001001000000001",
        "1001000000010011",
        "1000000101000001",
        "1000011100100011",
        "1000010010100001",
        "1000000101100001",
        "1110010000010011",
        "1000010001010001",
        "1101001101000001",
        "1111111111111111",
    ],
    "24x24_empty": [
        "111111111111111111111111",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "100000000000000000000001",
        "111111111111111111111111"
    ],
    "24x24_obstacles_v1_15p": [
        "111111111111111111111111",
        "100000000000010000000001",
        "110000000000000000100101",
        "110000000001000001010011",
        "101000000000000000000001",
        "100000100001000010000001",
        "111010110000000000110001",
        "100000000000010000010001",
        "100000000000000000001001",
        "100000000000010000001001",
        "100000000000100100000001",
        "100000000000001011000001",
        "101000000100000000001001",
        "100010010000000000000001",
        "100000001000110000000001",
        "110100000100000000000001",
        "100000110000000110000001",
        "110101000001011001010001",
        "101101001000001000000101",
        "100000010000000010000001",
        "110001100000000000000001",
        "110000010000001010010001",
        "100100100000000001000001",
        "111111111111111111111111",
    ],
    "24x24_obstacles_v1_25p": [
        "111111111111111111111111",
        "110100010000010000010001",
        "100110000100010001010001",
        "100000000100110010000001",
        "100010001101010100010001",
        "100000001100000100000001",
        "101000010000101111000001",
        "101000000000100000110011",
        "100001010111000101100011",
        "110010000101011000001001",
        "101001110001000010001001",
        "101001101011000010110001",
        "100100000100000001000101",
        "100001100000000011000001",
        "110000000001000001000001",
        "111000000001000000010011",
        "110000010001010000000011",
        "100101000000100101100001",
        "110100000011001000000011",
        "100000010000000000000001",
        "110011000000010001010001",
        "100110000010100010100001",
        "100000110100000000010111",
        "111111111111111111111111",
    ],
}


class RewardStrategy(Enum):
    DEFAULT = "default"
    NEGATIVE_BASED_ON_MAP_SIZE = "negative_based_on_map_size"
    NEGATIVE_BASED_ON_MAP_SIZE_WITH_PENALTIES = "negative_based_on_map_size_with_penalties"
    NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES = "negative_based_on_max_level_with_penalties"
    SPARSE = "sparse"


class ObservationEncodingStrategy(Enum):
    DEFAULT = "default"
    VIN = "vin"


class PartialObservabilityStrategy(Enum):
    FULL = "full"  # Full observability (current behavior)
    LOCAL_VIEW = "local_view"  # Local view around agent with configurable radius


class SimpleGridEnv(Env):
    """
    Simple Grid Environment

    The environment is a grid with obstacles (walls) and agents. The agents can move in one of the four cardinal directions. If they try to move over an obstacle or out of the grid bounds, they stay in place. Each agent has a unique color and a goal state of the same color. The environment is episodic, i.e. the episode ends when the agents reaches its goal.

    To initialise the grid, the user must decide where to put the walls on the grid. This can be done by either selecting an existing map or by passing a custom map. To load an existing map, the name of the map must be passed to the `obstacle_map` argument. Available pre-existing map names are "4x4" and "8x8". Conversely, if to load custom map, the user must provide a map correctly formatted. The map must be passed as a list of strings, where each string denotes a row of the grid and it is composed by a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. An example of a 4x4 map is the following:
    ["0000",
     "0101",
     "0001",
     "1000"]

    Assume the environment is a grid of size (nrow, ncol). A state s of the environment is an elemente of gym.spaces.Discete(nrow*ncol), i.e. an integer between 0 and nrow * ncol - 1. Assume nrow=ncol=5 and s=10, to compute the (x,y) coordinates of s on the grid the following formula are used: x = s // ncol  and y = s % ncol.

    The user can also decide the starting and goal positions of the agent. This can be done by through the `options` dictionary in the `reset` method. The user can specify the starting and goal positions by adding the key-value pairs(`starts_xy`, v1) and `goals_xy`, v2), where v1 and v2 are both of type int (s) or tuple (x,y) and represent the agent starting and goal positions respectively.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}
    FREE: int = 0
    OBSTACLE: int = 1
    MOVES: dict[int, tuple] = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),  # RIGHT
    }

    def __init__(
        self,
        obstacle_map: str | list[str],
        max_level: int,
        render_mode: str | None = None,
        reward_strategy: RewardStrategy = RewardStrategy.DEFAULT,
        observation_encoding_strategy: ObservationEncodingStrategy = ObservationEncodingStrategy.DEFAULT,
        partial_observability_strategy: PartialObservabilityStrategy = PartialObservabilityStrategy.FULL,
        view_radius: int = 3,
        is_slippery: bool = False,
        slippery_prob: float = 1/3,
        sticky_action_prob: float = 0.0,
        random_action_prob: float = 0.0,
        max_steps: int = 100,
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        agent_color: str
            Color of the agent. The available colors are: red, green, blue, purple, yellow, grey and black. Note that the goal cell will have the same color.
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from a set of pre-existing maps. The names of the available pre-existing maps are "4x4" and "8x8". If a list of strings is passed, the map provided by the user is parsed and loaded. The map must be a list of strings, where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.
            An example of a 4x4 map is the following:
            ["0000",
             "0101",
             "0001",
             "1000"]
        """

        # Env configuration
        self.obstacles = self.parse_obstacle_map(obstacle_map)  # walls
        self.nrow, self.ncol = self.obstacles.shape
        self.reward_strategy = reward_strategy
        self.observation_encoding_strategy = observation_encoding_strategy
        self.partial_observability_strategy = partial_observability_strategy
        self.view_radius = view_radius
        self.is_slippery = is_slippery
        self.slippery_prob = slippery_prob
        self.sticky_action_prob = sticky_action_prob
        self.random_action_prob = random_action_prob
        self.max_steps = max_steps
        self.step_count = 0
        self.reset_count = 0
        self.start_goal_count = 0
        self.start_goal_dataset = []
        self.last_action = None  # Track last action for sticky actions

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.internal_observation_space = spaces.Discrete(n=self.nrow * self.ncol)
        
        # Determine observation dimensions based on partial observability strategy
        if self.partial_observability_strategy == PartialObservabilityStrategy.FULL:
            obs_height, obs_width = self.nrow, self.ncol
        else:  # LOCAL_VIEW
            obs_height = obs_width = 2 * self.view_radius + 1
        
        # Observations are the encoding of the grid
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype="uint8",
        )

        # Rendering configuration
        self.render_mode = render_mode
        self.window = None
        self.agent_color = "yellow"
        self.tile_cache = {}
        self.fps = self.metadata["render_fps"]
        self.max_level = max_level
        # self.frames = []
        
        # Initialize random number generator (will be properly seeded in reset())
        self.np_random = np.random.default_rng()

    def reset(self, seed: int | None = None, options: Optional[dict] = None) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """
        if options is None:
            options = dict()
            # options = {
            #     'start_loc': (0, 2),
            #     'goal_loc': (3, 0),
            # }
            # start_locs = [(0, 0), (0, 7), (7, 0), (0, 4)]
            # # start_locs = [(0, 0)]
            # # # goal_locs = [(7, 7)]
            # goal_locs = [(7, 7), (5, 3), (7, 4), (2, 4)]
            #
            # start_loc = start_locs[self.reset_count % len(start_locs)]
            # goal_loc = goal_locs[self.reset_count % len(goal_locs)]
            start_goal_locs = self.start_goal_dataset[self.reset_count % len(self.start_goal_dataset)]
            # options = {
            #     "start_loc": start_loc,
            #     "goal_loc": goal_loc,
            # }
            options = {
                "start_loc": start_goal_locs[0],
                "goal_loc": start_goal_locs[1],
                "level": start_goal_locs[2],
            }
            #
            # options = {
            #     "start_loc": (0, 0),
            #     "goal_loc": (7, 7),
            # }

        # Set seed
        super().reset(seed=seed)
        
        # Update our random number generator with the provided seed
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # parse options
        self.start_xy = self.parse_state_option("start_loc", options)
        self.goal_xy = self.parse_state_option("goal_loc", options)
        self.level = options["level"]

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.last_action = None  # Reset last action for sticky actions

        # Check integrity
        self.integrity_checks()
        # assert (
        #     self.find_path() is not None
        # ), f"No path found from start position to goal position."

        # Step count since episode start
        self.step_count = 0

        self.reset_count += 1
        # if self.reset_count % len(goal_locs) == 0:
        #     self.start_goal_count += 1

        self.render()

        return self.encode(), self.get_info()

    def _get_perpendicular_actions(self, action: int) -> tuple[int, int]:
        """
        Get the two perpendicular actions for a given action.
        
        Args:
            action: The original action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            Tuple of two perpendicular actions
        """
        if action == 0 or action == 1:  # UP or DOWN
            return 2, 3  # LEFT, RIGHT
        else:  # LEFT or RIGHT
            return 0, 1  # UP, DOWN
    
    def _apply_stochastic_action(self, intended_action: int) -> int:
        """
        Apply stochastic effects to the intended action.
        
        Args:
            intended_action: The action the agent wants to take
            
        Returns:
            The actual action that will be executed
        """
        # First check for sticky action
        if self.sticky_action_prob > 0 and self.last_action is not None:
            if self.np_random.random() < self.sticky_action_prob:
                return self.last_action
        
        # Then check for random action selection
        if self.random_action_prob > 0:
            if self.np_random.random() < self.random_action_prob:
                return self.np_random.integers(0, len(self.MOVES))
        
        # Then check for slippery movement
        if self.is_slippery:
            rand_val = self.np_random.random()
            
            if rand_val < self.slippery_prob:
                # Execute intended action
                return intended_action
            else:
                # Split the remaining probability equally between perpendicular directions
                remaining_prob = 1.0 - self.slippery_prob
                perp_prob_each = remaining_prob / 2.0
                
                if rand_val < self.slippery_prob + perp_prob_each:
                    # Move in first perpendicular direction
                    perp1, _ = self._get_perpendicular_actions(intended_action)
                    return perp1
                else:
                    # Move in second perpendicular direction
                    _, perp2 = self._get_perpendicular_actions(intended_action)
                    return perp2
        
        return intended_action

    def step(self, action: int):
        """
        Take a step in the environment.
        """
        assert action in self.action_space

        self.step_count += 1

        # Apply stochastic effects to get the actual action
        actual_action = self._apply_stochastic_action(action)
        
        # Store the intended action for next step (for sticky actions)
        self.last_action = action

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[actual_action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)

        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(
            target_row, target_col
        ):
            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        self.render()

        return self.encode(), self.reward, self.done, False, self.get_info()

    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        >>> my_map = ["001", "010", "011]
        >>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 1, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list):
            map_str = np.asarray(obstacle_map, dtype="c")
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        elif isinstance(obstacle_map, str):
            map_str = MAPS[obstacle_map]
            map_str = np.asarray(map_str, dtype="c")
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        else:
            raise ValueError(
                f"You must provide either a map of obstacles or the name of an existing map. Available existing maps are {', '.join(MAPS.keys())}."
            )

    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple):
                return state
            else:
                raise TypeError(f"Allowed types for `{state_name}` are int or tuple.")
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(
                f"Key `{state_name}` not found in `options`. Random sampling a valid value for it:"
            )
            logger.info(f"...`{state_name}` has value: {state}")
            return state

    def sample_valid_state_xy(self) -> tuple:
        state = self.internal_observation_space.sample()
        pos_xy = self.to_xy(state)
        while not self.is_free(*pos_xy):
            state = self.internal_observation_space.sample()
            pos_xy = self.to_xy(state)
        return pos_xy

    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert (
            self.obstacles[self.start_xy] == self.FREE
        ), f"Start position {self.start_xy} overlaps with a wall."
        assert (
            self.obstacles[self.goal_xy] == self.FREE
        ), f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(
            *self.start_xy
        ), f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(
            *self.goal_xy
        ), f"Goal position {self.goal_xy} is out of bounds."
        assert (
            self.goal_xy != self.start_xy
        ), f"Goal position can't be same as start position"

    def to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.obstacles[row, col] == self.FREE

    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if self.reward_strategy == RewardStrategy.NEGATIVE_BASED_ON_MAP_SIZE:
            if (x, y) == self.goal_xy:
                return 1.0
            else:
                return -1.0 / (self.nrow * self.ncol)
        elif self.reward_strategy == RewardStrategy.NEGATIVE_BASED_ON_MAP_SIZE_WITH_PENALTIES or \
             self.reward_strategy == RewardStrategy.NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES:
            if (x, y) == self.goal_xy:
                return 1.0
            else:
                if self.reward_strategy == RewardStrategy.NEGATIVE_BASED_ON_MAP_SIZE_WITH_PENALTIES:
                    reward = -1.0 / (self.nrow * self.ncol)
                else:
                    reward = -1.0 / self.max_level

                if not self.is_in_bounds(x, y) or not self.is_free(x, y):
                    reward *= 3.0
                return reward
        elif self.reward_strategy == RewardStrategy.SPARSE:
            if (x, y) == self.goal_xy:
                return 1 - 0.9 * (self.step_count / self.max_steps)
            else:
                return 0.0
        else:
            if not self.is_in_bounds(x, y):
                return -1.0
            elif not self.is_free(x, y):
                return -1.0
            elif (x, y) == self.goal_xy:
                return 1.0
            else:
                return 0.0

    def get_obs(self) -> np.ndarray:
        return self.to_s(*self.agent_xy)

    def get_info(self) -> dict:
        return {"agent_xy": self.agent_xy}

    def close(self):
        """
        Close the environment.
        """
        if self.window:
            self.window.close()
        return None

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            img = self.render_frame()
            if not self.window:
                self.window = Window()
                self.window.show(block=False)
            caption = ""
            self.window.show_img(img, caption, self.fps)
            return None
        elif self.render_mode == "rgb_array":
            return self.render_frame()
        # elif mode == "rgb_array_list":
        #     img = self.render_frame(caption=caption)
        #     self.frames.append(img)
        #     return self.frames
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self, tile_size=r.TILE_PIXELS, highlight_mask=None):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly
        render it, we have to pass (y,x) to the grid.render method.

        tile_size: tile size in pixels
        """
        width = self.ncol
        height = self.nrow

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render grid with obstacles
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.obstacles[x, y] == self.OBSTACLE:
                    cell = r.Wall(color="black")
                else:
                    cell = None

                img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render start
        x, y = self.start_xy
        cell = r.ColoredTile(color="red")
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render goal
        x, y = self.goal_xy
        cell = r.ColoredTile(color="green")
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render agent
        x, y = self.agent_xy
        cell = r.Agent(color=self.agent_color)
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        return img

    def render_cell(
        self, obj: r.WorldObj, highlight=False, tile_size=r.TILE_PIXELS, subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (None, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = (
            np.zeros(
                shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
            )
            + 255
        )

        if obj != None:
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            r.highlight_img(img)

        # Draw the grid lines (top and left edges)
        r.fill_coords(img, r.point_in_rect(0, 0.031, 0, 1), (170, 170, 170))
        r.fill_coords(img, r.point_in_rect(0, 1, 0, 0.031), (170, 170, 170))

        # Downsample the image to perform supersampling/anti-aliasing
        img = r.downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img

        return img

    def update_cell_in_frame(self, img, x, y, cell, tile_size):
        """
        Parameters
        ----------
        img : np.ndarray
            Image to update.
        x : int
            x-coordinate of the cell to update.
        y : int
            y-coordinate of the cell to update.
        cell : r.WorldObj
            New cell to render.
        tile_size : int
            Size of the cell in pixels.
        """
        tile_img = self.render_cell(cell, tile_size=tile_size)
        height_min = x * tile_size
        height_max = (x + 1) * tile_size
        width_min = y * tile_size
        width_max = (y + 1) * tile_size
        img[height_min:height_max, width_min:width_max, :] = tile_img
        return img

    def encode(self):
        """
        Produce a compact numpy encoding of the grid
        """
        if self.partial_observability_strategy == PartialObservabilityStrategy.FULL:
            return self._encode_full_observation()
        else:  # LOCAL_VIEW
            return self._encode_partial_observation()
    
    def _encode_full_observation(self):
        """
        Encode the full grid observation (original behavior)
        """
        if self.observation_encoding_strategy == ObservationEncodingStrategy.DEFAULT:
            array = np.zeros((self.nrow, self.ncol, 3), dtype="uint8")

            for x in range(self.nrow):
                for y in range(self.ncol):
                    if self.obstacles[x, y] == self.OBSTACLE:
                        array[x, y, :] = r.Wall().encode()
                    else:
                        array[x, y, 0] = r.OBJECT_TO_IDX["empty"]
                        array[x, y, 1] = 0
                        array[x, y, 2] = 0

            # Render start
            x, y = self.start_xy
            cell = r.ColoredTile(color="red")
            array[x, y, :] = cell.encode()

            # Render goal
            x, y = self.goal_xy
            cell = r.ColoredTile(color="green")
            array[x, y, :] = cell.encode()
            array[x, y, 0] = r.OBJECT_TO_IDX["goal"]

            # Render agent
            x, y = self.agent_xy
            cell = r.Agent(color=self.agent_color)
            array[x, y, :] = cell.encode()
            array[x, y, 2] = 1

            return array
        else:
            agent_channel = np.zeros((self.nrow, self.ncol), dtype="uint8")
            x, y = self.agent_xy
            agent_channel[x, y] = 1

            goal_channel = np.zeros((self.nrow, self.ncol), dtype="uint8")
            x, y = self.goal_xy
            goal_channel[x, y] = 10

            obstacles_channel = np.copy(self.obstacles).astype("uint8")

            array = np.stack([agent_channel, goal_channel, obstacles_channel], axis=2)

            return array
    
    def _encode_partial_observation(self):
        """
        Encode a partial observation centered around the agent
        """
        obs_size = 2 * self.view_radius + 1
        agent_x, agent_y = self.agent_xy
        
        if self.observation_encoding_strategy == ObservationEncodingStrategy.DEFAULT:
            array = np.zeros((obs_size, obs_size, 3), dtype="uint8")
            
            # Fill the local view
            for i in range(obs_size):
                for j in range(obs_size):
                    # Calculate world coordinates
                    world_x = agent_x + i - self.view_radius
                    world_y = agent_y + j - self.view_radius
                    
                    # Check if within bounds
                    if self.is_in_bounds(world_x, world_y):
                        if self.obstacles[world_x, world_y] == self.OBSTACLE:
                            array[i, j, :] = r.Wall().encode()
                        else:
                            array[i, j, 0] = r.OBJECT_TO_IDX["empty"]
                            array[i, j, 1] = 0
                            array[i, j, 2] = 0
                        
                        # Check if goal is visible
                        if (world_x, world_y) == self.goal_xy:
                            cell = r.ColoredTile(color="green")
                            array[i, j, :] = cell.encode()
                            array[i, j, 0] = r.OBJECT_TO_IDX["goal"]
                        
                        # Check if start is visible (for reference)
                        if (world_x, world_y) == self.start_xy:
                            cell = r.ColoredTile(color="red")
                            array[i, j, :] = cell.encode()
                    else:
                        # Out of bounds - treat as wall
                        array[i, j, :] = r.Wall().encode()
            
            # Agent is always at center of observation
            center = self.view_radius
            cell = r.Agent(color=self.agent_color)
            array[center, center, :] = cell.encode()
            array[center, center, 2] = 1
            
            return array
        else:  # VIN encoding
            array = np.zeros((obs_size, obs_size, 3), dtype="uint8")
            
            # Fill the local view
            for i in range(obs_size):
                for j in range(obs_size):
                    # Calculate world coordinates
                    world_x = agent_x + i - self.view_radius
                    world_y = agent_y + j - self.view_radius
                    
                    # Channel 0: Agent position (always center in partial obs)
                    if i == self.view_radius and j == self.view_radius:
                        array[i, j, 0] = 10  # Agent
                    else:
                        array[i, j, 0] = 1   # Empty
                    
                    # Channel 1: Goal position (if visible)
                    if self.is_in_bounds(world_x, world_y) and (world_x, world_y) == self.goal_xy:
                        array[i, j, 1] = 2  # Goal color
                    else:
                        array[i, j, 1] = 0
                    
                    # Channel 2: Obstacles
                    if self.is_in_bounds(world_x, world_y):
                        array[i, j, 2] = self.obstacles[world_x, world_y]
                    else:
                        array[i, j, 2] = 1  # Out of bounds treated as obstacle
            
            return array

        #     width = self.ncol
        #     height = self.nrow

        #     if vis_mask is None:
        #         vis_mask = np.ones((width, height), dtype=bool)

        #     array = np.zeros((width, height, 3), dtype='uint8')

        #     for i in range(width):
        #         for j in range(height):
        #             if vis_mask[i, j]:
        #                 v = self.get(i, j)

        #                 if v is None:
        #                     array[i, j, 0] = r.OBJECT_TO_IDX['empty']
        #                     array[i, j, 1] = 0
        #                     array[i, j, 2] = 0

        #                 else:
        #                     array[i, j, :] = v.encode()

        #     return array
        pass

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        #     width, height, channels = array.shape
        #     assert channels == 3

        #     vis_mask = np.ones(shape=(width, height), dtype=bool)

        #     grid = SimpleGrid(width, height)
        #     for i in range(width):
        #         for j in range(height):
        #             type_idx, color_idx, state = array[i, j]
        #             v = WorldObj.decode(type_idx, color_idx, state)
        #             grid.set(i, j, v)
        #             vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        #     return grid, vis_mask
        pass

    def find_path(self):
        def is_valid_position(row: int, col: int) -> bool:
            # Check if the position is within the grid boundaries
            if not self.is_in_bounds(row, col):
                return False

            # Check if the position is a wall or an obstacle
            if not self.is_free(row, col):
                return False

            return True

        agent_pos = self.agent_xy
        goal_pos = self.goal_xy

        try:
            self.integrity_checks()
        except:
            return None

        queue = deque([agent_pos])
        visited = set()
        parent_map = {}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            current_pos = queue.popleft()

            if current_pos == goal_pos:
                # Reconstruct path
                path = []
                while current_pos:
                    path.append(current_pos)
                    current_pos = parent_map.get(current_pos)
                return path[::-1]  # Reverse path

            visited.add(current_pos)
            row, col = current_pos

            for dr, dc in directions:
                neighbor = (row + dr, col + dc)
                if (
                        neighbor not in visited
                        and is_valid_position(neighbor[0], neighbor[1])
                ):
                    queue.append(neighbor)
                    parent_map[neighbor] = current_pos
                    visited.add(neighbor)

        return None

    def get_path_length(self) -> int:
        """
        Calculate the length of the path from the start location to the goal location.

        Returns:
            int: The length of the path. If no path is found, returns -1.
        """
        path = self.find_path()
        if path is None:
            return -1
        else:
            return (
                len(path) - 1
            )  # Subtract 1 since the start position is included in the path
