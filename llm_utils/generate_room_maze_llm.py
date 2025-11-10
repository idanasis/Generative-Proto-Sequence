import random
import copy
import matplotlib.pyplot as plt
import numpy as np


BASE_MAZE = [
    "1111111111111111",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1111111111111111",  # <-- Horizontal dividing wall (row=7)
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1000000100000001",
    "1111111111111111"
    # ^ Outer boundary (row=0, row=15)
    #   and vertical dividing wall (col=7) in each row
]

def create_maze(seed_val):
    """
    Create a modified maze from the base structure using:
      - seeded random entrance selection
      - seeded random obstacle placement
      - ensuring each quadrant has exactly two entrances
      - placing 13 obstacles (5% of 256) not blocking entrances
    """
    rnd = random.Random(seed_val)

    # Convert BASE_MAZE to a mutable 2D list of characters
    maze_2d = [list(row) for row in BASE_MAZE]

    # ------------------------------------------------
    # 1) CREATE ENTRANCES
    # ------------------------------------------------
    #
    # We pick:
    #   - top-left vs top-right:   row in [1..6],   col=7
    #   - top-left vs bottom-left: row=7,          col in [1..6]
    #   - top-right vs bottom-right: row=7,        col in [8..14]
    #   - bottom-left vs bottom-right: col=7,      row in [8..13]
    #
    # That gives each quadrant exactly two openings.

    # Top-Left <-> Top-Right
    tl_tr_row = rnd.randint(1, 6)
    maze_2d[tl_tr_row][7] = '0'  # open the wall

    # Top-Left <-> Bottom-Left
    tl_bl_col = rnd.randint(1, 6)
    maze_2d[7][tl_bl_col] = '0'

    # Top-Right <-> Bottom-Right
    tr_br_col = rnd.randint(8, 14)
    maze_2d[7][tr_br_col] = '0'

    # Bottom-Left <-> Bottom-Right
    bl_br_row = rnd.randint(8, 13)
    maze_2d[bl_br_row][7] = '0'

    # Collect the newly created entrances for adjacency checks
    entrances = [
        (tl_tr_row, 7),
        (7, tl_bl_col),
        (7, tr_br_col),
        (bl_br_row, 7)
    ]

    # ------------------------------------------------
    # 2) PLACE OBSTACLES
    # ------------------------------------------------
    #
    # We want to place ~13 obstacles (5% of 256).
    # We'll distribute them anywhere in open cells (0),
    # with these conditions:
    #  - do not overwrite an entrance
    #  - do not place adjacent to entrances
    #  - do not overwrite existing walls
    #  - keep track that we only place up to 13
    #

    num_obstacles_needed = 13
    obstacles_placed = 0

    # A small helper to check adjacency to entrances (including the cell itself)
    def is_adjacent_to_entrance(r, c):
        for (er, ec) in entrances:
            if abs(r - er) <= 1 and abs(c - ec) <= 1:
                return True
        return False

    # Collect all valid candidate cells for obstacles
    candidates = []
    for r in range(16):
        for c in range(16):
            if maze_2d[r][c] == '0':
                # Not a wall, so it's open space
                # Check if it's an entrance or adjacent to one
                if (r, c) not in entrances and not is_adjacent_to_entrance(r, c):
                    candidates.append((r, c))

    # Shuffle candidates for random placement
    rnd.shuffle(candidates)

    # Try placing up to num_obstacles_needed obstacles
    for (r, c) in candidates:
        if obstacles_placed >= num_obstacles_needed:
            break
        # Place an obstacle
        maze_2d[r][c] = '1'
        obstacles_placed += 1

    # If needed, one could do a BFS check here to ensure connectivity
    # For simplicity, we assume it remains connected as we do not block
    # the newly created entrances or their immediate neighbors.

    # ------------------------------------------------
    # 3) CONVERT BACK TO STRINGS
    # ------------------------------------------------
    updated_rows = ["".join(row_list) for row_list in maze_2d]

    return {
        'maze': updated_rows
    }


if __name__ == "__main__":
    # Generate 5 mazes using different seeds
    seeds = [42, 1234, 9999, 2024, 2025]

    all_mazes = []
    for s in seeds:
        maze_dict = create_maze(s)
        all_mazes.append(maze_dict)
        # Convert maze data to a NumPy array
        maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])

        # Plot the maze
        plt.figure(figsize=(8, 8))
        plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
        plt.title("Maze Visualization")
        plt.axis('off')  # Remove axes for cleaner visualization
        plt.show()

    # Print them out or process as needed
    # Here we just print each maze in the required dictionary format
    for i, maze_dict in enumerate(all_mazes):
        print(f"=== Maze #{i+1} (seed={seeds[i]}) ===")
        print(maze_dict)
        print()
