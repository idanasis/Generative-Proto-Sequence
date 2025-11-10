# import matplotlib.pyplot as plt
# import numpy as np
#
# def create_maze():
#     """
#     Creates a 16x16 maze with corridors at rows 0, 5, 10, 15 and columns 0, 4, 8, 12, 15.
#     Returns a dictionary with a single key 'maze', mapping to a list of binary strings.
#     """
#     maze_rows = []
#
#     for row in range(16):
#         row_cells = []
#         for col in range(16):
#             # If this cell is in one of the 'corridor rows' or 'corridor columns', it's 0 (open).
#             if row in [0, 5, 10, 15] or col in [0, 4, 8, 12, 15]:
#                 row_cells.append('0')
#             else:
#                 row_cells.append('1')
#
#         # Join the row's cells into a binary string.
#         maze_rows.append("".join(row_cells))
#
#     # Wrap in a dictionary as requested
#     maze_dict = {
#         'maze': maze_rows
#     }
#
#     return maze_dict
#
#
# if __name__ == "__main__":
#     maze_dict = create_maze()
#
#     # Print the resulting maze dictionary
#     print(maze_dict['maze'])
#
#     maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
#
#     # Plot the maze
#     plt.figure(figsize=(8, 8))
#     plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
#     plt.title("Maze Visualization")
#     plt.axis('off')  # Remove axes for cleaner visualization
#     plt.show()
#

# output1
#
# import random
#
# def generate_maze(seed):
#     """
#     Generates a 16x16 maze layout as a dictionary with a single key 'maze'.
#     The value is a list of 16 binary strings, each 16 characters wide.
#     0 = open corridor cell, 1 = wall.
#     """
#     random.seed(seed)
#
#     # 1) Create a 16x16 grid of walls
#     grid = [[1]*16 for _ in range(16)]
#
#     # 2) Carve a perimeter corridor
#     for i in range(16):
#         grid[0][i] = 0        # top row
#         grid[15][i] = 0       # bottom row
#         grid[i][0] = 0        # left column
#         grid[i][15] = 0       # right column
#
#     # 3) Randomly choose some interior rows to open fully
#     #    (you can adjust how many are chosen)
#     num_horizontal_corridors = 4
#     chosen_rows = random.sample(range(1, 15), num_horizontal_corridors)
#     for r in chosen_rows:
#         for c in range(16):
#             grid[r][c] = 0
#
#     # 4) Randomly choose some interior columns to open fully
#     #    (you can adjust how many are chosen)
#     num_vertical_corridors = 4
#     chosen_cols = random.sample(range(1, 15), num_vertical_corridors)
#     for c in chosen_cols:
#         for r in range(16):
#             grid[r][c] = 0
#
#     # Convert the 2D grid of 0/1 into a list of binary strings
#     maze_rows = []
#     for row in grid:
#         maze_rows.append("".join(str(x) for x in row))
#
#     return {
#         'maze': maze_rows
#     }
#
# if __name__ == "__main__":
#     # Example seeds for generating 5 different mazes
#     seeds = [42, 123, 999, 2022, 314]
#
#     # Generate and print each maze
#     for s in seeds:
#         maze_dict = generate_maze(s)
#         print(f"--- Maze for seed={s} ---")
#         # Print the dictionary directly
#         print(maze_dict)
#         print()
#
#
#         import matplotlib.pyplot as plt
#         import numpy as np
#
#         maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
#         # Plot the maze
#         plt.figure(figsize=(8, 8))
#         plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
#         plt.title("Maze Visualization")
#         plt.axis('off')  # Remove axes for cleaner visualization
#         plt.show()



# output2
# import random
#
# def generate_maze(seed=0, size=16, widen_prob=0.20):
#     """
#     Generates a connected maze on a size×size grid using a randomized DFS.
#     Each cell in the final maze is either:
#       0 -> open corridor
#       1 -> wall
#
#     `widen_prob`: Probability that, when carving, we thicken the corridor
#                   to two cells wide in the direction carved.
#     """
#
#     random.seed(seed)
#
#     # -- Initialize all cells as walls. --
#     # grid[r][c] == 1 means wall; 0 means corridor (open).
#     grid = [[1]*size for _ in range(size)]
#
#     # -- A simple stack-based DFS for maze carving. --
#     def neighbors(r, c):
#         """Valid 2-step neighbors (up, down, left, right) for DFS."""
#         directions = [(-1,0), (1,0), (0,-1), (0,1)]
#         random.shuffle(directions)  # Randomize exploration order
#         for dr, dc in directions:
#             rr = r + 2*dr
#             cc = c + 2*dc
#             if 0 <= rr < size and 0 <= cc < size:
#                 yield (rr, cc, dr, dc)
#
#     # Pick a random start cell on the grid such that it is in a "valid"
#     # region for 2-step moves. For simplicity, pick from odd indices or
#     # any cell that leaves enough room to carve outwards.
#     start_r = random.randrange(size)
#     start_c = random.randrange(size)
#
#     # Carve the starting cell
#     grid[start_r][start_c] = 0
#     stack = [(start_r, start_c)]
#
#     while stack:
#         r, c = stack[-1]
#         # Attempt to find an unvisited neighbor (2 cells away)
#         found_unvisited = False
#         for (rr, cc, dr, dc) in neighbors(r, c):
#             if grid[rr][cc] == 1:  # Not yet carved
#                 # Carve corridor between (r,c) and (rr,cc)
#                 grid[r + dr][c + dc] = 0   # the wall between
#                 grid[rr][cc] = 0          # the neighbor cell
#
#                 # Optionally, thicken corridor in that direction
#                 # with probability widen_prob
#                 if random.random() < widen_prob:
#                     # Extend "sideways" or along the path by one cell
#                     # to create a 2-cell-wide corridor
#                     # We'll do a simple approach: try to open a cell
#                     # adjacent to the newly opened path.
#                     # For instance, if we carved vertically, open left or right cell.
#                     if dr != 0:  # carved vertically
#                         # random choice to open left or right side
#                         side_dc = random.choice([-1, 1])
#                         side_c = c + dc + side_dc
#                         if 0 <= side_c < size:
#                             grid[r + dr][side_c] = 0
#                             # Also consider the neighbor's side
#                             side_c2 = cc + side_dc
#                             if 0 <= side_c2 < size:
#                                 grid[rr][side_c2] = 0
#                     else:
#                         # carved horizontally
#                         side_dr = random.choice([-1, 1])
#                         side_r = r + dr + side_dr
#                         if 0 <= side_r < size:
#                             grid[side_r][c + dc] = 0
#                             # Also the neighbor's side
#                             side_r2 = rr + side_dr
#                             if 0 <= side_r2 < size:
#                                 grid[side_r2][cc] = 0
#
#                 # Push neighbor on stack
#                 stack.append((rr, cc))
#                 found_unvisited = True
#                 break
#
#         if not found_unvisited:
#             # If none of the neighbors is unvisited, backtrack
#             stack.pop()
#
#     # Convert to list of strings
#     maze_rows = ["".join(str(cell) for cell in row) for row in grid]
#     return maze_rows
#
#
# if __name__ == "__main__":
#     """
#     Generate 5 distinct mazes with seeds 0..4.  Each is guaranteed
#     to be a single connected set of corridors (0's) surrounded by walls
#     (or partial walls), ensuring that from any 0-cell you can
#     reach any other 0-cell via up/down/left/right moves.
#     """
#
#     NUM_MAZES = 5
#     for s in range(NUM_MAZES):
#         maze_data = generate_maze(seed=s, size=16, widen_prob=0.20)
#         maze_dict = {
#             "maze": maze_data
#         }
#         # Print them out
#         print(f"MAZE with seed={s}")
#         print(maze_dict)
#         print()
#
#         import matplotlib.pyplot as plt
#         import numpy as np
#
#         maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
#         # Plot the maze
#         plt.figure(figsize=(8, 8))
#         plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
#         plt.title("Maze Visualization")
#         plt.axis('off')  # Remove axes for cleaner visualization
#         plt.show()


# import random
# import math
# import copy
#
# GRID_SIZE = 16
# INNER_TOP = 1
# INNER_LEFT = 1
# INNER_BOTTOM = GRID_SIZE - 2
# INNER_RIGHT = GRID_SIZE - 2
#
# def create_empty_grid():
#     """Create a 16x16 grid filled with walls (1). The outer border remains walls."""
#     grid = [[1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
#     return grid
#
# def can_place_block(grid, top, left, height, width):
#     """
#     Check if a block of given dimensions can be placed with its top-left corner at (top, left)
#     while ensuring a 1-cell-thick wall separates it from any other open cell (or block).
#     We check an expanded region (the block plus a 1-cell border around it).
#     """
#     # Determine the expanded region coordinates
#     exp_top = top - 1
#     exp_left = left - 1
#     exp_bottom = top + height
#     exp_right = left + width
#     # Make sure the expanded region is within bounds of the grid.
#     if exp_top < 0 or exp_left < 0 or exp_bottom >= GRID_SIZE or exp_right >= GRID_SIZE:
#         return False
#     # Check that every cell in the expanded region is still a wall (1)
#     for r in range(exp_top, exp_bottom + 1):
#         for c in range(exp_left, exp_right + 1):
#             if grid[r][c] == 0:
#                 return False
#     return True
#
# def place_block(grid, top, left, height, width):
#     """Carve out a rectangular block (set cells to 0) in the grid."""
#     for r in range(top, top + height):
#         for c in range(left, left + width):
#             grid[r][c] = 0
#
# def get_block_center(top, left, height, width):
#     """
#     Return the center (row, col) coordinate of the block.
#     (Using integer division; corridors will be carved from these approximate centers.)
#     """
#     center_row = top + height // 2
#     center_col = left + width // 2
#     return (center_row, center_col)
#
# def carve_corridor(grid, start, end):
#     """
#     Carve a 1-cell-wide corridor from start to end.
#     We use a simple “L‑shaped” corridor: horizontal then vertical.
#     """
#     (r1, c1) = start
#     (r2, c2) = end
#     # Horizontal corridor from c1 to c2 at row r1
#     if c1 <= c2:
#         for c in range(c1, c2 + 1):
#             grid[r1][c] = 0
#     else:
#         for c in range(c2, c1 + 1):
#             grid[r1][c] = 0
#     # Vertical corridor from r1 to r2 at column c2
#     if r1 <= r2:
#         for r in range(r1, r2 + 1):
#             grid[r][c2] = 0
#     else:
#         for r in range(r2, r1 + 1):
#             grid[r][c2] = 0
#
# def connect_blocks(grid, centers):
#     """
#     Connect all block centers by computing a Minimum Spanning Tree (MST) on the complete graph
#     of centers (using Manhattan distance), and then carve corridors along the MST edges.
#     """
#     n = len(centers)
#     # Use Prim’s algorithm
#     connected = [False] * n
#     min_dist = [math.inf] * n
#     parent = [-1] * n
#
#     # Start with block 0
#     min_dist[0] = 0
#
#     for _ in range(n):
#         # Choose the next vertex with smallest distance
#         u = None
#         for i in range(n):
#             if not connected[i] and (u is None or min_dist[i] < min_dist[u]):
#                 u = i
#         connected[u] = True
#
#         # Update distances for remaining vertices
#         for v in range(n):
#             if not connected[v]:
#                 # Manhattan distance between centers
#                 dist = abs(centers[u][0] - centers[v][0]) + abs(centers[u][1] - centers[v][1])
#                 if dist < min_dist[v]:
#                     min_dist[v] = dist
#                     parent[v] = u
#
#     # Carve corridors along the MST edges
#     for i in range(1, n):
#         if parent[i] != -1:
#             carve_corridor(grid, centers[i], centers[parent[i]])
#
# def grid_to_output_format(grid):
#     """Convert the grid (a list of lists) to the required output format: a list of 16 strings."""
#     return ["".join(str(cell) for cell in row) for row in grid]
#
# def generate_maze(seed):
#     random.seed(seed)
#     grid = create_empty_grid()
#
#     blocks = []    # Will store tuples: (top, left, height, width)
#     centers = []   # Block center positions for corridor connection
#
#     attempts = 0
#     max_attempts = 2000
#     while len(blocks) < 12 and attempts < max_attempts:
#         attempts += 1
#         # Choose random block dimensions (at least 2 cells in each direction)
#         height = random.randint(2, 4)
#         width = random.randint(2, 4)
#         # Choose a random position that is within the inner area and such that the block fits
#         top = random.randint(INNER_TOP, INNER_BOTTOM - height + 1)
#         left = random.randint(INNER_LEFT, INNER_RIGHT - width + 1)
#         if can_place_block(grid, top, left, height, width):
#             place_block(grid, top, left, height, width)
#             blocks.append((top, left, height, width))
#             centers.append(get_block_center(top, left, height, width))
#
#     if len(blocks) < 12:
#         raise ValueError("Failed to place 12 blocks with the given constraints.")
#
#     # Connect the blocks via corridors
#     connect_blocks(grid, centers)
#
#     # Make sure the outer border is walls (reinforce the frame)
#     for r in range(GRID_SIZE):
#         grid[r][0] = 1
#         grid[r][GRID_SIZE - 1] = 1
#     for c in range(GRID_SIZE):
#         grid[0][c] = 1
#         grid[GRID_SIZE - 1][c] = 1
#
#     return {'maze': grid_to_output_format(grid)}
#
# # Generate five mazes with different seeds
# mazes = []
# for seed in range(5):
#     maze_dict = generate_maze(seed)
#     mazes.append(maze_dict)
#
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
#     # Plot the maze
#     plt.figure(figsize=(8, 8))
#     plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
#     plt.title("Maze Visualization")
#     plt.axis('off')  # Remove axes for cleaner visualization
#     plt.show()
#
#
# # Output the five maze dictionaries.
# for i, maze in enumerate(mazes):
#     print(f"Maze {i+1}:")
#     print("{")
#     print("    'maze': [")
#     for row in maze['maze']:
#         print(f'        "{row}",')
#     print("    ]")
#     print("}")
#     print()





#!/usr/bin/env python3
# import random
# from collections import deque
# import copy
#
#
# def bfs_connected(grid):
#     """Return True if every open cell (0) in grid is reachable from any open cell.
#        Moves allowed: up, down, left, right."""
#     n = len(grid)
#     m = len(grid[0])
#     # find any open cell as starting point
#     start = None
#     for i in range(n):
#         for j in range(m):
#             if grid[i][j] == 0:
#                 start = (i, j)
#                 break
#         if start is not None:
#             break
#     if start is None:
#         return False  # no open cell found
#
#     seen = set([start])
#     dq = deque([start])
#     while dq:
#         i, j = dq.popleft()
#         for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             ni, nj = i + di, j + dj
#             if 0 <= ni < n and 0 <= nj < m:
#                 if grid[ni][nj] == 0 and (ni, nj) not in seen:
#                     seen.add((ni, nj))
#                     dq.append((ni, nj))
#     # Count open cells
#     total_open = sum(row.count(0) for row in grid)
#     return len(seen) == total_open
#
#
# def random_partition(total, parts, minimum):
#     """
#     Partition the integer (total) into (parts) parts, each at least (minimum).
#     Return a list of length parts that sums to total.
#     For example: total=11, parts=4, minimum=2  (since 4*2 = 8, remainder 3 distributed).
#     """
#     base = [minimum] * parts
#     remainder = total - minimum * parts
#     # Distribute the remainder randomly among the parts.
#     for _ in range(remainder):
#         i = random.randrange(parts)
#         base[i] += 1
#     return base
#
#
# def generate_maze(random_seed=None):
#     """
#     Generate one maze as a dictionary with key 'maze' and a list of 16 strings.
#
#     The maze has:
#       - A 16×16 grid.
#       - Outer border is walls (1).
#       - Exactly 12 rectangular corridor wall blocks arranged in 4 block‐rows and 3 block‐columns.
#         (Inside the inner 14×14 area, we reserve alternating “block‐rows” and “corridor‐rows” so that
#          each block is separated from its neighbours by at least one open cell.)
#       - The “corridor cells” (the open cells outside the blocks) are arranged in full rows or full columns,
#         so that every open cell is connected by up/down/left/right moves.
#     """
#     if random_seed is not None:
#         random.seed(random_seed)
#     else:
#         random.seed()
#
#     N = 16  # overall grid size
#     # initialize grid with 0’s (open) then later we will “paint” blocks as 1.
#     grid = [[0 for _ in range(N)] for _ in range(N)]
#     # force the outer border to be walls
#     for i in range(N):
#         grid[0][i] = 1
#         grid[N - 1][i] = 1
#         grid[i][0] = 1
#         grid[i][N - 1] = 1
#
#     # Our inner area is rows 1..14 and cols 1..14 (14×14)
#     # We will partition this inner area into alternating rows: block row, corridor row, block row, corridor, ...
#     #
#     # We want 4 block rows and (4-1)=3 horizontal corridors inserted between them.
#     # Thus the total number of inner rows used for blocks is 14 - 3 = 11.
#     # Each block row must have at least 2 rows.
#     block_row_heights = random_partition(11, 4, 2)
#     # Similarly, for the columns: we want 3 block columns and 2 vertical corridors.
#     # Total inner columns used for blocks: 14 - 2 = 12, each block column at least 2 columns.
#     block_col_widths = random_partition(12, 3, 2)
#
#     # Now decide the exact row indices for each block row and corridor row.
#     # We will fill rows 1..14. They will come in alternating segments:
#     # [block row 1] then 1 full corridor row, then [block row 2] then corridor, then [block row 3] then corridor, then [block row 4].
#     row_ptr = 1
#     block_row_ranges = []
#     corridor_rows = []  # list of row indices that are corridors (completely open)
#     for r in range(4):
#         start = row_ptr
#         end = row_ptr + block_row_heights[r] - 1
#         block_row_ranges.append((start, end))
#         row_ptr = end + 1
#         if r < 3:
#             corridor_rows.append(row_ptr)
#             row_ptr += 1
#
#     # Similarly for columns 1..14. They come as: block col 1, corridor column, block col 2, corridor, block col 3.
#     col_ptr = 1
#     block_col_ranges = []
#     corridor_cols = []
#     for c in range(3):
#         start = col_ptr
#         end = col_ptr + block_col_widths[c] - 1
#         block_col_ranges.append((start, end))
#         col_ptr = end + 1
#         if c < 2:
#             corridor_cols.append(col_ptr)
#             col_ptr += 1
#
#     # For debugging you could print the ranges:
#     # print("Block row ranges:", block_row_ranges)
#     # print("Corridor rows:", corridor_rows)
#     # print("Block col ranges:", block_col_ranges)
#     # print("Corridor cols:", corridor_cols)
#
#     # Now “paint” the blocks.
#     # Each block is the rectangle given by one block row range and one block col range.
#     # (The blocks do not extend into the corridor rows/columns.)
#     for (rstart, rend) in block_row_ranges:
#         for (cstart, cend) in block_col_ranges:
#             for i in range(rstart, rend + 1):
#                 for j in range(cstart, cend + 1):
#                     grid[i][j] = 1
#
#     # (The corridors remain 0.) By construction the blocks are separated by at least one row or column that is not painted.
#     # The outer border remains intact.
#     #
#     # In many maze‐algorithms one would “carve” extra openings to connect regions. In our construction the corridors
#     # (which occur in full rows/cols – the ones not part of any block) guarantee that every open cell (corridor cell)
#     # is connected via moves in the four cardinal directions.
#     #
#     # As a final “random touch” we will optionally open one extra cell in one random wall along each corridor boundary
#     # between adjacent blocks. (This does not change connectivity but varies the maze slightly.)
#     #
#     # For each horizontal corridor row (which separates two block rows), choose one column (inside the inner area)
#     # that lies between two blocks (i.e. not in a corridor column) and set that cell to open.
#     for r in corridor_rows:
#         # possible columns: between col 1 and 14 that are not already corridor columns.
#         possible_cols = [j for j in range(1, N - 1) if j not in corridor_cols]
#         if possible_cols:
#             c = random.choice(possible_cols)
#             grid[r][c] = 0
#     # Similarly for each vertical corridor column:
#     for c in corridor_cols:
#         possible_rows = [i for i in range(1, N - 1) if i not in corridor_rows]
#         if possible_rows:
#             i = random.choice(possible_rows)
#             grid[i][c] = 0
#
#     # Finally, verify that every open cell is connected.
#     if not bfs_connected(grid):
#         # In the (very unlikely) event connectivity fails,
#         # we open one extra random cell in one of the corridor rows.
#         # (In our construction connectivity should always hold.)
#         for r in corridor_rows:
#             for j in range(1, N - 1):
#                 grid[r][j] = 0
#
#     # Format the maze as a list of strings.
#     maze_rows = [''.join(str(cell) for cell in row) for row in grid]
#     return {'maze': maze_rows}
#
#
# def main():
#     # Generate five distinct mazes using different seeds.
#     mazes = []
#     for seed in [17, 42, 123, 2020, 777, 10, 20, 30, 40]:
#         maze_dict = generate_maze(random_seed=seed)
#         # Optionally, check connectivity:
#         if not bfs_connected([list(map(int, list(row))) for row in maze_dict['maze']]):
#             print("Error: Maze not fully connected!")
#         mazes.append(maze_dict)
#
#         import matplotlib.pyplot as plt
#         import numpy as np
#
#         maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
#         # Plot the maze
#         plt.figure(figsize=(8, 8))
#         plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
#         plt.title("Maze Visualization")
#         plt.axis('off')  # Remove axes for cleaner visualization
#         plt.show()
#
#     # Print the mazes
#     for idx, m in enumerate(mazes, start=1):
#         print(f"\nMaze #{idx}:")
#         for row in m['maze']:
#             print(row)
#
#
# if __name__ == '__main__':
#     main()


import random


def choose_corridors(num_choices, min_val, max_val):
    """
    Choose num_choices positions between min_val and max_val (inclusive)
    such that no two chosen positions are consecutive.
    This function uses a rejection loop.
    """
    while True:
        # Pick random unique positions
        positions = sorted(random.sample(range(min_val, max_val + 1), num_choices))
        # Check that no two are consecutive
        if all(positions[i + 1] - positions[i] >= 2 for i in range(len(positions) - 1)):
            return positions


def generate_maze():
    # Maze dimensions
    ROWS, COLS = 16, 16

    # Create grid filled with walls ("1")
    grid = [["1" for _ in range(COLS)] for _ in range(ROWS)]

    # Inner indices (exclude outer border, which must remain walls)
    inner_min, inner_max = 1, 14  # indices 1 to 14 (since 0 and 15 are the border)

    # Choose how many corridors to carve: 2 to 4 for both vertical and horizontal.
    num_vertical = random.randint(2, 4)
    num_horizontal = random.randint(2, 4)

    # Choose non-consecutive vertical corridor column indices from inner columns
    vertical_corridors = choose_corridors(num_vertical, inner_min, inner_max)
    # Choose non-consecutive horizontal corridor row indices from inner rows
    horizontal_corridors = choose_corridors(num_horizontal, inner_min, inner_max)

    # Carve vertical corridors (except for the outer border)
    for col in vertical_corridors:
        for row in range(inner_min, inner_max + 1):
            grid[row][col] = "0"

    # Carve horizontal corridors (except for the outer border)
    for row in horizontal_corridors:
        for col in range(inner_min, inner_max + 1):
            grid[row][col] = "0"

    # Optionally: Additional branches can be added here (for a more interesting maze)
    # For now, we keep only the main corridors.

    # Convert each row of the grid to a string
    maze_rows = ["".join(row) for row in grid]

    # Return the maze as a dictionary.
    return {'maze': maze_rows}


def main():
    # Generate five distinct mazes using different random seeds.
    mazes = []
    for seed in range(1, 6):
        random.seed(seed)  # Set seed for reproducibility.
        maze_dict = generate_maze()
        mazes.append(maze_dict)
        import matplotlib.pyplot as plt
        import numpy as np

        maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
        # Plot the maze
        plt.figure(figsize=(8, 8))
        plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
        plt.title("Maze Visualization")
        plt.axis('off')  # Remove axes for cleaner visualization
        plt.show()

    # Print the mazes. You can also use them further in your code.
    for idx, maze_dict in enumerate(mazes, start=1):
        print(f"Maze {idx}:")
        for row in maze_dict['maze']:
            print(row)
        print("\n" + "=" * 20 + "\n")


if __name__ == "__main__":
    main()



