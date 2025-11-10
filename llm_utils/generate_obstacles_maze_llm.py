import random


def is_connected(maze):
    """
    Checks if all open spaces (0s) in the maze are connected.
    Returns True if a single connected component covers all open cells.
    """
    rows, cols = len(maze), len(maze[0])
    total_open = 0
    start = None

    # Identify an open cell to start the flood-fill and count total open cells.
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 0:
                total_open += 1
                if start is None:
                    start = (i, j)

    if start is None:
        return False  # No open cell found

    # Depth-first search (DFS) from the starting open cell.
    stack = [start]
    visited = {start}
    while stack:
        i, j = stack.pop()
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if maze[ni][nj] == 0 and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    stack.append((ni, nj))

    return len(visited) == total_open


def generate_maze(seed):
    """
    Generates a 16x16 maze with a wall frame (outer border) and 15% randomly
    placed obstacles (1s) inside the inner 14x14 grid, using the provided seed.
    """
    random.seed(seed)
    size = 16
    maze = [[0 for _ in range(size)] for _ in range(size)]

    # Set the outer border to walls (1s)
    for i in range(size):
        maze[0][i] = 1
        maze[size - 1][i] = 1
        maze[i][0] = 1
        maze[i][size - 1] = 1

    # List of inner cell coordinates (excluding border)
    inner_positions = [(i, j) for i in range(1, size - 1) for j in range(1, size - 1)]
    # Exactly 15% of the inner cells should be obstacles.
    obstacle_count = int(len(inner_positions) * 0.15)  # 15% of 14x14 = 29 obstacles

    # Randomly select positions for obstacles.
    obstacle_positions = random.sample(inner_positions, obstacle_count)
    for i, j in obstacle_positions:
        maze[i][j] = 1

    return maze


def maze_to_dict(maze):
    """
    Converts the 2D maze list into a dictionary with a list of row strings.
    """
    maze_rows = [''.join(str(cell) for cell in row) for row in maze]
    return {'maze': maze_rows}


def main():
    # Generate five distinct mazes using different seeds.
    mazes = []
    base_seeds = [0, 1, 2, 3, 4]

    for base in base_seeds:
        # To ensure connectivity, we may need to try several times per seed.
        attempts = 0
        while True:
            # Modify seed slightly on each attempt to get a different random layout
            current_seed = base + attempts * 100
            maze = generate_maze(current_seed)
            if is_connected(maze):
                break
            attempts += 1
        mazes.append(maze_to_dict(maze))

    # Output the five mazes.
    for index, maze_dict in enumerate(mazes, start=1):
        print(f"Maze {index}:")
        for row in maze_dict['maze']:
            print(row)
        print()

        import matplotlib.pyplot as plt
        import numpy as np

        maze_array = np.array([[int(cell) for cell in row] for row in maze_dict['maze']])
        # Plot the maze
        plt.figure(figsize=(8, 8))
        plt.imshow(maze_array, cmap='Greys', interpolation='nearest')
        plt.title("Maze Visualization")
        plt.axis('off')  # Remove axes for cleaner visualization
        plt.show()


if __name__ == "__main__":
    main()
