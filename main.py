import numpy as np
import math
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
size_x = 120
size_y = 120
density = 0.1
board_setup = {(39, 40), (39, 41), (40, 39), (40, 40), (41, 40)}

class GameOfLife:
    def __init__(self, board_size_x, board_size_y, initial_grid):
        self.size_x = board_size_x
        self.size_y = board_size_y
        self.state = initial_grid

    def clear_board(self):
        self.state.clear()

    def randomize(self, density):
        self.clear_board()
        num_cells = math.floor(self.size_x * self.size_y * density)
        for cell in range(num_cells):
            self.state.add((rnd.randint(0, self.size_x), rnd.randint(0, self.size_y)))

    def get_neighbors(self, cell):
        neighbors = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                if not (x == 0 and y == 0):
                    if (0 <= (cell[0] + x) <= self.size_x) and (0 <= (cell[1] + y) <= self.size_y):
                        neighbors.append((cell[0]+x, cell[1]+y))
        return neighbors

    def next_state(self):
        alive_neighbors = {}
        for cell in self.state:
            if cell not in alive_neighbors:
                alive_neighbors[cell] = 0
            neighbors = self.get_neighbors(cell)
            for neighbor in neighbors:
                if neighbor not in alive_neighbors:
                    alive_neighbors[neighbor] = 1
                else:
                    alive_neighbors[neighbor] += 1
        for cell in alive_neighbors:
            if alive_neighbors[cell] < 2 or alive_neighbors[cell] > 3:
                self.state.discard(cell)
            elif alive_neighbors[cell] == 3:
                self.state.add(cell)


game = GameOfLife(size_x, size_y, board_setup)

fig = plt.figure()
fig.canvas.set_window_title('Conway\'s Game of Life')
fig.set_size_inches(9, 6)
axs = fig.add_subplot(1, 1, 1)
game.randomize(density)


def animate(t):
    alive_cells = np.array(list(game.state))
    fig.suptitle('Iteration: ' + str(t), size=12)
    axs.clear()
    axs.set_xlim(0, size_x)
    axs.set_ylim(0, size_y)
    axs.axis("off")
    if alive_cells.ndim > 1:
        axs.scatter(
            alive_cells[:, :1],
            alive_cells[:, 1:],
            c='0.2',
            marker='o',
            s=(100**2/(size_x*size_y))*30
        )

    game.next_state()


ani = FuncAnimation(fig, animate, interval=1)
plt.show()