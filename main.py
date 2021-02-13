import numpy as np
import math
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
size_x = 100
size_y = 100
density = 0.1
# the board state is a set of tuples that will contain only the coordinates of the cells which are alive
board_setup = {(39, 40), (39, 41), (40, 39), (40, 40), (41, 40)}


class GameOfLife:
    def __init__(self, board_size_x, board_size_y, initial_grid):
        self.size_x = board_size_x
        self.size_y = board_size_y
        self.alive_cells = initial_grid

    def clear_board(self):
        self.alive_cells.clear()

    # create a random state considering the density parameter,
    # if density = 1 then the number of cells will be the full board
    def randomize(self, density_):
        self.clear_board()
        num_cells = math.floor(self.size_x * self.size_y * density_)
        for cell in range(num_cells):
            self.alive_cells.add((rnd.randint(0, self.size_x), rnd.randint(0, self.size_y)))

    # obtain the coordinates for all the neighbors of a given cell
    # the neighbors must be within the board size and exclude the cell in question
    def get_neighbors(self, cell):
        neighbors = []
        # loop x from -1 to +1
        for x in range(-1, 2, 1):
            # loop y from -1 to +1
            for y in range(-1, 2, 1):
                # avoid the cell in question
                if not (x == 0 and y == 0):
                    # only add the neighbor coordinates if those coordinates are not outside of the canvas size
                    if (0 <= (cell[0] + x) <= self.size_x) and (0 <= (cell[1] + y) <= self.size_y):
                        neighbors.append((cell[0]+x, cell[1]+y))
        return neighbors

    # calculate the next state of the board
    def next_state(self):
        # initialize a dictionary that will contain all the cells which have alive neighbors and
        # how many each of them have
        alive_neighbors = {}

        # loop through all of the cells which are alive
        for cell in self.alive_cells:
            # if the cell is not already in alive_neighbors then add it
            # since this is a cell that is alive in the previous state but we don't know yet
            # how many neighbors it has, we will set the number of neighbors to 0
            if cell not in alive_neighbors:
                alive_neighbors[cell] = 0
            # get the neighbors of the current cell
            neighbors = self.get_neighbors(cell)
            # loop through all the neighbors returned
            for neighbor in neighbors:
                # if the neighbor is not already in alive_neighbors then add it;
                # since we know that this cell is at least a neighbor to the current cell (alive)
                # then start with its number of alive neighbors at 1
                if neighbor not in alive_neighbors:
                    alive_neighbors[neighbor] = 1
                # else if the neighbor was already in alive_neighbors it means that it was either
                # one of the alive cells or a neighbor to another cell, since we know this is also a
                # neighbor to the current cell then increase its alive neighbors counter by 1
                else:
                    alive_neighbors[neighbor] += 1
        # loop through all the cells recorded in the previous loop within alive_neighbors
        # alive_neighbors should now contain all the cells in the current state which have
        # at least one neighbor plus the cells that were previously alive but have no neighbors
        for cell in alive_neighbors:
            # if the number of neighbors of this particular cell is less than two or more than 3
            # then discard it from the set of alive cells, if it was not part of the set,
            #  nothing happens; but if it was already alive it will delete it (the cell will die)
            if alive_neighbors[cell] < 2 or alive_neighbors[cell] > 3:
                self.alive_cells.discard(cell)
            # else if the number of neighbors is exactly 3, then add the cell to the alive set
            # if the cell was already there (it was alive in the previous state) it will do
            # nothing
            elif alive_neighbors[cell] == 3:
                self.alive_cells.add(cell)
            # from the previous logic, if a cell was alive and has between 2 and 3 neighbors it will continue
            # to be alive, since we did not discard it


# create the game
game = GameOfLife(size_x, size_y, board_setup)

# randomize the game, if a particular state is wanted, just comment this line
game.randomize(density)


# create the matplotlib figure to show the game on the screen
fig = plt.figure()
fig.canvas.set_window_title('Conway\'s Game of Life')
fig.set_size_inches(9, 6)
axs = fig.add_subplot(1, 1, 1)


# the animate function is what the matplotlib animation library loops through while animating the graph
def animate(t):
    # convert the set of tuples of the alive cells into a numpy array
    # note that to do so we need to convert it first to a python list
    alive_cells = np.array(list(game.alive_cells))

    # prepare the plot area for the graph
    fig.suptitle('Iteration: ' + str(t), size=12)
    axs.clear()
    axs.set_xlim(0, size_x)
    axs.set_ylim(0, size_y)
    axs.axis("off")

    # if there are still cells within the alive set then plot a scatter graph with all the alive
    # cells, use numpy index slicing to select the X and Y coordinates for the alive cells
    # select the size of the markers based on the number of cells set (size_x and size_y)
    if alive_cells.ndim > 1:
        axs.scatter(
            alive_cells[:, :1],
            alive_cells[:, 1:],
            c='0.2',
            marker='o',
            s=(100**2/(size_x*size_y))*30
        )
    # once the alive cells are displayed on the screen, generate the next state
    game.next_state()


# start the loop for the animation
ani = FuncAnimation(fig, animate, interval=1)
plt.show()
