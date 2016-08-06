import unittest
import Maze
import matplotlib.pyplot as plt
import numpy as np

class Test_Maze(unittest.TestCase):


    maze = Maze.Maze(binSize = 1)

    def test_geometry(self):
        self.maze.sigma = .1
        self.maze.visualize_maze()


    def test_discretization(self):
        assert(np.mod(self.maze.Nstates,10)==0), "Number of states must be a multiple of 10"


    def test_get_state_from_position(self):
        original_y = self.maze.y_position
        original_x = self.maze.x_position

        for y,x in zip([0, 30, 59.9],[59.9, 30, 0]):
            self.maze.y_position = y
            self.maze.x_position = x

        self.maze.y_position = original_y
        self.maze.x_position = original_x

    def test_training(self):
        print("Testing Learning...")
        self.maze.run(N_trials=2000, N_runs=1, verbose=False)
        plt.figure()
        plt.plot((self.maze.get_learning_curve()))
        plt.title("Latencies")
        self.maze.plot_Q()
        plt.show()
        self.maze.navigation_map()
        plt.show()
