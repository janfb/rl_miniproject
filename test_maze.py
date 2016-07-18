import unittest
import Maze
import matplotlib.pyplot as plt
import numpy as np

class Test_wetter_com(unittest.TestCase):

    maze = Maze.Maze()

    def test_geometry(self):
        print("Testing geometry")
        self.maze.visualize_maze(plot=True)

    def test_discretization(self):
        print("Testing discretization")
        assert(np.mod(self.maze.Nstates,10)==0)

    def test_statemapping(self):
        # test 100 random position in the field for state mapping 
        for i in range(100):
            self.maze.x_position = np.random.uniform(low=0, high=2*self.maze.armX+self.maze.armY)
            self.maze.y_position = np.random.uniform(low=0, high=self.maze.armX+self.maze.armY)
            self.maze.action=0
            self.maze._update_state()
