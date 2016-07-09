import unittest
import Maze
import matplotlib.pyplot as plt
import numpy as np

class Test_wetter_com(unittest.TestCase):

    maze = Maze.Maze()

    def test_geometry(self):
        print("Testing geometry")
        self.maze.visualize_maze(plot=False)

    def test_discretization(self):
        print("Testing discretization")
        assert(np.mod(self.maze.Nstates,10)==0)
