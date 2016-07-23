import unittest
import Maze
import matplotlib.pyplot as plt
import numpy as np

class Test_Maze(unittest.TestCase):


    maze = Maze.Maze(binSize = 1)

    def test_geometry(self):
        print("Testing geometry")
        self.maze.sigma = .1
        self.maze.visualize_maze(plot=False, plot_Pos=False)


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


    def test_get_state_from_position(self):
        original_y = self.maze.y_position
        original_x = self.maze.x_position

        for y,x in zip([0, 30, 59.9],[59.9, 30, 0]):
            self.maze.y_position = y
            self.maze.x_position = x

            #print(" pos (y, x): ", y, ", ", x, ": ", self.maze._get_state_from_pos())


        self.maze.y_position = original_y
        self.maze.x_position = original_x


    def test_training(self):
        print("Testing Learning...")
        self.maze.run(N_trials=10)
        print(self.maze.get_learning_curve())



