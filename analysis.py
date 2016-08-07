'''
This script provides the analysis tool for the T-maze project. It uses the Maze
to run the simulation and then analizes the results using the functions below.
'''

import numpy as np
import matplotlib.pyplot as plt
import Maze
import analysisTools as tools

# initial analysis: 400 trials, 10 rats
maze = Maze.Maze()
trials = 400
runs = 10
maze.run(N_trials=trials, N_runs=runs, verbose=False)
tools.plot_learningCurve(maze, suffix='t{}r{}'.format(trials, runs))
tools.navigation_map(maze)
#plt.show()

# development of the navigation map over trials
maze.run(N_trials=10, N_runs=1)
tools.navigation_map(maze, '_10trials')
#plt.show()
maze.run(N_trials=100, N_runs=1)
tools.navigation_map(maze, '_100trials')
#plt.show()
maze.run(N_trials=300, N_runs=1)
tools.navigation_map(maze, '_300trials')
#plt.show()

# how does the learning curve depend on lambda?
maze = Maze.Maze(lambda_=0)
maze.run(N_trials=400, N_runs=1)
tools.plot_learningCurve(maze, '_l0')
#plt.show()
maze = Maze.Maze(lambda_=.95)
maze.run(N_trials=400, N_runs=1)
tools.plot_learningCurve(maze, '_l095')
#plt.show()

# how does it depend on the number of actions?
actions = [4, 8, 16]
colors = ['r', 'g', 'b']
trials = 250
curves = np.zeros((len(actions), trials))
plt.figure(figsize=(15,10))
maze = Maze.Maze()
for i,a in enumerate(actions):
    maze.Nactions = a
    maze.run(N_trials=trials, N_runs=1)
    curves[i] = np.array(maze.get_learning_curve())
    plt.subplot(2,2,i+1)
    plt.plot(curves[i])
    plt.title('Nactions = {}'.format(a))
    plt.xlabel('trials')
    plt.ylabel('escape latency')
plt.subplot(224)
plt.plot(curves.mean(axis=1), 'o-', color='k')
plt.title('Mean escapte latency over actions')
plt.xlabel('Nactions')
plt.ylabel('Mean escape latency')
#plt.show()
plt.savefig('figures/learning_curve_actions.png')
