'''
This script provides the analysis tool for the T-maze project. It uses the Maze
to run the simulation and then analizes the results using the functions below.
'''

import numpy as np
import matplotlib.pyplot as plt
import Maze

def navigation_map(maze):
    """
    Plot the direction with the highest Q-value for every place cells position.
    """
    maze.x_direction_pickup = np.zeros(maze.Nin)
    maze.y_direction_pickup = np.zeros(maze.Nin)
    maze.x_direction_target = np.zeros(maze.Nin)
    maze.y_direction_target = np.zeros(maze.Nin)

    maze.preferred_actions = np.zeros((2,maze.Nin))
    for alpha in range(maze.preferred_actions.shape[0]):
        for cell in range(maze.preferred_actions.shape[1]):
            # get input rates for cell centers
            rates = maze.calculate_input_rates(maze.centers[cell,0], maze.centers[cell,1])
            # get corresponding output rates
            tmpQ = maze.w[alpha].dot(rates)
            # get preferred actions
            maze.preferred_actions[alpha, cell] = tmpQ.argmax()

    maze.y_direction_pickup[maze.preferred_actions[0]==0] = 1.
    maze.y_direction_pickup[maze.preferred_actions[0]==2] = -1.

    maze.x_direction_pickup[maze.preferred_actions[0]==1] = 1.
    maze.x_direction_pickup[maze.preferred_actions[0]==3] = -1.

    maze.y_direction_target[maze.preferred_actions[0]==0] = 1.
    maze.y_direction_target[maze.preferred_actions[0]==2] = -1.

    maze.x_direction_target[maze.preferred_actions[0]==1] = 1.
    maze.x_direction_target[maze.preferred_actions[0]==3] = -1.

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title("Navigation Map to pickup area, alpha=0")
    plt.quiver(maze.centers[:,0], maze.centers[:,1], maze.x_direction_pickup,maze.y_direction_pickup, color='b', alpha=.5, label = 'alpha = 0')
    maze.plot_Tmaze()
    plt.axis([-5, 115, -5, 65])
    plt.subplot(122)
    plt.title("Navigation Map to target area, alpha=1")
    plt.quiver(maze.centers[:,0], maze.centers[:,1], maze.x_direction_target,maze.y_direction_target, color='r', alpha=.5, label = 'alpha = 0')
    maze.plot_Tmaze()
    plt.axis([-5, 115, -5, 65])

def plot_Q(maze):
    """
    Plot the dependence of the Q-values on position.
    """
    directionStr = ['North', 'East', 'South', 'West']
    # bring Q in shape
    all_rates = np.reshape(maze.stateAct, (maze.stateAct.shape[0],
                                maze.stateAct.shape[1]*maze.stateAct.shape[2]))
    plottingQ0 = maze.w[0].dot(all_rates)
    plottingQ1 = maze.w[1].dot(all_rates)
    Q0 = np.reshape(plottingQ0, (maze.Nactions, maze.stateAct.shape[1], maze.stateAct.shape[2]))
    Q1 = np.reshape(plottingQ1, (maze.Nactions, maze.stateAct.shape[1], maze.stateAct.shape[2]))

    plt.figure(figsize=(20, 10))
    minval = Q0.min()
    maxval = Q0.max()
    for i in range(maze.Nactions):
        plt.subplot(np.sqrt(maze.Nactions),2*np.sqrt(maze.Nactions),i+1)
        im = plt.imshow(Q0[i,:,:],interpolation='None',origin='lower', vmin=minval, vmax=maxval)
        plt.title(directionStr[i]+ ' to pickup')
        plt.colorbar(im, fraction = 0.045)
        maze.plot_Tmaze()
    minval = Q1.min()
    maxval = Q1.max()
    for i in range(maze.Nactions):
        plt.subplot(np.sqrt(maze.Nactions),2*np.sqrt(maze.Nactions),i+1+maze.Nactions)
        im = plt.imshow(Q1[i,:,:],interpolation='None',origin='lower', vmin=minval, vmax=maxval)
        plt.title(directionStr[i]+ ' to target')
        plt.colorbar(im, fraction = 0.045)
        maze.plot_Tmaze()

def plot_Tmaze():
    '''
    Plots the tmaze boudaries as lines
    '''
    ax = plt.gca()
    ax.vlines(x=50, ymin=0, ymax=50, color='k')
    ax.vlines(x=60, ymin=0, ymax=50, color='k')
    ax.vlines(x=0, ymin=50, ymax=60, color='k')
    ax.vlines(x=110, ymin=50, ymax=60, color='k')
    ax.vlines(x=110, ymin=50, ymax=60, linestyle = '--', linewidth=3, color='k')
    ax.vlines(x=0, ymin=50, ymax=60, linestyle = '--', linewidth=3, color='k')
    ax.vlines(x=90, ymin=50, ymax=60, linestyle = '--', linewidth=3, color='k')
    ax.vlines(x=20, ymin=50, ymax=60, linestyle = '--', linewidth=3, color='k')

    ax.hlines(y=50, xmin=0, xmax=50, color='k')
    ax.hlines(y=60, xmin=0, xmax=110, color='k')
    ax.hlines(y=50, xmin=60, xmax=110, color='k')
    ax.hlines(y=50, xmin=0, xmax=20, linestyle = '--', linewidth=3, color='k')
    ax.hlines(y=60, xmin=0, xmax=20, linestyle = '--', linewidth=3, color='k')
    ax.hlines(y=50, xmin=90, xmax=110, linestyle = '--', linewidth=3, color='k')
    ax.hlines(y=60, xmin=90, xmax=110, linestyle = '--', linewidth=3, color='k')
    ax.hlines(y=0, xmin=50, xmax=60, color='k')
    ax.set_aspect('equal', adjustable='box');

def visualize_maze(maze, plot_Maze=False, plot_Act = False, plot_Pos = False):
    """
    Scatter plot the place cell centers
    """
    if np.any((plot_Maze, plot_Act, plot_Pos)):
        plt.figure(figsize=(10,10))
        plt.axis('equal')
    if plot_Maze:
        plt.scatter(0, 0, color='or')
        plt.scatter(maze.centers[:,0], maze.centers[:,1])
    if plot_Act:
        plt.imshow(np.repeat(np.repeat(maze.stateAct.sum(axis = 0), 2, axis = 1), 2, axis = 0))
        plt.gca().invert_yaxis()
    if plot_Pos:
        plt.scatter(maze.statePos[...,0], maze.statePos[...,1])

maze = Maze.Maze()
maze.run(N_trials=100, N_runs=10)
plot_Q(maze)
navigation_map(maze)
