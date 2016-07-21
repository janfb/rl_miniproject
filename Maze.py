import numpy as np
import matplotlib.pyplot as plt

class Maze:
    """
    A class that implements a T-maze.

    Methods:

    set_centers()              : Set the centers of the place fiels
    visualize_maze()           : Scatter plot the place field centers
    update_activity()          : update the activity of the input layer neurons
    reset()                    : Make the agent forget everything he has learned.
    plot_Q()                   : Plot of the Q-values .
    learning_curve()           : Plot the time it takes the agent to reach the target
                                    as a function of trial number.
    navigation_map()           : Plot the movement direction with the highest
                                    Q-value for all positions.
    """

    def __init__(self, binSize=1, lambda_eligibility=0., epsilon=0.99):
        """
        Creates a T-maze with pickup and reward area
        """

        # length and width of the T-maze arms
        self.armX = 50
        self.armY = 10

        # length of the pickup and reward area
        self.area_length = 20

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 20.
        self.reward_at_wall   = -1.
        # the target area starts 20cm before the end of the left arm
        self.targetAreaBegin = 20 #cm
        # the pickup area start 20cm before the end of the right arm
        self.pickupAreaBegin = 90 #cm
        # initialize pick up flag
        self.alpha = 0

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid. It is close to
        # at the beginning and then decreases throughout the trial
        self.epsilon = epsilon

        # learning rate
        self.eta = 0.1

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.95

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility


        # set up the place cells
        self.sigma = 5 # the width of the place field
        self.spacing = 5 # 5 cm spacing between the place cells
        self.Nin = 64 # number of input layer neurons
        self.set_centers()

        # initialize the activity of the input layer neurons
        self.r = np.zeros((self.Nin, 2)) # second idx is beta

        # initialize the output layer neurons
        self.Nactions = 4 # number of output layer neurons
        self.directions = np.linspace(0, 2*np.pi, self.Nactions)

        # choose the discretization of the Tmaze
        # should be , i.e., 1, 2, 2.5, 5
        self.binSize = binSize # choose the size of a state in the maze: quadratic bin
        #self.Nstates = 3*self.armX*self.armY/(self.binSize**2) + 10*10/(self.binSize**2)
        self.Nstates = (2*self.armX+self.armY)*(self.armX + self.armY)/(self.binSize**2)

        # initialize state mat for postitio -> state mapping
        self.stateMat = np.reshape(np.arange(self.Nstates), ((self.armX + self.armY)/self.binSize, (2*self.armX+self.armY)/self.binSize))
        # build stereotype state positions in cm and corresponding activity of
        # input neurons
        self.statePos, self.stateAct = self._build_statePos()

        # initialize the Q-values etc.
        self._init_run()

    def set_centers(self):
        """
        Place the centers of the place cells evenly distribution across the maze
        """
        self.centers = np.zeros((self.Nin,2))
        # middle arm up to top
        self.centers[:24, 0] = np.tile([52.5, 57.5], 12)
        self.centers[:24, 1] = np.repeat(np.arange(2.5, self.armX+10, self.spacing),2)
        # right arm with pickup area
        self.centers[24:44, 0] = np.repeat(np.arange(self.centers[21,0] + self.spacing,
                                                     self.centers[21,0] + self.spacing + self.armX,
                                                     self.spacing),2)
        self.centers[24:44, 1] = np.tile([self.centers[21,1], self.centers[23,1]], 10)
        # left arm with target area
        self.centers[44:64, 0] = np.repeat(np.arange(2.5, self.centers[20,0], self.spacing)[::-1],2)
        self.centers[44:64, 1] = np.tile([self.centers[20,1], self.centers[22,1]], 10)

    def _build_statePos(self):
        """
        set stereotype x and y position in cm for every state in stateMat.
        """
        # build stereotype positions
        x = np.arange(self.binSize/2., 2*self.armX + self.armY, self.binSize)
        y = np.arange(self.binSize/2., self.armX + self.armY, self.binSize)
        X1, X2 = np.meshgrid(x,y)
        Z = np.dstack((X1, X2))
        # get corresponding activities
        R = np.zeros((self.Nin, Z.shape[0], Z.shape[1]))
        for n in range(self.Nin):
            R[n,] = np.exp(- ((self.centers[n,0] - Z[:,:,0])**2 + (self.centers[n,1] - Z[:,:,1])**2) / (2*self.sigma**2))
        return Z, R

    def visualize_maze(self, plot=False, plot_Act = False, plot_Pos = False):
        """
        Scatter plot the place cell centers
        """
        plt.figure(figsize=(10,10))
        #  plt.scatter(0, 0, color='or')
        plt.scatter(self.centers[:,0], self.centers[:,1])
        if plot_Act:
            plt.imshow(np.repeat(np.repeat(self.stateAct.sum(axis = 0), 2, axis = 1), 2, axis = 0))
            plt.gca().invert_yaxis()
        if plot_Pos:
            plt.scatter(self.statePos[...,0], self.statePos[...,1])
        plt.axis('equal')
        if plot: plt.show()


    def update_activity(self, x_pos=None, y_pos=None):
        """
        Update the input layer activity based in the current position of the agent
        """
        # update activity of current phase, the other one is set to 0 (Kronecker Delta)
        self.r[:,0] = (self.alpha==0) * np.exp(-((self.centers[:,0]-self.x_position)**2
                                      +(self.centers[:,1]-self.y_position)**2)
                                      /2*self.sigma**2)
        self.r[:,1] = (self.alpha==1) * np.exp(-((self.centers[:,0]-self.x_position)**2
                                    +(self.centers[:,1]-self.y_position)**2)
                                    /2*self.sigma**2)

    def get_neurons_activity(locX, locY):
        pass

    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize weights
        self.w = np.random.uniform(low=-1, high=1, size=(self.Nactions, self.Nin, 2))

        # initialize the Q-values and the eligibility trace
        self.Q = 0.01 * np.random.rand(self.Nstates, self.Nactions, 2) + 0.1
        self.e = np.zeros((self.Nactions, self.Nin))

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_position = None
        self.y_position = None
        self.action = None

    def run(self,N_trials=10,N_runs=1):
        self.latencies = np.zeros(N_trials)

        for _ in range(N_runs):
            self._init_run()
            latencies = self._learn_run(N_trials=N_trials)
            self.latencies += latencies/N_runs

    def _learn_run(self,N_trials=10):
        """
        Run a learning period consisting of N_trials trials.

        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.

        """
        for _ in range(N_trials):
            # run a trial and store the time it takes to the target
            latency = self._run_trial()
            self.latency_list.append(latency)

        return np.array(self.latency_list)

    def _run_trial(self):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.
        """
        # Initialize the latency (time to reach the target) for this trial
        latency = 0.

        # Choose a the initial position at the bottom of the maze
        self.x_position = self.y_position = 0
        self.state = self._get_state_from_pos()

        # make initial move
        self._choose_action()
        # Run the trial by choosing an action and repeatedly applying SARSA
        # until the reward has been reached.
        while not(self._arrived()):
            # update state
            self._update_state()
            # choose new action
            self._choose_action()
            # update Q-values
            self._update_Q()
            # count moves
            latency += 1

        return latency

    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # Update the eligibility trace
        self.e *= self.gamma * self.lambda_eligibility # let all memories decay
        self.e[self.action_old, :] += self.r[:, self.alpha] # strengthen current state memory

        # Update the Q-values
        # deltaQ = eta * e * [r - (Q_old - gamma * Q)]
        Qold = self.Q[self.state_old, self.action_old, self.alpha_old]
        Qnew = self.Q[self.state, self.action, self.alpha]
        tdiff = [self._reward() - (Qold - self.gamma*Qnew)]


        # update weights
        self.w[:,:,self.alpha_old] += self.eta * tdiff * self.e

        # Update the Q-values
        self.Q = self.w.dot(self.r)

        # Needed here:
        # self.action, self.x_position, self.y_position, self._reward
        # plus _old versions of above and more.

        self.Q += deltaQ

        # Finally we visualize the state if requested by calling code.
        self._visualize_current_state()
    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        # Be sure to store the old action before choosing a new one.
        self.action_old = self.action
        # get greedy action as the index of the largest Q value at the current state
        greedy_action = np.argmax(self.Q[self.state, :, self.alpha])
        # choose greedy action with prob 1-epsilon , choose random action else
        self.action = greedy_action if (1-self.epsilon) > np.random.rand(1)[0] else np.random.randint(self.Nactions)

    def _picked_up(self):
        """
        Check if the agent has visited the pick up area
        """
        # if it has not picked up yet, check whether it arrived in the area
        if not(self.alpha):
            self.alpha = (self.x_position > self.pickupAreaBegin)
        return self.alpha

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        # we only check the x coordinate because y is constant at the target area begin
        # rat needs
        return (self.x_position < self.targetAreaBegin) and self.alpha

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the
        chosen action at the current location
        """
        if self._arrived():
            return self.reward_at_target

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action
        # the stepsize has Gaussian noise
        stepsize = np.random.normal(loc=3, scale=1.5)
        # the directions are given by the angle in the directions repertoire
        # by convention the animal looks in the direction of the T, i.e., the 0
        # angle points upwards. That is why sine gives x and cosine gives y.
        self.x_position += np.sin(self.directions[self.action])*stepsize
        self.y_position += np.cos(self.directions[self.action])*stepsize

        # check if the agent has bumped into a wall.
        self._wall_touch = self._is_wall()
        if self._wall_touch:
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old

        # update the state: the current bin
        self.state_old = self.state
        self.state = self._get_state_from_pos()
        self.alpha_old = self.alpha
        self.alpha = self._picked_up()

    def _get_state_from_pos(self, ):
        """
        get the state index given the current position of the animal
        :return : the state index
        """
        row_index = int(self.stateMat.shape[0] - self.y_position/self.binSize)
        col_index = int(self.x_position/self.binSize)
        if self.y_position == 0:
            row_index = self.stateMat.shape[0]-1
        if self.x_position == 110:
            col_index = self.stateMat.shape[1]-1

        return self.stateMat[row_index, col_index]

    def _is_wall(self,x_position=None,y_position=None):
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_position
            y_position = self.y_position

        # check if the agent is trying to leave the Tmaze
        if (y_position < 0) or (y_position > self.armX+self.armY): # the agent is below the maze
            return True
        if(y_position < self.armX): # the agent is in the vertical arm
            if x_position < 50 or x_position > 60:
                return True
        if(y_position > self.armX): # the agent is in the horizontal arm
            if x_position < 0 or x_position > 110:
                return True
        # if none of the above is the case, this position is not a wall
        return False

    def learning_curve(self,log=False,filter_t=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter_t=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        plt.figure()
        plt.xlabel('trials')
        plt.ylabel('time to reach target')
        latencies = np.array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter_t'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter_t)

        if not log:
            plt.plot(self.latencies)
        else:
            plt.semilogy(self.latencies)

    def navigation_map(self):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        self.x_direction = np.zeros((self.Nstates,self.Nstates))
        self.y_direction = np.zeros((self.Nstates,self.Nstates))

        self.actions = np.argmax(self.Q[:,:,:],axis=2)
        self.y_direction[self.actions==0] = 1.
        self.y_direction[self.actions==1] = -1.
        self.y_direction[self.actions==2] = 0.
        self.y_direction[self.actions==3] = 0.

        self.x_direction[self.actions==0] = 0.
        self.x_direction[self.actions==1] = 0.
        self.x_direction[self.actions==2] = 1.
        self.x_direction[self.actions==3] = -1.

        plt.figure()
        plt.quiver(self.x_direction,self.y_direction)
        plt.axis([-0.5, self.Nstates - 0.5, -0.5, self.Nstates - 0.5])

    def reset(self):
        """
        Reset the Q-values (and the latency_list).

        Instant amnesia -  the agent forgets everything he has learned before
        """
        self.Q = 0.01 * np.random.rand(self.Nstates, self.Nstates, self.Nactions) + 0.1
        self.latency_list = []

    def plot_Q(self):
        """
        Plot the dependence of the Q-values on position.
        The figure consists of 4 subgraphs, each of which shows the Q-values
        colorcoded for one of the actions.
        """
        if np.mod(np.sqrt(self.Nactions),1)==0:
            plt.figure()
            for i in range(self.Nactions):
                plt.subplot(np.sqrt(self.Nactions),np.sqrt(self.Nactions),i+1)
                plt.imshow(self.Q[:,:,i],interpolation='nearest',origin='lower',vmax=1.1)
                plt.title('Action {}'.format(i+1))
                plt.colorbar()
        else:
            print("No plotting possible because number of actions is not quadratic")
