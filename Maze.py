import numpy as np

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

    def __init__(self, binSize=1):
        """
        Creates a T-maze with pickup and reward area
        """

        # length and width of the T-maze arms
        self.armX = 50
        self.armY = 10

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 20.
        self.reward_at_wall   = -1.
        # the target area starts 20cm before the end of the left arm
        self.targetAreaBegin = 20 #cm
        self.pickup_area_begin = 90

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid. It is close to
        # at the beginning and then decreases throughout the trial
        self.epsilon = 1

        # learning rate
        self.eta = 0.1

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.95

        # the decay factor for the eligibility trace
        self.lambda_eligibility = 0.5


        # set up the place cells
        self.sigma = 5 # the width of the place field
        self.spacing = 5 # 5 cm spacing between the place cells
        self.Nin = 64 # number of input layer neurons
        self.set_centers()

        # initialize the output layer neurons
        self.Nactions = 4 # number of output layer neurons
        self.directions = np.arange(0, 2*np.pi, 2*np.pi/self.Nactions)

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

    def calculate_input_rates(self, x_position=None, y_position=None):
        """
        Calculate the activity of the input neurons given a x-y position. If no
        position is given, then the current position of the animal is used.
        :param x_position: custom x position
        :param y_position: custom y position
        :return rates: the rates of all input neurons, e.g., (64,)
        """
        if x_position==None or y_position==None:
            x_position = self.x_position
            y_position = self.y_position

        rates = np.exp(-((self.centers[:,0]-x_position)**2
                                      +(self.centers[:,1]-y_position)**2)
                                      /(2*self.sigma**2))
        return rates

    def calculate_output_rates(self):
        '''
        Calculates the ouput activity of the output layer neurons (q-values)
        '''
        return self.w[self.alpha].dot(self.input_rates)

    def _update_activities(self):
        '''
        Updates the activities of the input and the output layer
        '''
        self.input_rates_old = np.copy(self.input_rates)
        self.output_rates_old = np.copy(self.output_rates)
        self.input_rates = self.calculate_input_rates(self.x_position, self.y_position)
        self.output_rates = self.calculate_output_rates()

    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize weights
        self.w = np.random.rand(2, self.Nactions, self.Nin)
        self.alpha = 0
        # initialize the activity of the input layer neurons
        self.input_rates = self.calculate_input_rates(55, 0)
        self.input_rates_old = None
        self.output_rates = self.calculate_output_rates()
        self.output_rates_old = None

        # initialize the Q-values and the eligibility trace
        self.Q = np.zeros((self.Nactions))
        self.e = np.zeros((self.Nactions, self.Nin))
        self.epsilon = 1

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_position = None
        self.y_position = None
        self.action = None

    def run(self,N_trials=10,N_runs=1, verbose=False):
        self.latencies = np.zeros(N_trials)

        for r in range(N_runs):
            self._init_run()
            print("RUN %s"%r)
            latencies = self._learn_run(N_trials=N_trials, verbose=verbose)
            self.latencies += latencies/N_runs

    def _learn_run(self,N_trials=10, verbose=False):
        """
        Run a learning period consisting of N_trials trials.

        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.

        """
        decay_factor = ((0.1)/self.epsilon) ** (1/(0.7*N_trials))
        for t in range(N_trials):
            # run a trial and store the time it takes to the target
            latency = self._run_trial()
            if verbose:
                print('Finished trial %s in %s steps with epsilon= %.4s'%(t, latency, self.epsilon))
            # let epsilon decay exponentially to 0.1
            if self.epsilon>0.1:
                self.epsilon *= decay_factor
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
        self.x_position = 55.
        self.y_position = 0.
        self.alpha = 0

        # make initial move
        self._choose_action()
        # Run the trial by choosing an action and repeatedly applying SARSA
        # until the reward has been reached.
        while not(self._arrived()):
            # update state
            self._update_state()
            self._choose_action()
            self._update_activities()
            # update weights
            self._update_weights()
            # count moves
            latency += 1
        return latency

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
        if np.random.random_sample() > self.epsilon:
            self.action = self.output_rates.argmax()
        else:
            self.action = np.random.randint(0, self.output_rates.size)

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position
        self.alpha_old = self.alpha

        # update the agents position according to the action
        # the stepsize has Gaussian noise
        stepsize = np.random.normal(loc=3, scale=1.5)
        # NOTE: the directions are given by the angle in the directions repertoire
        # by convention the animal looks in the direction of the T, i.e., the 0
        # angle points upwards or 'north'. That is why sine gives x and cosine gives y.
        self.x_position += np.sin(self.directions[self.action])*stepsize
        self.y_position += np.cos(self.directions[self.action])*stepsize

        # calculate reward of the performed action
        self.reward = self._get_reward(self.x_position, self.y_position)

        # check for pickup
        if (self.alpha == 0 and self._in_pickup(self.x_position, self.y_position)):
            self.alpha = 1

        # check if the agent has bumped into a wall.
        self._wall_touch = self._is_wall()
        if self._wall_touch:
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old

    def _update_weights(self):
        '''
        Updates weights according to SARSA and returns the resulting weights
        '''
        # Determine candidate weight change
        self.e *= self.gamma * self.lambda_eligibility # let all memories decay
        self.e[self.action_old, :] += self.input_rates_old

        # get old and new Q values for time difference
        Qold = self.output_rates_old[self.action_old]
        Qnew = self.output_rates[self.action]
        # calculate time difference
        tdiff = np.array([self.reward + self.gamma*Qnew - Qold])

        # apply weight change
        self.w[self.alpha_old] += self.eta * tdiff * self.e

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        # we only check the x coordinate because y is constant at the target area begin
        # rat needs
        return (self.x_position <= self.targetAreaBegin) and self.alpha

    def _in_pickup(self, x, y):
        """
        Check for pickup
        """
        if self.alpha:
            return 1
        else:
            return not(self._is_wall(x,y)) and (x >= self.pickup_area_begin)

    def _get_reward(self, x, y):
        """
        Evaluates how much reward should be administered when performing the
        chosen action at the current location
        """
        # default reward
        reward = 0
        # reward at target
        if self._arrived():
            reward = self.reward_at_target
        # reward at wall
        if self._is_wall(x, y):
            reward = self.reward_at_wall
        return reward

    def _is_wall(self,x_position=None,y_position=None):
        """
        This function returns, if the given position is outside of the maze.

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

    def get_learning_curve(self, filter_t=1.):
        """
        Calculate running average of the time it takes the agent to reach the target location.

        Options:
        filter_t=1. : timescale of the running average.
        """
        latencies = np.array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter_t'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter_t)

        return self.latencies
