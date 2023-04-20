import functools
import random
from copy import copy
import numpy as np
import matlab.engine
from collections import deque

from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box

from pettingzoo.utils.env import ParallelEnv

MAX_ACTIONS = 100
NUM_AGENTS = 3
NONE_PATH = [-1,-1,-1,-1,-1,-1]
STARTING_POINTS = [7,20,9]
END_POINTS = [16,5,18]
NUM_ROWS = 3.0
NUM_COLS = 8.0
NUM_SPACES = NUM_ROWS*NUM_COLS+1
OBSTACLES = [10,12,13,15]

eng = matlab.engine.start_matlab()

class CustomEnvironment(ParallelEnv):
    def __init__(self):
        super(CustomEnvironment, self).__init__()
        self.metadata = {'render.modes': ['human', 'ansi']}
        s = eng.genpath('lib')
        eng.addpath(s, nargout=0)
        self.possible_agents = [str(r) for r in range(NUM_AGENTS)]
        eng.workspace['cycle_method'] = 'R2'
        self.agents = self.possible_agents[:]
        eng.workspace['myflag'] = 0
        # self.agents = np.arange(0,6,1)
        # self.params = {}
        # params = {
        #     'position': MultiDiscrete([10,10]),
        #     'path': MultiDiscrete([4,10,4,10,4,10,4,10,4,10])
        # }
        # for i in self.agents:
        #     self.params[i] = {i: copy.deepcopy(params)}

    def reset(self, seed=None, return_info=False, options=None):
        # eng.workspace['myflag'] = 0
        self.agents = self.possible_agents[:]
        self.done=False
        self.numRows = NUM_ROWS
        self.numCols = NUM_COLS
        # self.Paths = [7,8,11,14,17,16]
        # self.Paths = []
        # for a in self.agents:
        #     self.Paths.append(np.array(NONE_PATH))
        self.Paths = []
        # self.Paths.append(np.array([7, 8, 11, 14, 17, 16]))
        # self.Paths.append(np.array([20, 17, 14, 11, 8, 5]))
        # self.Paths.append(np.array([9, 8, 11, 14, 17, 18]))
        self.N = len(self.agents)
        self.obstacles = np.array(OBSTACLES)
        # self.initial_locations = np.zeros((1,self.N))
        self.initial_locations = np.array(STARTING_POINTS)
        # self.final_locations = np.zeros((1,self.N))
        self.final_locations = np.array(END_POINTS)

        self.ws = eng.create_workspace(self.numRows,self.numCols,self.obstacles)
        
        self.Agents = eng.cell(1,self.N)
        self.runs_completed = np.zeros((self.N,))
        self.time_elapsed = np.zeros((self.N,))
        self.positions = self.initial_locations

        # self.prev_actions = deque(maxlen=MAX_ACTIONS)
        # for _ in range(MAX_ACTIONS):
        #     self.prev_actions(-1)

        # # self.observation = [self.Paths, self.initial_locations, self.final_locations] + list(self.prev_actions)
        NONE_OBS = {'paths': self.Paths, 'initial locations': self.initial_locations,'final locations': self.final_locations}
        observations = {agent: NONE_OBS for agent in self.agents}
        return observations
    
    def step(self, actions):
        info = {}
        # if not actions:
        #     self.agents = []
        #     return {}, {}, {}, {}, {}
        # self.Paths = actions
        self.Paths = []
        for agent in actions:
            actions[agent].insert(0,STARTING_POINTS[int(agent)])
            actions[agent].append(END_POINTS[int(agent)])
            self.Paths.append(np.array(actions[agent]))
        # self.Paths.append(np.array([7, 8, 11, 14, 17, 16]))
        # self.Paths.append(np.array([20, 17, 14, 11, 8, 5]))
        # self.Paths.append(np.array([9, 8, 11, 14, 17, 18]))
        self.myflag = 0
        for i in range(0,self.N):
            self.Agents[i] = eng.agent(i+1,self.Paths,1)
        eng.plot_ws(self.ws, self.initial_locations, self.final_locations, self.Paths,nargout=0)
        # print(eng.get(self.Agents[0]))
        # eng.createBottlesSharedWith(self.Agents[2],self.Agents,nargout=0)
        for i in range(0,self.N):
            eng.createBottlesSharedWith(self.Agents[i],self.Agents,nargout=0)
            for j in range(self.N):
                flag = eng.getfield(self.Agents[j],'myflag')
                if flag == 1:
                    self.myflag = 1
        if eng.workspace['cycle_method'] == 'R2':
            eng.find_rainbow_cycles_all(self.Agents,self.Paths,2,nargout=0)
        else:
            for i in range(0,self.N):
                eng.findDrinkingSessions(self.Agents[i],eng.workspace['cycle_method'])
        eng.workspace['myflag'] = eng.set_initial_conditions(self.Agents,nargout=1)
        if eng.workspace['myflag'] == 1:
            self.myflag = 1

        if not self.myflag:
            self.random_order = eng.randperm(self.N)
            while np.sum(self.runs_completed) < self.N:
                for m in range(0,self.N):
                    n = m
                    # self.random_order[m]
                    print(self.runs_completed)
                    if not self.runs_completed[n]:
                        # self.Agents[n].move_philospher()
                        eng.move_philosopher(self.Agents[n],nargout=0)
                        
                        self.time_elapsed[n] = self.time_elapsed[n]+1
                        if eng.getfield(self.Agents[n],'curr_pos_idx') == eng.length(eng.getfield(self.Agents[n],'path')):
                            self.runs_completed[n] = 1
                for n in range(self.N):
                    curr_pos_idx_field = eng.getfield(self.Agents[n],'curr_pos_idx')
                    mypath = eng.getfield(self.Agents[n],'path')
                    # print('mypath=',mypath[0])
                    # print(curr_pos_idx_field)
                    # self.positions[n] = self.Agents[n].path(self.Agents[n].curr_pos_idx)
                    self.positions[n] = mypath[0][int(curr_pos_idx_field)-1]
                eng.plot_ws(self.ws, self.positions, self.final_locations, [],nargout=0)
                eng.drawnow

        #Determine Observation
        obs = {'paths': self.Paths, 'initial locations': self.initial_locations,'final locations': self.final_locations}
        observations = {agent: obs for agent in self.agents}

        #Calculate reward
        rewards = {}
        for i,a in enumerate(self.agents):
            if self.myflag:
                rewards[a] = -10000
            else:
                rewards[a] = -1*self.time_elapsed[i]

        terminations = {agent: True for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}


        return observations,rewards,terminations,truncations,infos
    

    def render(self):
        pass
    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent):
        # obs_space_agent = Dict({
        #     'position': MultiDiscrete([10,10]),
        #     'path': MultiDiscrete([4,10,4,10,4,10,4,10,4,10])
        # })

        path_space = MultiDiscrete([NUM_SPACES,NUM_SPACES,NUM_SPACES,NUM_SPACES,NUM_SPACES,NUM_SPACES])
        init_space = MultiDiscrete([NUM_SPACES,NUM_SPACES,NUM_SPACES])
        final_space = MultiDiscrete([NUM_SPACES,NUM_SPACES,NUM_SPACES])
        obs_space_agent = Dict({'paths': path_space, 'initial locations': init_space,'final locations': final_space})
        # self.observation_spaces = Dict({0: obs_space_agent,1:obs_space_agent,2:obs_space_agent})
        # for agent in self.agents:
        #     obs[agent] = obs_space_agent
        # return Dict(obs)
        return obs_space_agent
    # return self.observation_spaces

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #possible_paths = np.array(np.meshgrid([0,1,2,3],np.arange(1,11,1),[0,1,2,3],np.arange(1,11,1),[0,1,2,3],np.arange(1,11,1),[0,1,2,3],np.arange(1,11,1),[0,1,2,3],np.arange(1,11,1))).T.reshape(-1,10)
        #print(possible_paths.shape)
        # self.action_spaces[agent] = MultiDiscrete([4,10,4,10,4,10,4,10,4,10])
        # return self.action_spaces[agent]
        action_space = MultiDiscrete([NUM_SPACES]*6)
        return action_space