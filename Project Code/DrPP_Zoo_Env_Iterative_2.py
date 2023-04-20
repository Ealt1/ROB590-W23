import functools
import random
from copy import copy
import numpy as np
import matlab.engine
from collections import deque
import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box
from typing import Union

from pettingzoo.utils.env import ParallelEnv

MAX_ACTIONS = 100
NUM_AGENTS = 3
NONE_PATH = [-1]*6
STARTING_POINTS = [7,20,9]
END_POINTS = [16,5,18]
NUM_ROWS = 3.0
NUM_COLS = 8.0
NUM_SPACES = NUM_ROWS*NUM_COLS
OBSTACLES = [10.0,12.0,13.0,15.0]
MAX_MOVES = 8

eng = matlab.engine.start_matlab()

def manhattan_dist(point1,point2):
    return abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])

def in_bounds(point):
    # if point[0]*point[1] <= NUM_SPACES:
    #     if point[0] > 0 and point[1] > 0:
    if point[0] >= 1 and point[0] <= NUM_ROWS and point[1] >= 1 and point[1] <= NUM_COLS:
        # print("Evaluated Point", point, "True")
        return True
    else:
        # print("Evaluated Point", point, "False")
        return False

def calculate_valid_space(end_point,current_point,actions_remaining):
    #TODO: Consider removing this condition to promote different learning
    valid_actions = [0,0,0,0,0]
    if end_point == current_point:
        valid_actions = [1,0,0,0,0]
        print("THE END IS REACHED!!",current_point)
        # return valid_actions
    else:
        end_point_2D = [0,0]
        current_point_2D = [0,0]
        end_point_2D[0],end_point_2D[1] = eng.ind2sub(np.array([NUM_ROWS,NUM_COLS]),end_point,nargout=2)
        current_point_2D[0],current_point_2D[1] = eng.ind2sub(np.array([NUM_ROWS,NUM_COLS]),current_point,nargout=2)

        for i in range(5):
            my_point_temp_2D = copy(current_point_2D)
            # my_point_temp_2D = [ int(x) for x in my_point_temp_2D]
            my_point_temp_2D = np.asarray(my_point_temp_2D)
            # print("Before:",my_point_temp_2D)
            if i == 0:
                pass
            elif i == 1:
                my_point_temp_2D[1] = my_point_temp_2D[1]-1
            elif i == 2:
                my_point_temp_2D[0] = my_point_temp_2D[0]-1
            elif i == 3:
                my_point_temp_2D[1] = my_point_temp_2D[1]+1
            elif i == 4:
                my_point_temp_2D[0] = my_point_temp_2D[0]+1
            # print("After:",my_point_temp_2D)
            if manhattan_dist(my_point_temp_2D,end_point_2D) < actions_remaining and in_bounds(my_point_temp_2D):
                valid_actions[i] = 1
        # print(valid_actions)
    # print(current_point,current_point_2D,valid_actions)
    return np.array(valid_actions)
            

def action_to_space(action,point):
    # print("\n\n\n",point,"\n\n\n")
    current_point_2D = [0,0]
    current_point_2D[0],current_point_2D[1] = eng.ind2sub(np.array([NUM_ROWS,NUM_COLS]),point,nargout=2)
    my_point_temp_2D = copy(current_point_2D)
    # my_point_temp_2D = [ int(x) for x in my_point_temp_2D]
    my_point_temp_2D = np.asarray(my_point_temp_2D)
    # print("\n\n\n",my_point_temp_2D,"\n\n\n")
    if action == 1:
        my_point_temp_2D[1] = my_point_temp_2D[1]-1
    elif action == 2:
        my_point_temp_2D[0] = my_point_temp_2D[0]-1
    elif action == 3:
        my_point_temp_2D[1] = my_point_temp_2D[1]+1
    elif action == 4:
        my_point_temp_2D[0] = my_point_temp_2D[0]+1
    # print(current_point_2D,my_point_temp_2D,action)
    return eng.sub2ind(np.array([NUM_ROWS,NUM_COLS]),my_point_temp_2D[0],my_point_temp_2D[1],nargout=1)

    

class CustomEnvironment(ParallelEnv):
    def __init__(self):
        super(CustomEnvironment, self).__init__()
        self.metadata = {'render.modes': ['human', 'ansi']}
        s = eng.genpath(r"C:\Users\ezraa\OneDrive\Documents\ROB590-W23\Project Code\lib")
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
        self._agent_ids = self.agents
        self.done=False
        self.numRows = NUM_ROWS
        self.numCols = NUM_COLS
        # self.Paths = [7,8,11,14,17,16]
        # self.Paths = []
        # for a in self.agents:
        #     self.Paths.append(np.array(NONE_PATH))
        self.Paths = np.zeros((3,MAX_MOVES))
        self.Paths.fill(-1)
        # self.Paths.append(np.array([7, 8, 11, 14, 17, 16]))
        # self.Paths.append(np.array([20, 17, 14, 11, 8, 5]))
        # self.Paths.append(np.array([9, 8, 11, 14, 17, 18]))
        self.N = len(self.agents)
        self.obstacles = np.array(OBSTACLES)
        # self.initial_locations = np.zeros((1,self.N))
        self.initial_locations = np.array(STARTING_POINTS)
        self.current_locations = copy(self.initial_locations)
        # self.final_locations = np.zeros((1,self.N))
        self.final_locations = np.array(END_POINTS)
        self.paths_ready= 0
        self.moves_remaining = MAX_MOVES
        self.ws = eng.create_workspace(self.numRows,self.numCols,self.obstacles)
        
        self.Agents = eng.cell(1,self.N)
        self.runs_completed = np.zeros((self.N,))
        self.time_elapsed = np.zeros((self.N,))
        self.positions = self.initial_locations
        self.pathflags = np.zeros((self.N,6))
        self.myflag = 0
        self.time_limite_elapsed = 0

        # self.prev_actions = deque(maxlen=MAX_ACTIONS)
        # for _ in range(MAX_ACTIONS):
        #     self.prev_actions(-1)

        # # self.observation = [self.Paths, self.initial_locations, self.final_locations] + list(self.prev_actions)
        observations = {}
        for agent in self.agents:
            valid_actions = calculate_valid_space(self.final_locations[int(agent)],self.current_locations[int(agent)],self.moves_remaining)
            
            NONE_OBS = {
                "action_mask": valid_actions,
                "observations": {'paths': self.Paths, 'initial locations': self.initial_locations,'current locations': self.current_locations,'final locations': self.final_locations,'paths ready': self.paths_ready}
            }
            # print(list(NONE_OBS["observations"].values()))
            # print(self.observation_space_test())
            mylist = list(NONE_OBS["observations"].values())
            mylist[0] = mylist[0].reshape((-1,))
            mylist[4] = np.array(mylist[4])
            mylist[4] = mylist[4].reshape((1,))
            myarray = np.concatenate(mylist)
            # print(myarray.shape)
            NONE_OBS['observations'] = gymnasium.spaces.utils.flatten(self.observation_space_test(), myarray)
            # NONE_OBS['observations'] = np.array([NONE_OBS['observations']])c
            observations[agent] = NONE_OBS   
        # observations = {agent: NONE_OBS for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        info = {}
        # if not actions:
        #     self.agents = []
        #     return {}, {}, {}, {}, {}
        # self.Paths = actions
        # self.Paths = []
        # for agent in actions:
        #     actions[agent].insert(0,STARTING_POINTS[int(agent)])
        #     actions[agent].append(END_POINTS[int(agent)])
        #     self.Paths.append(np.array(actions[agent]))
        paths_complete = 0
        prev_locations = copy(self.current_locations)
        print(prev_locations,actions)
        for agent in actions:
            self.current_locations[int(agent)] = action_to_space(actions[agent],self.current_locations[int(agent)])
            path_temp = np.array(copy(self.Paths[int(agent)]))
            path_temp = np.where(path_temp == -1)[0]
            if np.any(path_temp):
                self.Paths[int(agent),path_temp[0]] = self.current_locations[int(agent)]
            else:
                paths_complete = 1
        # print(prev_locations,self.current_locations,actions)
        self.moves_remaining = self.moves_remaining - 1
        print(self.moves_remaining)
            # actions[agent] = np.insert(actions[agent],0,STARTING_POINTS[int(agent)])
            # actions[agent] = np.append(actions[agent],END_POINTS[int(agent)])
            # self.Paths.append(np.array(actions[agent]))

        # self.Paths.append(np.array([7, 8, 11, 14, 17, 16]))
        # self.Paths.append(np.array([20, 17, 14, 11, 8, 5]))
        # self.Paths.append(np.array([9, 8, 11, 14, 17, 18]))
        # prev_point = [0,0]
        # current_point = [0,0]
        # for path_ind, path in enumerate(self.Paths):
        #     for ind, point in enumerate(path):
        #         if ind == 0:
        #             prev_point[0],prev_point[1] = eng.ind2sub(np.array([NUM_ROWS,NUM_COLS]),point,nargout=2)
        #         else:
        #             current_point[0],current_point[1] = eng.ind2sub(np.array([NUM_ROWS,NUM_COLS]),point,nargout=2)
        #             # print(current_point,prev_point)
        #             if abs(prev_point[0]-current_point[0]) + abs(current_point[0]-current_point[1]) > 1:
        #                 self.pathflags[path_ind,ind] = 1
        #             prev_point = copy(current_point)
        # paths_mask = self.pathflags == 1
                    # print(threshold_mask)
        # path_error = 0
        # if True in paths_mask:
        #     path_error = 1
        #     print(self.pathflags)



        if paths_complete:
            mypaths = eng.cell(1,self.N)
            for val in range(self.N):
                mypaths[val] = self.Paths[val]
            self.myflag = 0
            for i in range(0,self.N):
                self.Agents[i] = eng.agent(i+1,mypaths,1)
            eng.plot_ws(self.ws, self.initial_locations, self.final_locations, mypaths,nargout=0)
            # print(eng.get(self.Agents[0]))
            # eng.createBottlesSharedWith(self.Agents[2],self.Agents,nargout=0)
            for i in range(0,self.N):
                eng.createBottlesSharedWith(self.Agents[i],self.Agents,nargout=0)
                for j in range(self.N):
                    flag = eng.getfield(self.Agents[j],'myflag')
                    if flag == 1:
                        self.myflag = 1
            if eng.workspace['cycle_method'] == 'R2':
                eng.find_rainbow_cycles_all(self.Agents,mypaths,2,nargout=0)
            else:
                for i in range(0,self.N):
                    eng.findDrinkingSessions(self.Agents[i],eng.workspace['cycle_method'])
            eng.workspace['myflag'] = eng.set_initial_conditions(self.Agents,nargout=1)
            if eng.workspace['myflag'] == 1:
                self.myflag = 1

            self.time_limite_elapsed = False
            if not self.myflag:
                self.random_order = eng.randperm(self.N)
                while np.sum(self.runs_completed) < self.N and not self.time_limite_elapsed:
                    for m in range(0,self.N):
                        n = m
                        # self.random_order[m]
                        # print(self.runs_completed)
                        if not self.runs_completed[n]:
                            # self.Agents[n].move_philospher()
                            eng.move_philosopher(self.Agents[n],nargout=0)
                            
                            self.time_elapsed[n] = self.time_elapsed[n]+1
                            if eng.getfield(self.Agents[n],'curr_pos_idx') == eng.length(eng.getfield(self.Agents[n],'path')):
                                self.runs_completed[n] = 1
                        threshold_mask = self.time_elapsed > 15
                        # print(threshold_mask)
                        if True in threshold_mask:
                            self.time_limite_elapsed = True
                        print(self.time_elapsed)
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
        # obs = {'paths': np.array(self.Paths), 'initial locations': self.initial_locations,'final locations': self.final_locations}
        # observations = {agent: obs for agent in self.agents}
        observations = {}
        
        for agent in self.agents:
            # print("Evaluating safe actions for ", agent)
            valid_actions = calculate_valid_space(self.final_locations[int(agent)],self.current_locations[int(agent)],self.moves_remaining)
            obs = {
                "action_mask": valid_actions,
                "observations": {'paths': self.Paths, 'initial locations': self.initial_locations,'current locations': self.current_locations,'final locations': self.final_locations,'paths ready': self.paths_ready}
            }
            # obs['observations'] = np.array([obs['observations']])
            mylist = list(obs["observations"].values())
            mylist[0] = mylist[0].reshape((-1,))
            mylist[4] = np.array(mylist[4])
            mylist[4] = mylist[4].reshape((1,))
            myarray = np.concatenate(mylist)
            obs['observations'] = gymnasium.spaces.utils.flatten(self.observation_space_test(), myarray)
            observations[agent] = obs
        my_mask = []
            # print('keys:',observations.keys())
        for x in range(3):
            my_mask.append(observations[str(x)]["action_mask"])
        # print("\n\n",my_mask)
            

        #Calculate reward
        rewards = {}
        if paths_complete:
            for i,a in enumerate(self.agents):
                if self.myflag:
                    rewards[a] = -1000
                else:
                    rewards[a] = 100-1*np.max(self.time_elapsed)
        else:
            rewards = {agent: 0 for agent in self.agents}


        if paths_complete:
            if self.time_limite_elapsed:
                truncations = {agent: True for agent in self.agents}
                terminations = {agent: False for agent in self.agents}
            else:
                truncations = {agent: False for agent in self.agents}
                terminations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}
            terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # print(mask for )
        return observations,rewards,terminations,truncations,infos
    

    def render(self):
        pass
    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self,agents=None):
        # obs_space_agent = Dict({
        #     'position': MultiDiscrete([10,10]),
        #     'path': MultiDiscrete([4,10,4,10,4,10,4,10,4,10])
        # })

        # path_space = MultiDiscrete([NUM_SPACES]*6)
        path_space = Box(low=-1.0, high=NUM_SPACES, shape=(3, MAX_MOVES), dtype=np.float32)
        # init_space = MultiDiscrete([NUM_SPACES,NUM_SPACES,NUM_SPACES])
        # final_space = MultiDiscrete([NUM_SPACES,NUM_SPACES,NUM_SPACES])
        init_space = Box(low=-1.0, high=NUM_SPACES, shape=(3,), dtype=np.float32)
        final_space = Box(low=-1.0, high=NUM_SPACES, shape=(3,), dtype=np.float32)
        current_space = Box(low=-1.0, high=NUM_SPACES, shape=(3,), dtype=np.float32)
        # mask_space = Box(low=-1.0, high=NUM_SPACES, shape=(5,), dtype=np.float32)
        mask_space = Box(0.0, 1.0, shape=(self.action_space().n,))
        is_ready = Box(low=0, high=1, dtype=np.intc)
        # {'paths': self.Paths, 'initial locations': self.initial_locations,'current locations': self.current_locations,'final locations': self.final_locations,'paths ready': self.paths_ready,"action_mask": valid_actions}
        obs_space_agent = Dict({'paths': path_space, 'initial locations': init_space, 'current locations': current_space, 'final locations': final_space, 'paths ready':is_ready})
        # obs_space_agent = gymnasium.spaces.utils.flatten_space(obs_space_agent)
        # self.observation_spaces = Dict({0: obs_space_agent,1:obs_space_agent,2:obs_space_agent})
        # for agent in self.agents:
        #     obs[agent] = obs_space_agent
        # return Dict(obs)
        obs = Dict({"action_mask": mask_space, "observations": gymnasium.spaces.utils.flatten_space(obs_space_agent)})
        return obs
    # return self.observation_spaces

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        # 0: Stay still
        # 1: Move left
        # 2: Move up
        # 3: Move right
        # 4: Move down
        action_space = Discrete(5)
        return action_space
    
    def observation_space_test(self,agents=None):
        path_space = Box(low=-1.0, high=NUM_SPACES, shape=(3, MAX_MOVES), dtype=np.float32)
        init_space = Box(low=-1.0, high=NUM_SPACES, shape=(3,), dtype=np.float32)
        final_space = Box(low=-1.0, high=NUM_SPACES, shape=(3,), dtype=np.float32)
        current_space = Box(low=-1.0, high=NUM_SPACES, shape=(3,), dtype=np.float32)
        is_ready = Box(low=0, high=1, dtype=np.intc)
        obs_space_agent = Dict({'paths': path_space, 'initial locations': init_space, 'current locations': current_space, 'final locations': final_space, 'paths ready':is_ready})
        obs = gymnasium.spaces.utils.flatten_space(obs_space_agent)
        return obs