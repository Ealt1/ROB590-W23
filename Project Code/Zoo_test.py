import numpy as np
from pettingzoo.test import parallel_api_test
from DrPP_Zoo_Env_Iterative import CustomEnvironment
import gymnasium

myenv = CustomEnvironment()
observations,infos = myenv.reset()
actions = {'0':1, '1':1, '2':1}
observations,rewards,terminations,truncations,infos = myenv.step(actions)
print(observations['0'])
# observations,rewards,terminations,truncations,infos = myenv.step(actions)
# print(observations['0'])
# print(observations['0'])
# print(myenv.observation_space())
# print(observations["0"])
# print(myenv.observation_space().shape)