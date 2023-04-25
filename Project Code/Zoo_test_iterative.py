import numpy as np
from pettingzoo.test import parallel_api_test
from DrPP_Zoo_Env_Iterative import CustomEnvironment
import gymnasium
from random import choice

myenv = CustomEnvironment()
observations,infos = myenv.reset()
actions = {'0':1, '1':1, '2':1}
observations,rewards,terminations,truncations,infos = myenv.step(actions)

for count in range(1000):
    for agent in actions:
        i = np.where(observations[agent]["action_mask"] == 1)
        # print("my action mask:",i)
        action = np.random.choice(i[0])
        actions[agent] = action
    observations,rewards,terminations,truncations,infos = myenv.step(actions)
    if True in truncations.values() or True in terminations.values():
        myenv.reset()



# print(observations['0'])
# observations,rewards,terminations,truncations,infos = myenv.step(actions)
# print(observations['0'])
# print(observations['0'])
# print(myenv.observation_space())
# print(observations["0"])
# print(myenv.observation_space().shape)