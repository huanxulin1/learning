import gym
import random
from math import pow

# test1   测试gym#
# env = gym.make('MountainCar-v0', render_mode = 'human')
# for i_episode in range(10):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info, _ = env.step(action)
#     if done:
#         print("Episode finished over after {} timesteps".format(t+1))
#         break
# env.close

# test2    近似求pi#
# m = 0
# n = 100000
# for i in range(n):
#     x = random.uniform(-1, 1)
#     y = random.uniform(-1, 1)
#     r = (pow(x, 2) + pow(y, 2))
#     if  r < 1 or r == 1:
#         m = m + 1
# print(m)
# print((4*m)/n)
