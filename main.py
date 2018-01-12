import numpy as np
from world import World
import random as rand

maze_map =  np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]])

start = np.array([4,5])
goal = np.array([6,6])
maze = World(maze_map, start, goal, step_cost_type='punish', state_trans_prob=0.7)

possible_states = [[i, j] for i in range(maze_map.shape[0]) for j in range(maze_map.shape[1]) if not maze_map[i, j]]
actions = ['up', 'down', 'left', 'right']

policy_map = np.full(maze_map.shape, 'down')    #arbitrary init value
value_map = np.full(maze_map.shape, 0)  #arbitrary init value
discount = 0.9
threshold = 0.01

n=0
while n < 10000:  # repeat until convergence of value map
    n += 1
    value_map_old = value_map
    for state in possible_states:
        for action in actions:




