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
actions = {'up': [-1, 0], 'down': [1, 0], 'left': [0, -1], 'right': [0, 1]}
maze = World(maze_map, start, goal, actions, step_cost_type='punish', state_trans_prob=0.7)

possible_states = [[i, j] for i in range(maze_map.shape[0]) for j in range(maze_map.shape[1]) if not maze_map[i, j]]

policy_map = np.full(maze_map.shape, "X")
value_map = np.full(maze_map.shape, 9.0)  #arbitrary init value
discount = 0.9
threshold = 0.01


#VALUE ITERATION
lim=100
n=0
val_diff = 99
while n < lim:  # repeat until convergence of value map
    n += 1
    value_map_old = value_map.copy()
    for state in possible_states:
        neighbour_states = [np.add(state, actions[command]) for command in actions.keys()]
        neighbour_states = [neighbour for neighbour in neighbour_states if neighbour.tolist() in possible_states]
        neighbour_states.append(state)
        action_values = {}
        for action in actions.keys():
            step_cost = maze.step_cost(state, action)
            expected_next_val = discount*sum([maze.state_trans_prob_function(state, neighbour, action)*value_map_old[neighbour[0], neighbour[1]] for neighbour in neighbour_states])
            val = step_cost + expected_next_val
            action_values[action] = val
        best_action = min(action_values, key=action_values.get)
        policy_map[state[0], state[1]] = best_action
        value_map[state[0], state[1]] = action_values[best_action]

    diff = np.absolute(value_map_old - value_map)
    val_diff = np.amax(diff)
    print('max cange: {}'.format(val_diff))

print('finished after {} iterations'.format(n))
print(policy_map)



