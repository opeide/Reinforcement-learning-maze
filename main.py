import numpy as np
from world import World
import random as rand
import copy

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
maze = World(maze_map, start, goal, actions, step_cost_type='reward', state_trans_prob=0.7)
discount = 0.9
threshold = 0.0001
iteration_limit = 1000


def value_iteration(maze, discount, threshold, iteration_limit):
    possible_states = maze.possible_states
    actions = maze.actions
    policy_map = np.full(maze_map.shape, "X", dtype=object)
    value_map = np.full(maze_map.shape, 9.0)  #arbitrary init value

    n=0
    val_change = 99
    while n < iteration_limit and val_change > threshold:  # repeat until convergence of value map
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
        val_change = np.amax(diff)

    print('finished policy iteration after {} iterations'.format(n))
    return policy_map, value_map


def policy_iteration(maze, discount,threshold, iteration_limit):
    maze_map = maze.map_mat
    policy_map = np.full(maze_map.shape, "down", dtype=object)

    n = 0
    while n < iteration_limit:
        n += 1
        old_policy_map = policy_map.copy()
        value_map = policy_evaluation(maze, policy_map, discount, threshold, iteration_limit)
        policy_map = policy_improvement(maze, old_policy_map.copy(), value_map, discount)
        if np.array_equal(policy_map, old_policy_map):
            print('Finished policy iteration after {} iterations'.format(n))
            return policy_map, value_map
    print('Policy iteration loop limit reached')



def policy_evaluation(maze, policy_map, discount, threshold, iteration_limit):
    value_map = np.full(maze_map.shape, 9.0)  # arbitrary init value
    possible_states = maze.possible_states
    n = 0
    val_change = 99
    while n < iteration_limit and val_change > threshold:  # repeat until convergence of value map
        n += 1
        value_map_old = value_map.copy()
        for state in possible_states:
            neighbour_states = [np.add(state, actions[command]) for command in actions.keys()]
            neighbour_states = [neighbour for neighbour in neighbour_states if neighbour.tolist() in possible_states]
            neighbour_states.append(state)

            action = policy_map[state[0], state[1]]
            step_cost = maze.step_cost(state, action)
            expected_next_val = discount * sum([maze.state_trans_prob_function(state, neighbour, action) *
                                                value_map_old[neighbour[0], neighbour[1]] for neighbour in
                                                neighbour_states])
            value_map[state[0], state[1]] = step_cost + expected_next_val

        diff = np.absolute(value_map_old - value_map)
        val_change = np.amax(diff)
    return value_map


def policy_improvement(maze, policy_map, value_map, discount):
    possible_states = maze.possible_states
    actions = maze.actions
    for state in possible_states:
        neighbour_states = [np.add(state, actions[command]) for command in actions.keys()]
        neighbour_states = [neighbour for neighbour in neighbour_states if neighbour.tolist() in possible_states]
        neighbour_states.append(state)

        action_values = {}
        for action in actions.keys():
            step_cost = maze.step_cost(state, action)
            expected_next_val = discount * sum(
                [maze.state_trans_prob_function(state, neighbour, action) * value_map[neighbour[0], neighbour[1]]
                 for neighbour in neighbour_states])
            val = step_cost + expected_next_val
            action_values[action] = val
        best_action = min(action_values, key=action_values.get)
        policy_map[state[0], state[1]] = best_action
    return policy_map


policy_ground, value_ground = value_iteration(maze, discount, threshold, iteration_limit)

policy2, value2 = policy_iteration(maze, discount, threshold, iteration_limit)

