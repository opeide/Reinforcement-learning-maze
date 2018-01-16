import numpy as np
import random as rand
import matplotlib.pyplot as plt

class World:
    def __init__(self, map_matrix, start_pos, goal_pos, action_set, step_cost_type='punish', state_trans_prob=1):
        if step_cost_type == 'reward':
            self.step_cost = self._step_cost_reward
        elif step_cost_type == 'punish':
            self.step_cost = self._step_cost_punish
        else:
            raise ValueError('Cost type must be reward or punish')

        self.map_mat = map_matrix
        self.pos = start_pos
        self.goal = goal_pos

        self.possible_states = [[i, j] for i in range(map_matrix.shape[0]) for j in range(map_matrix.shape[1]) if not map_matrix[i, j]]

        if self._is_wall(self.goal):
            raise ValueError('Goal cannot be inside wall')
        if self._is_wall(self.pos):
            raise ValueError('Start pos cannot be inside wall')

        self.actions = action_set
        self.state_trans_prob = state_trans_prob

    def move(self, direction):
        new_pos = np.add(self.get_pos(), np.array(self.dirs[direction]))
        cost = self.step_cost(self.get_pos(), np.array(self.dirs[direction]))
        if not self._is_wall(new_pos):
            if rand.random() < self.state_trans_prob:
                self.set_pos(new_pos)
        return cost

    def get_pos(self):
        return self.pos

    def set_pos(self, new_pos):
        self.pos = new_pos

    def _is_wall(self, pos):
        return self.map_mat[pos[0], pos[1]]

    def _is_goal(self, pos):
        return np.array_equal(pos, self.goal)

    def _step_cost_reward(self, pos, direction):
        step = np.array(self.actions[direction])
        end = np.add(pos, step)
        if self._is_goal(end):
            return -1
        return 0

    def _step_cost_punish(self, pos, direction):
        step = np.array(self.actions[direction])
        end = np.add(pos, step)
        if self._is_goal(end):
            return 0
        return 1

    def state_trans_prob_function(self, pos0, pos1, dir):
        if self._is_wall(pos0):
            raise ValueError('pos0, starting position, cannot be inside wall')
        if self._is_goal(pos0):
            return 0
        if self._is_wall(pos1):
            return 0
        pos_dir = np.add(pos0, np.array(self.actions[dir]))
        if np.array_equal(pos0, pos1):
            if self._is_wall(pos_dir):
                return 1
            return 1-self.state_trans_prob
        if np.array_equal(pos_dir, pos1):
            return self.state_trans_prob
        return 0







def value_iteration(maze, discount, threshold, iteration_limit):
    possible_states = maze.possible_states
    actions = maze.actions
    policy_map = np.full(maze_map.shape, "down", dtype=object)
    value_map = np.full(maze_map.shape, 0.0)  #arbitrary init value

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

    print('Finished value iteration after {} iterations'.format(n))
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
    value_map = np.full(maze_map.shape, 0.0)  # arbitrary init value
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



def plot_value(value_map, desc='value plot'):
    plt.imshow(value_map, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.suptitle(desc)
    plt.show()

def plot_policy(policy_map, actions, desc='policy plot'):
    x = []
    y = []
    xv = []
    yv = []
    action_map = np.empty(policy_map.shape, list)
    for m in range(policy_map.shape[0]):
        for n in range(policy_map.shape[1]):
            vect = actions[policy_map[m][n]]
            x.append(n)
            y.append(policy_map.shape[0]-1-m)
            xv.append(vect[1])
            yv.append(-vect[0])
    plt.quiver(x,y,xv,yv)
    plt.axis('off')
    plt.suptitle(desc)
    plt.show()

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
discount = 0.9
threshold = 0.0001
iteration_limit = 1000


vi_policy, vi_value = value_iteration(maze, discount, threshold, iteration_limit)
pi_policy, pi_value = policy_iteration(maze, discount, threshold, iteration_limit)

plot_value(pi_value, desc='discount {}'.format(discount))
plot_policy(pi_policy, actions, desc='discount {}'.format(discount))


