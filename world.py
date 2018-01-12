import numpy as np
import random as rand


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

        if self._is_wall(self.goal):
            raise ValueError('Goal cannot be inside wall')
        if self._is_wall(self.pos):
            raise ValueError('Start pos cannot be inside wall')

        self.dirs = action_set
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
        step = np.array(self.dirs[direction])
        end = np.add(pos, step)
        if self._is_goal(end):
            return -1
        return 0

    def _step_cost_punish(self, pos, direction):
        step = np.array(self.dirs[direction])
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
        pos_dir = np.add(pos0, np.array(self.dirs[dir]))
        if np.array_equal(pos0, pos1):
            if self._is_wall(pos_dir):
                return 1
            return 1-self.state_trans_prob
        if np.array_equal(pos_dir, pos1):
            return self.state_trans_prob
        return 0
