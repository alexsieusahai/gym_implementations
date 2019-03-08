from typing import Tuple, List

import numpy as np

from tabular_q import TabularQ


class Policy:
    def __init__(self, action_list: List):
        """
        :param action_list: A list of actions which can be taken.
        """
        self.pi = {}
        self.action_list = action_list

    def get_next_action(self, s: Tuple[float, float, float, float]):
        """
        Samples the next action from the distributiion associated with s.
        """
        if s not in self.pi:
            return self.action_list[np.random.randint(0, len(self.action_list))]

        dist_dict = self.pi[s]
        dist = [(key, dist_dict[key]) for key in dist_dict]
        dist = [(None, 0)] + sorted(dist, key = lambda x: x[1]) + [(None, 1)]
        selected_value = np.random.random()
        accumulator = 0
        for i in range(1, len(dist)):
            if selected_value >= accumulator and selected_value <= dist[i][1] + accumulator:
                #print(dist, selected_value, dist[i][0])
                return dist[i][0]
            accumulator += dist[i][1]

    def generate_new_policy(self, Q: TabularQ, epsilon: float):
        """
        Given a state value function Q, do policy improvement by making 
        an epsilon-greedy policy.
        """
        pi = {}
        for state in Q.q.keys():
            max_action = None
            max_exp_return = -np.inf

            pi[state] = {}

            for action in self.action_list:
                exp_return = Q.get(state, action)
                if exp_return > max_exp_return:
                    max_exp_return = exp_return
                    max_action = action

                pi[state][action] = epsilon / len(self.action_list)

            pi[state][max_action] = 1 - epsilon + (epsilon / len(self.action_list))
        self.pi = pi
