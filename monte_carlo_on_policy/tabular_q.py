import numpy as np 

from typing import Tuple

class TabularQ:
    """
    State-value function implementation for Monte Carlo control.
    This implementation abuses the fact that most states are never visited, 
    so it doesn't waste memory storing states that haven't been visited yet.
    """
    def __init__(self):
        """
        :attr q: Tabular q value lookup table. Maps states to a dictionary which maps actions to the expected
            discounted return given S_t = state, A_t = action :attr count: The amount of times a tuple of state, action has been seen.
        """
        self.q = {}
        self.count = {}
        for player_count in range(1, 22):
            for dealer_count in range(1, 12):
                for ace in [False, True]:
                    state = (player_count, dealer_count, ace)
                    self.q[state] = {}
                    self.q[state][0] = np.random.random()
                    self.q[state][1] = np.random.random()
                    self.count[state] = {}
                    self.count[state][0] = 0
                    self.count[state][1] = 0

    def get(self, state, action):
        """
        Wrapper for the logic that handles when state or action is not defined yet.
        """
        return self.q[state][action]

    def accumulate(self, state: Tuple[int, int, bool], action: int, reward: float):
        """
        The averaging procedure for returns.
        """
        tot = self.q[state][action] * self.count[state][action]
        tot += reward
        self.count[state][action] += 1
        self.q[state][action] = tot / self.count[state][action]
