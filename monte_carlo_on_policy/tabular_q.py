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

    def get(self, state, action):
        """
        Wrapper for the logic that handles when state or action is not defined yet.
        """
        try:
            return self.q[state][action]
        except KeyError:
            return 0

    def accumulate(self, state: Tuple[int, int, bool], action: int, G: float):
        """
        The averaging procedure for returns.
        """
        if state not in self.q or action not in self.q[state]:
            if state not in self.q:
                self.q[state] = {}
                self.count[state] = {}
            self.q[state][action] = G 
            self.count[state][action] = 1
        else:
            tot = self.q[state][action] * self.count[state][action]
            tot += G 
            self.count[state][action] += 1
            self.q[state][action] = tot / self.count[state][action]
