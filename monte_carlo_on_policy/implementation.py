import pickle as pkl
from itertools import product
from typing import Dict, Tuple, List

import gym
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go


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

    def accumulate(self, state: Tuple[int, int, bool], action: int, reward: float):
        """
        The averaging procedure for returns.
        """
        if state not in self.q or action not in self.q[state]:
            if state not in self.q:
                self.q[state] = {}
                self.count[state] = {}
            self.q[state][action] = reward
            self.count[state][action] = 1

        else:
            tot = self.q[state][action] * self.count[state][action]
            tot += reward
            self.count[state][action] += 1
            self.q[state][action] = tot / self.count[state][action]


class Policy:
    def __init__(self, action_list: List):
        """
        :param action_list: A list of actions which can be taken.
        """
        self.pi = {}
        self.action_list = action_list

    def get_action_distribution(self, state: Tuple[float, float, float, float]):
        """
        Gets the distribution of actions given a state.
        """
        return self.pi[state].copy()

    def get_probability(self, a, s: Tuple[float, float, float, float]):
        """
        Gets the probability of action a given state s.
        """
        return self.pi[s][a]

    def get_next_action(self, s: Tuple[float, float, float, float]):
        """
        Samples the next action from the distributiion associated with s.
        """
        if s not in self.pi:
            return self.action_list[np.random.randint(0, len(self.action_list))]

        dist_dict = self.pi[s]
        dist = [(key, dist_dict[key]) for key in dist_dict]
        dist = sorted(dist, key = lambda x: x[1])
        selected_value = np.random.random()
        for i in range(1, len(dist)):
            if selected_value >= dist[i-1][1] and selected_value <= dist[i][1]:
                return dist[i][0]
        return dist[-1][0]

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

            actions = [0, 1]

            for action in actions:
                exp_return = Q.get(state, action)
                if exp_return > max_exp_return:
                    max_exp_return = exp_return
                    max_action = action

                pi[state][action] = epsilon / len(actions)

            pi[state][max_action] = 1 - epsilon + (epsilon / len(actions))
        self.pi = pi


def generate_episode(env, pi: Policy, render: bool = False) -> List[Tuple[Tuple[float, float, float, float], float, float]]:
    """
    Using the environment env and policy pi, generates a sample from the environment.
    :param env: The environment to interact with. I'm assuming it's CartPole for this specific example.
    :param pi: The policy to follow; it's a discrete probability distribution represented as a mapping
        from states to a mapping from actions to the probability of taking that action given the state
        which the mapping is from; pi(a | s) is implemented as p[s][a]. 
    :param render: Choose whether or not to render the generation of this episode.

    Returns a list of state at t-1, action at t-1, and reward at t.
    """
    state = env.reset()
    done = False
    episode = []

    while not done:
        action = pi.get_next_action(state)
        tmp_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = tmp_state 

    return episode


def on_policy_mc_control(env, epsilon: float, gamma: float, num_episodes: int, 
        pi: Policy = None, Q: TabularQ = None, print_every=100):
    """
    An implementation of on-policy first-visit Monte Carlo control for epsilon soft policies.
    I do policy improvement on an episode by episode basis.
    :param epsilon: The epsilon from which the epsilon soft policy will be generated.
    :param gamma: The discount factor to use.
    :param num_episodes: The number of episodes to run before returning the generated policy.
    :param pi: The policy which can be passed in.
    :param Q: The value function which can be passed in.
    """
    if pi is None:
        pi = Policy([0, 1])
    if Q is None:
        Q = TabularQ()

    returns = []
    print_every_avg_returns = 0

    for i in range(1, num_episodes + 1):
        episode = generate_episode(env, pi)
        G = 0
        # don't have to consider seen or not seen for first visit in this environment
        for (state, action, reward) in episode[::-1]:
            G = G*gamma + reward
            Q.accumulate(state, action, reward)

        pi.generate_new_policy(Q, epsilon=epsilon)
        returns.append(G)
        print_every_avg_returns += G

        if i % print_every == 0:
            print(i, print_every_avg_returns / print_every)
            print_every_avg_returns = 0

    generate_episode(env, pi)

    return Q, pi, returns


def get_policy_graph(pi: Policy, eps, gamma):
    """
    Prints out two grids, which says stick or hit for each value on a grid.
    """
    for usable_ace in [False, True]:
        hits = [[], []]
        sticks = [[], []]
        for player_sum in range(11, 22):
            for dealer_val in range(1, 11):
                state = (player_sum, dealer_val, usable_ace)
                try:
                    if pi.pi[state][0] > pi.pi[state][1]:
                        sticks[0].append(dealer_val)
                        sticks[1].append(player_sum)
                    else:
                        hits[0].append(dealer_val)
                        hits[1].append(player_sum)
                except KeyError:
                    continue

            data = [go.Scatter(x=hits[0], y=hits[1], mode='markers', name='Hit'),
                     go.Scatter(x=sticks[0], y=hits[1], mode='markers', name='Stick')]

        url = py.iplot(data, filename=f'{usable_ace}_eps_{eps}_gamma_{gamma}_policy_viz')
        print(url)
    


env = gym.make('Blackjack-v0')
num_episodes = 100000

for eps in [0.1, 0.75, 0.1]:
    for gamma in [1]:
        print(eps, gamma)
        Q, pi, returns = on_policy_mc_control(env, eps, gamma, num_episodes, print_every=10000)
        get_policy_graph(pi, eps, gamma)

        break
        pkl.dump(Q, open(f'Q_{eps}_{gamma}.pkl', 'wb'))
        pkl.dump(pi, open(f'pi_{eps}_{gamma}.pkl', 'wb'))
        pkl.dump(returns, open(f'returns_{eps}_{gamma}.pkl', 'wb'))
    break
