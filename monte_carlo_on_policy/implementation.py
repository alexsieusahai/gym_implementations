import pickle as pkl
from itertools import product
from typing import Dict, Tuple, List

import gym
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

import sys
sys.path.append('..')
from tabular_q import TabularQ
from policy import Policy


def generate_episode(env, pi: Policy, render: bool = False) -> List[Tuple[Tuple[float, float, float, float], float, float]]:
    """
    Using the environment env and policy pi, generates a sample from the environment.
    :param env: The environment to interact with. I'm assuming it's CartPole for this specific example.
    :param pi: The policy to follow; it's a discrete probability distribution represented as a mapping
        from states to a mapping from actions to the probability of taking that action given the state
        which the mapping is from; pi(a | s) is implemented as p[s][a]. 
    :param render: Choose whether or not to render the generation of this episode.
    :returns: A list of state at t-1, action at t-1, and reward at t.
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

    states = set()
    for i in range(1, num_episodes + 1):
        episode = generate_episode(env, pi)
        G = 0
        # don't have to consider seen or not seen for first visit in this environment
        for (state, action, reward) in episode[::-1]:
            G = G*gamma + reward 
            Q.accumulate(state, action, G)
            states.add(state)

        pi.generate_new_policy(Q, epsilon=epsilon)
        returns.append(G)
        print_every_avg_returns += G

        if i % print_every == 0:
            print(i, print_every_avg_returns / print_every)
            print_every_avg_returns = 0

    return Q, pi, returns, states


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
                     go.Scatter(x=sticks[0], y=sticks[1], mode='markers', name='Stick')]

        url = py.iplot(data, filename=f'{usable_ace}_eps_{eps}_gamma_{gamma}_policy_viz')
        print(url)
    


env = gym.make('Blackjack-v0')
num_episodes = 1000000

for eps in [0.05]:
    for gamma in [1]:
        print(eps, gamma)
        Q, pi, returns, states = on_policy_mc_control(env, eps, gamma, num_episodes, print_every=10000)
        get_policy_graph(pi, eps, gamma)
        break

        pkl.dump(Q, open(f'Q_{eps}_{gamma}.pkl', 'wb'))
        pkl.dump(pi, open(f'pi_{eps}_{gamma}.pkl', 'wb'))
        pkl.dump(returns, open(f'returns_{eps}_{gamma}.pkl', 'wb'))
