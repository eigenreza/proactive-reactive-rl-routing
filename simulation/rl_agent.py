import numpy as np
from collections import defaultdict
from simulation.demand import sample_demands
from simulation.routing import (
    prune_routes, reoptimize_routes, total_distance, count_empty_trips
)

EMPTY_TRIP_BONUS = 80
REOPT_PENALTY = 50
ALPHA = 0.1
GAMMA = 0.95
N_TRAIN_EPISODES = 500
N_DAYS = 10


def compute_bins(actual_demands, mu):
    bins = []
    for i in range(len(mu)):
        r = actual_demands[i] / mu[i] if mu[i] > 0 else 0.0
        r = min(r, 2.0)
        if r < 0.5:
            bins.append(0)
        elif r <= 1.5:
            bins.append(1)
        else:
            bins.append(2)
    return tuple(bins)


def state_hash(bins):
    return hash(bins)


def _default_q():
    # Slight bias toward Action 1 (prune) for unseen states;
    # Action 2 (re-optimize) is expensive, so start it low.
    return np.array([-0.5, 0.5, -100.0])


class QLearningAgent:
    def __init__(self, network, proactive_routes, sigma):
        self.network = network
        self.proactive_routes = proactive_routes
        self.sigma = sigma
        self.Q = defaultdict(_default_q)
        self.training_rewards = []

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(3)
        q = self.Q[state]
        return int(np.argmax(q))

    def compute_reward(self, action, actual_demands, bins, proactive_dist, proactive_empty):
        network = self.network

        if action == 0:
            dist = proactive_dist
            empty = proactive_empty
        elif action == 1:
            pruned = prune_routes(network, self.proactive_routes, actual_demands, list(bins))
            dist = total_distance(network, pruned)
            empty = count_empty_trips(network, pruned, actual_demands)
        else:
            reopt = reoptimize_routes(network, actual_demands)
            dist = total_distance(network, reopt)
            empty = count_empty_trips(network, reopt, actual_demands)

        avoided_empty = proactive_empty - empty
        reward = -dist + EMPTY_TRIP_BONUS * avoided_empty
        if action == 2:
            reward -= REOPT_PENALTY
        return reward, dist, empty

    def train(self):
        rng = np.random.default_rng(0)
        mu = self.network.mu
        proactive_routes = self.proactive_routes

        epsilon_start = 1.0
        epsilon_end = 0.05

        episode_rewards = []

        for ep in range(N_TRAIN_EPISODES):
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * ep / (N_TRAIN_EPISODES - 1)
            total_reward = 0.0
            prev_state = None
            prev_action = None
            prev_reward = None

            for day in range(N_DAYS):
                actual = sample_demands(mu, self.sigma, rng)
                bins = compute_bins(actual, mu)
                state = state_hash(bins)

                proactive_dist = total_distance(self.network, proactive_routes)
                proactive_empty = count_empty_trips(self.network, proactive_routes, actual)

                action = self.get_action(state, epsilon)
                reward, dist, empty = self.compute_reward(
                    action, actual, bins, proactive_dist, proactive_empty
                )

                if prev_state is not None:
                    best_next = float(np.max(self.Q[state]))
                    self.Q[prev_state][prev_action] += ALPHA * (
                        prev_reward + GAMMA * best_next - self.Q[prev_state][prev_action]
                    )

                prev_state = state
                prev_action = action
                prev_reward = reward
                total_reward += reward

            if prev_state is not None:
                self.Q[prev_state][prev_action] += ALPHA * (
                    prev_reward - self.Q[prev_state][prev_action]
                )

            episode_rewards.append(total_reward)

        self.training_rewards = episode_rewards
        return episode_rewards

    def act_greedy(self, actual_demands):
        mu = self.network.mu
        bins = compute_bins(actual_demands, mu)
        state = state_hash(bins)
        q = self.Q[state]
        return int(np.argmax(q)), bins

    def get_action_distribution(self, sigma, n_episodes=200):
        rng = np.random.default_rng(99)
        counts = [0, 0, 0]
        for _ in range(n_episodes):
            for _ in range(N_DAYS):
                actual = sample_demands(self.network.mu, sigma, rng)
                action, _ = self.act_greedy(actual)
                counts[action] += 1
        total = sum(counts)
        return [c / total for c in counts]
