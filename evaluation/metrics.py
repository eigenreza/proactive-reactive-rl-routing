import numpy as np
from simulation.demand import sample_demands
from simulation.routing import (
    total_distance, count_empty_trips, reoptimize_routes, prune_routes
)
from simulation.rl_agent import compute_bins

N_DAYS = 10
N_EPISODES = 200


def evaluate_naive(network, sigma, n_episodes=N_EPISODES):
    rng = np.random.default_rng(7)
    naive_route = list(range(network.n_locations))
    all_routes = [naive_route, [], []]

    distances = []
    empty_rates = []

    for _ in range(n_episodes):
        day_dists = []
        day_empties = []
        for _ in range(N_DAYS):
            actual = sample_demands(network.mu, sigma, rng)
            dist = total_distance(network, all_routes)
            empty = count_empty_trips(network, all_routes, actual)
            total_visits = sum(len(r) for r in all_routes)
            day_dists.append(dist)
            day_empties.append(empty / total_visits if total_visits > 0 else 0)
        distances.append(np.mean(day_dists))
        empty_rates.append(np.mean(day_empties))

    return np.mean(distances), np.std(distances), np.mean(empty_rates), np.std(empty_rates)


def evaluate_proactive(network, proactive_routes, sigma, n_episodes=N_EPISODES):
    rng = np.random.default_rng(8)
    total_visits = sum(len(r) for r in proactive_routes)

    distances = []
    empty_rates = []

    for _ in range(n_episodes):
        day_dists = []
        day_empties = []
        for _ in range(N_DAYS):
            actual = sample_demands(network.mu, sigma, rng)
            dist = total_distance(network, proactive_routes)
            empty = count_empty_trips(network, proactive_routes, actual)
            day_dists.append(dist)
            day_empties.append(empty / total_visits if total_visits > 0 else 0)
        distances.append(np.mean(day_dists))
        empty_rates.append(np.mean(day_empties))

    return np.mean(distances), np.std(distances), np.mean(empty_rates), np.std(empty_rates)


def evaluate_two_stage(network, proactive_routes, agent, sigma, n_episodes=N_EPISODES):
    rng = np.random.default_rng(9)
    proactive_visits = sum(len(r) for r in proactive_routes)

    distances = []
    empty_rates = []

    for _ in range(n_episodes):
        day_dists = []
        day_empties = []
        for _ in range(N_DAYS):
            actual = sample_demands(network.mu, sigma, rng)
            action, bins = agent.act_greedy(actual)

            if action == 0:
                routes = proactive_routes
            elif action == 1:
                routes = prune_routes(network, proactive_routes, actual, list(bins))
            else:
                routes = reoptimize_routes(network, actual)

            dist = total_distance(network, routes)
            total_visits = sum(len(r) for r in routes)
            empty = count_empty_trips(network, routes, actual)
            day_dists.append(dist)
            day_empties.append(empty / total_visits if total_visits > 0 else 0)
        distances.append(np.mean(day_dists))
        empty_rates.append(np.mean(day_empties))

    return np.mean(distances), np.std(distances), np.mean(empty_rates), np.std(empty_rates)


def run_all_evaluations(network, proactive_routes, agent, sigma_values):
    results = {}
    for sigma in sigma_values:
        naive = evaluate_naive(network, sigma)
        proactive = evaluate_proactive(network, proactive_routes, sigma)
        two_stage = evaluate_two_stage(network, proactive_routes, agent, sigma)
        results[sigma] = {
            "naive": naive,
            "proactive": proactive,
            "two_stage": two_stage,
        }
    return results
