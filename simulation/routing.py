import numpy as np


VEHICLE_CAPACITY = 20
N_VEHICLES = 3


def greedy_assign(network, demands):
    demands = np.array(demands, dtype=int)
    unvisited = set(i for i in range(network.n_locations) if demands[i] > 0)
    routes = [[] for _ in range(N_VEHICLES)]
    remaining = [VEHICLE_CAPACITY] * N_VEHICLES

    for v in range(N_VEHICLES):
        current = network.depot
        while unvisited:
            best_node = None
            best_dist = float("inf")
            for node in unvisited:
                if demands[node] <= remaining[v]:
                    d = network.distance(current, network.locations[node])
                    if d < best_dist:
                        best_dist = d
                        best_node = node
            if best_node is None:
                break
            routes[v].append(best_node)
            remaining[v] -= demands[best_node]
            current = network.locations[best_node]
            unvisited.remove(best_node)

    return routes


def two_opt(network, route):
    if len(route) < 4:
        return route
    improved = True
    best = list(route)
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if network.route_distance(candidate) < network.route_distance(best):
                    best = candidate
                    improved = True
    return best


def build_proactive_routes(network):
    mu_demands = np.maximum(1, np.round(network.mu).astype(int))
    routes = greedy_assign(network, mu_demands)
    routes = [two_opt(network, r) for r in routes]
    return routes


def reoptimize_routes(network, actual_demands):
    demands = np.array(actual_demands, dtype=int)
    routes = greedy_assign(network, demands)
    routes = [two_opt(network, r) for r in routes]
    return routes


def prune_routes(network, proactive_routes, actual_demands, bins):
    pruned = []
    for route in proactive_routes:
        new_route = [
            loc for loc in route
            if not (bins[loc] == 0 and actual_demands[loc] == 0)
        ]
        pruned.append(new_route)
    return pruned


def total_distance(network, routes):
    return sum(network.route_distance(r) for r in routes)


def count_empty_trips(network, routes, actual_demands):
    count = 0
    for route in routes:
        for loc in route:
            if actual_demands[loc] == 0:
                count += 1
    return count
