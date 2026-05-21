import numpy as np


class Network:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.depot = np.array([100.0, 100.0])
        self.locations = rng.uniform(0, 200, size=(15, 2))
        self.mu = rng.uniform(2, 8, size=15)
        self.n_locations = 15

    def distance(self, a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def route_distance(self, route):
        if not route:
            return 0.0
        total = self.distance(self.depot, self.locations[route[0]])
        for i in range(len(route) - 1):
            total += self.distance(self.locations[route[i]], self.locations[route[i + 1]])
        total += self.distance(self.locations[route[-1]], self.depot)
        return total
