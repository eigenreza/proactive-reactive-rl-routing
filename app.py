```python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Proactive-Reactive Routing",
    page_icon="truck",
    layout="wide"
)

from simulation.network import Network
from simulation.routing import (
    greedy_assign, two_opt, build_proactive_routes,
    reoptimize_routes, prune_routes, total_distance, count_empty_trips
)
import simulation.routing as routing_mod
import simulation.rl_agent as rl_mod
from simulation.rl_agent import QLearningAgent, compute_bins
from simulation.demand import sample_demands
from evaluation.metrics import (
    evaluate_naive, evaluate_proactive, evaluate_two_stage
)


def make_network(n_nodes, seed=42):
    base = Network(seed=seed)
    net = Network.__new__(Network)
    rng = np.random.default_rng(seed)
    net.depot = np.array([100.0, 100.0])
    locs = rng.uniform(0, 200, size=(max(n_nodes, 15), 2))
    mu = rng.uniform(2, 8, size=max(n_nodes, 15))
    net.locations = locs[:n_nodes]
    net.mu = mu[:n_nodes]
    net.n_locations = n_nodes
    return net


def fig1_network(network, proactive_routes):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    mu = network.mu
    sizes = 50 + (mu - mu.min()) / (mu.max() - mu.min() + 1e-9) * 250
    locs = network.locations
    ax.scatter(locs[:, 0], locs[:, 1], s=sizes, c="steelblue", zorder=3, label="Collection site")
    for i, (x, y) in enumerate(locs):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color="gray")
    depot = network.depot
    ax.scatter(*depot, s=200, c="red", marker="s", zorder=4, label="Depot")
    ax.annotate("D", depot, textcoords="offset points", xytext=(4, 4), fontsize=10, color="red")
    v1_route = proactive_routes[0]
    if v1_route:
        path = [depot] + [locs[i] for i in v1_route] + [depot]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "g--", linewidth=1.5, zorder=2, label="Vehicle 1 route (Day 1)")
    ax.set_xlabel("X Coordinate (km)", fontsize=11)
    ax.set_ylabel("Y Coordinate (km)", fontsize=11)
    ax.set_title("EV Battery Collection Network and Sample Proactive Route", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def fig2_distance(sigma_values, results):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "naive": ("Naive Baseline", "red", "--", "o"),
        "proactive": ("Proactive Only", "steelblue", "-", "o"),
        "two_stage": ("Two-Stage", "green", "-", "o"),
    }
    for key, (label, color, ls, marker) in styles.items():
        means = np.array([results[s][key][0] for s in sigma_values])
        stds = np.array([results[s][key][1] for s in sigma_values])
        ax.plot(sigma_values, means, color=color, linestyle=ls, marker=marker,
                label=label, linewidth=1.8)
        ax.fill_between(sigma_values, means - stds, means + stds, color=color, alpha=0.2)
    ax.set_xlabel("Return Uncertainty (sigma)", fontsize=11)
    ax.set_ylabel("Mean Travel Distance per Day (km)", fontsize=11)
    ax.set_title("Policy Comparison: Mean Travel Distance vs. Return Uncertainty", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def fig3_empty_trips(sigma_values, results):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "naive": ("Naive Baseline", "red", "--", "o"),
        "proactive": ("Proactive Only", "steelblue", "-", "o"),
        "two_stage": ("Two-Stage", "green", "-", "o"),
    }
    for key, (label, color, ls, marker) in styles.items():
        means = np.array([results[s][key][2] for s in sigma_values])
        stds = np.array([results[s][key][3] for s in sigma_values])
        ax.plot(sigma_values, means, color=color, linestyle=ls, marker=marker,
                label=label, linewidth=1.8)
        ax.fill_between(sigma_values, np.maximum(0, means - stds), means + stds,
                        color=color, alpha=0.2)
    ax.set_xlabel("Return Uncertainty (sigma)", fontsize=11)
    ax.set_ylabel("Mean Empty Trip Rate", fontsize=11)
    ax.set_title("Policy Comparison: Empty Trip Rate vs. Return Uncertainty", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def fig4_training_curve(training_rewards):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    rewards = np.array(training_rewards)
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(rewards) + 1)
    ax.plot(x, smoothed, color="green", linewidth=1.8, label="Smoothed reward (20-ep window)")
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Cumulative Reward (smoothed)", fontsize=11)
    ax.set_title("Q-Learning Agent Training Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# Page title and description
st.title("Proactive-Reactive Routing under Stochastic Returns")
st.markdown("By [Reza Azad Gholami](https://github.com/eigenreza)")


st.write(
    "This app is a small routing experiment for used EV battery collection. The awkward part of "
    "the problem is that the routes have to be planned before the company knows the exact number "
    "of batteries waiting at each site. So the first plan is made from expected returns. Then, in "
    "the morning, the real counts are observed, and a Q-learning agent is allowed to decide whether "
    "the original plan should be kept or changed. The comparison is between three levels of "
    "decision-making: a naive rule, a proactive route plan, and a two-stage policy that can react "
    "after the uncertainty has partly disappeared."
)

with st.expander("How to use this simulation"):
    st.markdown("""
**What is being simulated**

The setting is a simple but realistic collection problem. There is one depot, 3 vehicles, and a set of battery drop-off locations scattered over a 200 by 200 km region. Some locations tend to produce more returned batteries than others, but the actual number on a given day is random.

This matters because a route that looks reasonable the evening before may look wasteful the next morning. A planned stop may have many batteries, a few batteries, or none at all. The simulation is built around that tension: plan early, but maybe revise once better information arrives.

**The three policies**

- **Naive Baseline:** The simplest possible rule. Visit every location every day. It is not clever, but it gives a useful lower-quality benchmark.
- **Proactive Only:** Build the route plan in advance using the expected battery count at each location ($\\mu_i$). Once the plan is made, it is followed as it is. Empty stops are not removed.
- **Two-Stage (Proactive + Reactive):** Start with the same evening plan. Then, after the morning counts are known, let the Q-learning agent decide whether to keep the routes, cut out empty stops, or re-optimize.

**What the sliders do**

- **Return uncertainty ($\\sigma$):** This is the most important control. Small values mean that daily returns stay fairly close to their averages. Large values mean that the morning can look quite different from the evening forecast. With high $\\sigma$, empty trips become much more likely.
- **Collection locations:** The number of drop-off sites in the network. More sites usually means a messier routing problem.
- **Vehicle capacity:** The maximum number of battery units a vehicle can carry.
- **RL training episodes:** The number of simulated days used to train the Q-learning agent before testing it. A larger value gives the agent more practice, but it also slows down the run.
- **Evaluation runs:** The number of repeated simulation runs used to compute the plotted averages. Higher values reduce random noise in the curves.

**Reading the figures**

- **Figure 1** is just the network layout. The depot is shown together with the collection sites, and one planned route is drawn as an example.
- **Figure 2** reports average daily travel distance. The lower the curve, the less driving the policy needs on average.
- **Figure 3** is the figure I would look at first. It measures empty trips: visits where the vehicle reaches a site and collects zero batteries. These are pure waste in this model.
- **Figure 4** gives the training curve for the Q-learning agent. It will not be perfectly smooth, but the trend should show whether the agent is improving.

**A quick test to try**

Run the model once with $\\sigma = 1.0$. Then run it again with something like $\\sigma = 3.5$. The difference should be most visible in Figure 3. When the returns are stable, the evening plan is often good enough. When returns are noisy, the morning correction becomes much more valuable.
    """)

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    sigma = st.slider("Return uncertainty (sigma)", min_value=0.5, max_value=4.0,
                      value=2.0, step=0.1, key="sigma")
    st.caption("Low σ gives fairly predictable daily counts. High σ creates more surprises, which is where the reactive policy has a chance to help.")
    n_nodes = st.slider("Collection locations", min_value=8, max_value=20,
                        value=15, step=1, key="n_nodes")
    capacity = st.slider("Vehicle capacity (units)", min_value=10, max_value=30,
                         value=20, step=5, key="capacity")
    n_episodes = st.slider("RL training episodes", min_value=100, max_value=1000,
                           value=500, step=100, key="n_episodes")
    st.caption("More training usually helps, but it also increases the runtime. 500 is a sensible starting value.")
    n_runs = st.slider("Evaluation runs", min_value=50, max_value=300,
                       value=200, step=50, key="n_runs")
    run_btn = st.button("Run Simulation", type="primary")

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

if run_btn:
    with st.spinner("Training the Q-learning agent and running the simulations. This may take a short while."):
        # Patch module-level constants to match slider values
        routing_mod.VEHICLE_CAPACITY = capacity
        rl_mod.N_TRAIN_EPISODES = n_episodes

        # 1. Generate network
        network = make_network(n_nodes, seed=42)

        # 2. Build proactive routes using expected demands and capacity
        proactive_routes = build_proactive_routes(network)

        # 3. Train Q-learning agent
        agent = QLearningAgent(network, proactive_routes, sigma=sigma)
        training_rewards = agent.train()

        # 4. Evaluate all three policies across sigma values
        results = {}
        for sv in SIGMA_VALUES:
            naive = evaluate_naive(network, sv, n_episodes=n_runs)
            proactive = evaluate_proactive(network, proactive_routes, sv, n_episodes=n_runs)
            two_stage = evaluate_two_stage(network, proactive_routes, agent, sv, n_episodes=n_runs)
            results[sv] = {
                "naive": naive,
                "proactive": proactive,
                "two_stage": two_stage,
            }

        # 5-6. Generate and display figures with captions
        f1 = fig1_network(network, proactive_routes)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f1, use_container_width=True)
        st.caption(
            "This is the simulated service area. The red square is the depot, and the blue points "
            "are collection sites. Bigger blue points mean higher expected battery returns. The "
            "green dashed line is one example of a route that is planned before the actual daily "
            "counts are known."
        )
        plt.close(f1)

        f2 = fig2_distance(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f2, use_container_width=True)
        st.caption(
            "Here I compare how much driving the three policies produce on average. When the "
            "uncertainty is low, the evening plan is already fairly reliable. As uncertainty grows, "
            "the two-stage policy has more opportunity to save distance because it can use the "
            "morning information before dispatch."
        )
        plt.close(f2)

        f3 = fig3_empty_trips(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f3, use_container_width=True)
        st.caption(
            "This is the most direct measure of wasted work in the simulation. An empty trip means "
            "that a vehicle visits a site but finds no batteries to collect. The reactive step is "
            "mainly useful because it can remove those confirmed-empty stops before the vehicles "
            "leave."
        )
        plt.close(f3)

        f4 = fig4_training_curve(training_rewards)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f4, use_container_width=True)
        st.caption(
            "This curve shows the reward obtained by the Q-learning agent during training, after "
            "smoothing over 20 episodes. Individual training episodes can jump around, so the main "
            "thing to watch is the direction of the curve. A rising pattern means that the agent is "
            "learning better morning adjustment decisions."
        )
        plt.close(f4)

# Methodology Notes
st.subheader("Methodology Notes")

st.markdown("""
The number of batteries ready for pickup at location $i$ on day $t$ is modeled as:

$$D_{i,t} = \\max\\left(0,\\ \\mathrm{round}\\left(\\mu_i + \\sigma \\cdot \\varepsilon_{i,t}\\right)\\right)$$

Here $\\mu_i$ is the typical return level at location $i$. It is drawn once from $\\mathrm{Uniform}(2, 8)$ and then kept fixed, so the locations are not all identical. The random term $\\varepsilon_{i,t}$ is an independent standard normal draw. It is there to represent the ordinary day-to-day variation in returned batteries. The parameter $\\sigma$ controls how strong that variation is. When $\\sigma$ is small, the realized counts stay close to the planned values. When $\\sigma$ is large, the evening plan becomes much less reliable, and some scheduled stops may turn out to have no batteries at all.
""")

st.markdown("""
The proactive route plan treats the expected counts $\\mu_i$ as deterministic demands. In other words, the first plan ignores the morning noise and solves the routing problem using the average return levels. The routes are built in two steps. First, a nearest-neighbor greedy rule assigns locations to vehicles while respecting capacity. Then a 2-opt local search tries to shorten each route by reversing parts of the route when doing so reduces Euclidean travel distance.

This is not meant to be an exact industrial vehicle-routing solver. It is a controlled way to produce a reasonable evening plan, which is enough for studying the main question here: how much is gained by adjusting that plan after the random counts are revealed?
""")

st.markdown("""
In the morning, the actual battery counts $D_{i,t}$ are observed before the vehicles depart. The Q-learning agent compares the realized counts with the expected counts through the ratio $r_i = D_{i,t} / \\mu_i$, using a discretized version of that information as its state. It then chooses one of three actions: follow the proactive plan unchanged, prune confirmed-empty stops while keeping the route order, or re-optimize the routes using the actual counts. The agent is trained via the Bellman update:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]$$

with learning rate $\\alpha = 0.1$ and discount factor $\\gamma = 0.95$. The reward penalizes travel distance and gives a bonus for avoiding empty trips. So the agent is not just trying to drive less; it is also learning to avoid sending vehicles to places where there is nothing to collect. The practical question is simple: once the morning information is available, is the old route still good enough, or is it worth changing?
""")
```
git status
