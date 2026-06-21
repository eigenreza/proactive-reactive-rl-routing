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
    "This simulation looks at a two-stage routing problem for collecting used electric vehicle "
    "batteries when daily return quantities are uncertain. In the first stage, routes are planned "
    "the evening before using the expected number of batteries at each collection point. This gives "
    "a proactive plan: useful, but still based on forecasts rather than actual counts. In the second "
    "stage, the actual battery counts are observed in the morning, and a Q-learning agent decides "
    "whether the original routes should be adjusted before the vehicles leave the depot. The goal is "
    "to compare this proactive-reactive approach with a proactive-only policy and a simple naive "
    "benchmark under different levels of uncertainty."
)

with st.expander("How to use this simulation"):
    st.markdown("""
**What this simulation models**

A logistics company has 3 vehicles operating from one central depot. The vehicles collect used electric vehicle batteries from several drop-off locations spread across a 200 by 200 km service area. Each location receives a random number of returned batteries each day. The company plans its routes the evening before, but the exact number of batteries at each location is only known the next morning. At that point, the company can either keep the original plan or revise it.

**The three policies compared**

- **Naive Baseline:** Every location is visited every day, whether or not batteries are actually available. There is no route optimization.
- **Proactive Only:** Routes are planned the evening before using the expected battery count at each location ($\\mu_i$). The next morning, the vehicles follow those routes exactly, even if some planned stops have no batteries.
- **Two-Stage (Proactive + Reactive):** Routes are first planned in the same way as the proactive policy. Then, once the actual morning counts are known, a Q-learning agent decides whether to keep the plan, remove empty stops, or re-optimize the routes.

**What the sliders control**

- **Return uncertainty ($\\sigma$):** This controls how much the daily battery counts fluctuate around their expected values. When $\\sigma = 0.5$, the counts are fairly predictable, so the proactive plan usually works well. When $\\sigma = 3.0$ or $\\sigma = 4.0$, the counts can change a lot from day to day, and planned stops are more likely to be empty. This is the key parameter to experiment with.
- **Collection locations:** This sets the number of drop-off locations in the network. More locations make the routing problem larger and harder.
- **Vehicle capacity:** This is the maximum number of battery units each vehicle can carry on a trip.
- **RL training episodes:** This controls how many simulated days the Q-learning agent trains on before evaluation. More training usually gives a better reactive policy, but it also increases runtime. 500 is a reasonable default.
- **Evaluation runs:** This controls how many independent simulation runs are used when computing the averages in the figures. More runs give smoother and more stable curves, but they also take longer.

**How to read the figures**

- **Figure 1** shows the collection network and one example proactive route for a single vehicle.
- **Figure 2** compares the average daily travel distance of the three policies as uncertainty increases. Lower values are better. At higher $\\sigma$ values, the green Two-Stage line should start to improve over the blue Proactive Only line.
- **Figure 3** is the main operational figure. It shows how often vehicles visit locations where no batteries are collected. Lower values are better. When uncertainty is high, the Two-Stage policy should remove most of these unnecessary empty visits.
- **Figure 4** shows the Q-learning agent's training progress. A rising curve means that the agent is learning to make better route-adjustment decisions over time.

**Suggested experiment**

Start with $\\sigma = 1.0$ and run the simulation. Look especially at the empty trip rates in Figure 3. Then increase $\\sigma$ to around $\\sigma = 3.5$ and run it again. The change in Figure 3 shows why reacting to morning information becomes much more valuable when battery returns are highly uncertain.
    """)

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    sigma = st.slider("Return uncertainty (sigma)", min_value=0.5, max_value=4.0,
                      value=2.0, step=0.1, key="sigma")
    st.caption("At σ = 0.5, daily counts are fairly predictable. At σ = 3.0 or higher, empty trips become much more likely, so reactive adjustment matters more.")
    n_nodes = st.slider("Collection locations", min_value=8, max_value=20,
                        value=15, step=1, key="n_nodes")
    capacity = st.slider("Vehicle capacity (units)", min_value=10, max_value=30,
                         value=20, step=5, key="capacity")
    n_episodes = st.slider("RL training episodes", min_value=100, max_value=1000,
                           value=500, step=100, key="n_episodes")
    st.caption("More episodes give the reactive agent more training, but they also increase runtime. 500 is a good starting point.")
    n_runs = st.slider("Evaluation runs", min_value=50, max_value=300,
                       value=200, step=50, key="n_runs")
    run_btn = st.button("Run Simulation", type="primary")

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

if run_btn:
    with st.spinner("Training the Q-learning agent and running the simulation."):
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
            "The network shows collection locations across a 200 by 200 km service area. Larger "
            "points represent locations with higher expected battery returns. The dashed green "
            "line shows one proactive route for Vehicle 1, planned before the actual daily battery "
            "counts are known."
        )
        plt.close(f1)

        f2 = fig2_distance(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f2, use_container_width=True)
        st.caption(
            "This figure shows the average total travel distance per collection day as return "
            "uncertainty increases. The values are averaged over the evaluation runs. At higher "
            "uncertainty levels, the two-stage policy can reduce travel distance by adapting the "
            "routes around locations where batteries are actually available."
        )
        plt.close(f2)

        f3 = fig3_empty_trips(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f3, use_container_width=True)
        st.caption(
            "This figure shows the share of location visits where no batteries are collected. "
            "These empty trips are costly because they use vehicle time and travel distance without "
            "producing any pickup. The two-stage policy keeps this rate close to zero by using the "
            "morning information before the vehicles depart."
        )
        plt.close(f3)

        f4 = fig4_training_curve(training_rewards)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f4, use_container_width=True)
        st.caption(
            "This figure shows the Q-learning agent's cumulative reward during training, smoothed "
            "with a 20-episode rolling window. The upward trend indicates that the agent is gradually "
            "learning when it is better to keep the original plan, prune empty stops, or re-optimize "
            "the routes."
        )
        plt.close(f4)

# Methodology Notes
st.subheader("Methodology Notes")

st.markdown("""
The number of batteries ready for pickup at location $i$ on day $t$ is modeled as:

$$D_{i,t} = \\max\\left(0,\\ \\mathrm{round}\\left(\\mu_i + \\sigma \\cdot \\varepsilon_{i,t}\\right)\\right)$$

Here, $\\mu_i$ is the average daily battery return for location $i$. It is drawn once from $\\mathrm{Uniform}(2, 8)$ and then kept fixed across the simulation. The random term $\\varepsilon_{i,t}$ is an independent standard normal variable that represents day-to-day variation. The slider value $\\sigma$ controls the strength of this variation. When $\\sigma$ is small, the realized counts stay close to their averages. When $\\sigma$ is large, the realized counts can move far away from the expected values, so a location that looked important in the evening plan may have no batteries the next morning.
""")

st.markdown("""
The proactive route plan is built by treating the expected battery counts $\\mu_i$ as deterministic demands in a capacitated vehicle routing problem. A nearest-neighbor greedy heuristic first assigns locations to vehicles while respecting capacity. Then a 2-opt local search improves each route by reversing route segments whenever this reduces Euclidean travel distance. This gives an evening route plan that is fixed before the random daily counts are observed. In that sense, it is an open-loop policy: the plan is made before uncertainty is resolved, and it does not react to the morning realization.
""")

st.markdown("""
Each morning, the actual battery counts $D_{i,t}$ are revealed before the vehicles leave the depot. The Q-learning agent observes how the realized counts compare with the expected counts, using a discretized state based on the ratio $r_i = D_{i,t} / \\mu_i$. It then chooses one of three actions: follow the proactive plan unchanged, remove confirmed-empty stops while keeping the existing route order, or fully re-optimize the routes using the actual counts. The agent is trained with the Bellman update:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]$$

where the learning rate is $\\alpha = 0.1$ and the discount factor is $\\gamma = 0.95$. The reward penalizes travel distance and gives credit for avoiding empty trips. This encourages the agent to skip locations where no batteries are ready for pickup. Operationally, this is a closed-loop version of the routing problem: the company still plans ahead, but it waits until better information is available before making the final dispatch decision.
""")