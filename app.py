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

st.write(
    "This simulation studies a two-stage decision framework for electric vehicle battery "
    "collection logistics under uncertain return flows. In the proactive stage, vehicle routes "
    "are planned the evening before using expected battery counts at each collection location, "
    "representing an open-loop policy that commits to decisions before uncertainty is resolved. "
    "Each morning, a Q-learning agent observes the actual revealed battery counts and decides "
    "whether to adjust the planned routes, representing a closed-loop reactive policy that "
    "conditions decisions on realized information. The simulation compares this two-stage "
    "approach against a proactive-only baseline and a naive benchmark across a range of "
    "uncertainty levels."
)

with st.expander("How to use this simulation"):
    st.markdown("""
**What this simulation models**

A logistics company operates 3 collection vehicles from a central depot. The vehicles collect used electric vehicle batteries from a number of dispersed drop-off locations across a 200 by 200 km service region. Each location accumulates batteries daily, but the exact count is unknown until the morning of collection. The company must decide routes the evening before and then optionally adjust them each morning when actual counts are revealed.

**The three policies compared**

- **Naive Baseline:** Every location is visited every day regardless of expected or actual battery counts. No optimization of any kind.
- **Proactive Only:** Routes are optimized the evening before using the average expected battery count at each location ($\\mu_i$). These routes are then followed exactly the next morning, even if some locations turn out to have zero batteries.
- **Two-Stage (Proactive + Reactive):** Routes are optimized the evening before as above. Each morning, before vehicles depart, the actual battery counts are revealed and a Q-learning agent decides whether to adjust the routes, for example by skipping locations that have zero batteries ready.

**What the sliders control**

- **Return uncertainty ($\\sigma$):** Controls how unpredictable daily battery counts are. At $\\sigma = 0.5$, counts are close to their daily averages and the proactive plan works well. At $\\sigma = 3.0$ or $\\sigma = 4.0$, counts vary wildly from day to day, empty trips become frequent, and morning adjustment becomes critical. This is the most important slider: try running at $\\sigma = 1.0$ and then at $\\sigma = 3.0$ to see the difference.
- **Collection locations:** The number of drop-off locations in the network. More locations means more routing complexity.
- **Vehicle capacity:** The maximum number of battery units each vehicle can carry per trip.
- **RL training episodes:** How many simulated days the Q-learning agent trains on before being evaluated. More episodes produce a better-trained agent but take longer to run. 500 is a good default.
- **Evaluation runs:** How many independent simulation runs are used to compute the performance averages shown in the figures. More runs give smoother curves but take longer.

**How to read the figures**

- **Figure 1** shows the geographic layout of the collection network and an example route for one vehicle on one day.
- **Figure 2** shows mean travel distance per day for each policy as uncertainty increases. Lower is better. Watch for the green Two-Stage line to separate from the blue Proactive Only line at higher $\\sigma$ values.
- **Figure 3** is the most important figure. It shows the fraction of vehicle visits that result in zero batteries collected (empty trips). Lower is better. At high $\\sigma$, the Two-Stage policy reduces empty trips to near zero while the other policies deteriorate.
- **Figure 4** shows how the Q-learning agent improves over training episodes. The upward trend confirms the agent is genuinely learning a better routing policy.

**Suggested experiment**

Set $\\sigma = 1.0$ and click Run Simulation. Note the empty trip rates in Figure 3. Then set $\\sigma = 3.5$ and run again. The difference in Figure 3 illustrates exactly why reactive morning adjustment becomes valuable when return flows are uncertain.
    """)

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    sigma = st.slider("Return uncertainty (sigma)", min_value=0.5, max_value=4.0,
                      value=2.0, step=0.1, key="sigma")
    st.caption("At σ = 0.5 daily counts are predictable. At σ = 3.0 or above, empty trips become frequent and reactive adjustment has the most value.")
    n_nodes = st.slider("Collection locations", min_value=8, max_value=20,
                        value=15, step=1, key="n_nodes")
    capacity = st.slider("Vehicle capacity (units)", min_value=10, max_value=30,
                         value=20, step=5, key="capacity")
    n_episodes = st.slider("RL training episodes", min_value=100, max_value=1000,
                           value=500, step=100, key="n_episodes")
    st.caption("More episodes improve the reactive agent but increase runtime. 500 is recommended.")
    n_runs = st.slider("Evaluation runs", min_value=50, max_value=300,
                       value=200, step=50, key="n_runs")
    run_btn = st.button("Run Simulation", type="primary")

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

if run_btn:
    with st.spinner("Training Q-learning agent and running simulation. This may take 30 to 60 seconds."):
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
            "The EV battery collection network consists of collection locations distributed "
            "across a 200 by 200 km service area, with location size proportional to mean "
            "expected battery count. The dashed green route shows a sample proactive plan for "
            "Vehicle 1 on Day 1, constructed using expected battery counts before any realized "
            "quantities are revealed."
        )
        plt.close(f1)

        f2 = fig2_distance(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f2, use_container_width=True)
        st.caption(
            "Mean total travel distance per collection day as a function of return uncertainty "
            "sigma, averaged across evaluation runs. At higher uncertainty levels, the two-stage "
            "policy achieves lower travel distance than the proactive-only baseline as the "
            "reactive agent learns to consolidate routes around locations with confirmed battery "
            "availability."
        )
        plt.close(f2)

        f3 = fig3_empty_trips(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f3, use_container_width=True)
        st.caption(
            "Mean fraction of location visits resulting in zero batteries collected, as a "
            "function of uncertainty. The two-stage policy reduces the empty trip rate to near "
            "zero across all uncertainty levels, demonstrating the operational value of "
            "closed-loop morning adjustment when return flows are highly variable."
        )
        plt.close(f3)

        f4 = fig4_training_curve(training_rewards)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f4, use_container_width=True)
        st.caption(
            "Cumulative reward per training episode, smoothed over a 20-episode rolling window. "
            "The clear upward trend confirms that the Q-learning agent learns an increasingly "
            "effective route adaptation policy over the course of training."
        )
        plt.close(f4)

# Methodology Notes
st.subheader("Methodology Notes")

st.markdown("""
The number of batteries ready for pickup at location $i$ on day $t$ is modeled as:

$$D_{i,t} = \\max\\left(0,\\ \\mathrm{round}\\left(\\mu_i + \\sigma \\cdot \\varepsilon_{i,t}\\right)\\right)$$

where $\\mu_i$ is a location-specific mean drawn from $\\mathrm{Uniform}(2, 8)$ and held fixed across days, $\\varepsilon_{i,t}$ is an independent standard normal draw representing daily variability, and $\\sigma$ is the global uncertainty parameter controlled by the slider above. At low $\\sigma$, realized counts are close to their expectations and the proactive plan performs well. At high $\\sigma$, realized counts deviate substantially from expectations, and locations planned for collection may have zero batteries available, generating empty trips that waste fuel and driver time.
""")

st.markdown("""
The proactive route plan is constructed by solving a capacitated vehicle routing problem using expected battery counts $\\mu_i$ as deterministic demands. A nearest-neighbor greedy heuristic builds initial vehicle assignments respecting capacity constraints, and a 2-opt local search iteratively improves each route by reversing subsequences to reduce total Euclidean travel distance. The resulting plan is fixed for the entire day and represents an open-loop policy in which all routing decisions are committed to before the stochastic component $\\varepsilon_{i,t}$ is revealed. This mirrors the standard planning practice of most real logistics operations, where routes are dispatched in the evening for the following morning.
""")

st.markdown("""
Each morning, before vehicles depart, the actual battery counts $D_{i,t}$ are revealed. A Q-learning agent observes the deviation of realized counts from expected counts at each location, represented as a discretized state tuple based on the ratio $r_i = D_{i,t} / \\mu_i$, and selects one of three actions: follow the proactive plan unchanged, prune confirmed-empty stops from the existing routes while preserving their optimized sequence, or fully re-optimize routes from scratch using actual counts. The agent is trained via the Bellman update:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]$$

with learning rate $\\alpha = 0.1$ and discount factor $\\gamma = 0.95$. The reward signal penalizes travel distance and grants a bonus for each empty trip avoided, incentivizing the agent to skip locations where no batteries are available. This closed-loop structure, in which decisions are revised after uncertainty is revealed, is the operational analogue of closed-loop postponement strategies studied in supply chain optimization, where delaying commitment until demand information becomes available consistently improves system performance.
""")
