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
    "This app simulates a battery collection problem where the company has to plan before it "
    "knows exactly how many used EV batteries will be waiting at each site. The evening plan is "
    "built from expected returns, so it is sensible but still imperfect. In the morning, the true "
    "battery counts are revealed. A Q-learning agent then gets a chance to keep the plan as it is "
    "or adjust it before the vehicles leave. The point of the simulation is to see when that second "
    "morning decision actually helps, and how it compares with a proactive-only plan and a very "
    "simple naive policy."
)

with st.expander("How to use this simulation"):
    st.markdown("""
**What this simulation models**

Think of a logistics company that sends out 3 collection vehicles from one depot. The vehicles pick up used electric vehicle batteries from drop-off locations spread over a 200 by 200 km area. Some locations usually have more batteries than others, but the exact number changes from day to day. The company can plan routes the evening before, but it only sees the actual battery counts the next morning.

That creates a practical planning problem: should the company trust last night's route plan, or should it revise the routes once the morning information is available?

**The three policies compared**

- **Naive Baseline:** The vehicles visit every location every day. It is easy to understand, but it wastes trips whenever a location has no batteries.
- **Proactive Only:** Routes are planned the evening before using the expected battery count at each location ($\\mu_i$). After that, the plan is followed exactly, even if the next morning shows that some planned stops are empty.
- **Two-Stage (Proactive + Reactive):** The first plan is made in the same proactive way. Then the actual morning counts are revealed, and the Q-learning agent chooses whether to keep the plan, remove empty stops, or rebuild the routes.

**What the sliders control**

- **Return uncertainty ($\\sigma$):** This is the main slider. It controls how much the daily battery counts move around their expected values. With $\\sigma = 0.5$, the day is fairly predictable, so the evening plan is usually close to what is needed. With $\\sigma = 3.0$ or $\\sigma = 4.0$, the situation is much noisier: a location that looked promising in the plan may have no batteries at all the next morning.
- **Collection locations:** This changes how many drop-off locations are included in the network. More locations make the routing problem less tidy and more expensive to solve.
- **Vehicle capacity:** This is the number of battery units a vehicle can carry on one trip.
- **RL training episodes:** This controls how many simulated training days the Q-learning agent sees before it is evaluated. More training can help the agent, but it also makes the run slower. 500 is a reasonable setting for a quick experiment.
- **Evaluation runs:** This controls how many independent simulation runs are averaged in the plots. A larger value gives smoother results, but it takes longer.

**How to read the figures**

- **Figure 1** gives a map of the collection sites and shows one example route for Vehicle 1.
- **Figure 2** compares the average daily travel distance for the three policies. Lower is better. The interesting part is what happens as $\\sigma$ gets larger.
- **Figure 3** shows the empty trip rate. This is the clearest operational measure here: it tells us how often a vehicle goes to a location and finds nothing to collect. A lower value means fewer wasted stops.
- **Figure 4** shows the learning curve for the Q-learning agent. If the curve moves upward over training, the agent is finding better adjustment decisions.

**Suggested experiment**

Run the simulation first with $\\sigma = 1.0$. Then increase it to about $\\sigma = 3.5$ and run it again. Figure 3 should make the difference easy to see: when returns are predictable, the evening plan is often good enough; when returns are noisy, the morning correction becomes much more useful.
    """)

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    sigma = st.slider("Return uncertainty (sigma)", min_value=0.5, max_value=4.0,
                      value=2.0, step=0.1, key="sigma")
    st.caption("Low σ means the daily counts stay close to their averages. High σ means more surprises, and therefore more chances for the reactive step to help.")
    n_nodes = st.slider("Collection locations", min_value=8, max_value=20,
                        value=15, step=1, key="n_nodes")
    capacity = st.slider("Vehicle capacity (units)", min_value=10, max_value=30,
                         value=20, step=5, key="capacity")
    n_episodes = st.slider("RL training episodes", min_value=100, max_value=1000,
                           value=500, step=100, key="n_episodes")
    st.caption("More episodes give the agent more practice, but the simulation will take longer. 500 is a useful default.")
    n_runs = st.slider("Evaluation runs", min_value=50, max_value=300,
                       value=200, step=50, key="n_runs")
    run_btn = st.button("Run Simulation", type="primary")

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

if run_btn:
    with st.spinner("Training the Q-learning agent and running the simulation. This can take a little while."):
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
            "The map shows the depot and the battery collection sites inside the service area. "
            "Larger blue points correspond to sites with higher expected battery returns. The "
            "dashed green line is one example of a route planned in advance, before the actual "
            "battery counts for the day are known."
        )
        plt.close(f1)

        f2 = fig2_distance(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f2, use_container_width=True)
        st.caption(
            "This plot compares the average distance driven per collection day. When uncertainty "
            "is low, the policies may look fairly similar because the evening forecast is already "
            "quite reliable. As uncertainty increases, the two-stage policy has more room to save "
            "distance by reacting to the actual morning counts."
        )
        plt.close(f2)

        f3 = fig3_empty_trips(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f3, use_container_width=True)
        st.caption(
            "This plot tracks empty visits: cases where a vehicle goes to a location and collects "
            "zero batteries. These trips are exactly the kind of waste the reactive step is meant "
            "to avoid. When the realized counts are very different from the evening expectations, "
            "removing confirmed-empty stops can make a large difference."
        )
        plt.close(f3)

        f4 = fig4_training_curve(training_rewards)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f4, use_container_width=True)
        st.caption(
            "This curve shows the agent's cumulative reward during training, smoothed over 20 "
            "episodes. The main thing to look for is the general direction rather than every small "
            "wiggle. An upward trend means the agent is gradually learning which morning adjustment "
            "is worth making."
        )
        plt.close(f4)

# Methodology Notes
st.subheader("Methodology Notes")

st.markdown("""
The number of batteries ready for pickup at location $i$ on day $t$ is modeled as:

$$D_{i,t} = \\max\\left(0,\\ \\mathrm{round}\\left(\\mu_i + \\sigma \\cdot \\varepsilon_{i,t}\\right)\\right)$$

In this expression, $\\mu_i$ is the usual return level for location $i$. It is sampled once from $\\mathrm{Uniform}(2, 8)$ and then kept fixed, so each location has its own typical demand level. The term $\\varepsilon_{i,t}$ is an independent standard normal draw, which adds day-to-day noise. The slider value $\\sigma$ controls how strong that noise is. With a small $\\sigma$, the realized counts stay close to the expected counts. With a large $\\sigma$, the realized counts can be quite different, and a stop that looked reasonable in the evening may turn out to be empty in the morning.
""")

st.markdown("""
The proactive plan treats the expected battery counts $\\mu_i$ as if they were known deterministic demands. The routes are built as a capacitated vehicle routing problem. First, a nearest-neighbor greedy heuristic assigns locations to vehicles while respecting the vehicle capacity. Then a 2-opt local search improves each route by reversing parts of the route whenever this shortens the total Euclidean travel distance. The important point is that this plan is made before the random term $\\varepsilon_{i,t}$ is observed. Once the plan is fixed, it represents the usual evening-dispatch logic used in many routing settings: plan from forecasts, then send the vehicles out according to that plan.
""")

st.markdown("""
Each morning, before the vehicles leave, the actual battery counts $D_{i,t}$ become available. The Q-learning agent compares these realized counts with the expected counts, using a discretized state based on the ratio $r_i = D_{i,t} / \\mu_i$. It then chooses one of three actions: follow the proactive plan unchanged, remove confirmed-empty stops while keeping the existing route order, or re-optimize the routes using the actual counts. The agent is trained via the Bellman update:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]$$

with learning rate $\\alpha = 0.1$ and discount factor $\\gamma = 0.95$. The reward penalizes travel distance and gives extra credit for avoiding empty trips. In practical terms, the agent is being trained to answer a simple dispatch question each morning: is last night's route still good enough, or is there now enough new information to justify changing it? This is the closed-loop part of the model, because the final routing decision is made after some of the uncertainty has been resolved.
""")