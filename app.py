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
    "This app is a small routing experiment for used EV battery collection. The basic problem is "
    "simple: the company has to make a route plan before it knows the exact number of batteries "
    "waiting at each location. So the first plan is made in the evening from expected battery "
    "returns. The next morning, the real counts are known, and the Q-learning agent gets one more "
    "chance to decide whether the plan should be kept, cleaned up, or rebuilt. I use the simulation "
    "to compare that two-step idea with a proactive-only plan and a naive visit-everything policy."
)

with st.expander("How to use this simulation"):
    st.markdown("""
**What this simulation models**

The setting is a simple battery pickup network. There is one depot, 3 vehicles, and several collection locations scattered over a 200 by 200 km region. Every location has its own usual level of battery returns, but the actual number changes from day to day. This means the evening plan is never completely wrong, but it is also never guaranteed to match the next morning perfectly.

That is the main point of the app. It asks whether it is worth making a second routing decision after the morning information becomes available.

**The three policies compared**

- **Naive Baseline:** This is the brute-force option. Every location is visited every day. It is simple, but it can easily send a vehicle to a place where nothing is waiting.
- **Proactive Only:** This policy plans routes in the evening using the expected battery count at each location ($\\mu_i$). After that, it does not react. The vehicles follow the plan even when the morning counts show that some stops are empty.
- **Two-Stage (Proactive + Reactive):** This policy starts with the same evening plan. Then, in the morning, the actual battery counts are revealed. The Q-learning agent chooses whether to keep the original routes, remove empty stops, or re-optimize the routes.

**What the sliders control**

- **Return uncertainty ($\\sigma$):** This is the most important knob in the app. Small values mean the daily counts stay close to their averages. Large values mean the day can look quite different from the evening forecast. At $\\sigma = 0.5$, the proactive plan usually has good information. At $\\sigma = 3.0$ or $\\sigma = 4.0$, some planned stops may suddenly be useless because no batteries are actually there.
- **Collection locations:** This sets how many drop-off locations are included. More locations usually means a messier routing problem.
- **Vehicle capacity:** This is how many battery units one vehicle can carry on a trip.
- **RL training episodes:** This is the number of simulated training days used for the Q-learning agent. More episodes give the agent more practice, but the simulation takes longer.
- **Evaluation runs:** This controls how many independent runs are averaged in the plots. A higher number gives smoother curves, but again it costs more time.

**How to read the figures**

- **Figure 1** is just the network map. It shows the depot, the collection locations, and one sample proactive route.
- **Figure 2** shows how much driving the policies produce on average. Lower is better. The interesting part is whether the two-stage policy starts to save distance when uncertainty becomes larger.
- **Figure 3** is the figure I would look at first. It shows empty trips: visits where a vehicle arrives and collects zero batteries. These are pure waste from an operational point of view.
- **Figure 4** shows the training curve for the Q-learning agent. It will not be perfectly smooth, and it should not be read too literally point by point. The useful thing is the overall direction.

**Suggested experiment**

Try $\\sigma = 1.0$ first and run the simulation. Then try something like $\\sigma = 3.5$. In the low-uncertainty case, the evening plan should already be quite reasonable. In the high-uncertainty case, the value of the morning correction should become much easier to see, especially in Figure 3.
    """)

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    sigma = st.slider("Return uncertainty (sigma)", min_value=0.5, max_value=4.0,
                      value=2.0, step=0.1, key="sigma")
    st.caption("Small σ gives a calmer day. Large σ gives more surprises, so the morning adjustment has more room to help.")
    n_nodes = st.slider("Collection locations", min_value=8, max_value=20,
                        value=15, step=1, key="n_nodes")
    capacity = st.slider("Vehicle capacity (units)", min_value=10, max_value=30,
                         value=20, step=5, key="capacity")
    n_episodes = st.slider("RL training episodes", min_value=100, max_value=1000,
                           value=500, step=100, key="n_episodes")
    st.caption("More training can improve the agent, but it also makes the run slower. 500 is a practical starting point.")
    n_runs = st.slider("Evaluation runs", min_value=50, max_value=300,
                       value=200, step=50, key="n_runs")
    run_btn = st.button("Run Simulation", type="primary")

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

if run_btn:
    with st.spinner("Running the training and evaluation now. This can take a little while."):
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
            "This is the simulated service area. The depot is in the middle, and the blue points "
            "are the collection sites. Bigger points mean higher expected battery returns. The "
            "green dashed line is one example of a route planned before the actual daily counts "
            "are known."
        )
        plt.close(f1)

        f2 = fig2_distance(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f2, use_container_width=True)
        st.caption(
            "Here I compare average driving distance. When uncertainty is small, the forecast-based "
            "plan is already fairly reliable, so there may not be much to gain. When uncertainty "
            "gets larger, the morning information becomes more useful, and the two-stage policy can "
            "avoid some unnecessary driving."
        )
        plt.close(f2)

        f3 = fig3_empty_trips(SIGMA_VALUES, results)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f3, use_container_width=True)
        st.caption(
            "This plot is about wasted visits. An empty trip means the vehicle went to a site and "
            "collected zero batteries. That is exactly the kind of mistake a fixed evening plan can "
            "make under uncertainty. The reactive step tries to remove those stops once the morning "
            "counts are known."
        )
        plt.close(f3)

        f4 = fig4_training_curve(training_rewards)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(f4, use_container_width=True)
        st.caption(
            "This shows the reward collected by the Q-learning agent during training, after smoothing "
            "over 20 episodes. I mainly use this as a sanity check. If the curve generally improves, "
            "the agent is picking up useful patterns instead of choosing adjustments at random."
        )
        plt.close(f4)

# Methodology Notes
st.subheader("Methodology Notes")

st.markdown("""
The number of batteries ready for pickup at location $i$ on day $t$ is modeled as:

$$D_{i,t} = \\max\\left(0,\\ \\mathrm{round}\\left(\\mu_i + \\sigma \\cdot \\varepsilon_{i,t}\\right)\\right)$$

Here, $\\mu_i$ is the typical return level for location $i$. It is drawn once from $\\mathrm{Uniform}(2, 8)$ and then kept fixed during the simulation. The random variable $\\varepsilon_{i,t}$ is an independent standard normal draw, so it gives the day-to-day fluctuation around that typical level. The parameter $\\sigma$ controls how large this fluctuation is. When $\\sigma$ is small, the realized counts stay close to what the evening plan expected. When $\\sigma$ is large, a location that looked worthwhile in the plan can easily turn out to have no batteries the next morning.
""")

st.markdown("""
The proactive plan is made by treating the expected counts $\\mu_i$ as deterministic demands. In other words, the evening planner temporarily ignores the random shock and solves the routing problem using the best available average information. The routes are built with a nearest-neighbor greedy heuristic that assigns locations to vehicles while respecting capacity. After that, a 2-opt local search improves the routes by reversing route segments when this reduces the total Euclidean distance. This is not meant to be an exact solver. It is a reasonable planning heuristic for producing a baseline route before the uncertain part of the day has been observed.
""")

st.markdown("""
In the morning, the actual values $D_{i,t}$ are known before the vehicles leave. The Q-learning agent compares the realized counts with the expected counts through the ratio $r_i = D_{i,t} / \\mu_i$, using a discretized state representation. It then chooses one of three actions: follow the proactive plan unchanged, prune confirmed-empty stops while keeping the route order, or fully re-optimize the routes using the actual counts. The agent is trained via the Bellman update:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]$$

with learning rate $\\alpha = 0.1$ and discount factor $\\gamma = 0.95$. The reward penalizes travel distance and rewards avoided empty trips. So the agent is not learning an abstract goal; it is learning a very practical dispatch rule. On a given morning, should the company trust the route plan from last night, remove obviously useless stops, or spend the effort to re-optimize? That is the closed-loop part of the model.
""")