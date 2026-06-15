# Proactive-Reactive Routing under Stochastic Returns: A Two-Stage Optimization Framework for EV Battery Collection Logistics

## Overview

Electric vehicle adoption has created a large-scale reverse logistics challenge: used EV batteries accumulate at dispersed collection locations at uncertain rates, and operators must dispatch vehicles daily to recover them. Companies typically plan routes the evening before using historical average battery counts and commit to those fixed routes, dispatching drivers without any morning adjustment. When actual battery quantities on a given day differ substantially from historical averages, drivers arrive at planned stops to find zero batteries available. These empty trips waste fuel, driver hours, and vehicle capacity, degrading operational efficiency precisely when return flows are most unpredictable.

This project implements and evaluates a two-stage routing framework that separates the planning decision from the dispatch decision. In the proactive stage, routes are constructed the evening before using expected battery counts, producing an open-loop plan that optimizes over anticipated demand. In the reactive stage, a Q-learning agent observes the actual revealed battery counts each morning and decides whether to follow the proactive plan, prune confirmed-empty stops, or fully re-optimize from scratch using realized data, representing a closed-loop policy that conditions decisions on information available at dispatch time. Simulation results show the two-stage policy reduces empty trip rates by over 99 percent at high uncertainty levels compared to proactive-only planning.

## Scenario

The simulation uses 15 collection locations served by 3 vehicles from a single depot, operating in a 200 by 200 km service area over a 10-day planning horizon. Battery counts at each location follow an independent normal distribution with a location-specific mean drawn from Uniform(2, 8) and held fixed across days, scaled by a shared uncertainty parameter sigma. An empty trip is defined as any vehicle visit to a location where zero batteries are available on that day.

## Methodology

### Proactive Stage (Open-Loop Optimization)

The proactive route plan is constructed by formulating a capacitated vehicle routing problem (CVRP) using expected battery counts as deterministic demands. A nearest-neighbor greedy heuristic builds initial vehicle route assignments, adding the closest feasible unvisited location to each vehicle in turn while respecting the capacity constraint. A 2-opt local search then iteratively improves each route by testing all pairwise segment reversals and accepting those that reduce total Euclidean travel distance, repeating until no improving swap exists.

The demand model for location i on day t is D(i,t) = max(0, round(mu_i + sigma * epsilon(i,t))), where mu_i is the location-specific mean and epsilon(i,t) is an independent standard normal draw. The proactive plan uses mu_i as the demand input and is committed before epsilon(i,t) is revealed, making it an open-loop policy. At low sigma the plan performs well because realized counts stay close to expectations. At high sigma the plan degrades because many locations receive zero batteries on days when the realized draw is sufficiently negative.

### Reactive Stage (Closed-Loop Q-Learning)

Each morning before vehicles depart, the actual battery counts D(i,t) are revealed. A Q-learning agent observes the deviation of each realized count from its location mean, computing the ratio D(i,t) / mu_i and discretizing it into three bins: below 0.5 (substantially below expectation), between 0.5 and 1.5 (near expectation), and above 1.5 (substantially above expectation). The resulting tuple of bin values across all locations forms the state representation, allowing the agent to distinguish days with widespread shortfalls from days with widespread surpluses.

The agent selects one of three actions: Action 0 (follow the proactive plan unchanged), Action 1 (prune confirmed-empty stops from the proactive routes while preserving the optimized sequence of remaining stops), or Action 2 (fully re-optimize routes from scratch using realized demand counts). The agent is trained via the Bellman update Q(s,a) = Q(s,a) + alpha * (R + gamma * max Q(s',a') - Q(s,a)) with learning rate alpha = 0.1 and discount factor gamma = 0.95. The reward function is R = negative travel distance + 30 times empty trips avoided minus 50 if Action 2 is selected, where the penalty on Action 2 reflects the cost of re-dispatching. This closed-loop structure conditions each day's dispatch decision on revealed demand information, directly analogous to closed-loop postponement strategies in supply chain optimization where delaying commitment until information arrives consistently reduces waste.

## Key Results

At sigma = 3.0, the two-stage policy reduces mean travel distance from 826 km to 792 km per day (4.1 percent reduction) and reduces the mean empty trip rate from 10.1 percent to 0.1 percent (99 percent reduction) compared to the proactive-only baseline. The performance advantage grows with sigma, as higher uncertainty creates more days with extreme deviations where morning adjustment is most valuable. At low uncertainty below sigma = 1.5, all three policies perform similarly on both metrics, confirming that reactive adaptation adds operational value primarily when return flows are highly variable and the gap between expected and realized battery counts is large.

## Live Demo

The interactive simulation is deployed at: https://proactive-reactive-rl-routing-dmcswxot8epaarzazyvbgm.streamlit.app. 

Adjust the sigma slider to compare policy performance at different uncertainty levels. Set sigma to 1.0 to see similar performance across policies. Set sigma to 3.0 or 4.0 to see the two-stage policy dramatically outperform the baselines on empty trip rate.

## Repository Structure

```
proactive-reactive-rl-routing/
├── app.py                        # Streamlit interactive simulation app
├── run_simulation.py             # Standalone script to run simulation and save figures
├── requirements.txt              # Python package dependencies
├── README.md                     # This file
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
├── simulation/
│   ├── __init__.py
│   ├── network.py                # Network and depot/location geometry
│   ├── demand.py                 # Stochastic demand sampling model
│   ├── routing.py                # CVRP heuristic, 2-opt, route utilities
│   ├── rl_agent.py               # Q-learning agent with deviation-based state
│   └── figures.py                # Figure generation functions (file-save mode)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # Policy evaluation functions across sigma values
└── figures/
    ├── figure1_network.png       # EV collection network map
    ├── figure2_distance.png      # Travel distance vs sigma
    ├── figure3_empty_trips.png   # Empty trip rate vs sigma
    └── figure4_training_curve.png# Q-learning training curve
```

## Requirements

```
streamlit>=1.32.0
numpy>=1.26.0
matplotlib>=3.8.0
scipy>=1.12.0
pandas>=2.2.0
seaborn>=0.13.0
```
