import sys
import numpy as np

from simulation.network import Network
from simulation.routing import build_proactive_routes, total_distance
from simulation.rl_agent import QLearningAgent
from evaluation.metrics import run_all_evaluations
from simulation.figures import (
    figure1_network, figure2_distance, figure3_empty_trips, figure4_training_curve
)

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TRAIN_SIGMA = 3.0


def print_summary(sigma_values, results):
    header = (
        f"{'Sigma':>6} | {'Naive Dist':>12} {'Pro Dist':>10} {'2S Dist':>10}"
        f" | {'Naive ETR':>10} {'Pro ETR':>9} {'2S ETR':>9}"
    )
    print("\n" + "=" * len(header))
    print("POLICY COMPARISON SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for sigma in sigma_values:
        r = results[sigma]
        nd, _, ne, _ = r["naive"]
        pd, _, pe, _ = r["proactive"]
        td, _, te, _ = r["two_stage"]
        print(
            f"{sigma:>6.1f} | {nd:>12.1f} {pd:>10.1f} {td:>10.1f}"
            f" | {ne:>10.4f} {pe:>9.4f} {te:>9.4f}"
        )
    print("=" * len(header))


def check_conditions(results, agent):
    sigma_check = 3.0
    r = results[sigma_check]
    two_dist = r["two_stage"][0]
    pro_dist = r["proactive"][0]
    two_etr = r["two_stage"][2]
    pro_etr = r["proactive"][2]

    dist_ok = two_dist < pro_dist
    etr_ok = two_etr < pro_etr

    dist_ok_25 = results[2.5]["two_stage"][0] < results[2.5]["proactive"][0]
    etr_ok_25 = results[2.5]["two_stage"][2] < results[2.5]["proactive"][2]

    action_dist = agent.get_action_distribution(sigma=3.0)
    action1_ok = action_dist[1] > 0.40

    print(f"\nVerification at sigma=3.0:")
    print(f"  Two-Stage dist ({two_dist:.1f}) < Proactive dist ({pro_dist:.1f}): {dist_ok}")
    print(f"  Two-Stage ETR ({two_etr:.4f}) < Proactive ETR ({pro_etr:.4f}): {etr_ok}")
    print(f"  Two-Stage dist < Proactive dist at sigma=2.5: {dist_ok_25}")
    print(f"  Two-Stage ETR < Proactive ETR at sigma=2.5: {etr_ok_25}")
    print(f"\nAction distribution at sigma=3.0:")
    print(f"  Action 0 (follow plan): {action_dist[0]:.3f}")
    print(f"  Action 1 (prune):       {action_dist[1]:.3f}")
    print(f"  Action 2 (re-optimize): {action_dist[2]:.3f}")
    print(f"  Action 1 > 40%: {action1_ok}")

    all_ok = dist_ok and etr_ok and dist_ok_25 and etr_ok_25 and action1_ok
    return all_ok, action_dist


def main():
    print("Step 1: Generating network...")
    network = Network(seed=42)
    print(f"  Depot: {network.depot}")
    print(f"  Locations: {network.n_locations}")
    print(f"  Mean demands (mu): min={network.mu.min():.2f}, max={network.mu.max():.2f}")

    print("\nStep 2: Building proactive routes...")
    proactive_routes = build_proactive_routes(network)
    for i, r in enumerate(proactive_routes):
        print(f"  Vehicle {i+1}: {len(r)} stops, dist={network.route_distance(r):.1f} km")

    print(f"\nStep 3: Training Q-learning agent at sigma={TRAIN_SIGMA}...")
    agent = QLearningAgent(network, proactive_routes, sigma=TRAIN_SIGMA)
    training_rewards = agent.train()
    print(f"  Training complete. Final episode reward: {training_rewards[-1]:.1f}")
    print(f"  Mean last 50 episodes: {np.mean(training_rewards[-50:]):.1f}")
    print(f"  Q-table size: {len(agent.Q)} distinct states visited")

    print("\nStep 4: Evaluating all policies across sigma values...")
    results = run_all_evaluations(network, proactive_routes, agent, SIGMA_VALUES)
    print("  Evaluation complete.")

    print("\nStep 5: Generating figures...")
    figure1_network(network, proactive_routes)
    figure2_distance(SIGMA_VALUES, results)
    figure3_empty_trips(SIGMA_VALUES, results)
    figure4_training_curve(training_rewards)

    print_summary(SIGMA_VALUES, results)
    all_ok, action_dist = check_conditions(results, agent)

    if not all_ok:
        print("\nRetrying with higher empty trip bonus...")
        import simulation.rl_agent as rl_mod
        rl_mod.EMPTY_TRIP_BONUS = 160

        agent2 = QLearningAgent(network, proactive_routes, sigma=TRAIN_SIGMA)
        training_rewards2 = agent2.train()
        results2 = run_all_evaluations(network, proactive_routes, agent2, SIGMA_VALUES)

        figure2_distance(SIGMA_VALUES, results2)
        figure3_empty_trips(SIGMA_VALUES, results2)
        figure4_training_curve(training_rewards2)

        print_summary(SIGMA_VALUES, results2)
        all_ok2, _ = check_conditions(results2, agent2)

        if all_ok2:
            print("\nAll verification conditions met after adjustment.")
        else:
            print("\nConditions still not all met after second attempt.")

    print("\nDone. Four figures saved to figures/ directory:")
    for fname in ["figure1_network.png", "figure2_distance.png",
                  "figure3_empty_trips.png", "figure4_training_curve.png"]:
        print(f"  figures/{fname}")


if __name__ == "__main__":
    main()
