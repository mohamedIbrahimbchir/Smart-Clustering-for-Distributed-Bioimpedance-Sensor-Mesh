"""
evaluate_my_work.py — Compare your algorithms against the built-in baselines.

Run:
    python student_work/evaluate_my_work.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import Simulator
from src.clustering import CLUSTERING_ALGOS
from src.sleep_scheduler import SCHEDULER_ALGOS
from src.metrics import print_metrics
from src import config as cfg

# Import student algorithms
from student_work.my_clustering import MY_CLUSTERING_ALGOS
from student_work.my_sleep_scheduler import MY_SCHEDULER_ALGOS


def evaluate_clustering_algorithms():
    """Compare student clustering algos vs built-in baselines."""
    print("\n" + "=" * 70)
    print("  CLUSTERING ALGORITHM EVALUATION")
    print("=" * 70)

    # Merge built-in + student algorithms
    all_algos = {}
    all_algos.update(CLUSTERING_ALGOS)
    student_names = set()
    for name, fn in MY_CLUSTERING_ALGOS.items():
        all_algos[name] = fn
        student_names.add(name)

    # Test scenarios
    scenarios = [
        ("Easy (30 nodes, random)", dict(num_nodes=30, deployment="random", seed=42)),
        ("Medium (50 nodes, random)", dict(num_nodes=50, deployment="random", seed=42)),
        ("Hard (30 nodes, clustered)", dict(num_nodes=30, deployment="clustered", seed=99)),
    ]

    for scenario_name, scenario_cfg in scenarios:
        print(f"\n{'─' * 60}")
        print(f"  Scenario: {scenario_name}")
        print(f"{'─' * 60}")

        results = {}
        for algo_name, algo_fn in all_algos.items():
            try:
                sim = Simulator(**scenario_cfg)
                result = sim.run(clustering_algo=algo_fn, scheduler="periodic",
                                 duration=600)
                results[algo_name] = result.metrics
                marker = " ★" if algo_name in student_names else "  "
                print(f"{marker} {algo_name:25s} │ clusters={result.metrics.num_clusters:2d} "
                      f"│ balance={result.metrics.cluster_balance:.3f} "
                      f"│ PDR={result.metrics.packet_delivery_ratio:.1%} "
                      f"│ efficiency={result.metrics.energy_efficiency:.1f}")
            except Exception as e:
                print(f" ✗ {algo_name:25s} │ ERROR: {e}")

        # Bar chart comparison
        if results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            names = list(results.keys())
            colors = ['#d62728' if n in student_names else '#1f77b4' for n in names]

            pdrs = [results[n].packet_delivery_ratio * 100 for n in names]
            axes[0].bar(names, pdrs, color=colors, edgecolor='black')
            axes[0].set_ylabel('PDR (%)')
            axes[0].set_title('Packet Delivery Ratio')
            axes[0].set_ylim(0, 105)

            bals = [results[n].cluster_balance for n in names]
            axes[1].bar(names, bals, color=colors, edgecolor='black')
            axes[1].set_ylabel('Balance (σ/μ)')
            axes[1].set_title('Cluster Balance (lower=better)')

            effs = [results[n].energy_efficiency for n in names]
            axes[2].bar(names, effs, color=colors, edgecolor='black')
            axes[2].set_ylabel('Pkts/Joule')
            axes[2].set_title('Energy Efficiency')

            for ax in axes:
                plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
                ax.grid(True, alpha=0.2, axis='y')

            fig.suptitle(f'Clustering Comparison — {scenario_name}',
                         fontsize=12, fontweight='bold')
            # Legend
            from matplotlib.patches import Patch
            axes[0].legend(handles=[
                Patch(facecolor='#1f77b4', label='Built-in'),
                Patch(facecolor='#d62728', label='★ Student'),
            ])
            fig.tight_layout()
            plt.show()


def evaluate_sleep_schedulers():
    """Compare student sleep schedulers vs built-in baselines."""
    print("\n" + "=" * 70)
    print("  SLEEP SCHEDULER EVALUATION")
    print("=" * 70)

    all_schedulers = {}
    all_schedulers.update(SCHEDULER_ALGOS)
    student_names = set()
    for name, fn in MY_SCHEDULER_ALGOS.items():
        all_schedulers[name] = fn
        student_names.add(name)

    results = {}
    for sched_name, sched_fn in all_schedulers.items():
        try:
            sim = Simulator(num_nodes=30, seed=42)
            result = sim.run(clustering_algo="leach", scheduler=sched_fn,
                             duration=3600)
            results[sched_name] = result
            marker = " ★" if sched_name in student_names else "  "
            m = result.metrics
            print(f"{marker} {sched_name:25s} │ alive={m.nodes_alive:2d} "
                  f"│ PDR={m.packet_delivery_ratio:.1%} "
                  f"│ energy_left={m.avg_energy_remaining_frac:.1%} "
                  f"│ efficiency={m.energy_efficiency:.1f}")
        except Exception as e:
            print(f" ✗ {sched_name:25s} │ ERROR: {e}")

    if results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        names = list(results.keys())
        colors = ['#d62728' if n in student_names else '#2ca02c' for n in names]

        # Alive at end
        axes[0].bar(names, [results[n].metrics.nodes_alive for n in names],
                    color=colors, edgecolor='black')
        axes[0].set_ylabel('Alive Nodes')
        axes[0].set_title('Nodes Alive After 1 Hour')

        # Energy remaining
        axes[1].bar(names,
                    [results[n].metrics.avg_energy_remaining_frac * 100 for n in names],
                    color=colors, edgecolor='black')
        axes[1].set_ylabel('Avg Energy (%)')
        axes[1].set_title('Energy Conservation')

        # PDR
        axes[2].bar(names,
                    [results[n].metrics.packet_delivery_ratio * 100 for n in names],
                    color=colors, edgecolor='black')
        axes[2].set_ylabel('PDR (%)')
        axes[2].set_title('Packet Delivery')
        axes[2].set_ylim(0, 105)

        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
            ax.grid(True, alpha=0.2, axis='y')

        fig.suptitle('Sleep Scheduler Comparison', fontsize=12, fontweight='bold')
        from matplotlib.patches import Patch
        axes[0].legend(handles=[
            Patch(facecolor='#2ca02c', label='Built-in'),
            Patch(facecolor='#d62728', label='★ Student'),
        ])
        fig.tight_layout()
        plt.show()


def main():
    print("\n" + "=" * 60)
    print("  Student Work Evaluator")
    print("=" * 60)
    print("  [1] Evaluate clustering algorithms")
    print("  [2] Evaluate sleep schedulers")
    print("  [3] Evaluate both")
    print("=" * 60)

    choice = input("\nChoice (1-3): ").strip()
    if choice == "1":
        evaluate_clustering_algorithms()
    elif choice == "2":
        evaluate_sleep_schedulers()
    elif choice == "3":
        evaluate_clustering_algorithms()
        evaluate_sleep_schedulers()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
