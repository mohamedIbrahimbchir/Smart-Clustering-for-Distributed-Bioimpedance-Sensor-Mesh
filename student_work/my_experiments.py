"""
my_experiments.py — Student sandbox for custom experiments.

Run:
    python student_work/my_experiments.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import Simulator
from src.metrics import print_metrics
from src import config as cfg

from student_work.my_clustering import MY_CLUSTERING_ALGOS
from student_work.my_sleep_scheduler import MY_SCHEDULER_ALGOS


def experiment_test_my_clustering():
    """Quick test: run your clustering on a small network."""
    print("\n── Testing My Clustering ──")
    for name, fn in MY_CLUSTERING_ALGOS.items():
        try:
            sim = Simulator(num_nodes=20, seed=42)
            result = sim.run(clustering_algo=fn, scheduler="periodic", duration=300)
            print(f"  ✓ {name}: {result.metrics.num_clusters} clusters, "
                  f"PDR={result.metrics.packet_delivery_ratio:.1%}")
            Simulator.plot_topology(result, title=f"My Clustering: {name}")
            plt.show()
        except Exception as e:
            print(f"  ✗ {name}: {e}")


def experiment_test_my_scheduler():
    """Quick test: run your scheduler on a small network."""
    print("\n── Testing My Scheduler ──")
    for name, fn in MY_SCHEDULER_ALGOS.items():
        try:
            sim = Simulator(num_nodes=20, seed=42)
            result = sim.run(clustering_algo="leach", scheduler=fn, duration=600)
            print(f"  ✓ {name}: alive={result.metrics.nodes_alive}, "
                  f"PDR={result.metrics.packet_delivery_ratio:.1%}, "
                  f"energy={result.metrics.avg_energy_remaining_frac:.1%}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")


def experiment_combined():
    """Test your clustering + your scheduler together."""
    print("\n── Combined: My Clustering + My Scheduler ──")
    for c_name, c_fn in MY_CLUSTERING_ALGOS.items():
        for s_name, s_fn in MY_SCHEDULER_ALGOS.items():
            try:
                sim = Simulator(num_nodes=30, seed=42)
                result = sim.run(clustering_algo=c_fn, scheduler=s_fn, duration=1800)
                m = result.metrics
                print(f"  {c_name} + {s_name}: "
                      f"PDR={m.packet_delivery_ratio:.1%}, "
                      f"alive={m.nodes_alive}, "
                      f"efficiency={m.energy_efficiency:.1f}")
            except Exception as e:
                print(f"  {c_name} + {s_name}: ERROR — {e}")


def experiment_custom():
    """Blank template — design your own experiment!"""
    print("\n── Custom Experiment ──")
    # TODO: Write your experiment here
    #
    # Example:
    sim = Simulator(num_nodes=40, deployment="clustered", seed=123)
    result = sim.run(
        clustering_algo=MY_CLUSTERING_ALGOS["student_clustering_2"],
        scheduler=MY_SCHEDULER_ALGOS["student_scheduler_2"],
        duration=7200
    )
    print_metrics(result.metrics)
    Simulator.plot_dashboard(result)
    plt.show()
    #print("  (empty — edit this function)")


EXPERIMENTS = {
    1: ("Test my clustering algorithms", experiment_test_my_clustering),
    2: ("Test my sleep schedulers", experiment_test_my_scheduler),
    3: ("Combined test (clustering + scheduler)", experiment_combined),
    4: ("Custom experiment (your sandbox)", experiment_custom),
}


def main():
    print("\n" + "=" * 60)
    print("  My Experiments — Student Sandbox")
    print("=" * 60)
    for num, (desc, _) in EXPERIMENTS.items():
        print(f"  [{num}] {desc}")
    print("=" * 60)

    choice = input("\nSelect (1-4): ").strip()
    if choice.isdigit() and int(choice) in EXPERIMENTS:
        EXPERIMENTS[int(choice)][1]()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
