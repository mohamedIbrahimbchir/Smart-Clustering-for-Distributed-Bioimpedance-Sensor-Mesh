"""
experiments.py — Pre-built demonstration experiments for Project 4.

Run:
    python experiments.py

Students should explore these FIRST to understand the simulator,
then build their own algorithms in student_work/.
"""

import sys
import os

from student_work.my_clustering import MY_CLUSTERING_ALGOS
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import Simulator
from src.metrics import print_metrics
from src import config as cfg

# ── Register student algorithms into the simulator registries ──────
from src.clustering import CLUSTERING_ALGOS
from src.sleep_scheduler import SCHEDULER_ALGOS
from student_work.my_clustering import MY_CLUSTERING_ALGOS
from student_work.my_sleep_scheduler import MY_SCHEDULER_ALGOS
CLUSTERING_ALGOS.update(MY_CLUSTERING_ALGOS)
SCHEDULER_ALGOS.update(MY_SCHEDULER_ALGOS)
# ──────────────────────────────────────────────────────────────────


def experiment_1_topology_visualization():
    """Experiment 1: Deploy nodes and visualise the mesh topology."""
    print("\n" + "=" * 60)
    print("  Experiment 1: Network Topology Visualisation")
    print("=" * 60)

    for deployment in ["random", "grid", "clustered"]:
        sim = Simulator(num_nodes=30, deployment=deployment)
        result = sim.run(clustering_algo="rssi_threshold", scheduler="periodic",
                         duration=60)
        fig = Simulator.plot_topology(result,
                                      title=f"Deployment: {deployment}")
        plt.show()
        print_metrics(result.metrics, title=f"{deployment} deployment")


def experiment_2_compare_clustering():
    """Experiment 2: Compare all built-in clustering algorithms."""
    print("\n" + "=" * 60)
    print("  Experiment 2: Clustering Algorithm Comparison")
    print("=" * 60)

    results = Simulator.compare_clustering(
        clustering_algos=["rssi_threshold", "kmedoids", "leach", "spectral"],
        scheduler="periodic",
        num_nodes=30,
        duration=1800,  # 30 minutes
    )
    return results


def experiment_3_compare_schedulers():
    """Experiment 3: Compare sleep/wake-up schedulers."""
    print("\n" + "=" * 60)
    print("  Experiment 3: Sleep Scheduler Comparison")
    print("=" * 60)

    schedulers = ["periodic", "synchronised", "on_demand", "adaptive"]
    results = {}

    for sched in schedulers:
        sim = Simulator(num_nodes=30, seed=42)
        result = sim.run(clustering_algo="leach", scheduler=sched,
                         duration=3600)
        results[sched] = result
        print_metrics(result.metrics, title=f"leach + {sched}")

    # Comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=cfg.FIGURE_DPI)
    names = list(results.keys())
    colors = plt.get_cmap('Set1')(np.linspace(0, 1, len(names)))

    # Alive nodes
    for i, name in enumerate(names):
        ts = results[name].time_series
        if ts["time"]:
            axes[0].plot(np.array(ts["time"]) / 60, ts["alive_count"],
                         label=name, linewidth=2, color=colors[i])
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Alive Nodes')
    axes[0].set_title('Network Lifetime')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PDR
    pdrs = [results[n].metrics.packet_delivery_ratio * 100 for n in names]
    axes[1].bar(names, pdrs, color=colors, edgecolor='black')
    axes[1].set_ylabel('PDR (%)')
    axes[1].set_title('Packet Delivery Ratio')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.2, axis='y')

    # Energy remaining
    fracs = [results[n].metrics.avg_energy_remaining_frac * 100 for n in names]
    axes[2].bar(names, fracs, color=colors, edgecolor='black')
    axes[2].set_ylabel('Avg Energy Remaining (%)')
    axes[2].set_title('Energy Conservation')
    axes[2].grid(True, alpha=0.2, axis='y')

    for ax in axes[1:]:
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    fig.suptitle('Sleep Scheduler Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show()
    return results


def experiment_4_node_scaling():
    """Experiment 4: How does network size affect performance?"""
    print("\n" + "=" * 60)
    print("  Experiment 4: Node Scaling Study")
    print("=" * 60)

    node_counts = [10, 20, 30, 50, 75]
    metrics_list = []

    for n in node_counts:
        sim = Simulator(num_nodes=n, seed=42)
        result = sim.run(clustering_algo="leach", scheduler="adaptive",
                         duration=1800)
        metrics_list.append(result.metrics)
        print(f"  N={n:3d}: PDR={result.metrics.packet_delivery_ratio:.1%}, "
              f"clusters={result.metrics.num_clusters}, "
              f"alive={result.metrics.nodes_alive}/{n}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=cfg.FIGURE_DPI)

    axes[0].plot(node_counts, [m.packet_delivery_ratio * 100 for m in metrics_list],
                 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Nodes')
    axes[0].set_ylabel('PDR (%)')
    axes[0].set_title('PDR vs Network Size')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(node_counts, [m.num_clusters for m in metrics_list],
                 'rs-', linewidth=2)
    axes[1].set_xlabel('Number of Nodes')
    axes[1].set_ylabel('Clusters Formed')
    axes[1].set_title('Clusters vs Network Size')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(node_counts, [m.avg_hops for m in metrics_list],
                 'g^-', linewidth=2)
    axes[2].set_xlabel('Number of Nodes')
    axes[2].set_ylabel('Avg Hops')
    axes[2].set_title('Hop Count vs Network Size')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Scaling Study', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show()


def experiment_5_rssi_threshold_sweep():
    """Experiment 5: Effect of RSSI threshold on clustering."""
    print("\n" + "=" * 60)
    print("  Experiment 5: RSSI Threshold Sweep")
    print("=" * 60)

    thresholds = [-90, -80, -70, -60, -50, -40]
    results = []

    for thr in thresholds:
        sim = Simulator(num_nodes=30, seed=42)
        result = sim.run(clustering_algo="rssi_threshold", scheduler="periodic",
                         duration=600, threshold_dbm=thr)
        results.append(result.metrics)
        print(f"  Threshold={thr:4d} dBm: "
              f"clusters={result.metrics.num_clusters}, "
              f"balance={result.metrics.cluster_balance:.2f}, "
              f"PDR={result.metrics.packet_delivery_ratio:.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=cfg.FIGURE_DPI)

    axes[0].plot(thresholds, [m.num_clusters for m in results], 'bo-', linewidth=2)
    axes[0].set_xlabel('RSSI Threshold (dBm)')
    axes[0].set_ylabel('Clusters')
    axes[0].set_title('Cluster Count vs Threshold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thresholds, [m.cluster_balance for m in results], 'rs-', linewidth=2)
    axes[1].set_xlabel('RSSI Threshold (dBm)')
    axes[1].set_ylabel('Balance (σ/μ)')
    axes[1].set_title('Cluster Balance vs Threshold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(thresholds, [m.packet_delivery_ratio * 100 for m in results],
                 'g^-', linewidth=2)
    axes[2].set_xlabel('RSSI Threshold (dBm)')
    axes[2].set_ylabel('PDR (%)')
    axes[2].set_title('PDR vs Threshold')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('RSSI Threshold Sweep', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show()


def experiment_6_full_dashboard():
    """Experiment 6: Full dashboard for a single long run."""
    print("\n" + "=" * 60)
    print("  Experiment 6: Full Dashboard (1-hour simulation)")
    print("=" * 60)

    sim = Simulator(num_nodes=30, seed=42)
    result = sim.run(clustering_algo="leach", scheduler="adaptive",
                     duration=3600)
    print_metrics(result.metrics, title="LEACH + Adaptive (1 hour)")
    fig = Simulator.plot_dashboard(result)
    plt.show()


def experiment_7_wearable_scenario():
    """
    Experiment 7: Design Challenge — Wearable Bioimpedance Array.

    Scenario: 20 bioimpedance patches on a patient's torso.
    Deployment area: 40cm × 50cm (body surface).
    Must deliver data to gateway (belt-mounted) every 5 seconds.
    Battery must last 8 hours minimum.

    FIX: area expressed in centimetres (40 x 50) instead of metres
    (0.4 x 0.5) so that the deploy_nodes margin of 5 units does not
    produce a negative range (high - low < 0).
    """
    print("\n" + "=" * 60)
    print("  Experiment 7: Wearable Bioimpedance Array Challenge")
    print("=" * 60)

    sim = Simulator(
        num_nodes=20,
        area=(40, 50),          # 40cm × 50cm — expressed in cm units
        gateway_pos=(20, 0),    # belt-mounted (centre of width, bottom)
        deployment="random",
        seed=42,
    )
    result = sim.run(
        clustering_algo="student_clustering_2",   # best PDR — 100%
        scheduler="student_scheduler_2",          # TDMA — zero collisions
        duration=8 * 3600,       # 8 hours
        report_interval=5.0,     # every 5 seconds
        recluster_interval=300,  # re-cluster every 5 min
    )
    print_metrics(result.metrics, title="Wearable Challenge (8 hours)")
    fig = Simulator.plot_dashboard(result)
    plt.show()

    # Check pass condition
    pdr = result.metrics.packet_delivery_ratio
    alive = result.metrics.nodes_alive
    total = len(result.nodes)
    print(f"\n  Challenge Check:")
    print(f"    PDR ≥ 95%?      {'✓ YES' if pdr >= 0.95 else '✗ NO'} ({pdr:.1%})")
    print(f"    All alive?      {'✓ YES' if alive == total else '✗ NO'} ({alive}/{total})")


# ═════════════════════════════════════════════════════════════════════
#  Menu
# ═════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    1: ("Network topology visualisation", experiment_1_topology_visualization),
    2: ("Compare clustering algorithms", experiment_2_compare_clustering),
    3: ("Compare sleep schedulers", experiment_3_compare_schedulers),
    4: ("Node scaling study", experiment_4_node_scaling),
    5: ("RSSI threshold sweep", experiment_5_rssi_threshold_sweep),
    6: ("Full dashboard (1 hour)", experiment_6_full_dashboard),
    7: ("Wearable bioimpedance challenge", experiment_7_wearable_scenario),
}


def main():
    print("\n" + "=" * 60)
    print("  Project 4 — Distributed Bioimpedance Mesh Simulator")
    print("  Pre-built Experiments")
    print("=" * 60)
    for num, (desc, _) in EXPERIMENTS.items():
        print(f"  [{num}] {desc}")
    print(f"  [0] Run ALL experiments")
    print("=" * 60)

    choice = input("\nSelect experiment (0-7): ").strip()
    if choice == "0":
        for num, (_, func) in EXPERIMENTS.items():
            func()
    elif choice.isdigit() and int(choice) in EXPERIMENTS:
        EXPERIMENTS[int(choice)][1]()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
