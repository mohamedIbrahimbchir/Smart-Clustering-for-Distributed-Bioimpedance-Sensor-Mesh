"""
metrics.py — Performance metrics for the distributed bioimpedance mesh.

Evaluates clustering quality, network lifetime, packet delivery, latency, and
energy efficiency.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .node import SensorNode, GatewayNode
from .mesh_network import Packet
from .rssi_model import true_distance
from . import config as cfg


@dataclass
class NetworkMetrics:
    """Complete set of network performance metrics."""
    # ── Clustering quality ───────────────────────────────────────────
    num_clusters: int = 0
    avg_cluster_size: float = 0.0
    cluster_balance: float = 0.0       # std / mean of cluster sizes
    coverage: float = 0.0              # fraction of alive nodes assigned
    avg_intra_cluster_rssi: float = 0.0  # mean RSSI within clusters

    # ── Network lifetime ─────────────────────────────────────────────
    first_node_death_sec: float = 0.0
    half_nodes_dead_sec: float = 0.0
    nodes_alive: int = 0
    nodes_dead: int = 0

    # ── Data delivery ────────────────────────────────────────────────
    packets_sent: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    packet_delivery_ratio: float = 0.0  # PDR = delivered / sent

    # ── Latency ──────────────────────────────────────────────────────
    avg_latency_sec: float = 0.0
    max_latency_sec: float = 0.0
    avg_hops: float = 0.0

    # ── Energy ───────────────────────────────────────────────────────
    total_energy_consumed_j: float = 0.0
    avg_energy_remaining_frac: float = 0.0
    energy_std: float = 0.0            # fairness indicator
    energy_efficiency: float = 0.0     # delivered packets / Joule


def compute_clustering_metrics(nodes: list[SensorNode],
                               clusters: dict[int, list[int]],
                               rssi_matrix: np.ndarray) -> dict:
    """Evaluate the quality of a clustering assignment."""
    n_alive = sum(1 for nd in nodes if nd.is_alive)
    if n_alive == 0 or not clusters:
        return {"num_clusters": 0, "avg_cluster_size": 0,
                "cluster_balance": 0, "coverage": 0,
                "avg_intra_cluster_rssi": -np.inf}

    sizes = [len(m) for m in clusters.values()]
    assigned = set()
    for members in clusters.values():
        assigned.update(members)

    # Intra-cluster RSSI
    intra_rssi = []
    for head, members in clusters.items():
        for m in members:
            if m != head and rssi_matrix[m, head] > -np.inf:
                intra_rssi.append(rssi_matrix[m, head])

    mean_size = np.mean(sizes) if sizes else 0
    std_size = np.std(sizes) if sizes else 0

    return {
        "num_clusters": len(clusters),
        "avg_cluster_size": float(mean_size),
        "cluster_balance": float(std_size / mean_size) if mean_size > 0 else 0,
        "coverage": len(assigned) / n_alive if n_alive > 0 else 0,
        "avg_intra_cluster_rssi": float(np.mean(intra_rssi)) if intra_rssi else -np.inf,
    }


def compute_energy_metrics(nodes: list[SensorNode]) -> dict:
    """Compute network-wide energy statistics."""
    fracs = [nd.energy_fraction for nd in nodes]
    consumed = [nd.energy_initial_j - nd.energy_remaining_j for nd in nodes]
    alive = sum(1 for nd in nodes if nd.is_alive)
    dead = len(nodes) - alive

    return {
        "nodes_alive": alive,
        "nodes_dead": dead,
        "total_energy_consumed_j": float(sum(consumed)),
        "avg_energy_remaining_frac": float(np.mean(fracs)),
        "energy_std": float(np.std(fracs)),
    }


def compute_delivery_metrics(packets: list[Packet]) -> dict:
    """Compute packet delivery ratio, latency, and hop statistics."""
    if not packets:
        return {"packets_sent": 0, "packets_delivered": 0,
                "packets_dropped": 0, "packet_delivery_ratio": 0,
                "avg_latency_sec": 0, "max_latency_sec": 0, "avg_hops": 0}

    delivered = [p for p in packets if p.delivered]
    dropped = [p for p in packets if p.dropped]
    n_sent = len(packets)
    n_delivered = len(delivered)
    pdr = n_delivered / n_sent if n_sent > 0 else 0

    latencies = [p.latency_sec for p in delivered] if delivered else [0]
    hops = [p.hops for p in delivered] if delivered else [0]

    return {
        "packets_sent": n_sent,
        "packets_delivered": n_delivered,
        "packets_dropped": len(dropped),
        "packet_delivery_ratio": float(pdr),
        "avg_latency_sec": float(np.mean(latencies)),
        "max_latency_sec": float(np.max(latencies)),
        "avg_hops": float(np.mean(hops)),
    }


def compute_all_metrics(nodes: list[SensorNode],
                        clusters: dict[int, list[int]],
                        rssi_matrix: np.ndarray,
                        packets: list[Packet]) -> NetworkMetrics:
    """Compute all metrics and return a single NetworkMetrics object."""
    cm = compute_clustering_metrics(nodes, clusters, rssi_matrix)
    em = compute_energy_metrics(nodes)
    dm = compute_delivery_metrics(packets)

    # Energy efficiency = packets delivered per Joule consumed
    total_e = em["total_energy_consumed_j"]
    efficiency = dm["packets_delivered"] / total_e if total_e > 0 else 0

    return NetworkMetrics(
        num_clusters=cm["num_clusters"],
        avg_cluster_size=cm["avg_cluster_size"],
        cluster_balance=cm["cluster_balance"],
        coverage=cm["coverage"],
        avg_intra_cluster_rssi=cm["avg_intra_cluster_rssi"],
        nodes_alive=em["nodes_alive"],
        nodes_dead=em["nodes_dead"],
        total_energy_consumed_j=em["total_energy_consumed_j"],
        avg_energy_remaining_frac=em["avg_energy_remaining_frac"],
        energy_std=em["energy_std"],
        energy_efficiency=float(efficiency),
        packets_sent=dm["packets_sent"],
        packets_delivered=dm["packets_delivered"],
        packets_dropped=dm["packets_dropped"],
        packet_delivery_ratio=dm["packet_delivery_ratio"],
        avg_latency_sec=dm["avg_latency_sec"],
        max_latency_sec=dm["max_latency_sec"],
        avg_hops=dm["avg_hops"],
    )


def print_metrics(m: NetworkMetrics, title: str = "Network Metrics"):
    """Pretty-print a NetworkMetrics object."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Clustering:")
    print(f"    Clusters          : {m.num_clusters}")
    print(f"    Avg cluster size  : {m.avg_cluster_size:.1f}")
    print(f"    Balance (σ/μ)     : {m.cluster_balance:.3f}")
    print(f"    Coverage          : {m.coverage:.1%}")
    print(f"    Avg intra RSSI    : {m.avg_intra_cluster_rssi:.1f} dBm")
    print(f"  Energy:")
    print(f"    Alive / Dead      : {m.nodes_alive} / {m.nodes_dead}")
    print(f"    Total consumed    : {m.total_energy_consumed_j:.2f} J")
    print(f"    Avg remaining     : {m.avg_energy_remaining_frac:.1%}")
    print(f"    Energy fairness σ : {m.energy_std:.4f}")
    print(f"    Efficiency        : {m.energy_efficiency:.1f} pkts/J")
    print(f"  Data Delivery:")
    print(f"    Sent / Delivered  : {m.packets_sent} / {m.packets_delivered}")
    print(f"    PDR               : {m.packet_delivery_ratio:.1%}")
    print(f"    Avg latency       : {m.avg_latency_sec * 1000:.2f} ms")
    print(f"    Max latency       : {m.max_latency_sec * 1000:.2f} ms")
    print(f"    Avg hops          : {m.avg_hops:.1f}")
    print(f"{'=' * 60}\n")
