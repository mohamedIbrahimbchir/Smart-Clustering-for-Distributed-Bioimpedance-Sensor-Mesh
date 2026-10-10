"""
simulator.py — Main simulation orchestrator and visualisation engine.

Ties together: node deployment, RSSI, clustering, mesh routing,
sleep scheduling, energy tracking, and metrics computation.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from dataclasses import dataclass, field

from . import config as cfg
from .node import SensorNode, GatewayNode, NodeState, deploy_nodes
from .rssi_model import (build_rssi_matrix, build_distance_matrix,
                         adjacency_from_rssi, true_distance)
from .clustering import CLUSTERING_ALGOS, ClusterFn
from .sleep_scheduler import SCHEDULER_ALGOS, SchedulerFn
from .mesh_network import (build_topology, find_route_to_gateway,
                           route_data_to_gateway, collect_data_in_cluster,
                           Packet)
from .energy_model import EnergyLog, estimate_network_lifetime
from .metrics import (compute_all_metrics, compute_clustering_metrics,
                      compute_energy_metrics, compute_delivery_metrics,
                      NetworkMetrics, print_metrics)


@dataclass
class SimulationResult:
    """Container for everything produced by a simulation run."""
    nodes: list[SensorNode]
    gateway: GatewayNode
    rssi_matrix: np.ndarray
    distance_matrix: np.ndarray
    clusters: dict[int, list[int]]
    packets: list[Packet]
    energy_logs: dict[int, EnergyLog]
    metrics: NetworkMetrics
    time_series: dict[str, list]      # time-indexed telemetry
    clustering_name: str = ""
    scheduler_name: str = ""


class Simulator:
    """
    Main simulation engine.

    Usage
    -----
    sim = Simulator()
    result = sim.run(clustering_algo='leach', scheduler='adaptive', duration=3600)
    sim.plot_topology(result)
    sim.plot_energy(result)
    """

    def __init__(self,
                 num_nodes: int = cfg.NUM_NODES,
                 area: tuple[float, float] = (cfg.AREA_WIDTH, cfg.AREA_HEIGHT),
                 gateway_pos: tuple[float, float] = cfg.GATEWAY_POS,
                 deployment: str = "random",
                 seed: int = cfg.RANDOM_SEED):
        self.num_nodes = num_nodes
        self.area = area
        self.gateway_pos = gateway_pos
        self.deployment = deployment
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self,
            clustering_algo: str | ClusterFn = "rssi_threshold",
            scheduler: str | SchedulerFn = "periodic",
            duration: float = cfg.SIM_DURATION_SEC,
            dt: float = cfg.TIME_STEP_SEC,
            report_interval: float = cfg.DATA_REPORT_INTERVAL,
            recluster_interval: float = cfg.CLUSTER_ROUND_INTERVAL,
            **kwargs) -> SimulationResult:
        """
        Run a full simulation.

        Parameters
        ----------
        clustering_algo : name (str) or callable
        scheduler       : name (str) or callable
        duration        : total simulation time [s]
        dt              : time step [s]
        report_interval : data reporting interval [s]
        recluster_interval : how often to re-cluster [s]
        **kwargs        : forwarded to clustering/scheduler functions
        """
        # ── Resolve algorithms ───────────────────────────────────────
        if isinstance(clustering_algo, str):
            clust_name = clustering_algo
            clust_fn = CLUSTERING_ALGOS[clustering_algo]
        else:
            clust_name = getattr(clustering_algo, "__name__", "custom")
            clust_fn = clustering_algo

        if isinstance(scheduler, str):
            sched_name = scheduler
            sched_fn = SCHEDULER_ALGOS[scheduler]
        else:
            sched_name = getattr(scheduler, "__name__", "custom")
            sched_fn = scheduler

        # ── Deploy nodes ─────────────────────────────────────────────
        nodes = deploy_nodes(self.num_nodes,
                             self.area[0], self.area[1],
                             self.rng, self.deployment)
        gateway = GatewayNode(position=self.gateway_pos)

        # ── Build RSSI & distance matrices ───────────────────────────
        positions = [nd.position for nd in nodes]
        rssi_matrix = build_rssi_matrix(positions, rng=self.rng)
        dist_matrix = build_distance_matrix(positions)

        # ── Initial clustering ───────────────────────────────────────
        clusters = clust_fn(nodes, rssi_matrix, self.gateway_pos, **kwargs)
        _apply_cluster_assignments(nodes, clusters)

        # ── Build mesh topology ──────────────────────────────────────
        adj_list = build_topology(nodes, rssi_matrix, gateway)

        # ── Energy logs ──────────────────────────────────────────────
        energy_logs: dict[int, EnergyLog] = {}
        for nd in nodes:
            energy_logs[nd.node_id] = EnergyLog(node_id=nd.node_id)

        # ── Time-series storage ──────────────────────────────────────
        ts = {"time": [], "alive_count": [], "packets_delivered": [],
              "total_energy_consumed": [], "pdr_cumulative": []}

        # ── Simulation loop ──────────────────────────────────────────
        all_packets: list[Packet] = []
        total_sent = 0
        total_delivered = 0
        round_num = 0
        next_report = report_interval
        next_recluster = recluster_interval

        t = 0.0
        while t < duration:
            # ── Re-cluster periodically ──────────────────────────────
            if t >= next_recluster:
                round_num += 1
                clusters = clust_fn(nodes, rssi_matrix, self.gateway_pos,
                                    round_num=round_num, **kwargs)
                _apply_cluster_assignments(nodes, clusters)
                adj_list = build_topology(nodes, rssi_matrix, gateway)
                next_recluster += recluster_interval

            # ── Sleep scheduling ─────────────────────────────────────
            actions = sched_fn(nodes, clusters, t, gateway, **kwargs)
            for nid, action in actions.items():
                if nid < 0 or nid >= len(nodes):
                    continue
                nd = nodes[nid]
                if action == "sleep":
                    nd.go_to_sleep()
                elif action == "wake":
                    nd.wake_up()
                elif action == "active":
                    nd.set_active()
                elif action == "idle":
                    nd.set_idle()

            # ── Data collection & routing at report intervals ────────
            if t >= next_report:
                for head_id, member_ids in clusters.items():
                    if not nodes[head_id].is_alive or not nodes[head_id].is_awake:
                        continue
                    # Collect data within cluster
                    data = collect_data_in_cluster(head_id, member_ids, nodes)
                    # Route to gateway
                    pkt = route_data_to_gateway(head_id, data, nodes,
                                                adj_list, gateway)
                    all_packets.append(pkt)
                    total_sent += 1
                    if pkt.delivered:
                        total_delivered += 1

                next_report += report_interval

            # ── Tick all nodes ───────────────────────────────────────
            for nd in nodes:
                nd.tick(dt)

            # ── Record telemetry ─────────────────────────────────────
            if int(t) % 10 == 0:  # every 10 seconds
                for nd in nodes:
                    energy_logs[nd.node_id].record(t, nd)
                ts["time"].append(t)
                ts["alive_count"].append(sum(1 for nd in nodes if nd.is_alive))
                ts["packets_delivered"].append(total_delivered)
                consumed = sum(nd.energy_initial_j - nd.energy_remaining_j
                               for nd in nodes)
                ts["total_energy_consumed"].append(consumed)
                ts["pdr_cumulative"].append(
                    total_delivered / total_sent if total_sent > 0 else 1.0)

            t += dt

        # ── Compute final metrics ────────────────────────────────────
        final_metrics = compute_all_metrics(nodes, clusters,
                                            rssi_matrix, all_packets)

        return SimulationResult(
            nodes=nodes, gateway=gateway,
            rssi_matrix=rssi_matrix, distance_matrix=dist_matrix,
            clusters=clusters, packets=all_packets,
            energy_logs=energy_logs, metrics=final_metrics,
            time_series=ts,
            clustering_name=clust_name, scheduler_name=sched_name,
        )

    # ═════════════════════════════════════════════════════════════════
    #  Visualisation
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def plot_topology(result: SimulationResult, title: str | None = None,
                      ax: plt.Axes | None = None):
        """Plot the mesh network topology with clusters colour-coded."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=cfg.FIGURE_DPI)
        else:
            fig = ax.figure

        nodes = result.nodes
        gw = result.gateway
        clusters = result.clusters
        rssi = result.rssi_matrix

        cmap = plt.get_cmap(cfg.CMAP_CLUSTER)
        cluster_list = list(clusters.keys())

        # Draw edges (connections above threshold)
        adj = adjacency_from_rssi(rssi, cfg.RSSI_THRESHOLD_DBM)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if adj[i, j] == 1:
                    ax.plot([nodes[i].position[0], nodes[j].position[0]],
                            [nodes[i].position[1], nodes[j].position[1]],
                            'k-', alpha=0.08, linewidth=0.5)

        # Draw nodes
        for ci, (head_id, members) in enumerate(clusters.items()):
            color = cmap(ci % 10)
            for mid in members:
                nd = nodes[mid]
                if not nd.is_alive:
                    ax.scatter(*nd.position, c='gray', s=30, marker='x',
                               alpha=0.5, zorder=3)
                elif mid == head_id:
                    ax.scatter(*nd.position, c=[color], s=160, marker='s',
                               edgecolors='black', linewidths=1.5, zorder=5,
                               label=f'CH {mid}' if ci < 8 else None)
                else:
                    ax.scatter(*nd.position, c=[color], s=cfg.NODE_SIZE,
                               marker='o', edgecolors='gray', linewidths=0.5,
                               zorder=4)

            # Lines from members to cluster head
            for mid in members:
                if mid != head_id and nodes[mid].is_alive:
                    ax.plot([nodes[mid].position[0], nodes[head_id].position[0]],
                            [nodes[mid].position[1], nodes[head_id].position[1]],
                            color=color, linestyle='--', alpha=0.4, linewidth=1)

        # Draw gateway
        ax.scatter(*gw.position, c='red', s=300, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=6)
        ax.annotate('Gateway', gw.position, textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight='bold', color='red')

        # Routes from cluster heads to gateway
        adj_list = build_topology(nodes, rssi, gw)
        for head_id in clusters:
            route = find_route_to_gateway(head_id, adj_list)
            if route and len(route) > 1:
                for step in range(len(route) - 1):
                    src = route[step]
                    dst = route[step + 1]
                    sx, sy = nodes[src].position if src >= 0 else gw.position
                    dx, dy = nodes[dst].position if dst >= 0 else gw.position
                    ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                                arrowprops=dict(arrowstyle='->', color='red',
                                                lw=1.5, alpha=0.6))

        ax.set_xlim(-2, cfg.AREA_WIDTH + 2)
        ax.set_ylim(-2, cfg.AREA_HEIGHT + 2)
        ax.set_xlabel('X (metres)')
        ax.set_ylabel('Y (metres)')
        ax.set_title(title or f'Network Topology — {result.clustering_name}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='□ Cluster Head'),
            mpatches.Patch(facecolor='white', edgecolor='gray', label='○ Member'),
            plt.Line2D([0], [0], marker='*', color='red', label='★ Gateway',
                       markersize=15, linestyle='None'),
            plt.Line2D([0], [0], marker='x', color='gray', label='✕ Dead',
                       markersize=8, linestyle='None'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_energy(result: SimulationResult, ax: plt.Axes | None = None):
        """Plot energy remaining per node over time."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=cfg.FIGURE_DPI)
        else:
            fig = ax.figure

        cmap = plt.get_cmap('viridis')
        for nid, log in result.energy_logs.items():
            if log.timestamps:
                frac = np.array(log.energy_levels) / result.nodes[nid].energy_initial_j
                color = cmap(nid / max(len(result.nodes) - 1, 1))
                label = f'N{nid}' if nid < 5 else None
                is_head = nid in result.clusters
                lw = 2.0 if is_head else 0.7
                ax.plot(np.array(log.timestamps) / 60, frac * 100,
                        color=color, linewidth=lw, alpha=0.7, label=label)

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Energy Remaining (%)')
        ax.set_title(f'Energy Depletion — {result.clustering_name} + {result.scheduler_name}')
        ax.set_ylim(-1, 101)
        ax.grid(True, alpha=0.3)
        if len(result.nodes) <= 10:
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_alive_nodes(result: SimulationResult, ax: plt.Axes | None = None):
        """Plot alive node count over time."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=cfg.FIGURE_DPI)
        else:
            fig = ax.figure

        ts = result.time_series
        if ts["time"]:
            t_min = np.array(ts["time"]) / 60
            ax.plot(t_min, ts["alive_count"], 'b-', linewidth=2)
            ax.axhline(y=len(result.nodes), color='green', linestyle='--',
                       alpha=0.5, label='Total nodes')
            ax.axhline(y=len(result.nodes) * 0.5, color='red', linestyle='--',
                       alpha=0.5, label='50% threshold')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Alive Nodes')
        ax.set_title('Network Lifetime')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_pdr(result: SimulationResult, ax: plt.Axes | None = None):
        """Plot cumulative packet delivery ratio over time."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=cfg.FIGURE_DPI)
        else:
            fig = ax.figure

        ts = result.time_series
        if ts["time"]:
            t_min = np.array(ts["time"]) / 60
            ax.plot(t_min, np.array(ts["pdr_cumulative"]) * 100,
                    'g-', linewidth=2)
            ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5,
                       label='90% target')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('PDR (%)')
        ax.set_title('Packet Delivery Ratio')
        ax.set_ylim(-1, 101)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_cluster_sizes(result: SimulationResult, ax: plt.Axes | None = None):
        """Bar chart of cluster sizes."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=cfg.FIGURE_DPI)
        else:
            fig = ax.figure

        clusters = result.clusters
        heads = list(clusters.keys())
        sizes = [len(m) for m in clusters.values()]
        colors = plt.get_cmap(cfg.CMAP_CLUSTER)(np.linspace(0, 1, len(heads)))

        bars = ax.bar([f'CH {h}' for h in heads], sizes, color=colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Cluster Head')
        ax.set_ylabel('Members')
        ax.set_title(f'Cluster Sizes — {result.clustering_name}')
        ax.grid(True, alpha=0.2, axis='y')

        for bar, s in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(s), ha='center', va='bottom', fontsize=9, fontweight='bold')

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_dashboard(result: SimulationResult):
        """Full 2×3 dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=cfg.FIGURE_DPI)

        Simulator.plot_topology(result, ax=axes[0, 0])
        Simulator.plot_cluster_sizes(result, ax=axes[0, 1])

        # RSSI heatmap
        ax = axes[0, 2]
        rssi_plot = result.rssi_matrix.copy()
        rssi_plot[rssi_plot == -np.inf] = np.nan
        im = ax.imshow(rssi_plot, cmap='RdYlGn', aspect='auto')
        ax.set_title('RSSI Matrix (dBm)')
        ax.set_xlabel('Node')
        ax.set_ylabel('Node')
        fig.colorbar(im, ax=ax, shrink=0.8)

        Simulator.plot_energy(result, ax=axes[1, 0])
        Simulator.plot_alive_nodes(result, ax=axes[1, 1])
        Simulator.plot_pdr(result, ax=axes[1, 2])

        fig.suptitle(f'Dashboard — {result.clustering_name} + {result.scheduler_name}',
                     fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout()
        return fig

    @staticmethod
    def compare_clustering(clustering_algos: list[str] | None = None,
                           scheduler: str = "periodic",
                           num_nodes: int = cfg.NUM_NODES,
                           duration: float = cfg.SIM_DURATION_SEC,
                           seed: int = cfg.RANDOM_SEED,
                           **kwargs) -> dict[str, SimulationResult]:
        """
        Run the same scenario with different clustering algorithms and compare.
        """
        if clustering_algos is None:
            clustering_algos = list(CLUSTERING_ALGOS.keys())

        results: dict[str, SimulationResult] = {}
        for name in clustering_algos:
            sim = Simulator(num_nodes=num_nodes, seed=seed)
            result = sim.run(clustering_algo=name, scheduler=scheduler,
                             duration=duration, **kwargs)
            results[name] = result
            print_metrics(result.metrics, title=f"{name} + {scheduler}")

        # Comparison bar chart
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=cfg.FIGURE_DPI)
        names = list(results.keys())
        colors = plt.get_cmap('Set2')(np.linspace(0, 1, len(names)))

        # PDR
        pdrs = [results[n].metrics.packet_delivery_ratio * 100 for n in names]
        axes[0].bar(names, pdrs, color=colors, edgecolor='black')
        axes[0].set_ylabel('PDR (%)')
        axes[0].set_title('Packet Delivery Ratio')
        axes[0].set_ylim(0, 105)

        # Energy efficiency
        effs = [results[n].metrics.energy_efficiency for n in names]
        axes[1].bar(names, effs, color=colors, edgecolor='black')
        axes[1].set_ylabel('Packets / Joule')
        axes[1].set_title('Energy Efficiency')

        # Cluster balance
        bals = [results[n].metrics.cluster_balance for n in names]
        axes[2].bar(names, bals, color=colors, edgecolor='black')
        axes[2].set_ylabel('σ/μ (lower=better)')
        axes[2].set_title('Cluster Balance')

        # Alive nodes at end
        alives = [results[n].metrics.nodes_alive for n in names]
        axes[3].bar(names, alives, color=colors, edgecolor='black')
        axes[3].set_ylabel('Nodes')
        axes[3].set_title('Alive at End')
        axes[3].axhline(y=num_nodes, color='gray', linestyle='--', alpha=0.5)

        for ax in axes:
            ax.grid(True, alpha=0.2, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        fig.suptitle('Clustering Algorithm Comparison', fontsize=13, fontweight='bold')
        fig.tight_layout()
        plt.show()
        return results


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _apply_cluster_assignments(nodes: list[SensorNode],
                               clusters: dict[int, list[int]]):
    """Update node.cluster_id and node.is_cluster_head from cluster dict."""
    # Reset
    for nd in nodes:
        nd.cluster_id = -1
        nd.is_cluster_head = False

    for head_id, members in clusters.items():
        if head_id < len(nodes):
            nodes[head_id].is_cluster_head = True
            nodes[head_id].cluster_id = head_id
        for mid in members:
            if mid < len(nodes):
                nodes[mid].cluster_id = head_id
