"""
mesh_network.py — Mesh topology, multi-hop routing, and data aggregation.

Builds a network graph from the RSSI matrix and clusters, then routes
aggregated data from cluster heads to the gateway via multi-hop paths.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from . import config as cfg
from .node import SensorNode, GatewayNode
from .rssi_model import true_distance, adjacency_from_rssi


@dataclass
class Packet:
    """A data packet travelling through the mesh."""
    source_id: int
    destination_id: int          # -1 = gateway
    payload: dict                # bioimpedance data
    hops: int = 0
    path: list[int] = field(default_factory=list)
    delivered: bool = False
    dropped: bool = False
    latency_sec: float = 0.0


def build_topology(nodes: list[SensorNode],
                   rssi_matrix: np.ndarray,
                   gateway: GatewayNode,
                   threshold_dbm: float = cfg.RSSI_THRESHOLD_DBM
                   ) -> dict[int, list[int]]:
    """
    Build adjacency list for the mesh (including gateway as node id -1).

    Returns
    -------
    adj_list : dict mapping node_id -> list of neighbour node_ids
    """
    n = len(nodes)
    adj = adjacency_from_rssi(rssi_matrix, threshold_dbm)
    adj_list: dict[int, list[int]] = {}

    for i in range(n):
        if not nodes[i].is_alive:
            continue
        neighbours = [j for j in range(n) if adj[i, j] == 1 and nodes[j].is_alive]
        # Check if gateway is in range
        d_gw = true_distance(nodes[i].position, gateway.position)
        if d_gw <= cfg.BLE_MAX_RANGE:
            neighbours.append(-1)  # gateway
        adj_list[i] = neighbours

    # Gateway adjacency
    gw_neighbours = [i for i in range(n)
                     if nodes[i].is_alive and
                     true_distance(nodes[i].position, gateway.position) <= cfg.BLE_MAX_RANGE]
    adj_list[-1] = gw_neighbours

    return adj_list


def find_route_to_gateway(source_id: int,
                          adj_list: dict[int, list[int]],
                          max_hops: int = cfg.MAX_HOPS) -> list[int] | None:
    """
    BFS shortest path from source_id to gateway (-1).

    Returns the path as a list of node ids (including source and gateway),
    or None if no route exists within max_hops.
    """
    if -1 in adj_list.get(source_id, []):
        return [source_id, -1]

    visited = {source_id}
    queue = [(source_id, [source_id])]

    while queue:
        current, path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for neighbour in adj_list.get(current, []):
            if neighbour == -1:
                return path + [-1]
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    return None  # no route found


def route_data_to_gateway(cluster_head_id: int,
                          aggregated_data: dict,
                          nodes: list[SensorNode],
                          adj_list: dict[int, list[int]],
                          gateway: GatewayNode) -> Packet:
    """
    Route aggregated data from a cluster head to the gateway.

    Each hop consumes energy (TX at sender, RX at receiver).
    """
    pkt = Packet(source_id=cluster_head_id, destination_id=-1,
                 payload=aggregated_data, path=[cluster_head_id])

    route = find_route_to_gateway(cluster_head_id, adj_list)
    if route is None:
        pkt.dropped = True
        return pkt

    for step in range(len(route) - 1):
        current_id = route[step]
        next_id = route[step + 1]
        pkt.hops += 1
        pkt.latency_sec += cfg.TX_TIME_PER_PACKET + cfg.RX_TIME_PER_PACKET

        # Energy for transmission
        if current_id >= 0:
            nodes[current_id].transmit_packet()

        # Energy for reception
        if next_id >= 0:
            nodes[next_id].receive_packet()
        else:
            gateway.receive_data(aggregated_data)

        pkt.path.append(next_id)

    pkt.delivered = (route[-1] == -1)
    if not pkt.delivered:
        pkt.dropped = True

    return pkt


def collect_data_in_cluster(cluster_head_id: int,
                            member_ids: list[int],
                            nodes: list[SensorNode]) -> dict:
    """
    Cluster head collects bioimpedance measurements from all members.

    Returns aggregated data dict with readings and stats.
    """
    readings = {}
    for mid in member_ids:
        if nodes[mid].is_alive and nodes[mid].is_awake:
            val = nodes[mid].measure_bioimpedance()
            if val is not None:
                readings[mid] = val
                nodes[mid].transmit_packet()  # member sends to head
                if mid != cluster_head_id:
                    nodes[cluster_head_id].receive_packet()

    # Head's own measurement
    if cluster_head_id not in readings:
        val = nodes[cluster_head_id].measure_bioimpedance()
        if val is not None:
            readings[cluster_head_id] = val

    aggregated = {
        "cluster_head": cluster_head_id,
        "n_readings": len(readings),
        "readings": readings,
        "mean_impedance": float(np.mean(list(readings.values()))) if readings else 0.0,
        "std_impedance": float(np.std(list(readings.values()))) if readings else 0.0,
    }
    return aggregated
