"""
energy_model.py — Energy accounting for the BLE mesh network.

Tracks per-node and network-wide energy consumption.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from . import config as cfg
from .node import SensorNode, NodeState


@dataclass
class EnergyLog:
    """Time-series energy log for one node."""
    node_id: int
    timestamps: list[float] = field(default_factory=list)
    energy_levels: list[float] = field(default_factory=list)
    states: list[str] = field(default_factory=list)

    def record(self, t: float, node: SensorNode):
        self.timestamps.append(t)
        self.energy_levels.append(node.energy_remaining_j)
        self.states.append(node.state.value)


def energy_for_sensing(duration_s: float = cfg.SENSE_TIME) -> float:
    """Energy to take one bioimpedance measurement [J]."""
    return cfg.SENSE_POWER_W * duration_s


def energy_for_tx(n_packets: int = 1) -> float:
    """Energy to transmit n packets [J]."""
    return cfg.TX_POWER_W * cfg.TX_TIME_PER_PACKET * n_packets


def energy_for_rx(n_packets: int = 1) -> float:
    """Energy to receive n packets [J]."""
    return cfg.RX_POWER_W * cfg.RX_TIME_PER_PACKET * n_packets


def energy_for_idle(duration_s: float) -> float:
    """Energy for idle listening [J]."""
    return cfg.IDLE_POWER_W * duration_s


def energy_for_sleep(duration_s: float) -> float:
    """Energy for deep sleep [J]."""
    return cfg.SLEEP_POWER_W * duration_s


def cluster_head_energy_per_round(n_members: int,
                                  aggregation_factor: float = 0.5) -> float:
    """
    Estimated energy a cluster head spends per data-collection round.

    - Receives from all members
    - Aggregates
    - Transmits aggregated packet to gateway / next hop
    """
    rx_cost = energy_for_rx(n_members)
    tx_cost = energy_for_tx(max(1, int(n_members * aggregation_factor)))
    sense_cost = energy_for_sensing()  # own measurement
    return rx_cost + tx_cost + sense_cost


def member_energy_per_round() -> float:
    """Energy a regular cluster member spends per data-collection round."""
    return energy_for_sensing() + energy_for_tx(1)


def estimate_network_lifetime(nodes: list[SensorNode],
                              clusters: dict[int, list[int]],
                              report_interval: float = cfg.DATA_REPORT_INTERVAL,
                              sleep_fraction: float = 1.0 - cfg.DUTY_CYCLE
                              ) -> float:
    """
    Estimate network lifetime (seconds until first node dies).

    Simplified steady-state model.
    """
    min_lifetime = np.inf
    for i, node in enumerate(nodes):
        if not node.is_alive:
            continue
        # Determine if this node is a cluster head
        is_head = i in clusters
        n_members = len(clusters[i]) - 1 if is_head else 0

        # Energy per round
        if is_head:
            e_round = cluster_head_energy_per_round(n_members)
        else:
            e_round = member_energy_per_round()

        # Rounds per second
        rounds_per_sec = 1.0 / report_interval

        # Idle/sleep power between rounds
        active_time = cfg.SENSE_TIME + cfg.TX_TIME_PER_PACKET
        sleep_time = report_interval - active_time
        e_between = (energy_for_sleep(sleep_time * sleep_fraction) +
                     energy_for_idle(sleep_time * (1 - sleep_fraction)))

        e_per_interval = e_round + e_between
        e_per_sec = e_per_interval / report_interval

        if e_per_sec > 0:
            lifetime = node.energy_remaining_j / e_per_sec
            min_lifetime = min(min_lifetime, lifetime)

    return min_lifetime if min_lifetime < np.inf else 0.0
