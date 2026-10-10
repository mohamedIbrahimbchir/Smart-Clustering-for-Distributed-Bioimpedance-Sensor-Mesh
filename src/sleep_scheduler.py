"""
sleep_scheduler.py — Sleep / wake-up scheduling strategies for the BLE mesh.

Every scheduler has the same interface:

    def scheduler(nodes, clusters, current_time, gateway, **kwargs)
        -> dict[int, str]      # node_id -> 'sleep' | 'wake' | 'active' | 'idle'

Students will implement their own schedulers.
"""

from __future__ import annotations
import numpy as np
from typing import Callable
from . import config as cfg
from .node import SensorNode, GatewayNode, NodeState

# Type alias
SchedulerFn = Callable[..., dict[int, str]]


# ═════════════════════════════════════════════════════════════════════
# 1.  Periodic Duty-Cycle
# ═════════════════════════════════════════════════════════════════════
def periodic_scheduler(nodes: list[SensorNode],
                       clusters: dict[int, list[int]],
                       current_time: float,
                       gateway: GatewayNode | None = None,
                       cycle_sec: float = cfg.SLEEP_CYCLE_SEC,
                       duty_cycle: float = cfg.DUTY_CYCLE,
                       **kwargs) -> dict[int, str]:
    """
    Simple periodic sleep/wake cycle.

    All nodes follow the same fixed schedule:
      - Awake for (duty_cycle * cycle_sec) seconds
      - Sleep for the rest
    Cluster heads stay awake longer (2x duty cycle).
    """
    actions: dict[int, str] = {}
    phase = (current_time % cycle_sec) / cycle_sec  # 0.0 to 1.0

    for i, node in enumerate(nodes):
        if not node.is_alive:
            actions[i] = "sleep"
            continue

        is_head = i in clusters
        effective_duty = min(duty_cycle * 2, 0.5) if is_head else duty_cycle

        if phase < effective_duty:
            actions[i] = "active" if is_head else "idle"
        else:
            actions[i] = "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
# 2.  Synchronised TDMA-like
# ═════════════════════════════════════════════════════════════════════
def synchronised_scheduler(nodes: list[SensorNode],
                           clusters: dict[int, list[int]],
                           current_time: float,
                           gateway: GatewayNode | None = None,
                           report_interval: float = cfg.DATA_REPORT_INTERVAL,
                           **kwargs) -> dict[int, str]:
    """
    Synchronised TDMA-style scheduler.

    Within each cluster, members wake up in assigned time slots.
    Cluster heads stay awake during the entire collection window.
    """
    actions: dict[int, str] = {}
    time_in_interval = current_time % report_interval

    for head_id, member_ids in clusters.items():
        n_members = len(member_ids)
        if n_members == 0:
            continue

        # Collection window: first 30% of interval
        collection_window = report_interval * 0.3
        slot_duration = collection_window / max(n_members, 1)

        # Cluster head awake during collection + transmission
        if time_in_interval < collection_window + 2.0:
            actions[head_id] = "active"
        else:
            actions[head_id] = "sleep"

        for slot_idx, mid in enumerate(member_ids):
            if mid == head_id:
                continue
            slot_start = slot_idx * slot_duration
            slot_end = slot_start + slot_duration

            if slot_start <= time_in_interval < slot_end:
                actions[mid] = "active"
            else:
                actions[mid] = "sleep"

    # Handle unassigned nodes
    for i, node in enumerate(nodes):
        if i not in actions:
            actions[i] = "sleep" if node.is_alive else "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
# 3.  On-Demand (Gateway Wake-Up)
# ═════════════════════════════════════════════════════════════════════
def on_demand_scheduler(nodes: list[SensorNode],
                        clusters: dict[int, list[int]],
                        current_time: float,
                        gateway: GatewayNode | None = None,
                        report_interval: float = cfg.DATA_REPORT_INTERVAL,
                        wakeup_duration: float = 5.0,
                        **kwargs) -> dict[int, str]:
    """
    On-demand wake-up scheduler.

    All nodes sleep by default.  At each report_interval, the gateway
    sends a wake-up signal.  Nodes wake up, send data, then go back to sleep.
    """
    actions: dict[int, str] = {}
    time_in_interval = current_time % report_interval

    # Wake-up window at the start of each interval
    is_wakeup_window = time_in_interval < wakeup_duration

    for i, node in enumerate(nodes):
        if not node.is_alive:
            actions[i] = "sleep"
        elif is_wakeup_window:
            actions[i] = "active" if i in clusters else "wake"
        else:
            actions[i] = "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
# 4.  Adaptive Energy-Aware
# ═════════════════════════════════════════════════════════════════════
def adaptive_scheduler(nodes: list[SensorNode],
                       clusters: dict[int, list[int]],
                       current_time: float,
                       gateway: GatewayNode | None = None,
                       cycle_sec: float = cfg.SLEEP_CYCLE_SEC,
                       energy_threshold_low: float = 0.2,
                       energy_threshold_crit: float = 0.05,
                       **kwargs) -> dict[int, str]:
    """
    Adaptive scheduler that adjusts duty cycle based on remaining energy.

    - High energy (> 50%): normal duty cycle
    - Medium energy (20-50%): reduced duty cycle
    - Low energy (5-20%): minimal duty cycle
    - Critical (< 5%): sleep-only mode (only wakes for urgent data)
    """
    actions: dict[int, str] = {}
    phase = (current_time % cycle_sec) / cycle_sec

    for i, node in enumerate(nodes):
        if not node.is_alive:
            actions[i] = "sleep"
            continue

        ef = node.energy_fraction
        is_head = i in clusters

        # Determine effective duty cycle
        if ef < energy_threshold_crit:
            duty = 0.02  # 2% — survival mode
        elif ef < energy_threshold_low:
            duty = 0.05  # 5% — low power
        elif ef < 0.5:
            duty = cfg.DUTY_CYCLE * 0.7  # 70% of normal
        else:
            duty = cfg.DUTY_CYCLE  # full duty

        # Cluster heads get extra awake time
        if is_head:
            duty = min(duty * 2.5, 0.6)

        if phase < duty:
            actions[i] = "active" if is_head else "idle"
        else:
            actions[i] = "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════
SCHEDULER_ALGOS: dict[str, SchedulerFn] = {
    "periodic":      periodic_scheduler,
    "synchronised":  synchronised_scheduler,
    "on_demand":     on_demand_scheduler,
    "adaptive":      adaptive_scheduler,
}


def get_scheduler(name: str) -> SchedulerFn:
    """Look up a sleep scheduler by name."""
    if name not in SCHEDULER_ALGOS:
        raise ValueError(f"Unknown scheduler '{name}'. "
                         f"Available: {list(SCHEDULER_ALGOS.keys())}")
    return SCHEDULER_ALGOS[name]
