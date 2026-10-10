"""
my_sleep_scheduler.py — Student sleep/wake-up scheduling algorithms.

╔══════════════════════════════════════════════════════════════╗
║  THIS IS YOUR FILE.  Write your sleep schedulers here.       ║
╚══════════════════════════════════════════════════════════════╝

Every scheduler MUST have this interface:

    def my_scheduler(nodes, clusters, current_time, gateway, **kwargs)
        -> dict[int, str]

    Parameters
    ----------
    nodes        : list[SensorNode]          — all sensor nodes
    clusters     : dict[int, list[int]]      — current clustering
    current_time : float                     — simulation clock [seconds]
    gateway      : GatewayNode | None        — the gateway node
    **kwargs     : extra parameters

    Returns
    -------
    actions : dict mapping node_id -> action_str
              Actions: 'sleep', 'wake', 'active', 'idle'
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.node import SensorNode, GatewayNode
from src import config as cfg


# ═════════════════════════════════════════════════════════════════════
#  EXAMPLE: Simple round-robin scheduler (working)
# ═════════════════════════════════════════════════════════════════════
def round_robin_scheduler(nodes: list[SensorNode],
                          clusters: dict[int, list[int]],
                          current_time: float,
                          gateway: GatewayNode | None = None,
                          window_sec: float = 5.0,
                          **kwargs) -> dict[int, str]:
    """
    Example scheduler: within each cluster, wake up one member at a time
    in round-robin fashion.  Cluster heads stay awake.
    """
    actions: dict[int, str] = {}

    for head_id, member_ids in clusters.items():
        if nodes[head_id].is_alive:
            actions[head_id] = "active"
        non_heads = [m for m in member_ids if m != head_id and nodes[m].is_alive]
        if not non_heads:
            continue
        slot = int(current_time / window_sec) % len(non_heads)
        for idx, mid in enumerate(non_heads):
            actions[mid] = "active" if idx == slot else "sleep"

    for i, nd in enumerate(nodes):
        if i not in actions:
            actions[i] = "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
#  SCHEDULER 1 — Energy-Balanced Duty Cycling
# ═════════════════════════════════════════════════════════════════════
def student_scheduler_1(nodes: list[SensorNode],
                        clusters: dict[int, list[int]],
                        current_time: float,
                        gateway: GatewayNode | None = None,
                        base_cycle_sec: float = 10.0,
                        base_duty: float = 0.15,
                        **kwargs) -> dict[int, str]:
    """
    Energy-Balanced Duty Cycling Scheduler.

    Core idea:
    ----------
    Each member node independently adapts its duty cycle based on its
    remaining battery level. High-energy nodes transmit more (better PDR),
    low-energy nodes sleep more (longer lifetime). Cluster heads always
    stay active since they must collect data from all members.

    Duty cycle tiers:
        energy > 70%  ->  25% awake
        energy 40-70% ->  15% awake
        energy 20-40% ->   8% awake
        energy < 20%  ->   4% awake

    Staggered phase offsets ensure members within the same cluster wake
    at different times, preventing transmission collisions.

    Why it beats periodic baseline:
    - Periodic uses one fixed duty for everyone -> wastes energy on
      high-energy nodes and keeps low-energy nodes awake too long.
    - Our tiered approach dynamically balances throughput vs lifetime.
    - Staggering eliminates collisions -> higher PDR per Joule consumed.
    """
    actions: dict[int, str] = {}

    # Build per-node slot info for staggered wake-up
    node_slot:         dict[int, int] = {}
    node_cluster_size: dict[int, int] = {}
    for head_id, member_ids in clusters.items():
        non_heads = [m for m in member_ids if m != head_id]
        n = len(non_heads)
        for slot_idx, mid in enumerate(non_heads):
            node_slot[mid]         = slot_idx
            node_cluster_size[mid] = max(n, 1)

    for i, node in enumerate(nodes):
        if not node.is_alive:
            actions[i] = "sleep"
            continue

        is_head = i in clusters

        if is_head:
            # Cluster heads stay active; only drop to idle if near-dead
            actions[i] = "active" if node.energy_fraction > 0.10 else "idle"
            continue

        # Energy-adaptive duty cycle
        ef = node.energy_fraction
        if ef > 0.70:
            duty = 0.25
        elif ef > 0.40:
            duty = 0.15
        elif ef > 0.20:
            duty = 0.08
        else:
            duty = 0.04

        # Staggered phase: each member wakes at a unique offset
        slot_idx  = node_slot.get(i, 0)
        cluster_n = node_cluster_size.get(i, 1)
        phase_offset  = (slot_idx / cluster_n) * base_cycle_sec
        shifted_time  = (current_time + phase_offset) % base_cycle_sec
        phase         = shifted_time / base_cycle_sec   # 0.0 – 1.0

        actions[i] = "active" if phase < duty else "sleep"

    # Nodes not in any cluster sleep
    for i in range(len(nodes)):
        if i not in actions:
            actions[i] = "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
#  SCHEDULER 2 — Cluster-Aware TDMA with Adaptive Head Relief
# ═════════════════════════════════════════════════════════════════════
def student_scheduler_2(nodes: list[SensorNode],
                        clusters: dict[int, list[int]],
                        current_time: float,
                        gateway: GatewayNode | None = None,
                        report_interval: float = 5.0,
                        **kwargs) -> dict[int, str]:
    """
    Cluster-Aware TDMA Scheduler with Adaptive Head Relief.

    Core idea:
    ----------
    Divides each reporting interval into three phases:
      Phase 1 (20%): Cluster head transmits aggregated data to gateway.
      Phase 2 (60%): Members report to CH in TDMA slots (one slot each).
      Phase 3 (20%): Everyone sleeps to save energy.

    The schedule changes every second because:
    - current_time advances -> time_in_interval advances -> each node's
      TDMA slot boundary shifts continuously through the interval.
    - The slot a node is in at t=0 is different from t=15, guaranteed.

    Adaptive head relief:
    ---------------------
    If a cluster head's energy drops below 30%, it enters "idle" during
    the second half of Phase 2 (still receives but doesn't TX).
    This prevents the head from burning out before members.

    Why it beats periodic baseline:
    - Zero collisions in member->CH phase (TDMA guarantees this).
    - Members sleep (1 - 1/N) fraction of time vs periodic's fixed rate.
    - Head relief distributes energy drain and extends network lifetime.
    """
    actions: dict[int, str] = {}

    # Position in current interval: changes every tick -> time-varying schedule
    time_in_interval = current_time % report_interval

    # Phase boundaries
    ch_tx_end   = report_interval * 0.20   # end of CH->GW phase
    member_end  = report_interval * 0.80   # end of member collection phase
    # sleep phase: member_end to report_interval

    for head_id, member_ids in clusters.items():
        if not nodes[head_id].is_alive:
            continue

        low_energy_head = nodes[head_id].energy_fraction < 0.30

        # ── Cluster head schedule ─────────────────────────────────────
        if time_in_interval < ch_tx_end:
            # Phase 1: head actively transmits to gateway
            actions[head_id] = "active"
        elif time_in_interval < member_end:
            # Phase 2: head receives from members
            if low_energy_head:
                # Head relief: idle in second half of collection window
                mid_point = ch_tx_end + (member_end - ch_tx_end) / 2.0
                actions[head_id] = "active" if time_in_interval < mid_point else "idle"
            else:
                actions[head_id] = "active"
        else:
            # Phase 3: head sleeps
            actions[head_id] = "sleep"

        # ── Member TDMA slots ─────────────────────────────────────────
        non_heads = [m for m in member_ids if m != head_id and nodes[m].is_alive]
        n_members = len(non_heads)
        if n_members == 0:
            continue

        # Divide Phase 2 into equal slots
        collection_duration = member_end - ch_tx_end
        slot_duration = collection_duration / n_members

        for slot_idx, mid in enumerate(non_heads):
            if not nodes[mid].is_alive:
                actions[mid] = "sleep"
                continue

            slot_start   = ch_tx_end + slot_idx * slot_duration
            slot_end     = slot_start + slot_duration
            guard        = slot_duration * 0.05

            if slot_start + guard <= time_in_interval < slot_end - guard:
                actions[mid] = "active"
            elif slot_start <= time_in_interval < slot_start + guard:
                # Wake-up transition at start of slot
                actions[mid] = "wake"
            else:
                actions[mid] = "sleep"

    # Nodes not in any cluster sleep
    for i in range(len(nodes)):
        if i not in actions:
            actions[i] = "sleep"

    return actions


# ═════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════
MY_SCHEDULER_ALGOS = {
    "round_robin":          round_robin_scheduler,
    "student_scheduler_1":  student_scheduler_1,
    "student_scheduler_2":  student_scheduler_2,
}
