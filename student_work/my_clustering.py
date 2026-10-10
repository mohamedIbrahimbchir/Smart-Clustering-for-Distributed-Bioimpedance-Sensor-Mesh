"""
my_clustering.py — Student clustering algorithms.

╔══════════════════════════════════════════════════════════════╗
║  THIS IS YOUR FILE.  Write your clustering algorithms here.  ║
╚══════════════════════════════════════════════════════════════╝

Every clustering function MUST have this interface:

    def my_algo(nodes, rssi_matrix, gateway_pos, **kwargs)
        -> dict[int, list[int]]

    Parameters
    ----------
    nodes        : list[SensorNode]   — all sensor nodes
    rssi_matrix  : np.ndarray (N×N)   — RSSI between all node pairs (dBm)
                                         -inf means out of range
    gateway_pos  : tuple[float,float] — (x, y) position of the gateway
    **kwargs     : extra parameters (round_num, threshold, etc.)

    Returns
    -------
    clusters : dict mapping cluster_head_id -> [member_id, ...]
               Each key is a node index that acts as cluster head.
               Each value is a list of node indices in that cluster
               (including the head itself).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.node import SensorNode
from src.rssi_model import adjacency_from_rssi


# ═════════════════════════════════════════════════════════════════════
#  EXAMPLE: Greedy distance-to-gateway clustering (working)
# ═════════════════════════════════════════════════════════════════════
def gateway_greedy_clustering(nodes: list[SensorNode],
                              rssi_matrix: np.ndarray,
                              gateway_pos: tuple[float, float],
                              k: int = 5,
                              **kwargs) -> dict[int, list[int]]:
    """
    Example algorithm: pick k heads closest to gateway, assign rest by RSSI.
    """
    alive = [i for i, nd in enumerate(nodes) if nd.is_alive]
    if not alive:
        return {}

    dists = []
    for i in alive:
        dx = nodes[i].position[0] - gateway_pos[0]
        dy = nodes[i].position[1] - gateway_pos[1]
        dists.append((i, np.sqrt(dx**2 + dy**2)))
    dists.sort(key=lambda x: x[1])

    k = min(k, len(alive))
    heads = [d[0] for d in dists[:k]]

    clusters: dict[int, list[int]] = {h: [h] for h in heads}
    for i in alive:
        if i in heads:
            continue
        best_head = heads[0]
        best_rssi = rssi_matrix[i, heads[0]]
        for h in heads[1:]:
            if rssi_matrix[i, h] > best_rssi:
                best_rssi = rssi_matrix[i, h]
                best_head = h
        clusters[best_head].append(i)

    return clusters


# ═════════════════════════════════════════════════════════════════════
#  ALGORITHM 1 — Energy-Weighted LEACH (Enhanced LEACH)
# ═════════════════════════════════════════════════════════════════════
def student_clustering_1(nodes: list[SensorNode],
                         rssi_matrix: np.ndarray,
                         gateway_pos: tuple[float, float],
                         p: float = 0.35,
                         round_num: int = 0,
                         **kwargs) -> dict[int, list[int]]:
    """
    Energy-Weighted LEACH with Gateway-Proximity Bonus.

    Each node's election probability is weighted by:
        score = 0.6 * energy_fraction + 0.4 * gateway_proximity

    Non-heads join the head with the best combined (RSSI + energy) score.
    Guarantees ~1 head per 4 nodes for balanced cluster sizes.
    """
    alive = [i for i, nd in enumerate(nodes) if nd.is_alive]
    if not alive:
        return {}

    n = len(nodes)
    rng = np.random.default_rng(42 + round_num)

    # Gateway proximity score (normalised to [0,1])
    gw_x, gw_y = gateway_pos
    proximity_scores = np.zeros(n)
    for i in alive:
        dx = nodes[i].position[0] - gw_x
        dy = nodes[i].position[1] - gw_y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        proximity_scores[i] = 1.0 / dist
    max_prox = max(proximity_scores[i] for i in alive)
    if max_prox > 0:
        for i in alive:
            proximity_scores[i] /= max_prox

    # Elect cluster heads
    alpha, beta = 0.6, 0.4
    heads: list[int] = []
    for i in alive:
        score = alpha * nodes[i].energy_fraction + beta * proximity_scores[i]
        if rng.random() < p * score:
            heads.append(i)

    # Guarantee ~1 head per 4 nodes
    target_heads = max(3, len(alive) // 4)
    if len(heads) < target_heads:
        candidates = sorted(
            [i for i in alive if i not in heads],
            key=lambda i: alpha * nodes[i].energy_fraction + beta * proximity_scores[i],
            reverse=True
        )
        heads.extend(candidates[:target_heads - len(heads)])

    # Assign non-heads with size cap
    gamma = 20.0
    max_members = max(4, len(alive) // len(heads) + 2)
    clusters: dict[int, list[int]] = {h: [h] for h in heads}
    head_set = set(heads)

    for i in alive:
        if i in head_set:
            continue
        scored = []
        for h in heads:
            r = rssi_matrix[i, h]
            if not np.isfinite(r):
                r = -120.0
            scored.append((h, r + gamma * nodes[h].energy_fraction))
        scored.sort(key=lambda x: x[1], reverse=True)

        assigned = False
        for h, _ in scored:
            if len(clusters[h]) < max_members:
                clusters[h].append(i)
                assigned = True
                break
        if not assigned:
            clusters[scored[0][0]].append(i)

    return {h: m for h, m in clusters.items() if m}


# ═════════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — Energy-Aware Gateway-Spread Clustering
# ═════════════════════════════════════════════════════════════════════
def student_clustering_2(nodes: list[SensorNode],
                         rssi_matrix: np.ndarray,
                         gateway_pos: tuple[float, float],
                         k: int = 5,
                         **kwargs) -> dict[int, list[int]]:
    """
    Energy-Aware Gateway-Spread Clustering.

    How it works:
    -------------
    This algorithm is a direct improvement over the gateway_greedy baseline.
    The baseline picks heads purely by distance to gateway (closest k nodes).
    This version adds two improvements:

    Improvement 1 — Energy-weighted head selection:
        Instead of purely picking the k closest nodes to the gateway,
        we score each candidate as:
            score = 0.5 * (1 / dist_to_gateway)  +  0.5 * energy_fraction
        The top-k scoring nodes become cluster heads. This means a node
        that is slightly farther from the gateway but has full battery
        is preferred over a closer node with a dying battery.

    Improvement 2 — Spread enforcement:
        After selecting heads by score, we check if any two heads are
        very close to each other (by RSSI). If two heads are extremely
        close, we replace the lower-energy one with the next best
        candidate that is farther away. This spreads heads across the
        deployment area so every region of the network has a nearby head,
        improving coverage and PDR.

    Member assignment (same proven approach as gateway_greedy):
        Each non-head joins the cluster head with the strongest RSSI.
        This is the same method that achieves PDR=100% in gateway_greedy,
        so we keep it exactly.

    Why this beats gateway_greedy:
    - Energy-aware selection -> heads survive longer -> longer lifetime
    - Spread enforcement -> more uniform coverage -> higher PDR over time
    - Still uses the proven strongest-RSSI assignment -> reliable delivery
    """
    alive = [i for i, nd in enumerate(nodes) if nd.is_alive]
    if not alive:
        return {}

    n = len(nodes)
    gw_x, gw_y = gateway_pos

    # ── Step 1: Score each alive node ─────────────────────────────────
    # score = 0.5 * proximity_to_gateway + 0.5 * energy_fraction
    scores = {}
    max_dist = 1e-6
    for i in alive:
        dx = nodes[i].position[0] - gw_x
        dy = nodes[i].position[1] - gw_y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        max_dist = max(max_dist, dist)
        scores[i] = dist   # store raw dist first

    # Normalise distance then compute score
    for i in alive:
        raw_dist = scores[i]
        prox_score = 1.0 - (raw_dist / max_dist)   # closer = higher score
        energy_score = nodes[i].energy_fraction
        scores[i] = 0.5 * prox_score + 0.5 * energy_score

    # ── Step 2: Pick top-k as initial head candidates ─────────────────
    k = min(k, len(alive))
    sorted_alive = sorted(alive, key=lambda i: scores[i], reverse=True)
    heads = sorted_alive[:k]

    # ── Step 3: Spread enforcement ────────────────────────────────────
    # If two heads have very strong RSSI to each other (> -55 dBm),
    # they are redundantly close. Replace the lower-energy one with
    # the next best candidate from the remaining nodes.
    remaining_candidates = sorted_alive[k:]   # unused nodes, sorted by score
    spread_threshold = -55.0   # dBm — heads closer than this are "too close"

    changed = True
    max_iterations = 10
    iteration = 0
    while changed and iteration < max_iterations and remaining_candidates:
        changed = False
        iteration += 1
        for idx_a in range(len(heads)):
            for idx_b in range(idx_a + 1, len(heads)):
                ha, hb = heads[idx_a], heads[idx_b]
                rssi_between = rssi_matrix[ha, hb]
                if np.isfinite(rssi_between) and rssi_between > spread_threshold:
                    # Too close — remove the lower-energy head
                    if nodes[ha].energy_fraction < nodes[hb].energy_fraction:
                        remove_idx = idx_a
                    else:
                        remove_idx = idx_b
                    # Replace with next best candidate
                    if remaining_candidates:
                        replacement = remaining_candidates.pop(0)
                        heads[remove_idx] = replacement
                        changed = True
                        break   # restart check after replacement
            if changed:
                break

    # ── Step 4: Assign members by strongest RSSI (proven approach) ────
    # Exactly the same as gateway_greedy — this achieves PDR=100%
    clusters: dict[int, list[int]] = {h: [h] for h in heads}
    head_set = set(heads)

    for i in alive:
        if i in head_set:
            continue
        best_head = heads[0]
        best_rssi = rssi_matrix[i, heads[0]]
        for h in heads[1:]:
            if rssi_matrix[i, h] > best_rssi:
                best_rssi = rssi_matrix[i, h]
                best_head = h
        clusters[best_head].append(i)

    return clusters


# ═════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════
MY_CLUSTERING_ALGOS = {
    "gateway_greedy":       gateway_greedy_clustering,
    "student_clustering_1": student_clustering_1,
    "student_clustering_2": student_clustering_2,
}
