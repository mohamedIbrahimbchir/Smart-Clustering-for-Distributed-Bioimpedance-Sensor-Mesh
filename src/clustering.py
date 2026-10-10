"""
clustering.py — Built-in clustering algorithms for the BLE mesh.

Every clustering function has the same interface:

    def algo(nodes, rssi_matrix, gateway_pos, **kwargs)
        -> dict[int, list[int]]

    Returns a mapping:  cluster_head_id -> [member_id, ...]

Students will study these as baselines and then implement their own.
"""

from __future__ import annotations
import numpy as np
from typing import Callable
from . import config as cfg
from .node import SensorNode
from .rssi_model import adjacency_from_rssi, true_distance

# Type alias for clustering functions
ClusterFn = Callable[..., dict[int, list[int]]]


# ═════════════════════════════════════════════════════════════════════
# 1.  RSSI-Threshold Clustering
# ═════════════════════════════════════════════════════════════════════
def rssi_threshold_clustering(nodes: list[SensorNode],
                              rssi_matrix: np.ndarray,
                              gateway_pos: tuple[float, float] = cfg.GATEWAY_POS,
                              threshold_dbm: float = cfg.RSSI_THRESHOLD_DBM,
                              **kwargs) -> dict[int, list[int]]:
    """
    Simple threshold-based clustering.

    1. Build adjacency from RSSI threshold.
    2. Node with the most neighbours becomes cluster head.
    3. Assign un-clustered neighbours to nearest cluster head.
    4. Repeat until all nodes assigned.
    """
    n = len(nodes)
    adj = adjacency_from_rssi(rssi_matrix, threshold_dbm)
    assigned = [False] * n
    clusters: dict[int, list[int]] = {}

    # Sort nodes by degree (most connected first)
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)

    for head_idx in order:
        head_id = int(head_idx)
        if assigned[head_id] or not nodes[head_id].is_alive:
            continue
        # This node becomes a cluster head
        members = [head_id]
        assigned[head_id] = True
        for j in range(n):
            if not assigned[j] and adj[head_id, j] == 1 and nodes[j].is_alive:
                members.append(j)
                assigned[j] = True
        clusters[head_id] = members

    # Assign any remaining orphans to the nearest cluster head
    for i in range(n):
        if not assigned[i] and nodes[i].is_alive:
            best_head = None
            best_rssi = -np.inf
            for h in clusters:
                if rssi_matrix[i, h] > best_rssi:
                    best_rssi = rssi_matrix[i, h]
                    best_head = h
            if best_head is not None:
                clusters[best_head].append(i)
                assigned[i] = True
            else:
                # Truly isolated — make it its own cluster
                clusters[i] = [i]
                assigned[i] = True

    return clusters


# ═════════════════════════════════════════════════════════════════════
# 2.  K-Medoids on RSSI Similarity
# ═════════════════════════════════════════════════════════════════════
def kmedoids_clustering(nodes: list[SensorNode],
                        rssi_matrix: np.ndarray,
                        gateway_pos: tuple[float, float] = cfg.GATEWAY_POS,
                        k: int = cfg.NUM_CLUSTERS,
                        max_iter: int = 50,
                        **kwargs) -> dict[int, list[int]]:
    """
    K-medoids clustering using RSSI-derived dissimilarity.

    Dissimilarity = -RSSI (higher RSSI = more similar = smaller distance).
    """
    alive = [i for i, nd in enumerate(nodes) if nd.is_alive]
    n_alive = len(alive)
    k = min(k, n_alive)

    # Build dissimilarity matrix (only among alive nodes)
    diss = np.zeros((n_alive, n_alive))
    for ii, i in enumerate(alive):
        for jj, j in enumerate(alive):
            if i == j:
                diss[ii, jj] = 0.0
            elif rssi_matrix[i, j] > -np.inf:
                diss[ii, jj] = -rssi_matrix[i, j]  # negate: high RSSI → low dissim
            else:
                diss[ii, jj] = 200.0  # out-of-range penalty

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    # Initialise medoids randomly
    medoid_indices = list(rng.choice(n_alive, size=k, replace=False))

    for _ in range(max_iter):
        # Assign each node to nearest medoid
        labels = np.zeros(n_alive, dtype=int)
        for ii in range(n_alive):
            best = 0
            best_d = diss[ii, medoid_indices[0]]
            for mi, m in enumerate(medoid_indices):
                if diss[ii, m] < best_d:
                    best_d = diss[ii, m]
                    best = mi
            labels[ii] = best

        # Update medoids
        changed = False
        for mi in range(k):
            members = np.where(labels == mi)[0]
            if len(members) == 0:
                continue
            # Pick the member with minimum total dissimilarity to others
            best_total = np.inf
            best_m = medoid_indices[mi]
            for c in members:
                total = sum(diss[c, m] for m in members)
                if total < best_total:
                    best_total = total
                    best_m = c
            if best_m != medoid_indices[mi]:
                medoid_indices[mi] = best_m
                changed = True

        if not changed:
            break

    # Build output dict
    clusters: dict[int, list[int]] = {}
    labels_final = np.zeros(n_alive, dtype=int)
    for ii in range(n_alive):
        best = 0
        best_d = diss[ii, medoid_indices[0]]
        for mi, m in enumerate(medoid_indices):
            if diss[ii, m] < best_d:
                best_d = diss[ii, m]
                best = mi
        labels_final[ii] = best

    for mi in range(k):
        head_local = medoid_indices[mi]
        head_global = alive[head_local]
        members_global = [alive[ii] for ii in range(n_alive) if labels_final[ii] == mi]
        if members_global:
            clusters[head_global] = members_global

    return clusters


# ═════════════════════════════════════════════════════════════════════
# 3.  LEACH-like Energy-Aware Clustering
# ═════════════════════════════════════════════════════════════════════
def leach_clustering(nodes: list[SensorNode],
                     rssi_matrix: np.ndarray,
                     gateway_pos: tuple[float, float] = cfg.GATEWAY_POS,
                     p: float = cfg.LEACH_P,
                     round_num: int = 0,
                     **kwargs) -> dict[int, list[int]]:
    """
    LEACH-inspired clustering.

    1. Each alive node elects itself as cluster head with probability p,
       weighted by remaining energy.
    2. Non-head nodes join the cluster head with the strongest RSSI.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED + round_num)
    n = len(nodes)
    heads: list[int] = []

    for i in range(n):
        if not nodes[i].is_alive:
            continue
        # Energy-weighted probability
        prob = p * nodes[i].energy_fraction
        if rng.random() < prob:
            heads.append(i)

    # Fallback: if no heads elected, pick the node with most energy
    if not heads:
        alive_nodes = [i for i in range(n) if nodes[i].is_alive]
        if alive_nodes:
            heads = [max(alive_nodes, key=lambda i: nodes[i].energy_fraction)]

    # Assign nodes to nearest cluster head (by RSSI)
    clusters: dict[int, list[int]] = {h: [h] for h in heads}
    for i in range(n):
        if not nodes[i].is_alive or i in heads:
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
# 4.  Spectral Clustering on RSSI Affinity
# ═════════════════════════════════════════════════════════════════════
def spectral_clustering(nodes: list[SensorNode],
                        rssi_matrix: np.ndarray,
                        gateway_pos: tuple[float, float] = cfg.GATEWAY_POS,
                        k: int = cfg.NUM_CLUSTERS,
                        **kwargs) -> dict[int, list[int]]:
    """
    Spectral clustering using the RSSI affinity matrix.

    1. Build affinity W from RSSI (Gaussian kernel on RSSI values).
    2. Compute normalised Laplacian.
    3. Use the first k eigenvectors for k-means.
    """
    alive = [i for i, nd in enumerate(nodes) if nd.is_alive]
    n_alive = len(alive)
    k = min(k, n_alive)

    # Build affinity matrix
    sigma = 10.0  # dB bandwidth
    W = np.zeros((n_alive, n_alive))
    for ii, i in enumerate(alive):
        for jj, j in enumerate(alive):
            if i == j:
                continue
            r = rssi_matrix[i, j]
            if r > -np.inf:
                W[ii, jj] = np.exp(-((-r) ** 2) / (2 * sigma ** 2))

    # Degree matrix
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))

    # Normalised Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    L = np.eye(n_alive) - D_inv_sqrt @ W @ D_inv_sqrt

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Take first k eigenvectors (smallest eigenvalues)
    U = eigenvectors[:, :k]
    # Normalise rows
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    U = U / norms

    # Simple k-means on the spectral embedding
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    centroids = U[rng.choice(n_alive, k, replace=False)]

    for _ in range(30):
        # Assign
        dists = np.linalg.norm(U[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) > 0:
                new_centroids[c] = U[members].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # Build output — pick cluster head as node with highest energy in each cluster
    clusters: dict[int, list[int]] = {}
    for c in range(k):
        member_local = np.where(labels == c)[0]
        if len(member_local) == 0:
            continue
        member_global = [alive[ii] for ii in member_local]
        head = max(member_global, key=lambda i: nodes[i].energy_fraction)
        clusters[head] = member_global

    return clusters


# ═════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════
CLUSTERING_ALGOS: dict[str, ClusterFn] = {
    "rssi_threshold": rssi_threshold_clustering,
    "kmedoids":       kmedoids_clustering,
    "leach":          leach_clustering,
    "spectral":       spectral_clustering,
}


def get_clustering(name: str) -> ClusterFn:
    """Look up a clustering algorithm by name."""
    if name not in CLUSTERING_ALGOS:
        raise ValueError(f"Unknown clustering algorithm '{name}'. "
                         f"Available: {list(CLUSTERING_ALGOS.keys())}")
    return CLUSTERING_ALGOS[name]
