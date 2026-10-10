"""
rssi_model.py — BLE RSSI propagation model.

Log-distance path-loss with log-normal shadowing and instantaneous noise.
"""

import numpy as np
from dataclasses import dataclass
from . import config as cfg


@dataclass
class RSSIReading:
    """A single RSSI observation between two nodes."""
    from_id: int
    to_id: int
    rssi_dbm: float
    distance_m: float       # true distance (hidden from students)
    timestamp: float


def true_distance(pos_a: tuple[float, float],
                  pos_b: tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) positions."""
    return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


def path_loss_db(distance: float,
                 pl_d0: float = cfg.PL_D0_DB,
                 d0: float = cfg.REF_DISTANCE,
                 n: float = cfg.PATH_LOSS_EXP) -> float:
    """
    Log-distance path loss model.

        PL(d) = PL(d0) + 10·n·log10(d / d0)

    Parameters
    ----------
    distance : float  — true distance [m]
    pl_d0    : float  — path loss at reference distance d0 [dB]
    d0       : float  — reference distance [m]
    n        : float  — path-loss exponent

    Returns
    -------
    Path loss in dB.
    """
    if distance < 0.01:
        distance = 0.01  # avoid log(0)
    return pl_d0 + 10.0 * n * np.log10(distance / d0)


def compute_rssi(distance: float,
                 tx_power_dbm: float = cfg.TX_POWER_DBM,
                 shadow_std: float = cfg.SHADOW_STD_DB,
                 noise_std: float = cfg.RSSI_NOISE_STD_DB,
                 rng: np.random.Generator | None = None) -> float:
    """
    Compute a noisy RSSI reading.

        RSSI = Tx_power − PL(d) + shadowing + noise

    Parameters
    ----------
    distance     : float — true distance [m]
    tx_power_dbm : float — transmit power [dBm]
    shadow_std   : float — log-normal shadowing σ [dB]
    noise_std    : float — instantaneous noise σ [dB]
    rng          : Generator — random number generator

    Returns
    -------
    RSSI in dBm.
    """
    if rng is None:
        rng = np.random.default_rng()
    pl = path_loss_db(distance)
    shadowing = rng.normal(0.0, shadow_std) if shadow_std > 0 else 0.0
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return tx_power_dbm - pl + shadowing + noise


def build_rssi_matrix(positions: list[tuple[float, float]],
                      rng: np.random.Generator | None = None,
                      max_range: float = cfg.BLE_MAX_RANGE
                      ) -> np.ndarray:
    """
    Build the N×N RSSI matrix for all node pairs.

    If a node is out of range (distance > max_range), the RSSI entry is -inf.
    Diagonal is 0 (self).

    Parameters
    ----------
    positions : list of (x, y) tuples for each node
    rng       : random generator
    max_range : maximum BLE range [m]

    Returns
    -------
    rssi_matrix : np.ndarray of shape (N, N), values in dBm.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
    n = len(positions)
    rssi_mat = np.full((n, n), -np.inf)
    for i in range(n):
        rssi_mat[i, i] = 0.0
        for j in range(i + 1, n):
            d = true_distance(positions[i], positions[j])
            if d <= max_range:
                r = compute_rssi(d, rng=rng)
                rssi_mat[i, j] = r
                rssi_mat[j, i] = r  # symmetric (same shadowing instance)
    return rssi_mat


def build_distance_matrix(positions: list[tuple[float, float]]) -> np.ndarray:
    """Build exact N×N distance matrix (ground truth, hidden from algorithms)."""
    n = len(positions)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = true_distance(positions[i], positions[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def rssi_to_estimated_distance(rssi_dbm: float,
                               tx_power_dbm: float = cfg.TX_POWER_DBM,
                               pl_d0: float = cfg.PL_D0_DB,
                               d0: float = cfg.REF_DISTANCE,
                               n: float = cfg.PATH_LOSS_EXP) -> float:
    """
    Estimate distance from an RSSI reading (inverse of path-loss model).

    d_est = d0 * 10^((Tx - RSSI - PL_d0) / (10·n))
    """
    exponent = (tx_power_dbm - rssi_dbm - pl_d0) / (10.0 * n)
    return d0 * (10.0 ** exponent)


def adjacency_from_rssi(rssi_matrix: np.ndarray,
                        threshold_dbm: float = cfg.RSSI_THRESHOLD_DBM
                        ) -> np.ndarray:
    """
    Build binary adjacency matrix: 1 if RSSI >= threshold, else 0.
    Diagonal is 0.
    """
    n = rssi_matrix.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and rssi_matrix[i, j] >= threshold_dbm:
                adj[i, j] = 1
    return adj
