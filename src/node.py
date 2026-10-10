"""
node.py — Sensor node model for the distributed bioimpedance mesh.

Each node has:
- A position (x, y)
- A bioimpedance sensor (with noise)
- A BLE radio (transmit / receive / idle / sleep states)
- An energy budget (battery)
- A state machine: ACTIVE → IDLE → SLEEP ↔ WAKE-UP
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from . import config as cfg


class NodeState(Enum):
    ACTIVE  = "active"       # sensing + transmitting
    IDLE    = "idle"         # radio listening, not sensing
    SLEEP   = "sleep"        # deep sleep, radio off
    WAKING  = "waking_up"   # transitioning from sleep


@dataclass
class SensorNode:
    """A single bioimpedance sensor node in the mesh."""

    node_id: int
    position: tuple[float, float]              # (x, y) in metres

    # ── Bioimpedance sensor ──────────────────────────────────────────
    bio_impedance_true: float = cfg.BIO_IMPEDANCE_BASE
    measurement_noise_std: float = cfg.MEASUREMENT_NOISE_STD

    # ── Energy ───────────────────────────────────────────────────────
    energy_remaining_j: float = cfg.BATTERY_CAPACITY_J
    energy_initial_j: float = cfg.BATTERY_CAPACITY_J

    # ── State ────────────────────────────────────────────────────────
    state: NodeState = NodeState.IDLE
    cluster_id: int = -1           # -1 = unassigned
    is_cluster_head: bool = False
    is_alive: bool = True

    # ── Statistics ───────────────────────────────────────────────────
    packets_sent: int = 0
    packets_received: int = 0
    packets_dropped: int = 0
    total_sleep_time: float = 0.0
    total_active_time: float = 0.0
    measurements_taken: int = 0

    # ── Internal ─────────────────────────────────────────────────────
    _wakeup_timer: float = 0.0
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng(cfg.RANDOM_SEED + self.node_id)

    # ── Sensing ──────────────────────────────────────────────────────
    def measure_bioimpedance(self) -> float | None:
        """Take a bioimpedance reading. Returns None if asleep or dead."""
        if not self.is_alive or self.state == NodeState.SLEEP:
            return None
        cost = cfg.SENSE_POWER_W * cfg.SENSE_TIME
        if not self._consume_energy(cost):
            return None
        self.measurements_taken += 1
        noise = self._rng.normal(0.0, self.measurement_noise_std)
        return self.bio_impedance_true + noise

    # ── Communication ────────────────────────────────────────────────
    def transmit_packet(self, packet_bytes: int = cfg.PACKET_SIZE_BYTES) -> bool:
        """Transmit a data packet.  Returns True if successful."""
        if not self.is_alive or self.state == NodeState.SLEEP:
            return False
        cost = cfg.TX_POWER_W * cfg.TX_TIME_PER_PACKET
        if not self._consume_energy(cost):
            self.packets_dropped += 1
            return False
        self.packets_sent += 1
        return True

    def receive_packet(self) -> bool:
        """Receive a data packet.  Returns True if successful."""
        if not self.is_alive or self.state == NodeState.SLEEP:
            return False
        cost = cfg.RX_POWER_W * cfg.RX_TIME_PER_PACKET
        if not self._consume_energy(cost):
            return False
        self.packets_received += 1
        return True

    # ── State transitions ────────────────────────────────────────────
    def go_to_sleep(self):
        """Transition to SLEEP state."""
        if self.is_alive:
            self.state = NodeState.SLEEP

    def wake_up(self):
        """Begin wake-up transition (takes WAKEUP_LATENCY_SEC)."""
        if self.is_alive and self.state == NodeState.SLEEP:
            self.state = NodeState.WAKING
            self._wakeup_timer = cfg.WAKEUP_LATENCY_SEC

    def set_active(self):
        """Transition to ACTIVE (sensing + transmitting)."""
        if self.is_alive:
            self.state = NodeState.ACTIVE

    def set_idle(self):
        """Transition to IDLE (listening only)."""
        if self.is_alive:
            self.state = NodeState.IDLE

    # ── Time step ────────────────────────────────────────────────────
    def tick(self, dt: float):
        """
        Advance the node by dt seconds.
        Consumes energy based on current state.
        """
        if not self.is_alive:
            return

        # Handle wake-up transition
        if self.state == NodeState.WAKING:
            self._wakeup_timer -= dt
            if self._wakeup_timer <= 0:
                self.state = NodeState.IDLE
                self._wakeup_timer = 0.0

        # Energy consumption per state
        if self.state == NodeState.SLEEP:
            cost = cfg.SLEEP_POWER_W * dt
            self.total_sleep_time += dt
        elif self.state == NodeState.ACTIVE:
            cost = (cfg.IDLE_POWER_W + cfg.SENSE_POWER_W * 0.1) * dt  # active overhead
            self.total_active_time += dt
        else:  # IDLE or WAKING
            cost = cfg.IDLE_POWER_W * dt
            self.total_active_time += dt

        self._consume_energy(cost)

    # ── Energy ───────────────────────────────────────────────────────
    def _consume_energy(self, joules: float) -> bool:
        """
        Consume energy.  If battery dies, mark node as dead.
        Returns True if the node is still alive after consumption.
        """
        self.energy_remaining_j -= joules
        if self.energy_remaining_j <= 0:
            self.energy_remaining_j = 0.0
            self.is_alive = False
            self.state = NodeState.SLEEP
            return False
        return True

    @property
    def energy_fraction(self) -> float:
        """Remaining energy as a fraction [0, 1]."""
        return self.energy_remaining_j / self.energy_initial_j

    @property
    def is_awake(self) -> bool:
        return self.is_alive and self.state != NodeState.SLEEP


@dataclass
class GatewayNode:
    """
    The gateway / sink node.  Always powered (unlimited energy).
    Collects data from cluster heads, can broadcast wake-up signals.
    """
    node_id: int = -1
    position: tuple[float, float] = cfg.GATEWAY_POS
    is_alive: bool = True
    packets_received: int = 0
    wakeup_signals_sent: int = 0

    def receive_data(self, data: dict) -> bool:
        """Receive aggregated data from a cluster head."""
        self.packets_received += 1
        return True

    def broadcast_wakeup(self) -> float:
        """
        Broadcast a wake-up signal.
        Returns the energy cost (charged to nodes that hear it, or to gateway power supply).
        """
        self.wakeup_signals_sent += 1
        return cfg.WAKEUP_SIGNAL_COST_J


def deploy_nodes(num_nodes: int = cfg.NUM_NODES,
                 width: float = cfg.AREA_WIDTH,
                 height: float = cfg.AREA_HEIGHT,
                 rng: np.random.Generator | None = None,
                 deployment: str = "random") -> list[SensorNode]:
    """
    Deploy sensor nodes in the area.

    Parameters
    ----------
    num_nodes  : number of nodes
    width, height : deployment area size
    rng        : random generator
    deployment : 'random' | 'grid' | 'clustered'

    Returns
    -------
    List of SensorNode instances.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.RANDOM_SEED)

    nodes = []
    if deployment == "grid":
        side = int(np.ceil(np.sqrt(num_nodes)))
        xs = np.linspace(5, width - 5, side)
        ys = np.linspace(5, height - 5, side)
        idx = 0
        for x in xs:
            for y in ys:
                if idx >= num_nodes:
                    break
                pos = (float(x), float(y))
                bio_true = cfg.BIO_IMPEDANCE_BASE + rng.normal(0, cfg.BIO_IMPEDANCE_VAR)
                nodes.append(SensorNode(node_id=idx, position=pos,
                                        bio_impedance_true=bio_true, _rng=rng))
                idx += 1
    elif deployment == "clustered":
        # Place nodes in natural clusters (3-5 groups)
        n_groups = rng.integers(3, 6)
        centres = [(rng.uniform(15, width - 15), rng.uniform(15, height - 15))
                    for _ in range(n_groups)]
        for idx in range(num_nodes):
            c = centres[idx % n_groups]
            x = np.clip(c[0] + rng.normal(0, 10), 0, width)
            y = np.clip(c[1] + rng.normal(0, 10), 0, height)
            bio_true = cfg.BIO_IMPEDANCE_BASE + rng.normal(0, cfg.BIO_IMPEDANCE_VAR)
            nodes.append(SensorNode(node_id=idx, position=(float(x), float(y)),
                                    bio_impedance_true=bio_true, _rng=rng))
    else:  # random
        for idx in range(num_nodes):
            x = rng.uniform(5, width - 5)
            y = rng.uniform(5, height - 5)
            bio_true = cfg.BIO_IMPEDANCE_BASE + rng.normal(0, cfg.BIO_IMPEDANCE_VAR)
            nodes.append(SensorNode(node_id=idx, position=(float(x), float(y)),
                                    bio_impedance_true=bio_true, _rng=rng))
    return nodes
