"""
config.py — Global simulation parameters for the Distributed Bioimpedance Mesh project.
"""

import numpy as np

# ── Random seed for reproducibility ──────────────────────────────────
RANDOM_SEED = 42

# ── Deployment area (metres) ─────────────────────────────────────────
AREA_WIDTH  = 100.0   # metres
AREA_HEIGHT = 100.0   # metres

# ── Node configuration ───────────────────────────────────────────────
NUM_NODES       = 30       # total sensor nodes (excluding gateway)
GATEWAY_POS     = (50.0, 50.0)  # gateway placed at centre

# ── Bioimpedance sensing ─────────────────────────────────────────────
BIO_IMPEDANCE_BASE   = 500.0    # ohms — nominal tissue impedance
BIO_IMPEDANCE_VAR    = 50.0     # ohms — variation across nodes
MEASUREMENT_NOISE_STD = 5.0     # ohms — measurement noise σ
MEASUREMENT_INTERVAL  = 1.0     # seconds between measurements

# ── BLE / RSSI parameters ───────────────────────────────────────────
TX_POWER_DBM       = 0.0     # BLE transmit power (dBm)
PATH_LOSS_EXP      = 3.0     # path-loss exponent (indoor / body)
REF_DISTANCE       = 1.0     # reference distance d0 (metres)
PL_D0_DB           = 40.0    # path loss at d0 (dB)
SHADOW_STD_DB      = 4.0     # log-normal shadowing σ (dB)
RSSI_NOISE_STD_DB  = 2.0     # instantaneous RSSI noise σ (dB)
BLE_MAX_RANGE      = 30.0    # max communication range (metres)

# ── Clustering defaults ──────────────────────────────────────────────
RSSI_THRESHOLD_DBM    = -70.0   # RSSI above this → "neighbour"
NUM_CLUSTERS          = 5       # default k for k-based algorithms
LEACH_P               = 0.1     # probability of becoming cluster head (LEACH)
CLUSTER_ROUND_INTERVAL = 60.0   # seconds between re-clustering

# ── Mesh / routing ───────────────────────────────────────────────────
MAX_HOPS            = 6        # maximum hops to gateway
PACKET_SIZE_BYTES   = 64       # bioimpedance data packet
ACK_SIZE_BYTES      = 8        # acknowledgement

# ── Energy model (typical BLE SoC, e.g. nRF52840) ───────────────────
BATTERY_CAPACITY_J     = 2700.0   # ~750 mAh coin cell @ 3 V → Joules
TX_POWER_W             = 0.030    # 30 mW active transmit
RX_POWER_W             = 0.025    # 25 mW active receive
IDLE_POWER_W           = 0.005    # 5 mW idle / listening
SLEEP_POWER_W          = 0.0001   # 0.1 mW deep sleep
SENSE_POWER_W          = 0.010    # 10 mW bioimpedance measurement
TX_TIME_PER_PACKET     = 0.002    # 2 ms to transmit one packet
RX_TIME_PER_PACKET     = 0.002    # 2 ms to receive one packet
SENSE_TIME             = 0.005    # 5 ms per impedance measurement

# ── Sleep / wake-up ──────────────────────────────────────────────────
SLEEP_CYCLE_SEC       = 10.0    # default periodic sleep cycle
DUTY_CYCLE            = 0.10    # fraction of time awake (10 %)
WAKEUP_SIGNAL_COST_J  = 0.0005  # energy cost for gateway wake-up beacon
WAKEUP_LATENCY_SEC    = 0.050   # time to transition sleep → active

# ── Simulation ───────────────────────────────────────────────────────
SIM_DURATION_SEC  = 3600.0   # total simulation time (1 hour)
TIME_STEP_SEC     = 1.0      # simulation resolution
DATA_REPORT_INTERVAL = 30.0  # how often nodes should send data to gateway

# ── Visualisation ────────────────────────────────────────────────────
FIGURE_DPI   = 120
CMAP_CLUSTER = "tab10"
NODE_SIZE    = 80
