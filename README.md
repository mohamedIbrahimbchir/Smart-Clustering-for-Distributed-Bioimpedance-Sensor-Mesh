# Project 4 — Smart Clustering for Distributed Bioimpedance Sensor Mesh

> **Core Idea:** A network of bioimpedance sensor nodes that self-organise into
> clusters using BLE RSSI, route data through a mesh to a gateway, and manage
> sleep/wake-up cycles to maximise network lifetime.

---

## What This Simulator Does

| Capability | Description |
|---|---|
| **Node deployment** | Random, grid, or naturally-clustered layouts |
| **Bioimpedance sensing** | Each node measures tissue impedance (with noise) |
| **BLE RSSI model** | Log-distance path loss + log-normal shadowing + noise |
| **Clustering** | 4 built-in algorithms: RSSI-threshold, k-medoids, LEACH, spectral |
| **Mesh routing** | Multi-hop BFS routing from cluster heads to gateway |
| **Sleep scheduling** | 4 built-in schedulers: periodic, synchronised, on-demand, adaptive |
| **Energy model** | Realistic BLE SoC power model (TX/RX/idle/sleep) + battery depletion |
| **Metrics** | PDR, cluster balance, energy efficiency, latency, network lifetime |
| **Visualisation** | Topology maps, energy plots, RSSI heatmaps, full dashboards |

---

## Repository Structure

```
Project_4_Distributed_Bioimpedance/
├── README.md                        ← You are here
├── requirements.txt                 ← pip install -r requirements.txt
├── auto_grader.py                   ← Automated pass/fail grading
│
├── src/                             ← Core simulator library
│   ├── config.py                    ← All simulation parameters
│   ├── node.py                      ← SensorNode + GatewayNode models
│   ├── rssi_model.py                ← BLE RSSI propagation model
│   ├── clustering.py                ← 4 built-in clustering algorithms
│   ├── mesh_network.py              ← Mesh topology + multi-hop routing
│   ├── energy_model.py              ← Energy accounting
│   ├── sleep_scheduler.py           ← 4 built-in sleep schedulers
│   ├── metrics.py                   ← Performance evaluation
│   └── simulator.py                 ← Main orchestrator + plots
│
├── experiments.py                   ← 7 pre-built demo experiments
│
└── student_work/                    ← ★ YOUR WORKSPACE ★
    ├── my_clustering.py             ← Write your clustering algorithms here
    ├── my_sleep_scheduler.py        ← Write your sleep schedulers here
    ├── evaluate_my_work.py          ← Compare yours vs built-in
    └── my_experiments.py            ← Sandbox for custom tests
```

---

## Background Theory

### Bioimpedance Sensing

Bioimpedance measures the electrical impedance of biological tissue by injecting a small AC current and measuring the resulting voltage. The impedance Z(ω) has a real (resistive) and imaginary (capacitive) component:

    Z(ω) = R + jX

where R is resistance (extracellular fluid) and X is reactance (cell membranes). Changes in bioimpedance indicate tissue composition changes — useful for medical monitoring, body composition analysis, and agricultural sensing.

### BLE RSSI Propagation Model

The received signal strength (RSSI) between two BLE nodes follows the **log-distance path-loss model**:

    RSSI(d) = Tx_power − PL(d) + X_σ + n

where:
- `PL(d) = PL(d₀) + 10·n·log₁₀(d/d₀)` — path loss at distance d
- `n` — path-loss exponent (2.0 free-space, 3.0 indoor, 3.5 body)
- `X_σ ~ N(0, σ²)` — log-normal shadowing
- `n ~ N(0, σ_n²)` — instantaneous noise

### Clustering for Wireless Sensor Networks

In a mesh-based architecture, **clustering** groups nearby nodes. Each cluster elects a **cluster head (CH)** that:
1. Collects bioimpedance data from members
2. Aggregates and compresses the data
3. Routes it through the mesh to the **gateway**

Good clustering minimises energy consumption by reducing long-range transmissions.

| Algorithm | Principle | Pros | Cons |
|---|---|---|---|
| RSSI-threshold | Neighbours above RSSI threshold | Simple, distributed | Unbalanced clusters |
| K-medoids | Minimise intra-cluster dissimilarity | Balanced | Needs global RSSI matrix |
| LEACH | Random CH election, energy-weighted | Energy-aware, distributed | Random balance |
| Spectral | Eigenvectors of RSSI Laplacian | Finds natural clusters | Computationally heavy |

### Sleep / Wake-Up Scheduling

Nodes spend most energy in radio TX/RX. **Sleep scheduling** puts nodes into deep sleep when not needed:

| Scheduler | Principle | Pros | Cons |
|---|---|---|---|
| Periodic | Fixed duty cycle (e.g. 10% on) | Simple, predictable | Rigid, not adaptive |
| Synchronised | TDMA slots within cluster | No collisions | Requires sync |
| On-demand | Gateway wake-up signal | Minimal idle | Wake-up latency |
| Adaptive | Duty cycle ∝ remaining energy | Extends lifetime | More complex |

### Performance Metrics

- **PDR** (Packet Delivery Ratio): `delivered / sent` — target ≥ 90%
- **Cluster Balance**: `σ/μ` of cluster sizes — lower is better
- **Energy Efficiency**: `packets_delivered / total_Joules` — higher is better
- **Network Lifetime**: time until first node dies
- **Avg Hops**: mean hop count to gateway — lower is better

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo experiments
python experiments.py

# 3. Write your algorithms
#    → Edit student_work/my_clustering.py
#    → Edit student_work/my_sleep_scheduler.py

# 4. Evaluate your work
python student_work/evaluate_my_work.py

# 5. Grade your work
python auto_grader.py
```

---

## Experiments Menu

| # | Experiment | What it shows |
|---|---|---|
| 1 | Topology visualisation | Random / grid / clustered deployments |
| 2 | Compare clustering algorithms | All 4 built-in algorithms side-by-side |
| 3 | Compare sleep schedulers | All 4 schedulers on energy + PDR |
| 4 | Node scaling study | Performance vs network size |
| 5 | RSSI threshold sweep | Effect of threshold on cluster formation |
| 6 | Full dashboard | Deep-dive single run (1 hour) |
| 7 | Wearable challenge | 20 nodes on body, 8-hour endurance |

---

## Student Tasks

### Where to Write Code

| Action | File | How to Run |
|---|---|---|
| Write clustering algos | `student_work/my_clustering.py` | — |
| Write sleep schedulers | `student_work/my_sleep_scheduler.py` | — |
| Evaluate vs built-ins | `student_work/evaluate_my_work.py` | `python student_work/evaluate_my_work.py` |
| Custom experiments | `student_work/my_experiments.py` | `python student_work/my_experiments.py` |
| **Grade your work** | `auto_grader.py` | `python auto_grader.py` |

### Task A — Explore the Simulator (Warm-up)
1. Run `python experiments.py` and try experiments 1, 2, and 3.
2. Observe how different clustering algorithms partition the network.
3. Note how sleep schedulers affect energy and PDR.

### Task B — Implement Your Clustering Algorithms
1. Open `student_work/my_clustering.py`.
2. Study the working example (`gateway_greedy_clustering`).
3. Implement `student_clustering_1` and `student_clustering_2`.
4. Register them in `MY_CLUSTERING_ALGOS`.

**Interface:**
```python
def my_clustering(nodes, rssi_matrix, gateway_pos, **kwargs) -> dict[int, list[int]]
```

### Task C — Implement Your Sleep Schedulers
1. Open `student_work/my_sleep_scheduler.py`.
2. Study the working example (`round_robin_scheduler`).
3. Implement `student_scheduler_1` and `student_scheduler_2`.
4. Register them in `MY_SCHEDULER_ALGOS`.

**Interface:**
```python
def my_scheduler(nodes, clusters, current_time, gateway, **kwargs) -> dict[int, str]
```

### Task D — Evaluate & Compare
Run `python student_work/evaluate_my_work.py` to compare your algorithms against
all built-in baselines across multiple scenarios. Record PDR, energy, and balance.

### Task E — Design Challenge: Wearable Bioimpedance Array
**Scenario:** 20 bioimpedance patches on a patient's torso (40cm × 50cm). Gateway on belt. Must deliver data every 5 seconds for 8 hours. **Design a clustering + scheduling strategy that keeps PDR ≥ 95% with all nodes alive.**

---

## Auto-Grader

Run at any time:
```bash
python auto_grader.py
```

### Clustering Tests (100 pts per algorithm)

| Test | What it checks | Points |
|---|---|---|
| T1 | Runs without crashing | 5 |
| T2 | Correct output format | 5 |
| T3 | All alive nodes assigned | 10 |
| T4 | Reasonable cluster count | 10 |
| T5 | Cluster balance (σ/μ) | 10 |
| T6 | PDR ≥ 70% in simulation | 10 |
| T7 | Energy efficiency vs baseline | 10 |
| T8 | Scales to 40 nodes | 10 |
| T9 | Beats threshold baseline | 15 |
| T10 | Wearable challenge (PDR ≥ 90%) | 15 |

### Scheduler Tests (100 pts per algorithm)

| Test | What it checks | Points |
|---|---|---|
| T1 | Runs without crashing | 5 |
| T2 | Correct output format | 5 |
| T3 | All nodes scheduled | 10 |
| T4 | Uses sleep mode | 10 |
| T5 | Cluster heads awake | 10 |
| T6 | Time-varying behaviour | 10 |
| T7 | PDR ≥ 60% in simulation | 10 |
| T8 | Energy conservation | 10 |
| T9 | Beats periodic baseline | 15 |
| T10 | 8-hour endurance test | 15 |

### Pass Criteria
- Each algorithm needs **≥ 70%** to pass.
- Must have **≥ 1 non-trivial clustering** + **≥ 1 non-trivial scheduler**.
- Partial credit for close results.

---

## Recommended Reading

1. H. Karl & A. Willig, *Protocols and Architectures for Wireless Sensor Networks*, Wiley.
2. W. Heinzelman et al., "LEACH: Energy-Efficient Communication Protocol for WSN", HICSS 2000.
3. S. Grimnes & Ø. G. Martinsen, *Bioimpedance and Bioelectricity Basics*, Academic Press.
4. Bluetooth SIG, *Bluetooth Core Specification v5.3* — mesh networking profile.
5. IEEE 802.15.4 — Low-rate WPAN standard.

---

*Sensors & Smart IoT — Project 4*
