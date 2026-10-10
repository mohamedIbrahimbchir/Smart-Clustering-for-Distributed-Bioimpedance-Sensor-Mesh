"""
auto_grader.py — Coursera-style automated grading for Project 4.

Run:
    python auto_grader.py

Imports student algorithms from student_work/, runs 10 automated tests,
and prints a pass/fail scorecard.
"""

import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import dataclass, field
from src.node import SensorNode, GatewayNode, deploy_nodes
from src.rssi_model import build_rssi_matrix
from src.clustering import CLUSTERING_ALGOS
from src.sleep_scheduler import SCHEDULER_ALGOS
from src.simulator import Simulator
from src.metrics import compute_all_metrics, compute_clustering_metrics
from src import config as cfg


# ═════════════════════════════════════════════════════════════════════
#  Grading infrastructure
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    points: int
    max_points: int
    detail: str = ""
    hint: str = ""


@dataclass
class AlgoGradeReport:
    algo_name: str
    algo_type: str            # "clustering" or "scheduler"
    tests: list[TestResult] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(t.points for t in self.tests)

    @property
    def max_score(self) -> int:
        return sum(t.max_points for t in self.tests)

    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0

    @property
    def passed(self) -> bool:
        return self.percentage >= 70.0


# ═════════════════════════════════════════════════════════════════════
#  Clustering tests
# ═════════════════════════════════════════════════════════════════════

def grade_clustering_algorithm(name: str, algo_fn) -> AlgoGradeReport:
    """Run 10 tests on a clustering algorithm."""
    report = AlgoGradeReport(algo_name=name, algo_type="clustering")
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # ── Set up test networks ─────────────────────────────────────────
    nodes_easy = deploy_nodes(20, cfg.AREA_WIDTH, cfg.AREA_HEIGHT, rng, "random")
    rssi_easy = build_rssi_matrix([n.position for n in nodes_easy], rng)

    rng2 = np.random.default_rng(99)
    nodes_hard = deploy_nodes(40, cfg.AREA_WIDTH, cfg.AREA_HEIGHT, rng2, "random")
    rssi_hard = build_rssi_matrix([n.position for n in nodes_hard], rng2)

    gateway_pos = cfg.GATEWAY_POS
    clusters_easy = None
    clusters_hard = None

    # T1: Runs without crashing (5 pts)
    try:
        clusters_easy = algo_fn(nodes_easy, rssi_easy, gateway_pos)
        report.tests.append(TestResult("T1: Runs without error", True, 5, 5,
                                       "Algorithm executed successfully."))
    except Exception as e:
        report.tests.append(TestResult("T1: Runs without error", False, 0, 5,
                                       f"Error: {e}",
                                       "Make sure your function handles all inputs."))
        # All remaining tests fail
        for i in range(2, 11):
            report.tests.append(TestResult(f"T{i}: Skipped", False, 0,
                                           [5,10,10,10,10,10,10,15,15][i-2],
                                           "Skipped due to T1 failure."))
        return report

    # T2: Returns correct format (5 pts)
    try:
        assert isinstance(clusters_easy, dict), "Must return a dict"
        for head, members in clusters_easy.items():
            assert isinstance(head, (int, np.integer)), f"Head {head} must be int"
            assert isinstance(members, list), f"Members for head {head} must be a list"
            assert head in members, f"Head {head} must be in its own member list"
        report.tests.append(TestResult("T2: Correct output format", True, 5, 5,
                                       "dict[int, list[int]] ✓"))
    except AssertionError as e:
        report.tests.append(TestResult("T2: Correct output format", False, 0, 5,
                                       str(e),
                                       "Return dict[cluster_head_id -> [member_ids]]. "
                                       "Head must be in its own member list."))

    # T3: All alive nodes assigned (10 pts)
    alive_ids = {i for i, n in enumerate(nodes_easy) if n.is_alive}
    assigned = set()
    for members in clusters_easy.values():
        assigned.update(members)
    coverage = len(alive_ids & assigned) / len(alive_ids) if alive_ids else 0
    if coverage >= 1.0:
        report.tests.append(TestResult("T3: Full coverage", True, 10, 10,
                                       f"All {len(alive_ids)} alive nodes assigned."))
    elif coverage >= 0.8:
        report.tests.append(TestResult("T3: Full coverage", True, 5, 10,
                                       f"Partial: {coverage:.0%} coverage (need 100%).",
                                       "Make sure every alive node is in some cluster."))
    else:
        report.tests.append(TestResult("T3: Full coverage", False, 0, 10,
                                       f"Only {coverage:.0%} coverage.",
                                       "Many nodes are unassigned — check your logic."))

    # T4: Reasonable cluster count (10 pts)
    n_clusters = len(clusters_easy)
    n_alive = len(alive_ids)
    if 2 <= n_clusters <= n_alive // 2:
        report.tests.append(TestResult("T4: Reasonable cluster count", True, 10, 10,
                                       f"{n_clusters} clusters for {n_alive} nodes."))
    elif n_clusters == 1 or n_clusters == n_alive:
        report.tests.append(TestResult("T4: Reasonable cluster count", False, 0, 10,
                                       f"{n_clusters} clusters — degenerate.",
                                       "1 cluster = no clustering. N clusters = each its own."))
    else:
        report.tests.append(TestResult("T4: Reasonable cluster count", True, 5, 10,
                                       f"{n_clusters} clusters — acceptable."))

    # T5: Cluster balance (10 pts)
    cm = compute_clustering_metrics(nodes_easy, clusters_easy, rssi_easy)
    balance = cm["cluster_balance"]
    if balance < 0.5:
        report.tests.append(TestResult("T5: Cluster balance", True, 10, 10,
                                       f"Balance σ/μ = {balance:.3f} (good)."))
    elif balance < 1.0:
        report.tests.append(TestResult("T5: Cluster balance", True, 5, 10,
                                       f"Balance σ/μ = {balance:.3f} (acceptable).",
                                       "Try to make clusters more even in size."))
    else:
        report.tests.append(TestResult("T5: Cluster balance", False, 0, 10,
                                       f"Balance σ/μ = {balance:.3f} (very uneven).",
                                       "Huge disparity in cluster sizes."))

    # T6: PDR > 70% in simulation (10 pts)
    try:
        sim = Simulator(num_nodes=20, seed=cfg.RANDOM_SEED)
        result = sim.run(clustering_algo=algo_fn, scheduler="periodic", duration=600)
        pdr = result.metrics.packet_delivery_ratio
        if pdr >= 0.70:
            report.tests.append(TestResult("T6: PDR ≥ 70%", True, 10, 10,
                                           f"PDR = {pdr:.1%}"))
        elif pdr >= 0.50:
            report.tests.append(TestResult("T6: PDR ≥ 70%", True, 5, 10,
                                           f"PDR = {pdr:.1%} (close, need ≥70%).",
                                           "Cluster heads may not reach gateway."))
        else:
            report.tests.append(TestResult("T6: PDR ≥ 70%", False, 0, 10,
                                           f"PDR = {pdr:.1%}.",
                                           "Many packets lost — check routing."))
    except Exception as e:
        report.tests.append(TestResult("T6: PDR ≥ 70%", False, 0, 10,
                                       f"Simulation error: {e}"))

    # T7: Energy efficiency better than random (10 pts)
    try:
        sim_baseline = Simulator(num_nodes=20, seed=cfg.RANDOM_SEED)
        result_baseline = sim_baseline.run(clustering_algo="rssi_threshold",
                                           scheduler="periodic", duration=600)
        baseline_eff = result_baseline.metrics.energy_efficiency
        student_eff = result.metrics.energy_efficiency
        if student_eff >= baseline_eff * 0.8:
            report.tests.append(TestResult("T7: Energy efficiency", True, 10, 10,
                                           f"Yours: {student_eff:.1f}, "
                                           f"baseline: {baseline_eff:.1f} pkts/J"))
        elif student_eff >= baseline_eff * 0.5:
            report.tests.append(TestResult("T7: Energy efficiency", True, 5, 10,
                                           f"Yours: {student_eff:.1f} vs "
                                           f"baseline: {baseline_eff:.1f}",
                                           "Close but not matching baseline."))
        else:
            report.tests.append(TestResult("T7: Energy efficiency", False, 0, 10,
                                           f"Yours: {student_eff:.1f} vs "
                                           f"baseline: {baseline_eff:.1f}",
                                           "Your clustering wastes energy."))
    except Exception as e:
        report.tests.append(TestResult("T7: Energy efficiency", False, 0, 10,
                                       f"Error: {e}"))

    # T8: Works on larger network (10 pts)
    try:
        clusters_hard = algo_fn(nodes_hard, rssi_hard, gateway_pos)
        assert isinstance(clusters_hard, dict) and len(clusters_hard) > 0
        report.tests.append(TestResult("T8: Scales to 40 nodes", True, 10, 10,
                                       f"{len(clusters_hard)} clusters on 40 nodes."))
    except Exception as e:
        report.tests.append(TestResult("T8: Scales to 40 nodes", False, 0, 10,
                                       f"Error: {e}",
                                       "Your algorithm must work for different network sizes."))

    # T9: Beats simple threshold baseline (15 pts)
    try:
        baseline_clusters = CLUSTERING_ALGOS["rssi_threshold"](
            nodes_easy, rssi_easy, gateway_pos)
        baseline_cm = compute_clustering_metrics(nodes_easy, baseline_clusters, rssi_easy)
        student_cm = compute_clustering_metrics(nodes_easy, clusters_easy, rssi_easy)

        # Score: better in at least 2 of 3 criteria
        wins = 0
        if student_cm["cluster_balance"] < baseline_cm["cluster_balance"]:
            wins += 1
        if student_cm["coverage"] >= baseline_cm["coverage"]:
            wins += 1
        if student_cm["avg_intra_cluster_rssi"] > baseline_cm["avg_intra_cluster_rssi"]:
            wins += 1

        if wins >= 2:
            report.tests.append(TestResult("T9: Beats threshold baseline", True, 15, 15,
                                           f"Wins {wins}/3 criteria vs threshold."))
        elif wins == 1:
            report.tests.append(TestResult("T9: Beats threshold baseline", True, 8, 15,
                                           f"Wins {wins}/3 criteria.",
                                           "Improve balance or intra-cluster RSSI."))
        else:
            report.tests.append(TestResult("T9: Beats threshold baseline", False, 0, 15,
                                           f"Wins {wins}/3 criteria.",
                                           "Your algorithm is worse than simple threshold."))
    except Exception as e:
        report.tests.append(TestResult("T9: Beats threshold baseline", False, 0, 15,
                                       f"Error: {e}"))

    # T10: Wearable challenge (15 pts)
    try:
        sim_wear = Simulator(num_nodes=20, area=(0.4, 0.5),
                             gateway_pos=(0.2, 0.0), seed=42)
        result_wear = sim_wear.run(clustering_algo=algo_fn, scheduler="adaptive",
                                   duration=3600, report_interval=5.0)
        pdr_wear = result_wear.metrics.packet_delivery_ratio
        alive_wear = result_wear.metrics.nodes_alive

        if pdr_wear >= 0.90 and alive_wear == 20:
            report.tests.append(TestResult("T10: Wearable challenge", True, 15, 15,
                                           f"PDR={pdr_wear:.1%}, all alive."))
        elif pdr_wear >= 0.75:
            report.tests.append(TestResult("T10: Wearable challenge", True, 8, 15,
                                           f"PDR={pdr_wear:.1%}, alive={alive_wear}.",
                                           "Need PDR≥90% and all nodes alive."))
        else:
            report.tests.append(TestResult("T10: Wearable challenge", False, 0, 15,
                                           f"PDR={pdr_wear:.1%}, alive={alive_wear}.",
                                           "Clustering doesn't suit body-area network."))
    except Exception as e:
        report.tests.append(TestResult("T10: Wearable challenge", False, 0, 15,
                                       f"Error: {e}"))

    return report


# ═════════════════════════════════════════════════════════════════════
#  Scheduler tests
# ═════════════════════════════════════════════════════════════════════

def grade_scheduler(name: str, sched_fn) -> AlgoGradeReport:
    """Run 10 tests on a sleep scheduler."""
    report = AlgoGradeReport(algo_name=name, algo_type="scheduler")

    # Set up test network
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    nodes = deploy_nodes(25, cfg.AREA_WIDTH, cfg.AREA_HEIGHT, rng, "random")
    rssi = build_rssi_matrix([n.position for n in nodes], rng)
    gateway = GatewayNode()

    # Make clusters using LEACH
    from src.clustering import leach_clustering
    clusters = leach_clustering(nodes, rssi, cfg.GATEWAY_POS)

    actions = None

    # T1: Runs without crashing (5 pts)
    try:
        actions = sched_fn(nodes, clusters, 0.0, gateway)
        report.tests.append(TestResult("T1: Runs without error", True, 5, 5))
    except Exception as e:
        report.tests.append(TestResult("T1: Runs without error", False, 0, 5,
                                       f"Error: {e}",
                                       "Make sure your function handles all inputs."))
        for i in range(2, 11):
            report.tests.append(TestResult(f"T{i}: Skipped", False, 0,
                                           [5,10,10,10,10,10,10,15,15][i-2]))
        return report

    # T2: Correct format (5 pts)
    try:
        assert isinstance(actions, dict), "Must return a dict"
        valid_actions = {"sleep", "wake", "active", "idle"}
        for nid, act in actions.items():
            assert act in valid_actions, f"Node {nid}: '{act}' not valid"
        report.tests.append(TestResult("T2: Correct format", True, 5, 5))
    except (AssertionError, Exception) as e:
        report.tests.append(TestResult("T2: Correct format", False, 0, 5,
                                       str(e),
                                       "Return dict[node_id -> 'sleep'|'wake'|'active'|'idle']"))

    # T3: Covers all nodes (10 pts)
    covered = len(set(actions.keys()) & {i for i in range(len(nodes))})
    total = len(nodes)
    if covered >= total:
        report.tests.append(TestResult("T3: All nodes scheduled", True, 10, 10))
    elif covered >= total * 0.8:
        report.tests.append(TestResult("T3: All nodes scheduled", True, 5, 10,
                                       f"{covered}/{total} covered.",
                                       "Some nodes have no schedule."))
    else:
        report.tests.append(TestResult("T3: All nodes scheduled", False, 0, 10,
                                       f"Only {covered}/{total} nodes.",
                                       "Schedule every node."))

    # T4: Some nodes sleep (10 pts)
    sleeping = sum(1 for a in actions.values() if a == "sleep")
    if sleeping > 0:
        report.tests.append(TestResult("T4: Uses sleep mode", True, 10, 10,
                                       f"{sleeping} nodes sleeping."))
    else:
        report.tests.append(TestResult("T4: Uses sleep mode", False, 0, 10,
                                       "No nodes sleeping.",
                                       "Scheduler must put some nodes to sleep to save energy!"))

    # T5: Cluster heads awake (10 pts)
    heads_awake = sum(1 for h in clusters if actions.get(h) in ("active", "idle"))
    total_heads = len(clusters)
    if heads_awake == total_heads:
        report.tests.append(TestResult("T5: Cluster heads awake", True, 10, 10))
    elif heads_awake > 0:
        report.tests.append(TestResult("T5: Cluster heads awake", True, 5, 10,
                                       f"{heads_awake}/{total_heads} heads awake.",
                                       "Cluster heads should generally stay awake."))
    else:
        report.tests.append(TestResult("T5: Cluster heads awake", False, 0, 10,
                                       "All cluster heads sleeping!",
                                       "If heads sleep, no data can be collected."))

    # T6: Time-varying behaviour (10 pts)
    try:
        actions_t0 = sched_fn(nodes, clusters, 0.0, gateway)
        actions_t1 = sched_fn(nodes, clusters, 15.0, gateway)
        differs = sum(1 for i in actions_t0 if actions_t0.get(i) != actions_t1.get(i))
        if differs > 0:
            report.tests.append(TestResult("T6: Time-varying schedule", True, 10, 10,
                                           f"{differs} nodes changed state over time."))
        else:
            report.tests.append(TestResult("T6: Time-varying schedule", False, 0, 10,
                                           "Same schedule at t=0 and t=15.",
                                           "Good schedulers change over time."))
    except Exception as e:
        report.tests.append(TestResult("T6: Time-varying schedule", False, 0, 10,
                                       f"Error: {e}"))

    # T7: Simulation PDR ≥ 60% (10 pts)
    try:
        sim = Simulator(num_nodes=25, seed=cfg.RANDOM_SEED)
        result = sim.run(clustering_algo="leach", scheduler=sched_fn, duration=1800)
        pdr = result.metrics.packet_delivery_ratio
        if pdr >= 0.60:
            report.tests.append(TestResult("T7: PDR ≥ 60%", True, 10, 10,
                                           f"PDR = {pdr:.1%}"))
        elif pdr >= 0.40:
            report.tests.append(TestResult("T7: PDR ≥ 60%", True, 5, 10,
                                           f"PDR = {pdr:.1%}",
                                           "Nodes may sleep during collection windows."))
        else:
            report.tests.append(TestResult("T7: PDR ≥ 60%", False, 0, 10,
                                           f"PDR = {pdr:.1%}",
                                           "Too much sleeping, data can't get through."))
    except Exception as e:
        report.tests.append(TestResult("T7: PDR ≥ 60%", False, 0, 10, f"Error: {e}"))

    # T8: Energy conservation (10 pts)
    try:
        energy_frac = result.metrics.avg_energy_remaining_frac
        if energy_frac >= 0.80:
            report.tests.append(TestResult("T8: Energy conservation", True, 10, 10,
                                           f"Avg energy remaining: {energy_frac:.1%}"))
        elif energy_frac >= 0.60:
            report.tests.append(TestResult("T8: Energy conservation", True, 5, 10,
                                           f"Avg energy: {energy_frac:.1%}",
                                           "Good but room for improvement."))
        else:
            report.tests.append(TestResult("T8: Energy conservation", False, 0, 10,
                                           f"Avg energy: {energy_frac:.1%}",
                                           "Nodes draining too fast."))
    except Exception as e:
        report.tests.append(TestResult("T8: Energy conservation", False, 0, 10,
                                       f"Error: {e}"))

    # T9: Beats periodic baseline (15 pts)
    try:
        sim_base = Simulator(num_nodes=25, seed=cfg.RANDOM_SEED)
        result_base = sim_base.run(clustering_algo="leach", scheduler="periodic",
                                   duration=1800)
        base_eff = result_base.metrics.energy_efficiency
        stud_eff = result.metrics.energy_efficiency
        base_pdr = result_base.metrics.packet_delivery_ratio
        stud_pdr = result.metrics.packet_delivery_ratio

        wins = 0
        if stud_eff >= base_eff * 0.9:
            wins += 1
        if stud_pdr >= base_pdr * 0.9:
            wins += 1
        if result.metrics.avg_energy_remaining_frac > result_base.metrics.avg_energy_remaining_frac:
            wins += 1

        if wins >= 2:
            report.tests.append(TestResult("T9: Beats periodic baseline", True, 15, 15,
                                           f"Wins {wins}/3 criteria."))
        elif wins == 1:
            report.tests.append(TestResult("T9: Beats periodic baseline", True, 8, 15,
                                           f"Wins {wins}/3.",
                                           "Improve energy or PDR."))
        else:
            report.tests.append(TestResult("T9: Beats periodic baseline", False, 0, 15,
                                           "Worse than periodic.",
                                           "Simple periodic duty-cycle is beating you."))
    except Exception as e:
        report.tests.append(TestResult("T9: Beats periodic baseline", False, 0, 15,
                                       f"Error: {e}"))

    # T10: Long-run endurance (8 hours) (15 pts)
    try:
        sim_long = Simulator(num_nodes=20, seed=42)
        result_long = sim_long.run(clustering_algo="leach", scheduler=sched_fn,
                                   duration=8 * 3600)
        alive_long = result_long.metrics.nodes_alive
        pdr_long = result_long.metrics.packet_delivery_ratio

        if alive_long == 20 and pdr_long >= 0.80:
            report.tests.append(TestResult("T10: 8-hour endurance", True, 15, 15,
                                           f"All alive, PDR={pdr_long:.1%}."))
        elif alive_long >= 15:
            report.tests.append(TestResult("T10: 8-hour endurance", True, 8, 15,
                                           f"alive={alive_long}/20, PDR={pdr_long:.1%}",
                                           "Need all 20 alive with PDR≥80%."))
        else:
            report.tests.append(TestResult("T10: 8-hour endurance", False, 0, 15,
                                           f"alive={alive_long}/20.",
                                           "Too many nodes died over 8 hours."))
    except Exception as e:
        report.tests.append(TestResult("T10: 8-hour endurance", False, 0, 15,
                                       f"Error: {e}"))

    return report


# ═════════════════════════════════════════════════════════════════════
#  Printing
# ═════════════════════════════════════════════════════════════════════

def print_report(report: AlgoGradeReport):
    """Pretty-print a grading report."""
    w = 70
    print(f"\n+{'─' * w}+")
    print(f"│{'':^{w}}│")
    label = f"{report.algo_name}  ({report.algo_type})"
    print(f"│{label:^{w}}│")
    print(f"│{'':^{w}}│")
    print(f"+{'─' * w}+")

    for t in report.tests:
        icon = "✓" if t.passed else "✗"
        pts = f"[{t.points}/{t.max_points}]"
        line = f"  {icon} {t.name:42s} {pts:>10s}"
        print(f"│{line:<{w}}│")
        if t.detail:
            det = f"     {t.detail}"
            print(f"│{det:<{w}}│")
        if not t.passed and t.hint:
            hint = f"     Hint: {t.hint}"
            # Wrap long hints
            while len(hint) > w:
                print(f"│{hint[:w]}│")
                hint = "     " + hint[w:]
            print(f"│{hint:<{w}}│")

    print(f"+{'─' * w}+")
    verdict = "PASSED" if report.passed else "NOT PASSED"
    icon = "✓" if report.passed else "✗"
    score_line = f"  Score: {report.score} / {report.max_score}  ({report.percentage:.1f}%)"
    verdict_line = f"  {icon} OVERALL: {verdict}"
    print(f"│{score_line:<{w}}│")
    print(f"│{verdict_line:<{w}}│")
    print(f"+{'─' * w}+")


def print_scorecard(reports: list[AlgoGradeReport]):
    """Print final scorecard."""
    w = 70
    print(f"\n+{'═' * w}+")
    print(f"│{'FINAL SCORECARD':^{w}}│")
    print(f"+{'═' * w}+")

    n_passed = 0
    n_nontrivial_clustering = 0
    n_nontrivial_scheduler = 0

    for r in reports:
        icon = "✓" if r.passed else "✗"
        status = "PASSED" if r.passed else "NOT PASSED"
        line = f"  {icon} {r.algo_name:30s} {r.score:3d}/{r.max_score}  ({r.percentage:5.1f}%)  {status}"
        print(f"│{line:<{w}}│")
        if r.passed:
            n_passed += 1
        if r.score > 20:  # non-trivial
            if r.algo_type == "clustering":
                n_nontrivial_clustering += 1
            else:
                n_nontrivial_scheduler += 1

    print(f"+{'─' * w}+")

    # Breadth check: need at least 1 non-trivial clustering + 1 scheduler
    clust_ok = n_nontrivial_clustering >= 1
    sched_ok = n_nontrivial_scheduler >= 1
    clust_icon = "✓" if clust_ok else "✗"
    sched_icon = "✓" if sched_ok else "✗"
    print(f"│  {clust_icon} Clustering: {n_nontrivial_clustering} non-trivial (need ≥ 1){'':<{w-52}}│")
    print(f"│  {sched_icon} Scheduler:  {n_nontrivial_scheduler} non-trivial (need ≥ 1){'':<{w-52}}│")

    overall = clust_ok and sched_ok and n_passed >= 2
    overall_str = "PASSED" if overall else "NOT YET PASSED"
    overall_icon = "✓" if overall else "✗"
    print(f"+{'═' * w}+")
    verdict = f"  {overall_icon} FINAL VERDICT:  {overall_str}"
    print(f"│{verdict:<{w}}│")
    print(f"+{'═' * w}+\n")


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Project 4 — Distributed Bioimpedance Mesh — AUTO-GRADER")
    print("=" * 70)

    reports: list[AlgoGradeReport] = []

    # Import student clustering algorithms
    print("\n  Loading student clustering algorithms...")
    try:
        from student_work.my_clustering import MY_CLUSTERING_ALGOS
        print(f"  Found: {list(MY_CLUSTERING_ALGOS.keys())}")
        for name, fn in MY_CLUSTERING_ALGOS.items():
            print(f"\n  Grading clustering: {name}...")
            report = grade_clustering_algorithm(name, fn)
            reports.append(report)
            print_report(report)
    except ImportError as e:
        print(f"  ERROR: Could not import student_work.my_clustering: {e}")

    # Import student schedulers
    print("\n  Loading student sleep schedulers...")
    try:
        from student_work.my_sleep_scheduler import MY_SCHEDULER_ALGOS
        print(f"  Found: {list(MY_SCHEDULER_ALGOS.keys())}")
        for name, fn in MY_SCHEDULER_ALGOS.items():
            print(f"\n  Grading scheduler: {name}...")
            report = grade_scheduler(name, fn)
            reports.append(report)
            print_report(report)
    except ImportError as e:
        print(f"  ERROR: Could not import student_work.my_sleep_scheduler: {e}")

    # Final scorecard
    print_scorecard(reports)


if __name__ == "__main__":
    main()
