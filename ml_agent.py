"""
SCEMARS ML Agent — ml_agent.py
================================
Drop-in module for the existing energy_monitor.py + Flask stack.

Architecture
------------
  ProcessProfiler   – builds per-process statistical profiles from CSV history
  AnomalyDetector   – flags sessions that deviate from the learned baseline
  RecommendationEngine – maps profiles → actionable advice (rule-augmented ML)
  SCEMARSAgent      – top-level class that ties everything together

Run standalone:  python ml_agent.py
Import from Flask:
    from ml_agent import SCEMARSAgent
    agent = SCEMARSAgent("energy_data_log.csv")
    report = agent.run()   # returns JSON-serialisable dict
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any


# ── Lightweight ML primitives (no sklearn dependency) ────────────────────────

class RunningStats:
    """Welford online mean / variance — works incrementally as new rows arrive."""
    __slots__ = ("n", "mean", "_M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._M2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self._M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, x: float) -> float:
        return (x - self.mean) / self.std if self.std > 0 else 0.0


class KMeans1D:
    """
    Minimal 1-D k-means (k=3) for segmenting processes into
    LOW / MEDIUM / HIGH load tiers without any external library.
    """
    def __init__(self, k: int = 3, max_iter: int = 50):
        self.k = k
        self.max_iter = max_iter
        self.centroids: list[float] = []
        self.labels: list[int] = []

    def fit(self, data: list[float]) -> "KMeans1D":
        if not data:
            return self
        mn, mx = min(data), max(data)
        # Evenly spaced initial centroids
        span = mx - mn
        self.centroids = [mn + span * i / (self.k - 1) for i in range(self.k)] if span > 0 else [mn] * self.k

        for _ in range(self.max_iter):
            clusters: list[list[float]] = [[] for _ in range(self.k)]
            self.labels = []
            for x in data:
                idx = min(range(self.k), key=lambda i: abs(x - self.centroids[i]))
                clusters[idx].append(x)
                self.labels.append(idx)
            new_centroids = [
                statistics.mean(c) if c else self.centroids[i]
                for i, c in enumerate(clusters)
            ]
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

        # Re-label so 0 = lowest centroid, k-1 = highest
        order = sorted(range(self.k), key=lambda i: self.centroids[i])
        remap = {old: new for new, old in enumerate(order)}
        self.labels = [remap[l] for l in self.labels]
        self.centroids = sorted(self.centroids)
        return self

    def predict(self, x: float) -> int:
        if not self.centroids:
            return 0
        return min(range(self.k), key=lambda i: abs(x - self.centroids[i]))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    row["_cpu"]   = float(row.get("cpu_usage_pct", 0))
                    row["_ram"]   = float(row.get("ram_usage_pct", 0))
                    row["_watts"] = float(row.get("total_draw_w", 0))
                    row["_ts"]    = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    rows.append(row)
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return rows


# ── ProcessProfiler ───────────────────────────────────────────────────────────

class ProcessProfile:
    """Accumulated statistics for one process name."""
    def __init__(self, name: str):
        self.name = name
        self.appearances = 0          # how many 5-s ticks it was #1
        self.cpu_stats  = RunningStats()
        self.ram_stats  = RunningStats()
        self.w_stats    = RunningStats()
        self.sessions: list[datetime] = []   # timestamps it appeared
        self.run_streaks: list[int] = []     # consecutive-tick streaks
        self._streak = 0

    def record(self, cpu: float, ram: float, watts: float, ts: datetime, is_consecutive: bool):
        self.appearances += 1
        self.cpu_stats.update(cpu)
        self.ram_stats.update(ram)
        self.w_stats.update(watts)
        self.sessions.append(ts)
        if is_consecutive:
            self._streak += 1
        else:
            if self._streak:
                self.run_streaks.append(self._streak)
            self._streak = 1

    def finalise(self):
        if self._streak:
            self.run_streaks.append(self._streak)

    @property
    def avg_cpu(self) -> float:
        return self.cpu_stats.mean

    @property
    def avg_watts(self) -> float:
        return self.w_stats.mean

    @property
    def max_streak(self) -> int:
        return max(self.run_streaks, default=0)

    @property
    def avg_streak(self) -> float:
        return statistics.mean(self.run_streaks) if self.run_streaks else 0.0

    @property
    def persistence_score(self) -> float:
        """
        0-100 score combining frequency + streaks.
        High score = process that hogs the top slot for long runs.
        """
        freq_norm = min(self.appearances / 100, 1.0)
        streak_norm = min(self.avg_streak / 50, 1.0)
        return round((freq_norm * 0.5 + streak_norm * 0.5) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "appearances": self.appearances,
            "avg_cpu_pct": round(self.avg_cpu, 1),
            "avg_watts":   round(self.avg_watts, 1),
            "max_streak_ticks": self.max_streak,
            "avg_streak_ticks": round(self.avg_streak, 1),
            "persistence_score": self.persistence_score,
        }


class ProcessProfiler:
    def __init__(self):
        self.profiles: dict[str, ProcessProfile] = {}
        self._prev_app: str | None = None
        self._prev_ts: datetime | None = None
        self.TICK_GAP = timedelta(seconds=12)  # 2× capture interval = consecutive

    def ingest(self, rows: list[dict]) -> "ProcessProfiler":
        for row in sorted(rows, key=lambda r: r["_ts"]):
            app = row.get("top_application", "Unknown")
            ts  = row["_ts"]
            if app not in self.profiles:
                self.profiles[app] = ProcessProfile(app)

            is_consec = (
                self._prev_app == app
                and self._prev_ts is not None
                and (ts - self._prev_ts) <= self.TICK_GAP
            )
            self.profiles[app].record(row["_cpu"], row["_ram"], row["_watts"], ts, is_consec)
            self._prev_app = app
            self._prev_ts  = ts

        for p in self.profiles.values():
            p.finalise()
        return self

    def ranked(self, by: str = "avg_watts", top_n: int = 15) -> list[ProcessProfile]:
        return sorted(self.profiles.values(), key=lambda p: getattr(p, by), reverse=True)[:top_n]


# ── AnomalyDetector ───────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Builds a system-level baseline (rolling mean/std of total_draw_w),
    then flags rows where wattage is > threshold standard deviations above mean.
    Also detects bursty processes using per-process z-scores.
    """
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
        self.system_stats = RunningStats()
        self.anomalies: list[dict] = []

    def fit_predict(self, rows: list[dict], profiles: dict[str, ProcessProfile]) -> "AnomalyDetector":
        # First pass: build baseline
        for row in rows:
            self.system_stats.update(row["_watts"])

        # Second pass: flag anomalies
        for row in rows:
            z = self.system_stats.z_score(row["_watts"])
            if z > self.z_threshold:
                app = row.get("top_application", "Unknown")
                per_proc_z = profiles[app].w_stats.z_score(row["_watts"]) if app in profiles else 0.0
                self.anomalies.append({
                    "timestamp":   row["timestamp"],
                    "process":     app,
                    "watts":       row["_watts"],
                    "system_z":    round(z, 2),
                    "proc_z":      round(per_proc_z, 2),
                    "efficiency":  row.get("efficiency_score", "?"),
                })
        return self

    def summary(self) -> dict:
        return {
            "baseline_mean_w":  round(self.system_stats.mean, 1),
            "baseline_std_w":   round(self.system_stats.std, 1),
            "anomaly_count":    len(self.anomalies),
            "anomalies":        self.anomalies[:20],  # cap for API response
        }


# ── RecommendationEngine ──────────────────────────────────────────────────────

# Static knowledge base: maps known process names → recommendation metadata
KNOWN_HEAVYWEIGHTS = {
    # browsers
    "Google Chrome":         {"alt": "Firefox or Brave (same tab, lower idle draw)", "save_w": 8,  "category": "Browser"},
    "Google Chrome Helper":  {"alt": "close background tabs or use Tab Suspender extension", "save_w": 5, "category": "Browser"},
    "brave.exe":             {"alt": "Brave is already efficient — check extension list", "save_w": 3, "category": "Browser"},
    "msedgewebview2.exe":    {"alt": "reduce apps using Edge WebView (often Teams, VS Code)", "save_w": 6, "category": "System Helper"},
    # editors / dev
    "Code.exe":              {"alt": "use lightweight profiles, disable LSP when not coding", "save_w": 10, "category": "IDE"},
    "language_server_macos_arm": {"alt": "restart or disable LSP idle polling in VS Code settings", "save_w": 7, "category": "IDE Helper"},
    # system
    "dwm.exe":               {"alt": "reduce display refresh rate or disable transparency effects", "save_w": 5, "category": "OS Process"},
    "SearchApp.exe":         {"alt": "disable Windows Search indexing during battery sessions", "save_w": 4, "category": "OS Process"},
    "dllhost.exe":           {"alt": "check for rogue COM Surrogate instances in Task Manager", "save_w": 4, "category": "OS Process"},
    # media / helpers
    "Antigravity Helper (Renderer)": {"alt": "close Codeshot / Antigravity when not actively using it", "save_w": 6, "category": "App Helper"},
    "com.apple.WebKit.WebContent":   {"alt": "limit Safari tabs or use Safari's Low Power Mode", "save_w": 5, "category": "Browser Helper"},
    # python
    "python.exe":            {"alt": "profile script with cProfile — likely a hot loop; optimise or schedule off-peak", "save_w": 15, "category": "Script"},
    "python3.13":            {"alt": "same as python.exe — hot loop candidate", "save_w": 12, "category": "Script"},
}

TIER_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

EFFICIENCY_ADVICE = {
    "D": "Critical — system is under severe load. Close non-essential apps immediately.",
    "C": "Poor — consider closing background processes and heavy apps.",
    "B": "Below average — a few heavy processes are dragging efficiency down.",
    "B+": "Moderate — some optimisation headroom remains.",
    "A": "Good — system is running efficiently. Minor gains possible.",
    "A+": "Excellent — no immediate action needed.",
}


class RecommendationEngine:
    def __init__(self, profiles: dict[str, ProcessProfile], anomaly_summary: dict):
        self.profiles = profiles
        self.anomaly_summary = anomaly_summary
        self.recommendations: list[dict] = []
        self._tiers: dict[str, str] = {}

    def _assign_tiers(self):
        names = list(self.profiles.keys())
        watts = [self.profiles[n].avg_watts for n in names]
        if len(watts) >= 3:
            km = KMeans1D(k=3).fit(watts)
            for name, label in zip(names, km.labels):
                self._tiers[name] = TIER_LABELS[label]
        else:
            for name in names:
                self._tiers[name] = "LOW"

    def generate(self) -> list[dict]:
        self._assign_tiers()
        seen = set()

        for profile in self.profiles.values():
            if profile.name in seen:
                continue
            seen.add(profile.name)

            tier   = self._tiers.get(profile.name, "LOW")
            kb     = KNOWN_HEAVYWEIGHTS.get(profile.name, {})
            reason = []
            priority = 0

            # Rule 1: high wattage tier
            if tier == "HIGH":
                reason.append(f"Consistently in the HIGH energy tier (avg {profile.avg_watts:.0f}W while dominant).")
                priority += 3

            # Rule 2: frequent + long-running
            if profile.appearances > 30:
                reason.append(f"Top process for {profile.appearances} ticks (~{profile.appearances * 5 // 60} min recorded).")
                priority += 2
            if profile.avg_streak > 20:
                reason.append(f"Runs continuously — avg streak of {profile.avg_streak:.0f} ticks without yielding.")
                priority += 2

            # Rule 3: high CPU
            if profile.avg_cpu > 60:
                reason.append(f"High average CPU: {profile.avg_cpu:.0f}%.")
                priority += 2
            elif profile.avg_cpu > 30:
                reason.append(f"Moderate CPU usage: {profile.avg_cpu:.0f}%.")
                priority += 1

            # Rule 4: known heavyweight
            if kb:
                reason.append(f"Known energy-intensive process category: {kb['category']}.")
                priority += 1

            if not reason:
                continue  # don't recommend for benign processes

            rec = {
                "process":          profile.name,
                "energy_tier":      tier,
                "avg_watts":        round(profile.avg_watts, 1),
                "avg_cpu_pct":      round(profile.avg_cpu, 1),
                "appearances":      profile.appearances,
                "persistence_score": profile.persistence_score,
                "priority":         priority,
                "reasons":          reason,
                "action":           kb.get("alt", "Investigate with Task Manager / Activity Monitor and close if not needed."),
                "estimated_saving_w": kb.get("save_w", max(1, round(profile.avg_watts * 0.15))),
                "category":         kb.get("category", "Unknown"),
            }
            self.recommendations.append(rec)

        # Sort: priority desc, then persistence desc
        self.recommendations.sort(key=lambda r: (-r["priority"], -r["persistence_score"]))
        return self.recommendations


# ── SCEMARSAgent ──────────────────────────────────────────────────────────────

class SCEMARSAgent:
    """
    Main agent. Call run() to get a full JSON-serialisable report.
    Call incremental_update(new_row) to feed live data without reloading.
    """
    def __init__(self, csv_path: str = "energy_data_log.csv"):
        self.csv_path = csv_path
        self._last_mtime: float = 0.0
        self._rows: list[dict] = []
        self.profiler    = ProcessProfiler()
        self.detector    = AnomalyDetector()
        self.recommender: RecommendationEngine | None = None

    # ── Internal ──

    def _reload_if_stale(self):
        try:
            mtime = os.path.getmtime(self.csv_path)
        except FileNotFoundError:
            return
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            self._rows = load_csv(self.csv_path)
            self.profiler  = ProcessProfiler().ingest(self._rows)
            self.detector  = AnomalyDetector().fit_predict(self._rows, self.profiler.profiles)
            self.recommender = RecommendationEngine(self.profiler.profiles, self.detector.summary())

    # ── Public API ──

    def run(self) -> dict:
        """Full analysis pass. Returns JSON-serialisable report."""
        self._reload_if_stale()

        if not self._rows:
            return {"error": "No data found. Start the monitor first."}

        recs = self.recommender.generate() if self.recommender else []
        top_procs = [p.to_dict() for p in self.profiler.ranked(by="avg_watts", top_n=10)]

        # Session-level stats
        watts_all = [r["_watts"] for r in self._rows]
        cpu_all   = [r["_cpu"]   for r in self._rows]
        ts_list   = [r["_ts"]    for r in self._rows]
        eff_scores = Counter(r.get("efficiency_score", "?") for r in self._rows)

        # Estimate potential savings
        total_saving = sum(r["estimated_saving_w"] for r in recs[:5])

        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(self._rows),
            "session_span_minutes": round((max(ts_list) - min(ts_list)).total_seconds() / 60, 1) if len(ts_list) > 1 else 0,
            "system_summary": {
                "avg_total_draw_w":  round(statistics.mean(watts_all), 1),
                "max_total_draw_w":  round(max(watts_all), 1),
                "min_total_draw_w":  round(min(watts_all), 1),
                "avg_cpu_pct":       round(statistics.mean(cpu_all), 1),
                "efficiency_distribution": dict(eff_scores),
                "dominant_process":  self.profiler.ranked(by="appearances")[0].name if self.profiler.profiles else "N/A",
            },
            "anomaly_summary": self.detector.summary(),
            "top_processes": top_procs,
            "recommendations": recs[:8],   # top 8 for UI
            "potential_saving_w": total_saving,
            "efficiency_advice": EFFICIENCY_ADVICE,
        }

    def get_live_recommendation(self, process_name: str, cpu: float, watts: float) -> dict:
        """
        Called with a single live reading (from /api/metrics).
        Returns a quick recommendation without full reload.
        """
        kb = KNOWN_HEAVYWEIGHTS.get(process_name, {})
        profile = self.profiler.profiles.get(process_name)

        risk = "LOW"
        if watts > 80 or cpu > 70:
            risk = "HIGH"
        elif watts > 55 or cpu > 40:
            risk = "MEDIUM"

        return {
            "process": process_name,
            "risk_level": risk,
            "action": kb.get("alt", "Monitor — no known optimisation for this process.") if risk != "LOW" else "No action needed.",
            "estimated_saving_w": kb.get("save_w", 0) if risk == "HIGH" else 0,
            "historical_avg_w": round(profile.avg_watts, 1) if profile else None,
        }


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "energy_data_log.csv"
    print(f"\n🔍  SCEMARS ML Agent — analysing {csv_file!r}\n")
    agent = SCEMARSAgent(csv_file)
    report = agent.run()
    print(json.dumps(report, indent=2, default=str))
