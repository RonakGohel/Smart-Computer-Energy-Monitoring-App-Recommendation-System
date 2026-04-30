"""
energy_monitor.py  ─  SCEMARS backend  (ML-enhanced)
======================================================
Adds two new endpoints to the original Flask server:
  GET /api/ml/report      → full ML analysis report (cached 30s)
  GET /api/ml/live        → quick live recommendation for current process

Drop ml_agent.py next to this file, then run:
    python energy_monitor.py
"""

import os
import psutil
import time
import csv
import json
import threading
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS

# ── ML Agent ─────────────────────────────────────────────────────────────────
try:
    from ml_agent import SCEMARSAgent
    _agent = SCEMARSAgent("energy_data_log.csv")
    _ml_available = True
    print("[ml]  SCEMARSAgent loaded successfully.")
except ImportError:
    _ml_available = False
    print("[ml]  ml_agent.py not found — ML endpoints disabled.")

_ml_cache: dict = {}
_ml_cache_time: float = 0.0
_ML_CACHE_TTL = 30  # seconds


# ── Configuration ─────────────────────────────────────────────────────────────
LOG_FILE        = "energy_data_log.csv"
CAPTURE_INTERVAL = 5
API_PORT         = 5050

latest_metrics = {}
metrics_lock   = threading.Lock()

app = Flask(__name__)
CORS(app)


# ── Existing routes ───────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def serve_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scemars_ml_dashboard.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scemars_website3.html")
    return send_file(html_path)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(get_system_metrics())


@app.route("/api/history", methods=["GET"])
def get_history():
    rows = []
    try:
        with open(LOG_FILE, "r", newline="") as f:
            rows = list(csv.DictReader(f))[-12:]
    except FileNotFoundError:
        pass
    return jsonify(rows)


# ── ML routes ─────────────────────────────────────────────────────────────────

@app.route("/api/ml/report", methods=["GET"])
def ml_report():
    """Full ML analysis — cached for 30s to avoid re-reading CSV on every poll."""
    global _ml_cache, _ml_cache_time
    if not _ml_available:
        return jsonify({"error": "ML agent not available"}), 503

    now = time.time()
    if now - _ml_cache_time > _ML_CACHE_TTL:
        _ml_cache      = _agent.run()
        _ml_cache_time = now

    return jsonify(_ml_cache)


@app.route("/api/ml/live", methods=["GET"])
def ml_live():
    """Fast per-request recommendation for whatever process is hottest right now."""
    if not _ml_available:
        return jsonify({"error": "ML agent not available"}), 503

    m   = get_system_metrics()
    rec = _agent.get_live_recommendation(
        m["top_application"], m["cpu_usage_pct"], m["total_draw_w"]
    )
    rec["current_watts"] = m["total_draw_w"]
    rec["current_cpu"]   = m["cpu_usage_pct"]
    return jsonify(rec)


# ── Metric collection ─────────────────────────────────────────────────────────

def _estimate_wattage(cpu_pct, ram_pct):
    cpu_w  = 15 + (cpu_pct / 100) * 50
    gpu_w  = 5  + (cpu_pct / 100) * 25
    base_w = 10 + (ram_pct / 100) * 5
    return round(cpu_w + gpu_w + base_w, 1), round(cpu_w, 1), round(gpu_w, 1)


def _efficiency_score(total_w):
    if total_w < 40:  return "A+"
    if total_w < 55:  return "A"
    if total_w < 70:  return "B+"
    if total_w < 85:  return "B"
    if total_w < 100: return "C"
    return "D"


def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=None)
    memory      = psutil.virtual_memory()
    ram_percent = memory.percent

    battery_percent = "N/A"
    power_plugged   = "N/A"
    try:
        battery = psutil.sensors_battery()
        if battery:
            battery_percent = battery.percent
            power_plugged   = battery.power_plugged
    except Exception:
        pass

    top_process_name = "System"
    max_cpu = 0
    for proc in psutil.process_iter(["name", "cpu_percent"]):
        try:
            c = proc.info["cpu_percent"]
            if c and c > max_cpu:
                max_cpu = c
                top_process_name = proc.info["name"]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    total_w, cpu_w, gpu_w = _estimate_wattage(cpu_percent, ram_percent)
    return {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_usage_pct":    cpu_percent,
        "ram_usage_pct":    ram_percent,
        "battery_pct":      battery_percent,
        "is_charging":      power_plugged,
        "top_application":  top_process_name,
        "total_draw_w":     total_w,
        "cpu_power_w":      cpu_w,
        "gpu_power_w":      gpu_w,
        "efficiency_score": _efficiency_score(total_w),
    }


def log_to_csv(data):
    csv_fields = [
        "timestamp", "cpu_usage_pct", "ram_usage_pct",
        "battery_pct", "is_charging", "top_application",
        "total_draw_w", "cpu_power_w", "gpu_power_w", "efficiency_score",
    ]
    row        = {k: data[k] for k in csv_fields}
    exists     = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def monitoring_loop():
    print(f"[monitor] Starting. Logging to {LOG_FILE} every {CAPTURE_INTERVAL}s …")
    psutil.cpu_percent(interval=1)
    while True:
        metrics = get_system_metrics()
        with metrics_lock:
            latest_metrics.update(metrics)
        log_to_csv(metrics)
        print(
            f"[{metrics['timestamp']}] "
            f"CPU {metrics['cpu_usage_pct']}% | "
            f"RAM {metrics['ram_usage_pct']}% | "
            f"~{metrics['total_draw_w']}W | "
            f"Top: {metrics['top_application']}"
        )
        time.sleep(CAPTURE_INTERVAL)


if __name__ == "__main__":
    t = threading.Thread(target=monitoring_loop, daemon=True)
    t.start()
    print(f"[api] SCEMARS running → http://127.0.0.1:{API_PORT}")
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
