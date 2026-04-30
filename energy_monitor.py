import os
import psutil
import time
import csv
import threading
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS

# ── Configuration ────────────────────────────────────────────────────────────
LOG_FILE = "energy_data_log.csv"
CAPTURE_INTERVAL = 5        # Seconds between each CSV log entry
API_PORT = 5050             # Port the Flask server listens on

# ── Shared state (updated by background thread) ───────────────────────────────
latest_metrics = {}
metrics_lock = threading.Lock()

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow the HTML page (file://) to call the API


@app.route("/", methods=["GET"])
def serve_ui():
    """Serve the SCEMARS website so it runs on the same origin as the API."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scemars_website3.html")
    return send_file(html_path)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return a fresh system-metrics snapshot as JSON."""
    return jsonify(get_system_metrics())


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return up to the last 12 CSV rows as JSON (for the bar chart)."""
    rows = []
    try:
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-12:]          # last 12 entries
    except FileNotFoundError:
        pass
    return jsonify(rows)


# ── Metric collection helpers ─────────────────────────────────────────────────

def _estimate_wattage(cpu_pct, ram_pct):
    """
    Very rough wattage estimate for demo purposes.
    CPU baseline ~15W idle, scales up to ~65W at 100%.
    GPU is approximated from CPU load as a proxy.
    Total adds ~10W base (RAM, disk, fans, etc.).
    """
    cpu_w  = 15 + (cpu_pct / 100) * 50
    gpu_w  = 5  + (cpu_pct / 100) * 25   # proxy
    base_w = 10 + (ram_pct / 100) * 5
    total  = cpu_w + gpu_w + base_w
    return round(total, 1), round(cpu_w, 1), round(gpu_w, 1)


def _efficiency_score(total_w):
    """Map total wattage to a simple letter grade."""
    if total_w < 40:  return "A+"
    if total_w < 55:  return "A"
    if total_w < 70:  return "B+"
    if total_w < 85:  return "B"
    if total_w < 100: return "C"
    return "D"


def get_system_metrics():
    """Captures CPU, RAM, Battery, top application, and estimated wattage."""
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
            if c is None:
                continue
            if c > max_cpu:
                max_cpu = c
                top_process_name = proc.info["name"]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    total_w, cpu_w, gpu_w = _estimate_wattage(cpu_percent, ram_percent)

    return {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_usage_pct":   cpu_percent,
        "ram_usage_pct":   ram_percent,
        "battery_pct":     battery_percent,
        "is_charging":     power_plugged,
        "top_application": top_process_name,
        # Derived display values consumed by the website
        "total_draw_w":    total_w,
        "cpu_power_w":     cpu_w,
        "gpu_power_w":     gpu_w,
        "efficiency_score": _efficiency_score(total_w),
    }


# ── CSV logging ──────────────────────────────────────────────────────────────

def log_to_csv(data):
    """Saves the captured metrics to a CSV file for future ML analysis."""
    # Only log the original fields to keep the CSV clean
    csv_fields = [
        "timestamp", "cpu_usage_pct", "ram_usage_pct",
        "battery_pct", "is_charging", "top_application",
        "total_draw_w", "cpu_power_w", "gpu_power_w", "efficiency_score",
    ]
    row = {k: data[k] for k in csv_fields}

    file_exists = False
    try:
        with open(LOG_FILE, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Background monitoring thread ──────────────────────────────────────────────

def monitoring_loop():
    """Runs forever: collect → store in shared dict → log to CSV → sleep."""
    global latest_metrics
    print(f"[monitor] Starting. Logging to {LOG_FILE} every {CAPTURE_INTERVAL}s ...")
    psutil.cpu_percent(interval=1)   # prime the counter

    while True:
        metrics = get_system_metrics()
        with metrics_lock:
            latest_metrics = metrics
        log_to_csv(metrics)
        print(
            f"[{metrics['timestamp']}] "
            f"CPU {metrics['cpu_usage_pct']}% | "
            f"RAM {metrics['ram_usage_pct']}% | "
            f"Total ~{metrics['total_draw_w']}W | "
            f"Top: {metrics['top_application']}"
        )
        time.sleep(CAPTURE_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start monitoring in a daemon thread so it doesn't block Flask
    t = threading.Thread(target=monitoring_loop, daemon=True)
    t.start()

    print(f"[api] SCEMARS running — open http://127.0.0.1:{API_PORT} in your browser")
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
