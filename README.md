# SCEMARS — Smart Computer Energy Monitor & Recommendation System

Real-time energy monitoring dashboard that tracks your CPU, RAM, battery, and top running process - displayed in a live web interface.

---

## Requirements

- **Python 3.8+** (any modern version works)
- A terminal (macOS Terminal, Windows CMD/PowerShell, Linux bash)

---

## Setup & Run (3 steps)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend server

```bash
python energy_monitor.py
```

You should see:
```
[monitor] Starting. Logging to energy_data_log.csv every 5s ...
[api] SCEMARS running - open http://127.0.0.1:5050 in your browser
 * Running on http://127.0.0.1:5050
```

### 3. Open the dashboard

Open your browser and go to:

```
http://127.0.0.1:5050
```

The dashboard will show **live metrics** from your machine, updating every 2 seconds.

---

## What it does

| Feature | Details |
|---|---|
| **Live Dashboard** | Total wattage, CPU power, GPU estimate, efficiency grade |
| **Top Process** | Shows which app is consuming the most CPU right now |
| **Battery Status** | Charge level + charging indicator |
| **History Chart** | Bar chart of the last 12 logged readings |
| **CSV Logging** | All readings saved to `energy_data_log.csv` for analysis |

---

## Files

```
ASEProject/
├── energy_monitor.py      # Backend — Flask API + psutil data collection
├── scemars_website.html   # Frontend — live dashboard UI
├── requirements.txt       # Python dependencies
└── energy_data_log.csv    # Auto-created when you run the server
```

---

## Stopping the server

Press `Ctrl+C` in the terminal where `energy_monitor.py` is running.
