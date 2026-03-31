import psutil
import time
import csv
from datetime import datetime

# Configuration
LOG_FILE = "energy_data_log.csv"
CAPTURE_INTERVAL = 10  # Seconds between each log

def get_system_metrics():
    """Captures CPU, RAM, Battery, and the most active application."""
    # CPU Usage percentage (system-wide)
    cpu_percent = psutil.cpu_percent(interval=None)
    
    # RAM Usage percentage
    memory = psutil.virtual_memory()
    ram_percent = memory.percent
    
    # Battery information (if available)
    battery = psutil.sensors_battery()
    battery_percent = battery.percent if battery else "N/A"
    power_plugged = battery.power_plugged if battery else "N/A"
    
    # Identify the 'Top App' by CPU consumption
    top_process_name = "System"
    max_cpu = 0
    for proc in psutil.process_iter(['name', 'cpu_percent']):
        try:
            # Note: The first call to cpu_percent might be 0; 
            # tracking over time is more accurate.
            current_cpu = proc.info['cpu_percent']
            if current_cpu > max_cpu:
                max_cpu = current_cpu
                top_process_name = proc.info['name']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_usage_pct": cpu_percent,
        "ram_usage_pct": ram_percent,
        "battery_pct": battery_percent,
        "is_charging": power_plugged,
        "top_application": top_process_name
    }

def log_to_csv(data):
    """Saves the captured metrics to a CSV file for future ML analysis."""
    file_exists = False
    try:
        with open(LOG_FILE, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def start_monitoring():
    print(f"Monitoring Started. Logging to {LOG_FILE}...")
    # Initial call to psutil.cpu_percent to initialize the counter
    psutil.cpu_percent(interval=1)
    
    try:
        while True:
            metrics = get_system_metrics()
            log_to_csv(metrics)
            print(f"Logged [{metrics['timestamp']}]: {metrics['top_application']} is most active.")
            time.sleep(CAPTURE_INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    start_monitoring()
