import json
import os

RESULTS_FILE = "src/experiment/benchmark_results_v2.json"

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
    
    # Remove the last entry if it is 15d LNS (regardless of status, but we expect TIMEOUT)
    if data and data[-1]["scenario"] == "15d" and data[-1]["mode"] == "LNS":
        print(f"Removing result: {data[-1]}")
        data.pop()
        
        with open(RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print("Cleaned results file.")
    else:
        print("Last entry does not match expected retry target. Doing nothing.")
else:
    print("Results file not found.")
