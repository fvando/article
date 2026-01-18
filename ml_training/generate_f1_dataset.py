import os
import json
import pandas as pd
import numpy as np

def load_matrix(path):
    return pd.read_csv(path, header=None).values

def process_instance(instance_path, instance_id):
    need = load_matrix(os.path.join(instance_path, "need.csv")).flatten()
    heur = load_matrix(os.path.join(instance_path, "heuristic_solution.csv"))
    opt = load_matrix(os.path.join(instance_path, "optimal_solution.csv"))

    num_drivers, num_periods = heur.shape

    rows = []

    for d in range(num_drivers):
        for t in range(num_periods):

            assigned_heur = int(heur[d, t])
            assigned_opt = int(opt[d, t])

            diff = assigned_opt - assigned_heur

            current_load = heur[:, t].sum()
            need_gap = need[t] - current_load

            rows.append({
                "instance_id": instance_id,
                "driver": d,
                "period": t,
                "assigned_heuristic": assigned_heur,
                "assigned_optimal": assigned_opt,
                "diff": diff,
                "current_load_period": float(current_load),
                "need_period": int(need[t]),
                "need_gap": float(need_gap),
                "total_workers": int(heur.sum()),
            })

    return rows


if __name__ == "__main__":

    input_root = "datasets"
    all_rows = []

    for instance_id in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, instance_id)
        if not os.path.isdir(path):
            continue

        rows = process_instance(path, instance_id)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv("output/training_f1_dataset.csv", index=False)

    print("Dataset f1 gerado com sucesso!")
