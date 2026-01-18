import os
import pandas as pd
import numpy as np
import json


def process_instance(instance_path, instance_id):

    history_path = os.path.join(instance_path, "lns_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)

    rows = []

    for it in history:

        before = it["objective_before"]
        after = it["objective_after"]

        rows.append({
            "instance_id": instance_id,
            "iteration": it["iteration"],
            "neighborhood_size": len(it["neighborhood"]["periods"]),
            "neighborhood_periods": str(it["neighborhood"]["periods"]),
            "removed_assignments": it["removed_assignments"],
            "objective_before": before,
            "objective_after": after,
            "improvement": before - after,
            "relaxation_level": it["relaxation_level"],
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
    df.to_csv("output/training_f2_dataset.csv", index=False)

    print("Dataset f2 gerado com sucesso!")
