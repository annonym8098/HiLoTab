import os
import json
import pandas as pd

root = "results"

rows = []

for dataset in sorted(os.listdir(root)):
    dataset_path = os.path.join(root, dataset)

    if not os.path.isdir(dataset_path):
        continue

    for file in os.listdir(dataset_path):
        if not file.endswith("_results.json"):
            continue

        file_path = os.path.join(dataset_path, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        perf = data["performance"]

        rows.append({
            "Model": data["model"],
            "Dataset": dataset,
            "ACC": (perf["accuracy_mean"], perf["accuracy_std"]),
            "AUC": (perf["auc_mean"], perf["auc_std"]),
            "F1": (perf["f1_mean"], perf["f1_std"]),
        })

df = pd.DataFrame(rows)

def format_metric(col):
    return col.apply(lambda x: f"{x[0]:.2f} ± {x[1]:.2f}")

df["ACC_str"] = format_metric(df["ACC"])
df["AUC_str"] = format_metric(df["AUC"])
df["F1_str"] = format_metric(df["F1"])

acc_df = df[["Model", "Dataset", "ACC_str"]].rename(columns={"ACC_str": "Result"})
auc_df = df[["Model", "Dataset", "AUC_str"]].rename(columns={"AUC_str": "Result"})
f1_df = df[["Model", "Dataset", "F1_str"]].rename(columns={"F1_str": "Result"})

acc_pivot = acc_df.pivot(index="Model", columns="Dataset", values="Result")
auc_pivot = auc_df.pivot(index="Model", columns="Dataset", values="Result")
f1_pivot = f1_df.pivot(index="Model", columns="Dataset", values="Result")

acc_pivot = acc_pivot.reset_index()
auc_pivot = auc_pivot.reset_index()
f1_pivot = f1_pivot.reset_index()

def reorder(df):
    cols = ["Model"] + sorted([c for c in df.columns if c != "Model"])
    return df[cols]

acc_pivot = reorder(acc_pivot)
auc_pivot = reorder(auc_pivot)
f1_pivot = reorder(f1_pivot)

print("\n=== ACC TABLE ===")
print(acc_pivot)

print("\n=== AUC TABLE ===")
print(auc_pivot)

print("\n=== F1 TABLE ===")
print(f1_pivot)

acc_pivot.to_csv("acc_table.csv", index=False)
auc_pivot.to_csv("auc_table.csv", index=False)
f1_pivot.to_csv("f1_table.csv", index=False)

print("\nSaved: acc_table.csv, auc_table.csv, f1_table.csv")