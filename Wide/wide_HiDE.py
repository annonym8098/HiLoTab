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

        res = data["resource_usage"]

        peak_gpu_mb = res.get("peak_gpu_mb", 0)
        peak_cpu_mb = res.get("peak_cpu_rss_mb", 0)

        rows.append({
            "Model": data["model"],
            "Dataset": dataset,
            "Train_Time": res["wall_clock_train_time_sec"],
            "Inference_Latency": res["inference_latency_sec_per_sample"],
            "Peak_Memory_MB": peak_gpu_mb + peak_cpu_mb,
        })

df = pd.DataFrame(rows)

df["Train_Time"] = df["Train_Time"].map(lambda x: f"{x:.4f}")
df["Inference_Latency"] = df["Inference_Latency"].map(lambda x: f"{x:.6f}")
df["Peak_Memory_MB"] = df["Peak_Memory_MB"].map(lambda x: f"{x:.2f}")

train_time_df = df[["Model", "Dataset", "Train_Time"]].rename(columns={"Train_Time": "Result"})
inference_df = df[["Model", "Dataset", "Inference_Latency"]].rename(columns={"Inference_Latency": "Result"})
memory_df = df[["Model", "Dataset", "Peak_Memory_MB"]].rename(columns={"Peak_Memory_MB": "Result"})

train_time_pivot = train_time_df.pivot(index="Model", columns="Dataset", values="Result")
inference_pivot = inference_df.pivot(index="Model", columns="Dataset", values="Result")
memory_pivot = memory_df.pivot(index="Model", columns="Dataset", values="Result")

train_time_pivot = train_time_pivot.reset_index()
inference_pivot = inference_pivot.reset_index()
memory_pivot = memory_pivot.reset_index()

def reorder(df):
    cols = ["Model"] + sorted([c for c in df.columns if c != "Model"])
    return df[cols]

train_time_pivot = reorder(train_time_pivot)
inference_pivot = reorder(inference_pivot)
memory_pivot = reorder(memory_pivot)

train_time_pivot.to_csv("train_time_table.csv", index=False)
inference_pivot.to_csv("inference_latency_table.csv", index=False)
memory_pivot.to_csv("peak_memory_table.csv", index=False)

print("Saved:")
print("train_time_table.csv")
print("inference_latency_table.csv")
print("peak_memory_table.csv")