import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
SUCCESS_DIR = os.path.join(RESULTS_DIR, "successful_datasets")
FAILED_DIR = os.path.join(RESULTS_DIR, "failed_datasets")
PLOTS_DIR = os.path.join(RESULTS_DIR, "average_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_plot(fig, filename):
    path = os.path.join(PLOTS_DIR, f"{filename}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.close(fig)


# ---- Successful Datasets Analysis ----
# loading csv
csv_files = []
for root, dirs, files in os.walk(SUCCESS_DIR):
    for file in files:
        if file == "benchmark_result.csv":
            csv_files.append(os.path.join(root, file))

df_all = pd.concat([pd.read_csv(f) for f in csv_files])

#bar plot for successful datasets
avg_scores = df_all.groupby("model")[["roc_auc", "pr_auc"]].mean().reset_index()
fig = plt.figure(figsize=(10,6))
x = np.arange(len(avg_scores))
width = 0.35
plt.bar(x - width/2, avg_scores["roc_auc"], width, label="ROC AUC", color='skyblue')
plt.bar(x + width/2, avg_scores["pr_auc"], width, label="PR AUC", color='salmon')
plt.xticks(x, avg_scores["model"], rotation=45)
plt.ylabel("Average Score")
plt.title("Average ROC AUC and PR AUC per Model")
plt.legend()
plt.tight_layout()
save_plot(fig, "avg_roc_pr_auc_per_model")


#scatter plot roc auc vs pr auc
fig = plt.figure(figsize=(7, 7))
plt.scatter(avg_scores["roc_auc"], avg_scores["pr_auc"], s=100)

for i, txt in enumerate(avg_scores["model"]):
    plt.annotate(txt, (avg_scores["roc_auc"][i]+0.005, avg_scores["pr_auc"][i]+0.005))

plt.xlabel("Average ROC AUC")
plt.ylabel("Average PR AUC")
plt.title("ROC AUC vs PR AUC (Per Model)")
plt.grid(True, linestyle='--', alpha=0.5)
save_plot(fig, "roc_vs_pr_auc_scatter")


# heatmap of model average ranks
df_ranks = df_all.copy()
df_ranks["rank"] = df_ranks.groupby("model")["roc_auc"].rank(ascending=False, method="average")
rank_summary = df_ranks.groupby("model")["rank"].mean().sort_values()

fig = plt.figure(figsize=(8, 6))
sns.heatmap(rank_summary.to_frame().T, annot=True, cmap="YlGnBu", cbar=False)
plt.title("Average ROC AUC Rank per Model (Lower = Better)")
save_plot(fig, "average_rank_per_model_heatmap")


# ---- Failed Datasets Analysis ----
#collecting failed datasets
failed_files = []
for root, dirs, files in os.walk(FAILED_DIR):
    for file in files:
        if file == "failed_models.json":
            failed_files.append(os.path.join(root, file))

failed_datasets = {}
failed_model_counts = {}

for f in failed_files:
    with open(f, "r") as file:
        failed = json.load(file)
        dataset_name = os.path.basename(os.path.dirname(f))
        failed_datasets[dataset_name] = len(failed)

        for model in failed:
            model_name = model["model"]
            if model_name not in failed_model_counts:
                failed_model_counts[model_name] = 0
            failed_model_counts[model_name] += 1

#bar plot failed models count per dataset
fig = plt.figure(figsize=(10,6))
plt.bar(failed_datasets.keys(), failed_datasets.values(), color='tomato')
plt.xticks(rotation=45)
plt.ylabel("Number of Failed Models")
plt.title("Failed Model Count Per Dataset")
plt.tight_layout()
save_plot(fig, "failed_model_count_per_dataset")


#bar plot which model fails most
fig = plt.figure(figsize=(10,6))
plt.bar(failed_model_counts.keys(), failed_model_counts.values(), color='orangered')
plt.xticks(rotation=45)
plt.ylabel("Failure Count")
plt.title("Failure Frequency per Model")
plt.tight_layout()
save_plot(fig, "failure_frequency_per_model")


#pie chart of success vs failed datasets
total_datasets = len(next(os.walk(SUCCESS_DIR))[1]) + len(next(os.walk(FAILED_DIR))[1])
success_count = len(next(os.walk(SUCCESS_DIR))[1])
failed_count = len(next(os.walk(FAILED_DIR))[1])

fig = plt.figure(figsize=(6,6))
plt.pie([success_count, failed_count], labels=["Success", "Failed"], autopct="%1.1f%%", colors=["green", "red"])
plt.title("Overall Dataset Success vs Failure")
save_plot(fig, "overall_success_vs_failure_pie")
