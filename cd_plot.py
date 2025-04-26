import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def collect_all_auc_scores(results_dir="results"):
    auc_scores = {}

    for dataset_folder in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        result_file = os.path.join(dataset_path, f"{dataset_folder}_benchmark_result.csv")
        if not os.path.isfile(result_file):
            continue

        df = pd.read_csv(result_file)
        for _, row in df.iterrows():
            model = row["model"]
            roc_auc = row["roc_auc"]

            if model not in auc_scores:
                auc_scores[model] = {}
            auc_scores[model][dataset_folder] = roc_auc

    return auc_scores

def generate_global_cd_plot(auc_scores):
    model_names = list(auc_scores.keys())
    datasets = list(next(iter(auc_scores.values())).keys())
    score_matrix = np.array([[auc_scores[model].get(dataset, 0.0) for model in model_names] for dataset in datasets])

    # Compute ranks (higher AUC = better rank)
    ranks = np.array([rankdata(-row) for row in score_matrix])
    avg_ranks = np.mean(ranks, axis=0)

    nemenyi_result = sp.posthoc_nemenyi_friedman(ranks)
    ranks_series = pd.Series(avg_ranks, index=model_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    sp.critical_difference_diagram(
        ranks_series, nemenyi_result, ax=ax,
        label_props={'fontsize': 14, 'fontweight': 'bold'},
        elbow_props={"linewidth": 2},
        color_palette={model: "black" for model in model_names}
    )

    plt.title("Global Critical Difference Plot (ROC AUC)", pad=10)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "global_cd_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Global CD Plot to {output_path}")

if __name__ == "__main__":
    auc_scores = collect_all_auc_scores(RESULTS_DIR)
    if auc_scores:
        generate_global_cd_plot(auc_scores)
    else:
        print("[WARNING] No benchmark results found to generate Global CD plot.")
