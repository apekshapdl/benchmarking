import os
import sys
from pyod.models.vae import VAE
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
from vae_models.svae import SparseVAE, train_sparse_vae, evaluate_sparse_vae
from vae_models.betaVAE import BetaVAE, train_beta_vae, evaluate_beta_vae
from vae_models.betatcvae import BetaTCVAE, TCVDiscriminator, train_tcvae, evaluate_tcvae
from vae_models.hvae import HierarchicalVAE, train_models, evaluate_model
from vae_models.cvae import ConditionalVAE, train_conditional_vae, evaluate_conditional_vae
from vae_models.fvae import FactorVAE, Discriminator, train_factor_vae, evaluate_factor_vae
from vae_models.hhvae import DeepHierarchicalVAE, train_models_deep_hvae, evaluate_model_deep_hvae
from vae_models.utils import compute_recon_error
import scikit_posthocs as sp
from scipy.stats import rankdata


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#f = np.load("benchmarking_vae_models\AIonosphere.npz")  
#x_train, x_test, y_test = f["x"], f["tx"], f["ty"]

def preprocess(dataset):
    f = np.load(dataset)  
    x_train, x_test, y_test = f["x"], f["tx"], f["ty"]

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test = y_test.astype(int)

    return x_train_tensor, x_test_tensor, y_test


def find_best_f1_threshold(y_true, scores):
    thresholds = np.linspace(0, 1, 200)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


#-----NORMAL VAE-----
def run_normal_vae_default(x_train, x_test, y_test, dataset):

    print("Running Normal Vae...")
    try: 
        vae = VAE(
            contamination=0.1,
            encoder_neuron_list=[128, 64, 32],
            decoder_neuron_list=[32, 64, 128],
            latent_dim=2,
            epoch_num=50,
            batch_size=32,
            lr=0.001,
            dropout_rate=0.1,
            beta=1.0,
            verbose=0
        )

        start_train = time.time()
        vae.fit(x_train)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores = vae.decision_function(x_test)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)
        #thresh, f1 = find_best_f1_threshold(y_test, scores)

        print(f"Normal VAE (Default) ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        result = {
            "model": "NormalVAE_Default",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }
        #save_summary(dataset, "normal_vae", pd.DataFrame(result))
        return result
    
    except Exception as e:
        print(f"[ERROR] Normal VAE failed on dataset {dataset}: {str(e)}")
        return {
            "model": "NormalVAE_Default",
            "roc_auc": 0,
            "pr_auc": 0,
            "train_time": 0,
            "inference_time": 0
        }


#-----OVERCOMPLETE VAE-----
def run_overcomplete_vae_default(x_train, x_test, y_test, dataset):
    print("Running Overcomplete VAE (Default)...")
    try: 

        input_dim = x_train.shape[1]
        latent_dim = max(2 * input_dim, 64)
        encoer_layers = [input_dim * 2, input_dim, int(input_dim/2)]
        decoder_layers = encoer_layers[::-1]

        vae = VAE(
            contamination=0.1,
            encoder_neuron_list=encoer_layers,
            decoder_neuron_list=decoder_layers,
            latent_dim=latent_dim,
            epoch_num=100,
            batch_size=32,
            lr=0.001,
            dropout_rate=0.01,
            beta=1.0,
            verbose=0
        )

        start_train = time.time()
        vae.fit(x_train)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores = vae.decision_function(x_test)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)
    # thresh, f1 = find_best_f1_threshold(y_test, scores)

        print(f"Overcomplete VAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        result = {
            "model": "OvercompleteVAE_Default",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

        #save_summary(dataset, "over_complete", pd.DataFrame(result))
        return result
    except Exception as e:
        print(f"[ERROR] Overcomplete VAE failed on dataset {dataset}: {str(e)}")
        return {
            "model": "OvercompleteVAE_Default",
            "roc_auc": 0,
            "pr_auc": 0,
            "train_time": 0,
            "inference_time": 0
        }



#-----HIERARCHIECAL VAE-----
def run_hvae_default(x_train, x_test, y_test, dataset):
    print("Running HVAE (Default)...")
    try: 
        input_dim = x_train.shape[1]
        start_train = time.time()
        results = train_models(
            X_train=x_train,
            X_test=x_test,
            y_test=y_test,
            input_dim=input_dim,
            z1_list=[16],
            z2_list=[8],
            epoch_list=[200],
            optimizer_type="adam"
        )
        end_train = time.time()
        train_time = end_train - start_train

        best_result = results[0]
        model = best_result["model"]

        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        start_infer = time.time()
        recon_errors, _ = evaluate_model(model, X_test_tensor, y_test)
        #recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
        #threshold, best_f1 = find_best_f1_threshold(y_test, recon_errors)
        #print(f"HVAE_Default - Optimal Threshold: {threshold:.4f}, Best F1 Score: {best_f1:.4f}")

        end_infer = time.time()
        inference_time = end_infer - start_infer

        roc_auc = best_result["roc_auc"]
        pr_auc = best_result["pr_auc"]

        print(f"Hierarchical VAE (Hyperparameters) - ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, "
            f"Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        result = {
            "model": "HVAE_Default",
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "train_time": train_time,
            "inference_time": inference_time
        }
        return result
    except Exception as e:
        print(f"[ERROR] HVAE failed on dataset {dataset}: {str(e)}")
        return {
            "model": "HVAE_Default",
            "roc_auc": 0,
            "pr_auc": 0,
            "train_time": 0,
            "inference_time": 0
        }


# ----- SPARSE VAE -----
def run_sparse_vae_default(x_train, x_test, y_test, dataset):
    print("Running SparseVAE...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SparseVAE(input_dim=x_train.shape[1], latent_dim=10, beta=1.0, l1_lambda=1e-3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        start_train = time.time()
        train_sparse_vae(model, train_loader, optimizer, device=device, epochs=50)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores = evaluate_sparse_vae(model, X_test_tensor)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)

        print(f"SparseVAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        return {
            "model": "SparseVAE",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

    except Exception as e:
        print(f"[ERROR] SparseVAE failed on dataset {dataset}: {str(e)}")
        return {"model": "SparseVAE", "roc_auc": 0, "pr_auc": 0, "train_time": 0, "inference_time": 0}
    


# ----- FACTOR VAE -----
def run_factor_vae_default(x_train, x_test, y_test, dataset):
    print("Running FactorVAE...")
    try:
        latent_dim = 10
        beta = 10.0
        discriminator = Discriminator(latent_dim)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FactorVAE(input_dim=x_train.shape[1], latent_dim=latent_dim)
        optimizer_vae = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        start_train = time.time()
        train_factor_vae(model, discriminator, train_loader, optimizer_vae, optimizer_disc, beta=beta, epochs=50)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores = evaluate_factor_vae(model, X_test_tensor)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)

        print(f"FactorVAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        return {
            "model": "FactorVAE",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

    except Exception as e:
        print(f"[ERROR] FactorVAE failed on dataset {dataset}: {str(e)}")
        return {"model": "FactorVAE", "roc_auc": 0, "pr_auc": 0, "train_time": 0, "inference_time": 0}
    


# ----- CONDITIONAL VAE -----
def run_cvae_default(x_train, x_test, y_test, dataset):
    print("Running ConditionalVAE...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = x_train.shape[1]
        cond_dim = 1

        y_train = np.zeros(len(x_train))  # Assume all training data is normal
        Y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        Y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        model = ConditionalVAE(input_dim=input_dim, latent_dim=10, cond_dim=cond_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(TensorDataset(x_train, Y_train_tensor), batch_size=64, shuffle=True)

        start_train = time.time()
        train_conditional_vae(model, train_loader, optimizer, device=device, epochs=50)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores, _ = evaluate_conditional_vae(model, X_test_tensor, Y_test_tensor, device=device)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)

        print(f"ConditionalVAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        return {
            "model": "ConditionalVAE",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

    except Exception as e:
        print(f"[ERROR] ConditionalVAE failed on dataset {dataset}: {str(e)}")
        return {"model": "ConditionalVAE", "roc_auc": 0, "pr_auc": 0, "train_time": 0, "inference_time": 0}
    


# ----- BETA VAE -----
def run_beta_vae_default(x_train, x_test, y_test, dataset):
    print("Running BetaVAE...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BetaVAE(input_dim=x_train.shape[1], latent_dim=10, beta=4.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        start_train = time.time()
        train_beta_vae(model, train_loader, optimizer, device=device, epochs=50)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores = evaluate_beta_vae(model, X_test_tensor)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)

        print(f"BetaVAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        return {
            "model": "BetaVAE",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

    except Exception as e:
        print(f"[ERROR] BetaVAE failed on dataset {dataset}: {str(e)}")
        return {"model": "BetaVAE", "roc_auc": 0, "pr_auc": 0, "train_time": 0, "inference_time": 0}
    


# ----- BETA TC VAE -----
def run_betatc_vae_default(x_train, x_test, y_test, dataset):
    print("Running Beta-TCVAE...")
    try:
        latent_dim = 16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BetaTCVAE(input_dim=x_train.shape[1], latent_dim=latent_dim).to(device)
        discriminator = TCVDiscriminator(latent_dim=latent_dim)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        optimizer_vae = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

        train_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)

        start_train = time.time()
        train_tcvae(model, discriminator, train_loader, optimizer_vae, optimizer_disc, beta=6.0, epochs=50)
        train_time = time.time() - start_train

        start_infer = time.time()
        scores= evaluate_tcvae(model, X_test_tensor)
        inference_time = time.time() - start_infer

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr = auc(recall, precision)

        print(f"Beta-TCVAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        return {
            "model": "BetaTCVAE",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

    except Exception as e:
        print(f"[ERROR] BetaTCVAE failed on dataset {dataset}: {str(e)}")
        return {"model": "BetaTCVAE", "roc_auc": 0, "pr_auc": 0, "train_time": 0, "inference_time": 0}



# ----- HIERARCHICAL DEEP VAE (z1->z2->z3) -----
def run_deep_hvae_default(x_train, x_test, y_test, dataset):
    print("Running Deep Hierarchical VAE...")
    try:
        input_dim = x_train.shape[1]
        start_train = time.time()
        results = train_models_deep_hvae(
            X_train=x_train.numpy(),
            X_test=x_test.numpy(),
            y_test=y_test,
            input_dim=input_dim,
            z1_list=[16],
            z2_list=[8],
            z3_list=[4],
            epoch_list=[200],
            optimizer_type="adam"
        )
        train_time = time.time() - start_train

        best_result = results[0]
        model = best_result["model"]
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        start_infer = time.time()
        recon_errors, _ = evaluate_model_deep_hvae(model, X_test_tensor, y_test)
        inference_time = time.time() - start_infer

        roc = best_result["roc_auc"]
        #precision, recall, _ = precision_recall_curve(y_test, recon_errors)
        pr = best_result["pr_auc"]

        print(f"Deep HVAE ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}, Train Time: {train_time:.2f}s, Inference Time: {inference_time:.2f}s")

        return {
            "model": "DeepHVAE",
            "roc_auc": roc,
            "pr_auc": pr,
            "train_time": train_time,
            "inference_time": inference_time
        }

    except Exception as e:
        print(f"[ERROR] DeepHVAE failed on dataset {dataset}: {str(e)}")
        return {"model": "DeepHVAE", "roc_auc": 0, "pr_auc": 0, "train_time": 0, "inference_time": 0}



#-----BASELINE MODELS: KNN AND ISOLATION FOREST-----
def run_baseline_models(x_train, x_test, y_test, dataset):
    print("Running baseline models...")
    baseline_models = {
        "kNN": KNN(),
        "Isolation Forest": IForest()
    }

    results = []

    try:
        for name, model in baseline_models.items():
            print(f"Training {name}...")
            model.fit(x_train)
            scores = model.decision_function(x_test)

            scores = (scores - scores.min()) / (scores.max() - scores.min())
            roc = roc_auc_score(y_test, scores)
            precision, recall, _ = precision_recall_curve(y_test, scores)
            pr = auc(recall, precision)
            thresh, f1 = find_best_f1_threshold(y_test, scores)

            print(f"{name} ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}")

            results.append({
                "model": name,
                "roc_auc": roc,
                "pr_auc": pr,
                "train_time": None,
                "inference_time": None,
            })

        return results  
    except Exception as e:
        print(f"[ERROR] Normal VAE failed on dataset {dataset}: {str(e)}")
        return {
            "model": "NormalVAE_Default",
            "roc_auc": 0,
            "pr_auc": 0,
            "train_time": 0,
            "inference_time": 0
        }


def save_summary(result_folder, file_name, df_result):
    os.makedirs(result_folder, exist_ok=True)
    output_file = os.path.join(result_folder, f"{file_name}.csv")
    df_result.to_csv(output_file, index=False)
    print(f"[SAVED] Summary to {output_file}")

def save_figures(fig, result_folder, figure_name):
    os.makedirs(result_folder, exist_ok=True)
    image_filename = f"{figure_name}.png"
    full_path = os.path.join(result_folder, image_filename)
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] Plot: {full_path}")


def run_all_for_one_dataset(dataset):
    try:
        print(f"[PID {os.getpid()}] Processing: {dataset}")

        x_train, x_test, y_test = preprocess(dataset)
        failed_models = []

        #helper to check and record failure
        def check_failure(result, model_name):
            if result["roc_auc"] == 0:
                failed_models.append({"model": model_name, "error": result.get("error", "Unknown (roc_auc=0)")})

        result_normal_default = run_normal_vae_default(x_train, x_test, y_test, dataset)
        check_failure(result_normal_default, "NormalVAE_Default")

        result_ocvae = run_overcomplete_vae_default(x_train, x_test, y_test, dataset)
        check_failure(result_ocvae, "OvercompleteVAE_Default")

        result_hvae = run_hvae_default(x_train, x_test, y_test, dataset)
        check_failure(result_hvae, "HVAE_Default")

        result_baseline_models = run_baseline_models(x_train, x_test, y_test, dataset)

        result_svae = run_sparse_vae_default(x_train, x_test, y_test, dataset)
        check_failure(result_svae, "SparseVAE")

        result_betavae = run_beta_vae_default(x_train, x_test, y_test, dataset)
        check_failure(result_betavae, "BetaVAE")
        
        result_betatcvae = run_betatc_vae_default(x_train, x_test, y_test, dataset)
        check_failure(result_betatcvae, "BetaTCVAE")
        
        result_cvae = run_cvae_default(x_train, x_test, y_test, dataset)
        check_failure(result_cvae, "ConditionalVAE")

        result_fvae = run_factor_vae_default(x_train, x_test, y_test, dataset)
        check_failure(result_fvae, "FactorVAE")

        result_dhvae = run_deep_hvae_default(x_train, x_test, y_test, dataset)
        check_failure(result_dhvae, "DeepHVAE")

        vae_results = [result_normal_default, result_ocvae, result_hvae, result_betatcvae, result_betavae, result_cvae, result_fvae, result_svae, result_dhvae]
        #vae_results = [ result_hvae,  result_betavae,  result_fvae ]

        # Determine result folder
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        if len(failed_models) == 0:
            print(f"[{dataset}] SUCCESS: All models ran fine.")
            result_folder = os.path.join(SCRIPT_DIR, "results", "successful_datasets", dataset_name)
        else:
            print(f"[{dataset}] FAILED MODELS: {failed_models}")
            result_folder = os.path.join(SCRIPT_DIR, "results", "failed_datasets", dataset_name)
        os.makedirs(result_folder, exist_ok=True)

        # VAE-only outputs
        df_results = pd.DataFrame(vae_results)
        df_results.sort_values(by="roc_auc", ascending=False, inplace=True)
        save_summary(result_folder, "benchmark_result", df_results)

        # VAE-only plots
        x = np.arange(len(df_results))
        labels = df_results["model"].tolist()

        # Combined outputs
        combined_results = vae_results + result_baseline_models
        combined_df = pd.DataFrame(combined_results)
        combined_df.sort_values(by="roc_auc", ascending=False, inplace=True)
        save_summary(result_folder, "benchmark_with_baselines", combined_df)

        # Combined plots
        x_all = np.arange(len(combined_df))
        labels_all = combined_df["model"].tolist()

        # ROC vs PR AUC
        create_then_save(
            x=x,
            values1=df_results["roc_auc"],
            values2=df_results["pr_auc"],
            labels=labels,
            title="ROC AUC vs PR AUC",
            ylabel="Score",
            result_folder=result_folder,
            plot_name="roc_vs_pr_auc",
            color1="skyblue",
            color2="salmon"
        )       
        
        # Inference Time vs train time
        create_then_save(
            x=x,
            values1=df_results["inference_time"],
            values2=df_results["train_time"],
            labels=labels,
            title="Inference Time vs Train timeper Model",
            ylabel="Seconds",
            result_folder=result_folder,
            plot_name="inference_time_vs_train_time",
            color1="seagreen",
            color2="skyblue"
        )
       # ROC vs PR AUC with baseline models
        create_then_save(
            x=x_all,
            values1=combined_df["roc_auc"],
            values2=combined_df["pr_auc"],
            labels=labels_all,
            title="ROC AUC vs PR AUC (All Models)",
            ylabel="Score",
            result_folder=result_folder,
            plot_name="roc_vs_pr_auc_all",
            color1="skyblue",
            color2="salmon"
        )

        # Save failed models info if any
        if len(failed_models) > 0:
            with open(os.path.join(result_folder, "failed_models.json"), "w") as f:
                json.dump(failed_models, f, indent=4)

    except Exception as e:
        print(f"[FATAL ERROR] Skipping dataset {dataset} due to: {e}")
        result_folder = os.path.join(SCRIPT_DIR, "results", "failed_datasets", os.path.splitext(os.path.basename(dataset))[0])
        os.makedirs(result_folder, exist_ok=True)
        with open(os.path.join(result_folder, "fatal_error.txt"), "w") as f:
            f.write(str(e))


def create_then_save(
    x,
    values1,
    labels,
    title,
    ylabel,
    result_folder,
    plot_name,
    color1="skyblue",
    values2=None,
    color2="salmon"):

    fig = plt.figure(figsize=(10, 5))
    width = 0.35
    if values2 is not None:
        plt.bar(x - width / 2, values1, width, label="ROC AUC", color=color1)
        plt.bar(x + width / 2, values2, width, label="PR AUC", color=color2)
        plt.legend()
    else:
        plt.bar(x, values1, color=color1)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_figures(fig, result_folder, plot_name)
    plt.close(fig)



def get_all_datasets(folder_path = ".", extension = ".npz"):
    files = []
    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isfile(full_path) and item.endswith(extension):
            files.append(item)
    return files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_script.py <dataset_file.npz>")
        sys.exit(1)

    dataset_file = sys.argv[1]
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Dataset file {dataset_file} not found!")
        sys.exit(1)

    print(f"Running benchmark on dataset: {dataset_file}")
    run_all_for_one_dataset(dataset_file)
