import numpy as np
import pandas as pd
import os
import random
import torch
import main
from metrics import ECGMetrics
from reporting import report_predictions, report_per_class_metrics

def seed_everything(seed=137):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
CORRUPTIONS = ["ra_la_reversal", "ra_ll_reversal", "la_ll_reversal", "v1_v2_swap", "v2_v3_swap"]
ELIGIBLE_SET = {
    "ra_la_reversal": ["I", "II", "III", "aVR", "aVL"],
    "ra_ll_reversal": ["I", "II", "III", "aVR", "aVF"],
    "la_ll_reversal": ["I", "II", "III", "aVL", "aVF"],
    "v1_v2_swap": ["V1", "V2"],
    "v2_v3_swap": ["V2", "V3"],
}

def run_experiment_5(results_dir="results/experiment5"):
    os.makedirs(results_dir, exist_ok=True)

    for corruption in CORRUPTIONS:
        print(f"{'=' * 60}\n")
        print(f"CORRUPTION: {corruption.upper()}")
        print(f"{'=' * 60}\n")

        selected_leads = []

        lead_selection_rows = []

        for size in range(1, len(ELIGIBLE_SET[corruption]) + 1):
            # 1. Select the subset with the highest validation F1
            best_candidate = None
            best_score = 0.0

            remaining_candidates = [l for l in ELIGIBLE_SET[corruption] if l not in selected_leads]

            for lead in remaining_candidates:
                if len(remaining_candidates) == 1:
                    best_candidate = lead
                    continue

                trial_leads = selected_leads + [lead]
                print(f"LEADS: {trial_leads}")
                exp_name = f"leads_{'_'.join(trial_leads)}_5_epochs"

                score_path = os.path.join("results", exp_name, "ckpts", "best_val_f1.txt")

                if os.path.isfile(score_path):
                    print(f"Checkpoint already exists for {exp_name}, skipping training.")
                    with open(score_path, "r") as f:
                        best_val_f1 = float(f.read())
                else:
                    seed_everything(137)
                    best_val_f1 = main.train(exp_name=exp_name, leads=trial_leads, max_epochs=5, batch_size=32)

                print(f"Candidate {trial_leads} Val F1: {best_val_f1}")
                row = {
                    "subset_size": size,
                    "candidate": '-'.join(trial_leads),
                    "intended_corruption": corruption,
                    "val_f1": best_val_f1,
                }

                lead_selection_rows.append(row)

                if best_val_f1 > best_score:
                    best_candidate = lead
                    best_score = best_val_f1

            # Lock in the winner of this greedy tier step
            selected_leads.append(best_candidate)
            print(f"Selected Leads: {selected_leads}")

            # 2. Evaluate performance under the corruption

            # Retrain the model with 10 epochs
            exp_name = f"leads_{'_'.join(selected_leads)}_10_epochs"
            ckpt_path = os.path.join("results", exp_name, "ckpts", "best.ckpt")

            if os.path.isfile(ckpt_path):
                print(f"Checkpoint already exists for {exp_name}, skipping training.")
            else:
                seed_everything(137)
                main.train(exp_name=exp_name, leads=selected_leads, max_epochs=10, batch_size=32)

            clean_results = main.test(exp_name=exp_name, leads=selected_leads, corruption=None)
            corr_results = main.test(exp_name=exp_name, leads=selected_leads, corruption=corruption)
            instability = ECGMetrics.compute_instability(clean_results["scores"], corr_results["scores"], CLASS_NAMES)

            # Clean
            report_predictions(
                outputs=clean_results["outputs"],
                save_dir=results_dir,
                leads=selected_leads,
                mode="clean",
                corruption=corruption,
            )
        
            report_per_class_metrics(
                scores=clean_results["scores"],
                save_dir=results_dir,
                leads=selected_leads,
                mode="clean",
                corruption=corruption,
                instability=None,
            )

            # Corrupted
            report_predictions(
                outputs=corr_results["outputs"],
                save_dir=results_dir,
                leads=selected_leads,
                mode="corrupted",
                corruption=corruption,
            )
        
            report_per_class_metrics(
                scores=corr_results["scores"],
                save_dir=results_dir,
                leads=selected_leads,
                mode="corrupted",
                corruption=corruption,
                instability=instability,
            )

        csv_path = os.path.join(results_dir, f"Experiment_5_{corruption}_lead_selection.csv")
        
        pd.DataFrame(lead_selection_rows).to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_experiment_5()