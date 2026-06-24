import numpy as np
import pandas as pd
import main
from metrics import ECGMetrics

ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CORRUPTIONS = ["ra_la_reversal", "ra_ll_reversal", "la_ll_reversal", "v1_v2_swap", "v2_v3_swap", "single_lead_polarity_inversion"]

def run_greedy_selection(mode="robust", target_sizes=[1, 3, 6]):
    """
    mode: "clean" (Experiment 3) or "robust" (Experiment 4/5)
    """
    selected_leads = []
    summary_history = []
    
    print(f"\n=========================================================")
    # Visual anchors for training logs
    print(f"🧬 RUNNING: {mode.upper()} GREEDY LEAD SELECTION")
    print(f"=========================================================")

    for size in range(1, max(target_sizes) + 1):
        best_score = -1.0
        best_candidate = None
        best_candidate_metrics = {}

        remaining_candidates = [l for l in ALL_LEADS if l not in selected_leads]
        
        for candidate in remaining_candidates:
            trial_leads = selected_leads + [candidate]
            exp_name = f"trial_{mode}_{size}_{candidate}"
            
            # 1. Train model configuration natively on the specific candidate subset
            main.train(exp_name=exp_name, leads=trial_leads, max_epochs=1, batch_size=32, accelerator="cpu")
            
            # 2. Test on Clean ECGs
            clean_results = main.test(exp_name=exp_name, leads=trial_leads, corruption=None, accelerator="cpu")
            
            if mode == "clean":
                # Experiment 3 Selection Criterion: Accuracy alone (F1-score)
                score = clean_results["f1"]
                penalty = 0.0
            else:
                # Experiment 4/5 Selection Criterion: Accuracy - Instability Penalty
                instability_penalties = []
                for corr in CORRUPTIONS:
                    corr_results = main.test(exp_name=exp_name, leads=trial_leads, corruption=corr, accelerator="cpu")
                    instability = ECGMetrics.compute_instability(clean_results, corr_results)
                    # Penalty = average of Diagnosis Flip Rate + Probability Shift
                    instability_penalties.append(instability["flip_rate"] + instability["prob_shift"])
                
                penalty = np.mean(instability_penalties)
                score = clean_results["f1"] - penalty  # 🔥 THE NOVEL ROBUST FORMULA!

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_candidate_metrics = {
                    "clean_f1": clean_results["f1"],
                    "clean_auroc": clean_results["auroc"],
                    "penalty": penalty,
                    "robust_score": score
                }

        # Lock in the winner of this greedy tier step
        selected_leads.append(best_candidate)
        if size in target_sizes:
            summary_history.append({
                "Subset Size": size,
                "Selected Leads": list(selected_leads),
                "Clean F1": best_candidate_metrics["clean_f1"],
                "Clean AUROC": best_candidate_metrics["clean_auroc"],
                "Instability Penalty": best_candidate_metrics["penalty"],
                "Final Robust Score": best_candidate_metrics["robust_score"]
            })
            print(f"🏆 Size {size} Locked: {selected_leads} | Score: {best_score:.4f}")

    return pd.DataFrame(summary_history)

if __name__ == "__main__":
    # Run Experiment 3 (Clean Selection Criterion)
    clean_df = run_greedy_selection(mode="clean", target_sizes=[1, 3, 6])
    
    # Run Experiment 4/5 (Robust Selection Criterion)
    robust_df = run_greedy_selection(mode="robust", target_sizes=[1, 3, 6])
    
    print("\n📊 EXPERIMENT 3 SUMMARY (CLEAN SELECTION):")
    print(clean_df.to_string(index=False))
    
    print("\n📊 EXPERIMENT 4/5 SUMMARY (ROBUST SELECTION):")
    print(robust_df.to_string(index=False))
