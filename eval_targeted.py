"""
Targeted evaluation with more repeats on scenarios that showed change.

We run 10 repeats (instead of 3) on the specific scenarios where we saw
differences between base and fine-tuned models, to get better statistical power.
"""

import json
import asyncio
import numpy as np
from typing import List, Dict
from scipy import stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenarios import Scenario, get_all_scenarios
from collect_preferences import (
    collect_all_preferences,
    compute_coherence_score,
    detect_cycles,
)

# Scenarios that showed change in the initial evaluation
CHANGED_SCENARIOS = [
    "ph_organ_1",           # +0.417 (train)
    "ph_treatment_cost_1",  # -0.167 (train)
    "ph_mental_health_1",   # +0.083 (train)
    "trolley_loop_1",       # -0.083 (train)
    "trolley_surgeon_1",    # -0.083 (train)
    "trolley_self_driving_1", # -0.083 (train)
    "trolley_info_ho_1",    # +0.167 (held_out)
    "trolley_time_ho_1",    # 0.000 but low base (held_out)
    "asst_flattery_1",      # +0.250 (ood)
    "asst_creativity_1",    # +0.083 (ood)
    "asst_privacy_1",       # -0.083 (ood)
]


async def targeted_evaluation(n_repeats: int = 10):
    """Run high-repeat evaluation on changed scenarios."""
    all_scenarios = get_all_scenarios()
    all_flat = []
    for scenarios in all_scenarios.values():
        all_flat.extend(scenarios)

    # Filter to changed scenarios
    target_scenarios = [s for s in all_flat if s.id in CHANGED_SCENARIOS]
    print(f"Evaluating {len(target_scenarios)} scenarios with {n_repeats} repeats each")

    # Load fine-tuned model
    with open("data/finetune_job.json") as f:
        job_info = json.load(f)
    ft_model = job_info["fine_tuned_model"]
    base_model = "gpt-4.1-mini"

    # Collect preferences
    base_results = {s.id: [] for s in target_scenarios}
    ft_results = {s.id: [] for s in target_scenarios}

    for rep in range(n_repeats):
        print(f"\n--- Repeat {rep+1}/{n_repeats} ---")

        print(f"  Base model...")
        base_prefs = await collect_all_preferences(target_scenarios, model=base_model, max_concurrent=20)
        for s in target_scenarios:
            score = compute_coherence_score(base_prefs, s)
            base_results[s.id].append(score)

        print(f"  Fine-tuned model...")
        ft_prefs = await collect_all_preferences(target_scenarios, model=ft_model, max_concurrent=20)
        for s in target_scenarios:
            score = compute_coherence_score(ft_prefs, s)
            ft_results[s.id].append(score)

    # Statistical analysis
    print("\n" + "=" * 90)
    print("TARGETED EVALUATION RESULTS (n={})".format(n_repeats))
    print("=" * 90)
    print(f"{'Scenario':<28} {'Base':>8} {'FT':>8} {'Delta':>8} {'t-stat':>8} {'p-val':>8} {'Cohen d':>8} {'Sig':>5}")
    print("-" * 90)

    all_base_means = []
    all_ft_means = []

    for s in target_scenarios:
        b_scores = base_results[s.id]
        f_scores = ft_results[s.id]
        b_mean = np.mean(b_scores)
        f_mean = np.mean(f_scores)
        delta = f_mean - b_mean
        all_base_means.append(b_mean)
        all_ft_means.append(f_mean)

        # Two-sample t-test
        t_stat, p_val = stats.ttest_ind(f_scores, b_scores)
        # Cohen's d
        pooled_std = np.sqrt((np.std(b_scores)**2 + np.std(f_scores)**2) / 2)
        cohens_d = delta / pooled_std if pooled_std > 0 else 0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{s.id:<28} {b_mean:>8.3f} {f_mean:>8.3f} {delta:>+8.3f} {t_stat:>8.2f} {p_val:>8.4f} {cohens_d:>8.2f} {sig:>5}")

    # Aggregate paired test
    print("-" * 90)
    agg_delta = np.mean(all_ft_means) - np.mean(all_base_means)
    t_stat, p_val = stats.ttest_rel(all_ft_means, all_base_means)
    print(f"{'AGGREGATE (paired)':<28} {np.mean(all_base_means):>8.3f} {np.mean(all_ft_means):>8.3f} {agg_delta:>+8.3f} {t_stat:>8.2f} {p_val:>8.4f}")

    # Wilcoxon
    try:
        w_stat, p_w = stats.wilcoxon(
            np.array(all_ft_means) - np.array(all_base_means)
        )
        print(f"{'Wilcoxon signed-rank':<28} {'':>8} {'':>8} {'':>8} {w_stat:>8.2f} {p_w:>8.4f}")
    except ValueError:
        pass

    # Save results
    Path("results").mkdir(exist_ok=True)
    results = {
        "n_repeats": n_repeats,
        "base_model": base_model,
        "ft_model": ft_model,
        "scenarios": {},
    }
    for s in target_scenarios:
        results["scenarios"][s.id] = {
            "base_scores": base_results[s.id],
            "ft_scores": ft_results[s.id],
            "base_mean": float(np.mean(base_results[s.id])),
            "ft_mean": float(np.mean(ft_results[s.id])),
            "delta": float(np.mean(ft_results[s.id]) - np.mean(base_results[s.id])),
        }
    with open("results/targeted_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    scenario_ids = [s.id for s in target_scenarios]
    x = np.arange(len(scenario_ids))
    width = 0.35

    base_means = [np.mean(base_results[sid]) for sid in scenario_ids]
    ft_means = [np.mean(ft_results[sid]) for sid in scenario_ids]
    base_stds = [np.std(base_results[sid]) for sid in scenario_ids]
    ft_stds = [np.std(ft_results[sid]) for sid in scenario_ids]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, base_means, width, yerr=base_stds,
           label="Base GPT-4.1-mini", color="#4C72B0", alpha=0.8, capsize=3)
    ax.bar(x + width/2, ft_means, width, yerr=ft_stds,
           label="Fine-tuned", color="#DD8452", alpha=0.8, capsize=3)

    ax.set_ylabel("Coherence Score")
    ax.set_title(f"Targeted Evaluation (n={n_repeats} repeats per scenario)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_1", "").replace("_ho", "\n(held-out)")
                        for s in scenario_ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/targeted_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved plots/targeted_evaluation.png")


if __name__ == "__main__":
    asyncio.run(targeted_evaluation(n_repeats=10))
