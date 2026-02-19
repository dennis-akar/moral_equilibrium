"""
Steps 6-8: Evaluate the fine-tuned model on:
  - Original training scenarios (step 6)
  - Held-out in-distribution scenarios (step 7)
  - Out-of-distribution scenarios (step 8)

Compare coherence scores (transitivity) between base and fine-tuned models.
"""

import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import asdict
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenarios import Scenario, get_all_scenarios
from collect_preferences import (
    collect_all_preferences,
    detect_cycles,
    compute_coherence_score,
    PairwiseResult,
)


async def evaluate_model(
    model: str,
    scenarios: List[Scenario],
    label: str,
    n_repeats: int = 3,
) -> Dict:
    """Evaluate a model's coherence across scenarios, repeated n times for statistics."""
    all_results = []

    for rep in range(n_repeats):
        print(f"\n  {label} — repeat {rep+1}/{n_repeats}")
        prefs = await collect_all_preferences(scenarios, model=model, max_concurrent=20)
        all_results.append(prefs)

    # Compute coherence per scenario per repeat
    coherence_data = {}
    for scenario in scenarios:
        scores = []
        for prefs in all_results:
            score = compute_coherence_score(prefs, scenario)
            scores.append(score)
        coherence_data[scenario.id] = {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "domain": scenario.domain,
            "split": scenario.split,
        }

    return coherence_data


async def run_full_evaluation(
    base_model: str = "gpt-4.1-mini",
    ft_model: str = None,
    n_repeats: int = 3,
):
    """Run evaluation comparing base and fine-tuned models."""
    all_scenarios = get_all_scenarios()

    if ft_model is None:
        # Try to load from job info
        try:
            with open("data/finetune_job.json") as f:
                job_info = json.load(f)
            ft_model = job_info.get("fine_tuned_model")
        except FileNotFoundError:
            pass

    results = {"base_model": base_model, "ft_model": ft_model}

    for split_name, scenarios in all_scenarios.items():
        print(f"\n=== Evaluating {split_name} scenarios ===")

        # Base model
        print(f"\nBase model ({base_model}):")
        base_coherence = await evaluate_model(base_model, scenarios, f"base/{split_name}", n_repeats)
        results[f"base_{split_name}"] = base_coherence

        # Fine-tuned model
        if ft_model:
            print(f"\nFine-tuned model ({ft_model}):")
            ft_coherence = await evaluate_model(ft_model, scenarios, f"ft/{split_name}", n_repeats)
            results[f"ft_{split_name}"] = ft_coherence

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compute and print summary statistics
    print_summary(results)

    # Generate plots
    if ft_model:
        generate_plots(results)

    return results


def print_summary(results: Dict):
    """Print a summary table of coherence scores."""
    ft_model = results.get("ft_model")
    print("\n" + "=" * 80)
    print("COHERENCE SUMMARY")
    print("=" * 80)

    for split in ["train", "held_out", "ood"]:
        base_key = f"base_{split}"
        ft_key = f"ft_{split}"

        if base_key not in results:
            continue

        print(f"\n--- {split.upper()} ---")
        print(f"{'Scenario':<25} {'Base Mean':>10} {'Base Std':>10}", end="")
        if ft_model and ft_key in results:
            print(f" {'FT Mean':>10} {'FT Std':>10} {'Delta':>10} {'p-value':>10}", end="")
        print()

        base_data = results[base_key]
        ft_data = results.get(ft_key, {})

        base_means = []
        ft_means = []

        for scenario_id, data in base_data.items():
            print(f"{scenario_id:<25} {data['mean']:>10.3f} {data['std']:>10.3f}", end="")
            base_means.append(data["mean"])

            if ft_model and scenario_id in ft_data:
                ft_d = ft_data[scenario_id]
                delta = ft_d["mean"] - data["mean"]
                # t-test for difference
                if len(data["scores"]) > 1 and len(ft_d["scores"]) > 1:
                    t_stat, p_val = stats.ttest_ind(ft_d["scores"], data["scores"])
                else:
                    p_val = float("nan")
                print(f" {ft_d['mean']:>10.3f} {ft_d['std']:>10.3f} {delta:>+10.3f} {p_val:>10.3f}", end="")
                ft_means.append(ft_d["mean"])
            print()

        # Aggregate
        print(f"{'AGGREGATE':<25} {np.mean(base_means):>10.3f} {np.std(base_means):>10.3f}", end="")
        if ft_means:
            delta = np.mean(ft_means) - np.mean(base_means)
            # Paired t-test on scenario means
            if len(base_means) > 1 and len(ft_means) == len(base_means):
                t_stat, p_val = stats.ttest_rel(ft_means, base_means)
            else:
                p_val = float("nan")
            print(f" {np.mean(ft_means):>10.3f} {np.std(ft_means):>10.3f} {delta:>+10.3f} {p_val:>10.3f}", end="")
        print()


def generate_plots(results: Dict):
    """Generate comparison plots."""
    Path("plots").mkdir(exist_ok=True)

    ft_model = results.get("ft_model", "fine-tuned")

    # Plot 1: Coherence comparison by split
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    split_names = ["train", "held_out", "ood"]
    split_labels = ["Training", "Held-Out (ID)", "Out-of-Distribution"]

    for ax, split, label in zip(axes, split_names, split_labels):
        base_key = f"base_{split}"
        ft_key = f"ft_{split}"

        if base_key not in results:
            continue

        base_data = results[base_key]
        ft_data = results.get(ft_key, {})

        scenarios = sorted(base_data.keys())
        base_means = [base_data[s]["mean"] for s in scenarios]
        base_stds = [base_data[s]["std"] for s in scenarios]

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax.bar(x - width/2, base_means, width, yerr=base_stds,
                       label="Base", color="#4C72B0", alpha=0.8, capsize=3)

        if ft_data:
            ft_means = [ft_data.get(s, {}).get("mean", 0) for s in scenarios]
            ft_stds = [ft_data.get(s, {}).get("std", 0) for s in scenarios]
            bars2 = ax.bar(x + width/2, ft_means, width, yerr=ft_stds,
                           label="Fine-tuned", color="#DD8452", alpha=0.8, capsize=3)

        ax.set_ylabel("Coherence Score")
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("_")[0] + "_" + s.split("_")[1][:3]
                            for s in scenarios], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/coherence_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/coherence_comparison.png")

    # Plot 2: Aggregate coherence by split
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = []
    base_agg = []
    ft_agg = []
    base_agg_std = []
    ft_agg_std = []

    for split, label in zip(split_names, split_labels):
        base_key = f"base_{split}"
        ft_key = f"ft_{split}"
        if base_key not in results:
            continue

        base_data = results[base_key]
        ft_data = results.get(ft_key, {})

        base_scores = [d["mean"] for d in base_data.values()]
        splits.append(label)
        base_agg.append(np.mean(base_scores))
        base_agg_std.append(np.std(base_scores) / np.sqrt(len(base_scores)))

        if ft_data:
            ft_scores = [d["mean"] for d in ft_data.values()]
            ft_agg.append(np.mean(ft_scores))
            ft_agg_std.append(np.std(ft_scores) / np.sqrt(len(ft_scores)))

    x = np.arange(len(splits))
    width = 0.35

    ax.bar(x - width/2, base_agg, width, yerr=base_agg_std,
           label="Base GPT-4.1-mini", color="#4C72B0", alpha=0.8, capsize=5)
    if ft_agg:
        ax.bar(x + width/2, ft_agg, width, yerr=ft_agg_std,
               label="Fine-tuned", color="#DD8452", alpha=0.8, capsize=5)

    ax.set_ylabel("Mean Coherence Score")
    ax.set_title("Preference Coherence: Base vs Fine-tuned Model")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="Perfect coherence")

    plt.tight_layout()
    plt.savefig("plots/aggregate_coherence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/aggregate_coherence.png")

    # Plot 3: Coherence by domain
    fig, ax = plt.subplots(figsize=(8, 5))
    domains = set()
    for split in split_names:
        base_key = f"base_{split}"
        if base_key in results:
            for d in results[base_key].values():
                domains.add(d["domain"])

    domains = sorted(domains)
    base_by_domain = {d: [] for d in domains}
    ft_by_domain = {d: [] for d in domains}

    for split in split_names:
        base_key = f"base_{split}"
        ft_key = f"ft_{split}"
        if base_key not in results:
            continue
        for scenario_id, data in results[base_key].items():
            base_by_domain[data["domain"]].append(data["mean"])
        if ft_key in results:
            for scenario_id, data in results[ft_key].items():
                ft_by_domain[data["domain"]].append(data["mean"])

    x = np.arange(len(domains))
    base_domain_means = [np.mean(base_by_domain[d]) if base_by_domain[d] else 0 for d in domains]
    base_domain_se = [np.std(base_by_domain[d])/np.sqrt(len(base_by_domain[d]))
                      if len(base_by_domain[d]) > 1 else 0 for d in domains]

    ax.bar(x - width/2, base_domain_means, width, yerr=base_domain_se,
           label="Base", color="#4C72B0", alpha=0.8, capsize=5)

    if any(ft_by_domain[d] for d in domains):
        ft_domain_means = [np.mean(ft_by_domain[d]) if ft_by_domain[d] else 0 for d in domains]
        ft_domain_se = [np.std(ft_by_domain[d])/np.sqrt(len(ft_by_domain[d]))
                        if len(ft_by_domain[d]) > 1 else 0 for d in domains]
        ax.bar(x + width/2, ft_domain_means, width, yerr=ft_domain_se,
               label="Fine-tuned", color="#DD8452", alpha=0.8, capsize=5)

    ax.set_ylabel("Mean Coherence Score")
    ax.set_title("Preference Coherence by Domain")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/coherence_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/coherence_by_domain.png")


async def evaluate_baseline_only(n_repeats: int = 3):
    """Evaluate just the baseline model (useful before fine-tuning)."""
    all_scenarios = get_all_scenarios()
    base_model = "gpt-4.1-mini"
    results = {"base_model": base_model, "ft_model": None}

    for split_name, scenarios in all_scenarios.items():
        print(f"\n=== Evaluating {split_name} scenarios ===")
        print(f"\nBase model ({base_model}):")
        base_coherence = await evaluate_model(base_model, scenarios, f"base/{split_name}", n_repeats)
        results[f"base_{split_name}"] = base_coherence

    Path("results").mkdir(exist_ok=True)
    with open("results/evaluation_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        asyncio.run(evaluate_baseline_only(n_repeats=int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    else:
        ft_model = sys.argv[1] if len(sys.argv) > 1 else None
        asyncio.run(run_full_evaluation(ft_model=ft_model))
