"""
Comprehensive analysis of the reflective equilibrium experiment.

Produces:
- Coherence comparison plots (base vs fine-tuned)
- Statistical tests (paired t-test, Wilcoxon signed-rank)
- Domain-level and split-level analysis
- Qualitative analysis of what changed
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def analyze_baseline(results: Dict):
    """Analyze baseline coherence patterns."""
    print("=" * 70)
    print("BASELINE COHERENCE ANALYSIS")
    print("=" * 70)

    # Collect scores by domain and split
    by_domain = defaultdict(list)
    by_split = defaultdict(list)
    all_scores = []

    for split in ["train", "held_out", "ood"]:
        key = f"base_{split}"
        if key not in results:
            continue
        for scenario_id, data in results[key].items():
            mean_score = data["mean"]
            domain = data["domain"]
            by_domain[domain].append(mean_score)
            by_split[split].append(mean_score)
            all_scores.append(mean_score)

    print(f"\nOverall mean coherence: {np.mean(all_scores):.3f} (std={np.std(all_scores):.3f})")
    print(f"Fraction of perfectly coherent scenarios: {sum(1 for s in all_scores if s >= 0.999)/len(all_scores):.1%}")

    print("\nBy split:")
    for split, scores in by_split.items():
        print(f"  {split:12s}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
              f"n={len(scores)}, perfect={sum(1 for s in scores if s >= 0.999)}/{len(scores)}")

    print("\nBy domain:")
    for domain, scores in sorted(by_domain.items()):
        print(f"  {domain:20s}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
              f"n={len(scores)}, perfect={sum(1 for s in scores if s >= 0.999)}/{len(scores)}")

    # Identify scenarios with lowest coherence
    print("\nScenarios with lowest coherence:")
    all_scenario_scores = []
    for split in ["train", "held_out", "ood"]:
        key = f"base_{split}"
        if key not in results:
            continue
        for scenario_id, data in results[key].items():
            all_scenario_scores.append((scenario_id, data["mean"], data["std"], split, data["domain"]))

    all_scenario_scores.sort(key=lambda x: x[1])
    for sid, mean, std, split, domain in all_scenario_scores[:10]:
        print(f"  {sid:30s} [{split:8s}/{domain:20s}] coherence={mean:.3f} +/- {std:.3f}")


def compare_models(results: Dict):
    """Compare base vs fine-tuned model."""
    ft_model = results.get("ft_model")
    if not ft_model:
        print("\nNo fine-tuned model results available.")
        return

    print("\n" + "=" * 70)
    print("BASE vs FINE-TUNED MODEL COMPARISON")
    print("=" * 70)

    for split in ["train", "held_out", "ood"]:
        base_key = f"base_{split}"
        ft_key = f"ft_{split}"
        if base_key not in results or ft_key not in results:
            continue

        base_data = results[base_key]
        ft_data = results[ft_key]

        base_means = []
        ft_means = []
        improvements = []

        print(f"\n--- {split.upper()} ---")
        for scenario_id in sorted(base_data.keys()):
            if scenario_id not in ft_data:
                continue
            b = base_data[scenario_id]
            f = ft_data[scenario_id]
            delta = f["mean"] - b["mean"]
            base_means.append(b["mean"])
            ft_means.append(f["mean"])
            improvements.append(delta)
            marker = "  " if abs(delta) < 0.01 else (" +" if delta > 0 else " -")
            print(f"  {scenario_id:30s} base={b['mean']:.3f}  ft={f['mean']:.3f}  delta={delta:+.3f} {marker}")

        if len(base_means) >= 2:
            # Paired t-test
            t_stat, p_val_t = stats.ttest_rel(ft_means, base_means)
            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, p_val_w = stats.wilcoxon(
                    np.array(ft_means) - np.array(base_means),
                    alternative="greater"
                )
            except ValueError:
                p_val_w = float("nan")

            mean_improvement = np.mean(improvements)
            print(f"\n  Aggregate: base_mean={np.mean(base_means):.3f}, ft_mean={np.mean(ft_means):.3f}")
            print(f"  Mean improvement: {mean_improvement:+.3f}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_val_t:.4f}")
            print(f"  Wilcoxon signed-rank (one-sided): p={p_val_w:.4f}")
            print(f"  Effect size (Cohen's d): {mean_improvement / (np.std(improvements) + 1e-10):.3f}")
            n_improved = sum(1 for d in improvements if d > 0.01)
            n_degraded = sum(1 for d in improvements if d < -0.01)
            n_same = len(improvements) - n_improved - n_degraded
            print(f"  Improved: {n_improved}/{len(improvements)}, Same: {n_same}/{len(improvements)}, Degraded: {n_degraded}/{len(improvements)}")


def generate_comprehensive_plots(results: Dict):
    """Generate publication-quality plots."""
    Path("plots").mkdir(exist_ok=True)
    ft_model = results.get("ft_model")
    has_ft = ft_model is not None

    # Color scheme
    BASE_COLOR = "#4C72B0"
    FT_COLOR = "#DD8452"

    # Plot 1: Heatmap of coherence scores per scenario
    fig, ax = plt.subplots(figsize=(10, 8))
    all_scenarios = []
    base_scores = []
    ft_scores = []

    for split in ["train", "held_out", "ood"]:
        base_key = f"base_{split}"
        if base_key not in results:
            continue
        for scenario_id in sorted(results[base_key].keys()):
            all_scenarios.append(f"[{split[:3]}] {scenario_id}")
            base_scores.append(results[base_key][scenario_id]["mean"])
            if has_ft:
                ft_key = f"ft_{split}"
                ft_scores.append(results.get(ft_key, {}).get(scenario_id, {}).get("mean", np.nan))

    y = np.arange(len(all_scenarios))
    width = 0.35

    ax.barh(y - width/2, base_scores, width, label="Base GPT-4.1-mini",
            color=BASE_COLOR, alpha=0.8)
    if has_ft and ft_scores:
        ax.barh(y + width/2, ft_scores, width, label="Fine-tuned",
                color=FT_COLOR, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(all_scenarios, fontsize=7)
    ax.set_xlabel("Coherence Score (1.0 = perfectly transitive)")
    ax.set_title("Preference Coherence by Scenario")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("plots/coherence_all_scenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/coherence_all_scenarios.png")

    # Plot 2: Aggregate by split with error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = []
    base_agg_means = []
    base_agg_ses = []
    ft_agg_means = []
    ft_agg_ses = []

    split_labels = {"train": "Training\n(ID)", "held_out": "Held-Out\n(ID)", "ood": "Out-of-\nDistribution"}

    for split in ["train", "held_out", "ood"]:
        base_key = f"base_{split}"
        if base_key not in results:
            continue
        splits.append(split_labels.get(split, split))
        scores = [d["mean"] for d in results[base_key].values()]
        base_agg_means.append(np.mean(scores))
        base_agg_ses.append(np.std(scores) / np.sqrt(len(scores)))

        if has_ft:
            ft_key = f"ft_{split}"
            if ft_key in results:
                ft_s = [d["mean"] for d in results[ft_key].values()]
                ft_agg_means.append(np.mean(ft_s))
                ft_agg_ses.append(np.std(ft_s) / np.sqrt(len(ft_s)))

    x = np.arange(len(splits))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_agg_means, width, yerr=base_agg_ses,
                   label="Base GPT-4.1-mini", color=BASE_COLOR, alpha=0.8, capsize=5)
    if has_ft and ft_agg_means:
        bars2 = ax.bar(x + width/2, ft_agg_means, width, yerr=ft_agg_ses,
                       label="Fine-tuned", color=FT_COLOR, alpha=0.8, capsize=5)

    ax.set_ylabel("Mean Coherence Score")
    ax.set_title("Preference Coherence: Base vs Fine-tuned Model")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9)
    if has_ft and ft_agg_means:
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/aggregate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/aggregate_comparison.png")

    # Plot 3: Improvement distribution (if fine-tuned model exists)
    if has_ft:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_deltas = {"train": [], "held_out": [], "ood": []}

        for split in ["train", "held_out", "ood"]:
            base_key = f"base_{split}"
            ft_key = f"ft_{split}"
            if base_key not in results or ft_key not in results:
                continue
            for sid in results[base_key]:
                if sid in results[ft_key]:
                    delta = results[ft_key][sid]["mean"] - results[base_key][sid]["mean"]
                    all_deltas[split].append(delta)

        positions = []
        data = []
        labels = []
        colors = []
        split_colors = {"train": "#4C72B0", "held_out": "#55A868", "ood": "#C44E52"}

        for i, (split, deltas) in enumerate(all_deltas.items()):
            if deltas:
                positions.append(i)
                data.append(deltas)
                labels.append(split_labels.get(split, split))
                colors.append(split_colors[split])

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Change in Coherence Score")
            ax.set_title("Distribution of Coherence Changes After Fine-tuning")

            plt.tight_layout()
            plt.savefig("plots/improvement_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved plots/improvement_distribution.png")

    # Plot 4: Domain-level comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    domain_base = defaultdict(list)
    domain_ft = defaultdict(list)

    for split in ["train", "held_out", "ood"]:
        base_key = f"base_{split}"
        ft_key = f"ft_{split}"
        if base_key in results:
            for sid, data in results[base_key].items():
                domain_base[data["domain"]].append(data["mean"])
        if has_ft and ft_key in results:
            for sid, data in results[ft_key].items():
                domain_ft[data["domain"]].append(data["mean"])

    domains = sorted(domain_base.keys())
    x = np.arange(len(domains))
    width = 0.35

    base_d_means = [np.mean(domain_base[d]) for d in domains]
    base_d_ses = [np.std(domain_base[d])/np.sqrt(len(domain_base[d])) if len(domain_base[d]) > 1 else 0
                  for d in domains]

    ax.bar(x - width/2, base_d_means, width, yerr=base_d_ses,
           label="Base", color=BASE_COLOR, alpha=0.8, capsize=5)

    if has_ft and domain_ft:
        ft_d_means = [np.mean(domain_ft[d]) if domain_ft[d] else 0 for d in domains]
        ft_d_ses = [np.std(domain_ft[d])/np.sqrt(len(domain_ft[d]))
                    if len(domain_ft[d]) > 1 else 0 for d in domains]
        ax.bar(x + width/2, ft_d_means, width, yerr=ft_d_ses,
               label="Fine-tuned", color=FT_COLOR, alpha=0.8, capsize=5)

    ax.set_ylabel("Mean Coherence Score")
    ax.set_title("Coherence by Domain")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.15)
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/coherence_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/coherence_by_domain.png")


def main():
    # Try to load full evaluation first, fall back to baseline
    try:
        results = load_results("results/evaluation.json")
        print("Loaded full evaluation results")
    except FileNotFoundError:
        try:
            results = load_results("results/evaluation_baseline.json")
            print("Loaded baseline evaluation results (no fine-tuned model)")
        except FileNotFoundError:
            print("No evaluation results found. Run evaluation first.")
            return

    analyze_baseline(results)
    compare_models(results)
    generate_comprehensive_plots(results)


if __name__ == "__main__":
    main()
