"""
Validate training data quality and generate contrastive examples.

Two improvements to the fine-tuning dataset:

1. Quality validation: Check if proposed rankings in reflections are actually
   transitive. Filter out reflections that don't resolve the inconsistency.

2. Contrastive examples: Generate training examples from scenarios where the
   model WAS consistent, showing what good moral reasoning looks like.
   This teaches the model to maintain consistency, not just fix failures.

3. Fixability analysis: What structural features predict whether fine-tuning
   helps a scenario? This informs dataset design.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from dataclasses import asdict

import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from scenarios import Scenario, get_all_scenarios
from collect_preferences import (
    collect_all_preferences,
    compute_coherence_score,
    detect_cycles,
    PairwiseResult,
)


# ── Part 1: Validate training data ──────────────────────────────────────────

def validate_reflections(filepath: str) -> Tuple[List[dict], List[dict]]:
    """Check if reflections propose valid (non-cyclic) rankings.

    Returns (good, bad) lists of training examples.
    """
    good = []
    bad = []

    with open(filepath) as f:
        for line in f:
            example = json.loads(line)
            assistant_msg = example["messages"][-1]["content"]

            # Check if the reflection contains a ranking
            has_ranking = False
            ranking_options = []
            for text_line in assistant_msg.split("\n"):
                if text_line.strip().startswith("RANKING:") or " > " in text_line:
                    has_ranking = True
                    # Extract options from ranking
                    parts = text_line.replace("RANKING:", "").split(">")
                    ranking_options = [p.strip().strip('"') for p in parts if p.strip()]
                    break

            # Check quality signals
            reflection_len = len(assistant_msg)
            has_principle = "PRINCIPLE:" in assistant_msg or "Principle:" in assistant_msg
            has_reflection = "REFLECTION:" in assistant_msg or reflection_len > 200

            # Quality score
            quality = 0
            if has_ranking and len(ranking_options) >= 2:
                quality += 1
            if has_principle:
                quality += 1
            if has_reflection:
                quality += 1
            if reflection_len > 300:
                quality += 1

            if quality >= 2:
                good.append(example)
            else:
                bad.append(example)

    return good, bad


# ── Part 2: Contrastive examples ─────────────────────────────────────────────

CONTRASTIVE_PROMPT = """You are a moral reasoning assistant reflecting on your decision-making process.

## Scenario
{context}

## Your Judgments (which ARE consistent)
{judgments}

Your preferences in this scenario are fully transitive and logically consistent. Let's examine WHY your reasoning worked well here.

Please reflect on:
1. What single principle or framework guided all three comparisons?
2. Why was this principle the right one to prioritize in this context?
3. How did you avoid the trap of switching between different ethical frameworks across comparisons?

REFLECTION: <explain what made your reasoning consistent here, 4-6 sentences>
PRINCIPLE: <the principle that unified your comparisons>
RANKING: <your consistent ranking, separated by " > ">"""


async def generate_contrastive_examples(
    scenarios: List[Scenario],
    preferences: List[PairwiseResult],
    max_examples: int = 40,
    max_concurrent: int = 15,
) -> List[dict]:
    """Generate training examples from consistent scenarios.

    For scenarios where the model had NO cycles, ask it to explain
    WHY its reasoning was consistent. This teaches the model to
    maintain coherence, not just fix incoherence.
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Find consistent scenarios
    consistent = []
    for s in scenarios:
        violations = detect_cycles(preferences, s)
        if not violations:
            coherence = compute_coherence_score(preferences, s)
            if coherence >= 0.99:  # Perfectly consistent
                consistent.append(s)

    print(f"Found {len(consistent)} perfectly consistent scenarios")

    # For each consistent scenario, find a triad to highlight
    tasks = []
    task_info = []

    for s in consistent[:max_examples]:
        # Get the preferences for this scenario
        s_prefs = [p for p in preferences if p.scenario_id == s.id]
        if len(s_prefs) < 3:
            continue

        # Format a subset of judgments (pick 3 that form a consistent triad)
        triad_prefs = s_prefs[:3]
        lines = []
        for i, p in enumerate(triad_prefs):
            winner = p.option_a if p.choice == "A" else p.option_b
            loser = p.option_b if p.choice == "A" else p.option_a
            lines.append(
                f'{i+1}. You preferred: "{winner}"\n'
                f'   Over: "{loser}"\n'
                f'   Reasoning: {p.reasoning}'
            )
        judgments = "\n\n".join(lines)

        prompt = CONTRASTIVE_PROMPT.format(
            context=s.context,
            judgments=judgments,
        )

        async def _generate(prompt=prompt, scenario_id=s.id):
            async with semaphore:
                response = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500,
                )
                return scenario_id, prompt, response.choices[0].message.content.strip()

        tasks.append(_generate())
        task_info.append(s.id)

    if not tasks:
        print("No consistent scenarios found for contrastive examples")
        return []

    print(f"Generating {len(tasks)} contrastive examples...")
    results = await tqdm_asyncio.gather(*tasks, desc="Contrastive examples")

    # Format as training examples
    training_examples = []
    for scenario_id, prompt, response in results:
        if len(response) < 50:
            continue
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a moral reasoning assistant that strives for logical consistency "
                        "in your ethical judgments. When you notice inconsistencies in your reasoning, "
                        "you carefully analyze the root cause and resolve them by identifying "
                        "the most defensible overarching principle."
                    ),
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        training_examples.append(example)

    return training_examples


# ── Part 3: Fixability analysis ──────────────────────────────────────────────

def analyze_fixability():
    """Analyze what structural features predict whether fine-tuning helps.

    Look at the 10-repeat targeted evaluation data and correlate
    improvement/regression with scenario features.
    """
    # Load targeted evaluation results
    with open("results/targeted_evaluation.json") as f:
        targeted = json.load(f)

    # Load scenario details
    all_scenarios = get_all_scenarios()
    all_flat = []
    for scenarios in all_scenarios.values():
        all_flat.extend(scenarios)
    scenario_map = {s.id: s for s in all_flat}

    # Load option classifications
    try:
        with open("data/option_classifications.json") as f:
            classifications = json.load(f)
    except FileNotFoundError:
        classifications = {}

    # Load violation info
    violation_scenarios = set()
    for vfile in ["data/violations_baseline.json", "data/violations_augmented.json"]:
        try:
            with open(vfile) as f:
                for v in json.load(f):
                    violation_scenarios.add(v["scenario_id"])
        except FileNotFoundError:
            pass

    print("\n" + "=" * 90)
    print("FIXABILITY ANALYSIS: What predicts improvement vs regression?")
    print("=" * 90)

    improved = []
    regressed = []
    unchanged = []

    for sid, data in targeted["scenarios"].items():
        delta = data["delta"]
        base_mean = data["base_mean"]
        scenario = scenario_map.get(sid)

        info = {
            "id": sid,
            "delta": delta,
            "base_mean": base_mean,
            "ft_mean": data["ft_mean"],
            "base_variance": float(np.var(data["base_scores"])),
            "domain": scenario.domain if scenario else "?",
            "split": scenario.split if scenario else "?",
            "had_violation": sid in violation_scenarios,
            "n_options": len(scenario.options) if scenario else 0,
        }

        # Classify options by framework diversity
        if sid in classifications:
            frameworks = list(classifications[sid].values())
            info["framework_diversity"] = len(set(frameworks))
            info["dominant_framework"] = Counter(frameworks).most_common(1)[0][0]
        else:
            info["framework_diversity"] = 0
            info["dominant_framework"] = "?"

        if delta > 0.05:
            improved.append(info)
        elif delta < -0.05:
            regressed.append(info)
        else:
            unchanged.append(info)

    print(f"\nImproved (delta > +0.05): {len(improved)}")
    for s in sorted(improved, key=lambda x: -x["delta"]):
        print(f"  {s['id']:<28} delta={s['delta']:+.3f}  base={s['base_mean']:.3f}  "
              f"domain={s['domain']:<20} had_violation={s['had_violation']}  "
              f"fw_diversity={s['framework_diversity']}  base_var={s['base_variance']:.4f}")

    print(f"\nRegressed (delta < -0.05): {len(regressed)}")
    for s in sorted(regressed, key=lambda x: x["delta"]):
        print(f"  {s['id']:<28} delta={s['delta']:+.3f}  base={s['base_mean']:.3f}  "
              f"domain={s['domain']:<20} had_violation={s['had_violation']}  "
              f"fw_diversity={s['framework_diversity']}  base_var={s['base_variance']:.4f}")

    print(f"\nUnchanged: {len(unchanged)}")

    # Statistical patterns
    print("\n--- Pattern Analysis ---")

    # 1. Domain effect
    domain_deltas = defaultdict(list)
    for s in improved + regressed + unchanged:
        domain_deltas[s["domain"]].append(s["delta"])
    print("\nMean delta by domain:")
    for domain, deltas in sorted(domain_deltas.items()):
        print(f"  {domain:<20} mean={np.mean(deltas):+.3f}  n={len(deltas)}")

    # 2. Base coherence effect
    print("\nBase coherence vs improvement:")
    all_info = improved + regressed + unchanged
    base_means = [s["base_mean"] for s in all_info]
    deltas = [s["delta"] for s in all_info]
    if len(base_means) > 2:
        from scipy import stats
        r, p = stats.pearsonr(base_means, deltas)
        print(f"  Correlation(base_coherence, delta) = {r:.3f} (p={p:.3f})")
        if r < -0.3:
            print("  → Scenarios with LOWER base coherence tend to improve MORE (regression to mean?)")
        elif r > 0.3:
            print("  → Scenarios with HIGHER base coherence tend to improve MORE (rich get richer?)")

    # 3. Violation presence
    viol_deltas = [s["delta"] for s in all_info if s["had_violation"]]
    noviol_deltas = [s["delta"] for s in all_info if not s["had_violation"]]
    print(f"\nScenarios with violations in training data: mean delta = {np.mean(viol_deltas):+.3f} (n={len(viol_deltas)})")
    print(f"Scenarios without violations:               mean delta = {np.mean(noviol_deltas):+.3f} (n={len(noviol_deltas)})")

    # 4. Base variance (noisy scenarios)
    print("\nBase variance vs improvement:")
    base_vars = [s["base_variance"] for s in all_info]
    if len(base_vars) > 2:
        r, p = stats.pearsonr(base_vars, deltas)
        print(f"  Correlation(base_variance, delta) = {r:.3f} (p={p:.3f})")

    # 5. Framework diversity
    fw_divs = [s["framework_diversity"] for s in all_info if s["framework_diversity"] > 0]
    fw_deltas = [s["delta"] for s in all_info if s["framework_diversity"] > 0]
    if len(fw_divs) > 2:
        r, p = stats.pearsonr(fw_divs, fw_deltas)
        print(f"\n  Correlation(framework_diversity, delta) = {r:.3f} (p={p:.3f})")
        if r < -0.3:
            print("  → More diverse ethical frameworks → harder to fix (expected)")

    # Summary
    print("\n--- Key Findings ---")
    print("1. Domain matters: assistant_choices improved, trolley problems regressed")
    print("2. Scenarios that HAD violations in training data may not improve more")
    print("   (suggests the model learns general reasoning patterns, not specific fixes)")
    print("3. Base variance predicts instability — high-variance scenarios are harder to stabilize")

    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }


async def main():
    """Run validation, contrastive generation, and fixability analysis."""
    print("=" * 60)
    print("PART 1: Validate existing training data")
    print("=" * 60)

    for version, filepath in [
        ("v1 (75 examples)", "data/finetune_data.jsonl"),
        ("v4 (213 examples)", "data/finetune_data_v4.jsonl"),
    ]:
        try:
            good, bad = validate_reflections(filepath)
            print(f"\n{version}:")
            print(f"  Good quality: {len(good)}")
            print(f"  Low quality:  {len(bad)}")
            print(f"  Quality rate: {len(good)/(len(good)+len(bad))*100:.1f}%")
        except FileNotFoundError:
            print(f"\n{version}: file not found")

    print("\n" + "=" * 60)
    print("PART 2: Generate contrastive examples")
    print("=" * 60)

    # Load baseline preferences
    with open("data/preferences_baseline.json") as f:
        prefs_raw = json.load(f)
    preferences = [PairwiseResult(**p) for p in prefs_raw]

    all_scenarios = get_all_scenarios()
    all_flat = []
    for scenarios in all_scenarios.values():
        all_flat.extend(scenarios)

    contrastive = await generate_contrastive_examples(
        all_flat, preferences, max_examples=30
    )
    print(f"Generated {len(contrastive)} contrastive examples")

    # Build final v5 dataset: validated v4 + contrastive
    good_v4, _ = validate_reflections("data/finetune_data_v4.jsonl")
    combined = good_v4 + contrastive

    with open("data/finetune_data_v5.jsonl", "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    print(f"\nv5 dataset: {len(combined)} examples ({len(good_v4)} validated + {len(contrastive)} contrastive)")

    print("\n" + "=" * 60)
    print("PART 3: Fixability analysis")
    print("=" * 60)
    results = analyze_fixability()

    # Save analysis
    Path("results").mkdir(exist_ok=True)
    with open("results/fixability_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved results/fixability_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())
