"""
Collect preferences and violations from generated scenarios,
then combine with existing data for an improved training dataset.
"""

import json
import asyncio
from pathlib import Path
from typing import List
from dataclasses import asdict

from scenarios import Scenario
from collect_preferences import (
    collect_all_preferences,
    detect_cycles,
    compute_coherence_score,
    PairwiseResult,
    TransitivityViolation,
)
from generate_reflections import generate_all_reflections, prepare_finetuning_data
from augment_violations import collect_high_temp


async def main():
    # Load generated scenarios
    with open("data/scenarios_generated.json") as f:
        gen_data = json.load(f)

    scenarios = []
    for s in gen_data["generated"]:
        scenarios.append(Scenario(**s))

    print(f"Loaded {len(scenarios)} generated scenarios")

    all_violations = []

    # Round 1: Normal temperature
    print("\n=== Normal temperature preferences ===")
    prefs = await collect_all_preferences(scenarios, max_concurrent=20)

    for s in scenarios:
        violations = detect_cycles(prefs, s)
        coherence = compute_coherence_score(prefs, s)
        if violations:
            all_violations.extend(violations)
            print(f"  {s.id}: coherence={coherence:.2f}, cycles={len(violations)}")

    # Round 2: High temperature
    print("\n=== High temperature preferences (T=0.7) ===")
    hi_prefs = await collect_high_temp(scenarios, temperature=0.7)

    for s in scenarios:
        violations = detect_cycles(hi_prefs, s)
        coherence = compute_coherence_score(hi_prefs, s)
        if violations:
            all_violations.extend(violations)
            print(f"  {s.id}: coherence={coherence:.2f}, cycles={len(violations)}")

    # Deduplicate
    seen = set()
    unique = []
    for v in all_violations:
        key = (v.scenario_id, tuple(sorted(v.cycle)))
        if key not in seen:
            seen.add(key)
            unique.append(v)

    print(f"\nTotal unique violations from generated scenarios: {len(unique)}")

    if unique:
        # Generate reflections
        reflections = await generate_all_reflections(
            unique, scenarios, reflections_per_violation=3
        )
        print(f"Generated {len(reflections)} reflections")

        # Prepare fine-tuning data
        new_ft_data = prepare_finetuning_data(reflections)

        # Combine with existing
        existing_data = []
        with open("data/finetune_data.jsonl") as f:
            for line in f:
                existing_data.append(json.loads(line))

        combined = existing_data + new_ft_data
        with open("data/finetune_data_v3.jsonl", "w") as f:
            for ex in combined:
                f.write(json.dumps(ex) + "\n")

        print(f"\nFinal dataset: {len(combined)} examples ({len(existing_data)} existing + {len(new_ft_data)} new)")
    else:
        print("No new violations found from generated scenarios at either temperature.")

    # Save violation info
    with open("data/violations_generated.json", "w") as f:
        json.dump([asdict(v) for v in unique], f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
