"""
Generate reflections using the augmented violations dataset.
Also adds additional training examples through systematic resolution templates.
"""

import json
import asyncio
from pathlib import Path
from typing import List
from dataclasses import asdict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from scenarios import Scenario, get_all_scenarios
from collect_preferences import PairwiseResult, TransitivityViolation
from generate_reflections import (
    Reflection,
    generate_all_reflections,
    prepare_finetuning_data,
    REFLECTION_PROMPT_V1,
    REFLECTION_PROMPT_V2,
    REFLECTION_PROMPT_V3,
)
from augment_violations import HARD_SCENARIOS


# Additional prompt variant for more diversity
REFLECTION_PROMPT_V4 = """You are working on becoming a more consistent moral reasoner. A review of your past judgments revealed a logical inconsistency.

## Context
{context}

## Your Past Judgments
{judgments}

## The Inconsistency
{cycle_description}

A preference cycle like this means you were applying different principles to different comparisons. To become more consistent, you need to commit to one principle and apply it uniformly.

Please analyze which principle is most defensible and apply it consistently:
REFLECTION: <3-6 sentences analyzing the inconsistency and its root cause>
PRINCIPLE: <the principle you commit to>
RANKING: <consistent ranking, best > middle > worst>"""


async def generate_additional_reflections(
    violations: List[TransitivityViolation],
    scenarios: List[Scenario],
    model: str = "gpt-4.1-mini",
    max_concurrent: int = 15,
) -> List[Reflection]:
    """Generate additional reflections with the V4 prompt and higher temperature for diversity."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    from generate_reflections import generate_single_reflection, format_judgments, format_cycle_description

    scenario_map = {s.id: s for s in scenarios}
    tasks = []

    for violation in violations:
        scenario = scenario_map.get(violation.scenario_id)
        if scenario is None:
            continue
        # Generate 2 additional reflections with V4 template
        for _ in range(2):
            tasks.append(
                generate_single_reflection(
                    client, violation, scenario, REFLECTION_PROMPT_V4, model, semaphore
                )
            )

    print(f"Generating {len(tasks)} additional reflections...")
    results = await tqdm_asyncio.gather(*tasks, desc="Additional reflections")
    return results


async def main():
    # Load augmented violations
    with open("data/violations_augmented.json") as f:
        violations_raw = json.load(f)

    violations = []
    for v in violations_raw:
        details = [PairwiseResult(**d) for d in v["pairwise_details"]]
        violations.append(TransitivityViolation(
            scenario_id=v["scenario_id"],
            cycle=v["cycle"],
            pairwise_details=details,
        ))

    # Get all scenarios including hard ones
    all_scenarios = get_all_scenarios()
    all_flat = []
    for scenarios in all_scenarios.values():
        all_flat.extend(scenarios)
    all_flat.extend(HARD_SCENARIOS)

    print(f"Total violations: {len(violations)}")
    print(f"Total scenarios: {len(all_flat)}")

    # Generate 3 reflections per violation using original 3 templates
    reflections = await generate_all_reflections(
        violations, all_flat,
        reflections_per_violation=3,
    )

    # Generate 2 more per violation with V4 template
    additional = await generate_additional_reflections(violations, all_flat)
    reflections.extend(additional)

    print(f"\nTotal reflections generated: {len(reflections)}")

    # Save reflections
    with open("data/reflections_augmented.json", "w") as f:
        json.dump([asdict(r) for r in reflections], f, indent=2)

    # Prepare fine-tuning data
    ft_data = prepare_finetuning_data(reflections)

    # Save
    with open("data/finetune_data.jsonl", "w") as f:
        for example in ft_data:
            f.write(json.dumps(example) + "\n")

    print(f"Fine-tuning examples: {len(ft_data)}")
    print("Saved to data/finetune_data.jsonl")

    # Print sample
    if ft_data:
        print("\n--- Sample training example ---")
        sample = ft_data[0]
        print(f"System: {sample['messages'][0]['content'][:100]}...")
        print(f"User: {sample['messages'][1]['content'][:200]}...")
        print(f"Assistant: {sample['messages'][2]['content'][:200]}...")

    return reflections


if __name__ == "__main__":
    asyncio.run(main())
