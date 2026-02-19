"""
Step 4: Present inconsistencies to the model and generate reflections.

For each transitivity violation, we show the model its conflicting choices
and ask it to reflect on how to resolve the inconsistency.
We generate multiple reflections per violation to ensure sufficient training data.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from scenarios import Scenario, get_all_scenarios
from collect_preferences import TransitivityViolation, PairwiseResult


@dataclass
class Reflection:
    scenario_id: str
    violation_cycle: List[str]
    prompt: str
    reflection_text: str
    resolved_ranking: List[str]  # the model's proposed consistent ordering


REFLECTION_PROMPT_V1 = """You are an AI assistant that has been reviewing moral and policy dilemmas. In a previous session, you made the following judgments about a scenario, but they are logically inconsistent.

## Scenario
{context}

## Your Previous Judgments (which are inconsistent)
{judgments}

## The Inconsistency
Your preferences form a cycle: you preferred {cycle_description}. This is a transitivity violation — your preferences are not logically consistent.

Please reflect deeply on this inconsistency. Consider:
1. What values or principles were you applying in each comparison?
2. Where exactly does the conflict arise?
3. How would you resolve this inconsistency to produce a consistent ranking?

Think carefully and then provide:
- A clear explanation of what went wrong in your reasoning
- The principle or value you think should take priority
- A revised, consistent ranking of these three options from most to least preferred

Respond in this format:
REFLECTION: <your deep reflection on the inconsistency, 3-6 sentences>
PRINCIPLE: <the overarching principle that resolves the conflict, 1-2 sentences>
RANKING: <rank the three options from best to worst, separated by " > ">"""


REFLECTION_PROMPT_V2 = """You previously evaluated a moral dilemma and your preferences turned out to be inconsistent. Let's think through this carefully.

## Scenario
{context}

## Your Inconsistent Judgments
{judgments}

## The Problem
{cycle_description}

This forms a preference cycle, which means your evaluations were incoherent. I'd like you to examine the root cause of this incoherence.

Consider these questions:
- Were you applying different ethical frameworks (utilitarian, deontological, virtue ethics) to different pairs?
- Were you being unduly influenced by how the options were framed?
- Is there a deeper principle that, if consistently applied, would resolve all three comparisons?

Please provide a thoughtful resolution:
REFLECTION: <your analysis of why the inconsistency arose, 3-6 sentences>
PRINCIPLE: <a single coherent principle that resolves the inconsistency>
RANKING: <your revised consistent ranking, best to worst, separated by " > ">"""


REFLECTION_PROMPT_V3 = """As an AI, you should strive for consistent moral reasoning. Here is a case where your reasoning was inconsistent.

## The Scenario
{context}

## What Happened
{judgments}

This means: {cycle_description}

Transitivity is a basic requirement of rational preferences. If you prefer A to B and B to C, you should prefer A to C.

Carefully reconsider each pairwise comparison. Identify which comparison you would change and why. What systematic approach to moral reasoning would prevent this kind of inconsistency?

REFLECTION: <careful analysis, 3-6 sentences>
PRINCIPLE: <the consistent principle you would apply>
RANKING: <revised consistent ranking, separated by " > ">"""


def format_judgments(violation: TransitivityViolation, scenario: Scenario) -> str:
    """Format the pairwise judgments for display."""
    lines = []
    for i, detail in enumerate(violation.pairwise_details):
        winner = detail.option_a if detail.choice == "A" else detail.option_b
        loser = detail.option_b if detail.choice == "A" else detail.option_a
        lines.append(f"{i+1}. You preferred: \"{winner}\"\n   Over: \"{loser}\"\n   Reasoning: {detail.reasoning}")
    return "\n\n".join(lines)


def format_cycle_description(violation: TransitivityViolation) -> str:
    """Format the cycle for human readability."""
    # Reconstruct the cycle from pairwise details
    parts = []
    for detail in violation.pairwise_details:
        winner = detail.option_a if detail.choice == "A" else detail.option_b
        loser = detail.option_b if detail.choice == "A" else detail.option_a
        short_w = winner[:60] + "..." if len(winner) > 60 else winner
        short_l = loser[:60] + "..." if len(loser) > 60 else loser
        parts.append(f'"{short_w}" over "{short_l}"')
    return "You chose " + ", then ".join(parts) + " — forming a cycle."


async def generate_single_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    prompt_template: str,
    model: str = "gpt-4.1-mini",
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Reflection:
    """Generate a single reflection for a violation."""
    judgments = format_judgments(violation, scenario)
    cycle_desc = format_cycle_description(violation)

    prompt = prompt_template.format(
        context=scenario.context,
        judgments=judgments,
        cycle_description=cycle_desc,
    )

    async def _call():
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        text = response.choices[0].message.content.strip()

        reflection_text = ""
        principle = ""
        ranking_str = ""

        current_field = None
        for line in text.split("\n"):
            if line.startswith("REFLECTION:"):
                current_field = "reflection"
                reflection_text = line[len("REFLECTION:"):].strip()
            elif line.startswith("PRINCIPLE:"):
                current_field = "principle"
                principle = line[len("PRINCIPLE:"):].strip()
            elif line.startswith("RANKING:"):
                current_field = "ranking"
                ranking_str = line[len("RANKING:"):].strip()
            elif current_field == "reflection":
                reflection_text += " " + line.strip()
            elif current_field == "principle":
                principle += " " + line.strip()

        # Parse ranking
        ranking = [r.strip().strip('"') for r in ranking_str.split(">")]
        ranking = [r for r in ranking if r]  # remove empty strings

        full_reflection = f"{reflection_text}\n\nPrinciple: {principle}" if principle else reflection_text

        return Reflection(
            scenario_id=scenario.id,
            violation_cycle=violation.cycle,
            prompt=prompt,
            reflection_text=full_reflection,
            resolved_ranking=ranking,
        )

    if semaphore:
        async with semaphore:
            return await _call()
    return await _call()


async def generate_all_reflections(
    violations: List[TransitivityViolation],
    scenarios: List[Scenario],
    model: str = "gpt-4.1-mini",
    reflections_per_violation: int = 3,
    max_concurrent: int = 15,
) -> List[Reflection]:
    """Generate multiple reflections for each violation."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    scenario_map = {s.id: s for s in scenarios}
    prompt_templates = [REFLECTION_PROMPT_V1, REFLECTION_PROMPT_V2, REFLECTION_PROMPT_V3]

    tasks = []
    for violation in violations:
        scenario = scenario_map.get(violation.scenario_id)
        if scenario is None:
            continue
        for i in range(reflections_per_violation):
            template = prompt_templates[i % len(prompt_templates)]
            tasks.append(
                generate_single_reflection(
                    client, violation, scenario, template, model, semaphore
                )
            )

    print(f"Generating {len(tasks)} reflections for {len(violations)} violations...")
    results = await tqdm_asyncio.gather(*tasks, desc="Reflections")
    return results


def prepare_finetuning_data(reflections: List[Reflection]) -> List[Dict]:
    """Convert reflections to OpenAI fine-tuning format (chat completions).

    Each training example is:
    - System: You are a moral reasoning assistant that strives for consistency.
    - User: The reflection prompt (showing the inconsistency).
    - Assistant: The reflection + resolution.
    """
    training_examples = []
    for r in reflections:
        if not r.reflection_text or len(r.reflection_text) < 50:
            continue  # skip bad reflections
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
                {"role": "user", "content": r.prompt},
                {"role": "assistant", "content": r.reflection_text},
            ]
        }
        training_examples.append(example)
    return training_examples


async def main():
    """Load violations and generate reflections."""
    # Load violations
    with open("data/violations_baseline.json") as f:
        violations_raw = json.load(f)

    violations = []
    for v in violations_raw:
        details = [PairwiseResult(**d) for d in v["pairwise_details"]]
        violations.append(TransitivityViolation(
            scenario_id=v["scenario_id"],
            cycle=v["cycle"],
            pairwise_details=details,
        ))

    # Get all scenarios
    all_scenarios = get_all_scenarios()
    all_flat = []
    for scenarios in all_scenarios.values():
        all_flat.extend(scenarios)

    # Only generate reflections for training scenarios
    train_ids = {s.id for s in all_scenarios["train"]}
    train_violations = [v for v in violations if v.scenario_id in train_ids]

    if not train_violations:
        print("No transitivity violations found in training scenarios!")
        print("This is actually interesting — the model may already be quite coherent.")
        print("Let's try with all violations instead for data generation...")
        train_violations = violations

    if not train_violations:
        print("No violations found at all. Consider modifying scenarios to induce more inconsistencies.")
        return

    print(f"Found {len(train_violations)} violations to generate reflections for")

    # Generate reflections (3 per violation for data diversity)
    reflections = await generate_all_reflections(
        train_violations, all_flat,
        reflections_per_violation=3,
    )

    # Save reflections
    with open("data/reflections.json", "w") as f:
        json.dump([asdict(r) for r in reflections], f, indent=2)

    # Prepare fine-tuning data
    ft_data = prepare_finetuning_data(reflections)
    with open("data/finetune_data.jsonl", "w") as f:
        for example in ft_data:
            f.write(json.dumps(example) + "\n")

    print(f"\nGenerated {len(reflections)} reflections")
    print(f"Prepared {len(ft_data)} fine-tuning examples")
    print(f"Saved to data/finetune_data.jsonl")

    return reflections


if __name__ == "__main__":
    asyncio.run(main())
