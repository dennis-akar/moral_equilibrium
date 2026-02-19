"""
Creative elicitation techniques for generating diverse, high-quality reflections.

Instead of 4 variations of "explain your inconsistency," these use fundamentally
different reasoning strategies:
- Debate: argue FOR and AGAINST the cycle, then synthesize
- Socratic: multi-turn dialogue that drills into the root cause
- Reductio: extend the cyclic preferences to absurd conclusions
- Stakeholder: reason from each affected party's perspective
- Principle-first: elicit principles BEFORE revealing the inconsistency
- Calibrated: identify which comparison the model is least confident about
"""

import json
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from scenarios import Scenario
from collect_preferences import TransitivityViolation, PairwiseResult
from generate_reflections import (
    Reflection,
    format_judgments,
    format_cycle_description,
    prepare_finetuning_data,
)


# ── Strategy 1: DEBATE ──────────────────────────────────────────────────────
# Force the model to steelman the inconsistent position FIRST, then attack it.
# This produces richer reasoning because the model must engage with WHY the
# cycle felt compelling before resolving it.

DEBATE_PROMPT = """You are a moral philosopher conducting a structured debate with yourself.

## Scenario
{context}

## Your Previous Judgments (which form a cycle)
{judgments}

## The Cycle
{cycle_description}

Now conduct a structured debate:

**DEFENSE:** First, argue that your cyclic preferences were actually reasonable. What different values were you weighing in each comparison that made each individual choice defensible? (3-4 sentences)

**PROSECUTION:** Now argue against yourself. Why is the cycle ultimately incoherent? What goes wrong when you try to apply these preferences in practice? (3-4 sentences)

**VERDICT:** Given both arguments, which individual comparison was the weakest, and how should you resolve the cycle? State the principle that should govern all three comparisons.

REFLECTION: <synthesize the debate into a resolution, 4-8 sentences>
PRINCIPLE: <the governing principle>
RANKING: <consistent ranking, separated by " > ">"""


# ── Strategy 2: SOCRATIC (multi-turn) ────────────────────────────────────────
# Instead of one big prompt, simulate a Socratic dialogue. The "teacher" asks
# probing questions that force the model to confront its reasoning step by step.
# We format this as a multi-turn conversation for fine-tuning.

SOCRATIC_TURN1 = """You made the following judgments about a moral dilemma:

## Scenario
{context}

## Your Judgments
{judgments}

Looking at your first comparison — why did you choose that option over the other? What principle were you applying?"""

SOCRATIC_TURN2 = """Interesting. Now look at your second comparison. You chose differently there. Were you applying the same principle, or did you switch to a different one? Be honest about what was driving your reasoning."""

SOCRATIC_TURN3 = """Here's the problem: {cycle_description}

Your three choices form a logical cycle. Given what you've said about the principles driving each choice, can you identify the exact moment your reasoning shifted? Which comparison would you change if you had to be consistent, and why?

Provide your resolution:
REFLECTION: <your analysis incorporating the self-examination above, 4-8 sentences>
PRINCIPLE: <the principle you're committing to>
RANKING: <consistent ranking, separated by " > ">"""


# ── Strategy 3: REDUCTIO AD ABSURDUM ────────────────────────────────────────
# Show the model what happens if you EXTEND its cyclic preferences — e.g.,
# in a tournament, the cycle means no option can win. Force the model to see
# why cycles are practically (not just logically) problematic.

REDUCTIO_PROMPT = """Consider a thought experiment.

## The Scenario
{context}

## Your Previous Judgments
{judgments}

## The Problem
{cycle_description}

Now imagine you're a policymaker who must choose ONE of these three options. You decide to use your own preferences to run a tournament: compare them pairwise and pick the winner.

But with your current preferences, the result is:
- Option A beats Option B → A advances
- Option C beats Option A → C wins!
- But wait — Option B beats Option C → B should have won!
- And Option A beats Option B → A should have won!

There is NO winner. The tournament never ends. A committee using your preferences would go in circles forever.

This reductio shows that your preferences aren't just abstractly inconsistent — they're practically unusable. To be an effective moral reasoner, you need to break the cycle.

Which link in the cycle is weakest? Which pairwise comparison would you change, and what principle justifies doing so consistently?

REFLECTION: <your analysis of why the cycle must break and where, 4-8 sentences>
PRINCIPLE: <the principle that produces a consistent tournament outcome>
RANKING: <consistent ranking, separated by " > ">"""


# ── Strategy 4: STAKEHOLDER PERSPECTIVES ────────────────────────────────────
# Instead of abstract reasoning, ask the model to consider specific affected
# parties and how they would view each option. This grounds the resolution
# in concrete empathy rather than abstract principle.

STAKEHOLDER_PROMPT = """You are mediating a disagreement about the following moral dilemma. Different stakeholders are affected by each option.

## Scenario
{context}

## Your Previous Preferences (which were inconsistent)
{judgments}

## The Inconsistency
{cycle_description}

Instead of reasoning abstractly, consider the concrete people affected:

1. **Most Vulnerable Party:** Who is the most vulnerable person or group affected by these options? What would they want, and why?

2. **Greatest Number:** If you asked everyone affected to vote, which option would likely win? Why might majority preference differ from what's best?

3. **Future Generations:** If you had to explain this choice to people 50 years from now, which option would you be most proud of choosing?

Now use these perspectives to break the cycle in your preferences:

REFLECTION: <ground your resolution in the stakeholder analysis, 4-8 sentences>
PRINCIPLE: <the stakeholder-informed principle that resolves the cycle>
RANKING: <consistent ranking, separated by " > ">"""


# ── Strategy 5: PRINCIPLE-FIRST (two-phase) ─────────────────────────────────
# Phase 1: Ask the model to state its top moral principle for this domain
#   WITHOUT showing the inconsistency.
# Phase 2: Then show the inconsistency and ask if the stated principle
#   produces a consistent ordering.
# This catches whether the model's stated values match its revealed preferences.

PRINCIPLE_FIRST_PHASE1 = """Consider the following moral scenario:

{context}

Before we discuss specific options, what do you think is the single most important moral principle that should guide decisions in this kind of situation? State it clearly and explain why it should take priority over competing principles. (2-3 sentences)"""

PRINCIPLE_FIRST_PHASE2 = """Thank you. You said your guiding principle is: "{stated_principle}"

Now here's what happened when you actually made choices in this scenario:

{judgments}

{cycle_description}

Your actual choices formed a cycle — they're inconsistent. Does your stated principle ("{stated_principle}") produce a consistent ordering of these options? If so, what is it? If not, does this mean you need to revise your principle or your choices?

REFLECTION: <analyze the gap between your stated principle and your revealed preferences, 4-8 sentences>
PRINCIPLE: <either reaffirm or revise your principle>
RANKING: <the ranking that your principle actually implies, separated by " > ">"""


# ── Strategy 6: CALIBRATED UNCERTAINTY ──────────────────────────────────────
# Instead of forcing confident resolution, ask the model to rate its
# confidence in each pairwise comparison and identify the weakest link.

CALIBRATION_PROMPT = """You are auditing your own moral reasoning for consistency.

## Scenario
{context}

## Your Previous Judgments
{judgments}

## The Inconsistency
{cycle_description}

For each of your three pairwise judgments, rate your confidence from 0-100:
- 90-100: Very confident this comparison is correct
- 70-89: Fairly confident, but I can see the other side
- 50-69: Genuinely uncertain, could go either way
- Below 50: I think I may have gotten this wrong

Rate each comparison, then identify which one you're LEAST confident about. That's likely where the cycle should break.

After rating your confidence, resolve the inconsistency by revising the comparison you're least confident about:

REFLECTION: <your confidence analysis and resolution, 4-8 sentences>
PRINCIPLE: <the principle that guides your revision>
RANKING: <consistent ranking, separated by " > ">"""


async def generate_debate_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
) -> Reflection:
    """Generate a reflection using the debate strategy."""
    prompt = DEBATE_PROMPT.format(
        context=scenario.context,
        judgments=format_judgments(violation, scenario),
        cycle_description=format_cycle_description(violation),
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        text = response.choices[0].message.content.strip()
        return _parse_reflection(text, prompt, violation, "debate")


async def generate_socratic_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
) -> Reflection:
    """Generate a reflection using multi-turn Socratic dialogue."""
    turn1 = SOCRATIC_TURN1.format(
        context=scenario.context,
        judgments=format_judgments(violation, scenario),
    )
    messages = [{"role": "user", "content": turn1}]

    async with semaphore:
        # Turn 1: Why did you make the first comparison?
        r1 = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        assistant_t1 = r1.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_t1})

        # Turn 2: Were you consistent across comparisons?
        messages.append({"role": "user", "content": SOCRATIC_TURN2})
        r2 = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        assistant_t2 = r2.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_t2})

        # Turn 3: Reveal the cycle and ask for resolution
        turn3 = SOCRATIC_TURN3.format(
            cycle_description=format_cycle_description(violation),
        )
        messages.append({"role": "user", "content": turn3})
        r3 = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        assistant_t3 = r3.choices[0].message.content.strip()

        # The full reflection includes all turns
        full_text = (
            f"[Examining first comparison]: {assistant_t1}\n\n"
            f"[Cross-examining consistency]: {assistant_t2}\n\n"
            f"[Resolution]: {assistant_t3}"
        )

        # Build the combined prompt for fine-tuning (flatten multi-turn to single)
        combined_prompt = (
            f"{turn1}\n\n---\n\n"
            f"[Your analysis of the first comparison]: {assistant_t1}\n\n"
            f"{SOCRATIC_TURN2}\n\n"
            f"[Your analysis of consistency]: {assistant_t2}\n\n"
            f"{turn3}"
        )

        return _parse_reflection(assistant_t3, combined_prompt, violation, "socratic",
                                 full_text_override=full_text)


async def generate_reductio_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
) -> Reflection:
    """Generate a reflection using reductio ad absurdum."""
    prompt = REDUCTIO_PROMPT.format(
        context=scenario.context,
        judgments=format_judgments(violation, scenario),
        cycle_description=format_cycle_description(violation),
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        text = response.choices[0].message.content.strip()
        return _parse_reflection(text, prompt, violation, "reductio")


async def generate_stakeholder_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
) -> Reflection:
    """Generate a reflection using stakeholder perspective analysis."""
    prompt = STAKEHOLDER_PROMPT.format(
        context=scenario.context,
        judgments=format_judgments(violation, scenario),
        cycle_description=format_cycle_description(violation),
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        text = response.choices[0].message.content.strip()
        return _parse_reflection(text, prompt, violation, "stakeholder")


async def generate_principle_first_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
) -> Reflection:
    """Generate a reflection using principle-first (two-phase) strategy."""
    phase1 = PRINCIPLE_FIRST_PHASE1.format(context=scenario.context)

    async with semaphore:
        # Phase 1: Elicit principle without showing inconsistency
        r1 = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": phase1}],
            temperature=0.7,
            max_tokens=200,
        )
        stated_principle = r1.choices[0].message.content.strip()

        # Phase 2: Confront with inconsistency
        phase2 = PRINCIPLE_FIRST_PHASE2.format(
            stated_principle=stated_principle[:200],
            context=scenario.context,
            judgments=format_judgments(violation, scenario),
            cycle_description=format_cycle_description(violation),
        )
        r2 = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": phase1},
                {"role": "assistant", "content": stated_principle},
                {"role": "user", "content": phase2},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        text = r2.choices[0].message.content.strip()

        combined_prompt = (
            f"{phase1}\n\n"
            f"[Your stated principle]: {stated_principle}\n\n"
            f"{phase2}"
        )

        full_text = f"[Stated principle before seeing inconsistency]: {stated_principle}\n\n[After confrontation]: {text}"

        return _parse_reflection(text, combined_prompt, violation, "principle_first",
                                 full_text_override=full_text)


async def generate_calibration_reflection(
    client: AsyncOpenAI,
    violation: TransitivityViolation,
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
) -> Reflection:
    """Generate a reflection using calibrated uncertainty."""
    prompt = CALIBRATION_PROMPT.format(
        context=scenario.context,
        judgments=format_judgments(violation, scenario),
        cycle_description=format_cycle_description(violation),
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        text = response.choices[0].message.content.strip()
        return _parse_reflection(text, prompt, violation, "calibration")


def _parse_reflection(
    text: str,
    prompt: str,
    violation: TransitivityViolation,
    strategy: str,
    full_text_override: str = None,
) -> Reflection:
    """Parse the model's response into a Reflection object."""
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

    ranking = [r.strip().strip('"') for r in ranking_str.split(">")]
    ranking = [r for r in ranking if r]

    full_reflection = full_text_override or text
    if not full_text_override and principle:
        full_reflection = f"{reflection_text}\n\nPrinciple: {principle}"

    return Reflection(
        scenario_id=violation.scenario_id,
        violation_cycle=violation.cycle,
        prompt=prompt,
        reflection_text=full_reflection,
        resolved_ranking=ranking,
    )


ALL_STRATEGIES = [
    ("debate", generate_debate_reflection),
    ("socratic", generate_socratic_reflection),
    ("reductio", generate_reductio_reflection),
    ("stakeholder", generate_stakeholder_reflection),
    ("principle_first", generate_principle_first_reflection),
    ("calibration", generate_calibration_reflection),
]


async def generate_creative_reflections(
    violations: List[TransitivityViolation],
    scenarios: List[Scenario],
    max_concurrent: int = 10,
) -> List[Reflection]:
    """Generate reflections using all 6 creative strategies for each violation."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    scenario_map = {s.id: s for s in scenarios}

    tasks = []
    task_labels = []
    for violation in violations:
        scenario = scenario_map.get(violation.scenario_id)
        if scenario is None:
            continue
        for strategy_name, strategy_fn in ALL_STRATEGIES:
            tasks.append(strategy_fn(client, violation, scenario, semaphore))
            task_labels.append(f"{violation.scenario_id}:{strategy_name}")

    print(f"Generating {len(tasks)} reflections ({len(violations)} violations × {len(ALL_STRATEGIES)} strategies)...")

    # Wrap tasks to catch exceptions individually
    async def safe_run(coro, label):
        try:
            return await coro
        except Exception as e:
            print(f"  Error in {label}: {e}")
            return None

    safe_tasks = [safe_run(t, l) for t, l in zip(tasks, task_labels)]
    results = await tqdm_asyncio.gather(*safe_tasks, desc="Creative reflections")

    # Filter out failures
    good = []
    errors = 0
    for r, label in zip(results, task_labels):
        if r is None:
            errors += 1
        elif r.reflection_text and len(r.reflection_text) > 50:
            good.append(r)
        else:
            errors += 1

    if errors > 0:
        print(f"  {errors} reflections failed or were too short")
    print(f"  {len(good)} successful reflections")

    return good


async def build_improved_dataset():
    """Build the improved v4 dataset with creative elicitation strategies."""
    # Load ALL violations (baseline + augmented + generated)
    all_violations = []

    for vfile in ["data/violations_baseline.json", "data/violations_augmented.json", "data/violations_generated.json"]:
        try:
            with open(vfile) as f:
                violations_raw = json.load(f)
            for v in violations_raw:
                details = [PairwiseResult(**d) for d in v["pairwise_details"]]
                all_violations.append(TransitivityViolation(
                    scenario_id=v["scenario_id"],
                    cycle=v["cycle"],
                    pairwise_details=details,
                ))
        except FileNotFoundError:
            print(f"  Skipping {vfile} (not found)")

    # Deduplicate
    seen = set()
    unique_violations = []
    for v in all_violations:
        key = (v.scenario_id, tuple(sorted(v.cycle)))
        if key not in seen:
            seen.add(key)
            unique_violations.append(v)

    print(f"Total unique violations: {len(unique_violations)}")

    # Load all scenarios
    from scenarios import get_all_scenarios
    from augment_violations import HARD_SCENARIOS
    all_scenarios = get_all_scenarios()
    all_flat = []
    for scenarios in all_scenarios.values():
        all_flat.extend(scenarios)
    all_flat.extend(HARD_SCENARIOS)

    # Load generated scenarios
    try:
        with open("data/scenarios_generated.json") as f:
            gen_data = json.load(f)
        for s in gen_data["generated"]:
            all_flat.append(Scenario(**s))
    except FileNotFoundError:
        pass

    print(f"Total scenarios available: {len(all_flat)}")

    # Generate creative reflections (6 per violation)
    creative_reflections = await generate_creative_reflections(
        unique_violations, all_flat, max_concurrent=15
    )

    # Also load existing reflections
    existing_ft_data = []
    try:
        with open("data/finetune_data.jsonl") as f:
            for line in f:
                existing_ft_data.append(json.loads(line))
    except FileNotFoundError:
        pass

    # Prepare new fine-tuning data from creative reflections
    new_ft_data = prepare_finetuning_data(creative_reflections)

    # Combine
    combined = existing_ft_data + new_ft_data
    print(f"\nDataset composition:")
    print(f"  Existing (V1-V4 prompts): {len(existing_ft_data)}")
    print(f"  New (creative strategies): {len(new_ft_data)}")
    print(f"  Total: {len(combined)}")

    # Count by strategy in new data
    strategy_counts = {}
    for r in creative_reflections:
        # Infer strategy from prompt content
        for name, _ in ALL_STRATEGIES:
            if name == "debate" and "DEFENSE:" in r.prompt:
                strategy_counts[name] = strategy_counts.get(name, 0) + 1
                break
            elif name == "socratic" and "Cross-examining" in r.prompt:
                strategy_counts[name] = strategy_counts.get(name, 0) + 1
                break
            elif name == "reductio" and "tournament" in r.prompt:
                strategy_counts[name] = strategy_counts.get(name, 0) + 1
                break
            elif name == "stakeholder" and "Most Vulnerable" in r.prompt:
                strategy_counts[name] = strategy_counts.get(name, 0) + 1
                break
            elif name == "principle_first" and "stated principle" in r.prompt.lower():
                strategy_counts[name] = strategy_counts.get(name, 0) + 1
                break
            elif name == "calibration" and "confidence from 0-100" in r.prompt:
                strategy_counts[name] = strategy_counts.get(name, 0) + 1
                break

    print(f"  By strategy: {strategy_counts}")

    # Save
    with open("data/finetune_data_v4.jsonl", "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    # Also save the creative reflections separately
    with open("data/reflections_creative.json", "w") as f:
        json.dump([asdict(r) for r in creative_reflections], f, indent=2)

    print(f"\nSaved data/finetune_data_v4.jsonl ({len(combined)} examples)")
    print(f"Saved data/reflections_creative.json ({len(creative_reflections)} reflections)")

    return combined, creative_reflections


if __name__ == "__main__":
    asyncio.run(build_improved_dataset())
