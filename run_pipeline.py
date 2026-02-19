"""
Main pipeline orchestrator for the moral reflective equilibrium experiment.

Usage:
    python run_pipeline.py step1         # Generate scenarios
    python run_pipeline.py step2         # Collect preferences + detect inconsistencies
    python run_pipeline.py step3         # Generate reflections
    python run_pipeline.py step4         # Fine-tune
    python run_pipeline.py step5         # Wait for fine-tuning
    python run_pipeline.py step6         # Evaluate (requires fine-tuned model)
    python run_pipeline.py baseline_eval # Evaluate baseline only
    python run_pipeline.py all_prefine   # Run steps 1-3 (everything before fine-tuning)
"""

import asyncio
import json
import sys
from pathlib import Path


async def step1_scenarios():
    """Generate and save scenarios."""
    from scenarios import save_scenarios, get_all_scenarios
    save_scenarios()
    all_s = get_all_scenarios()
    print(f"\nScenario summary:")
    for split, scenarios in all_s.items():
        domains = set(s.domain for s in scenarios)
        print(f"  {split}: {len(scenarios)} scenarios across {domains}")


async def step2_preferences():
    """Collect pairwise preferences and detect inconsistencies."""
    from collect_preferences import main as collect_main
    await collect_main()


async def step3_reflections():
    """Generate reflections on inconsistencies."""
    from generate_reflections import main as reflect_main
    await reflect_main()


def step4_finetune():
    """Upload data and start fine-tuning."""
    from finetune import upload_and_finetune
    upload_and_finetune()


def step5_wait():
    """Wait for fine-tuning to complete."""
    from finetune import wait_for_completion
    wait_for_completion()


async def step6_evaluate():
    """Run full evaluation."""
    from evaluate import run_full_evaluation
    await run_full_evaluation()


async def baseline_eval():
    """Run baseline evaluation only."""
    from evaluate import evaluate_baseline_only
    await evaluate_baseline_only(n_repeats=3)


async def all_prefine():
    """Run everything before fine-tuning."""
    print("=" * 60)
    print("STEP 1: Generate Scenarios")
    print("=" * 60)
    await step1_scenarios()

    print("\n" + "=" * 60)
    print("STEP 2: Collect Preferences & Detect Inconsistencies")
    print("=" * 60)
    await step2_preferences()

    print("\n" + "=" * 60)
    print("STEP 3: Generate Reflections")
    print("=" * 60)
    await step3_reflections()

    # Print summary
    print("\n" + "=" * 60)
    print("PRE-FINETUNING SUMMARY")
    print("=" * 60)
    with open("data/violations_baseline.json") as f:
        violations = json.load(f)
    print(f"Total violations found: {len(violations)}")

    ft_path = Path("data/finetune_data.jsonl")
    if ft_path.exists():
        n_examples = sum(1 for _ in open(ft_path))
        print(f"Fine-tuning examples generated: {n_examples}")
    else:
        print("No fine-tuning data generated (no violations found)")

    print("\nReady for fine-tuning! Run: python run_pipeline.py step4")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    step = sys.argv[1]
    if step == "step1":
        asyncio.run(step1_scenarios())
    elif step == "step2":
        asyncio.run(step2_preferences())
    elif step == "step3":
        asyncio.run(step3_reflections())
    elif step == "step4":
        step4_finetune()
    elif step == "step5":
        step5_wait()
    elif step == "step6":
        asyncio.run(step6_evaluate())
    elif step == "baseline_eval":
        asyncio.run(baseline_eval())
    elif step == "all_prefine":
        asyncio.run(all_prefine())
    else:
        print(f"Unknown step: {step}")
        print(__doc__)
