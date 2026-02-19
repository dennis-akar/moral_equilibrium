"""
Moral Reflective Equilibrium for LLMs

Entry point — delegates to run_pipeline.py for the full pipeline,
or run individual modules directly (e.g. python evaluate.py).

See WRITEUP.md for experiment description and results.
See run_pipeline.py for step-by-step pipeline usage.
"""

from run_pipeline import *

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python main.py <step>")
        print("  Steps: step1, step2, step3, step4, step5, step6, baseline_eval, all_prefine")
        print("\nOr run individual modules:")
        print("  python generate_scenarios.py    # Generate scenarios programmatically")
        print("  python creative_elicitation.py  # Generate reflections with creative strategies")
        print("  python eval_targeted.py         # 10-repeat targeted evaluation")
        print("  python analyze_convergence.py   # Ethical framework convergence analysis")
    else:
        import asyncio
        step = sys.argv[1]
        steps = {
            "step1": step1_scenarios,
            "step2": step2_preferences,
            "step3": step3_reflections,
            "step4": lambda: step4_finetune(),
            "step5": lambda: step5_wait(),
            "step6": step6_evaluate,
            "baseline_eval": baseline_eval,
            "all_prefine": all_prefine,
        }
        fn = steps.get(step)
        if fn is None:
            print(f"Unknown step: {step}")
        elif asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
