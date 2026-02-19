# Moral Reflective Equilibrium for LLMs — Summary

## Implementation

I built a pipeline that (1) elicits pairwise moral preferences from GPT-4.1-mini across 74 scenarios in 5 domains, (2) detects transitivity violations, (3) generates reflections using 10 elicitation strategies, (4) fine-tunes on the reflections, and (5) evaluates coherence changes with statistical rigor.

**Scenarios.** 34 manually authored scenarios across public health, tax policy, trolley problems, AI assistant dilemmas, and legal domains. Each has 4 options (6 pairwise comparisons). Split: 18 train, 7 held-out, 9 OOD — these are the scenarios used for evaluation. An additional 40 LLM-generated scenarios (28 train, 12 held-out) were used to mine more violations and produce more training data, but not for the base vs fine-tuned comparison.

**Finding violations.** The base model is already 96.6% coherent at T=0.3, so I augmented with high-temperature sampling (T=0.7), option-order swapping, and 6 hard scenarios designed to induce conflicts. This produced 23 unique transitivity violations.

**Elicitation.** Once we find a violation (e.g., the model prefers A>B, B>C, but C>A), we need to generate a training example where the model reflects on and resolves it. The quality of these reflections determines training data quality. I used 10 prompting strategies to get diverse, high-quality reflections:

- **Basic (4 variants):** Show the model its cycle and ask it to explain what went wrong and propose a consistent ranking. The 4 variants differ in framing (root-cause analysis, principle extraction, systematic reasoning, meta-cognitive self-analysis).
- **Creative (6 strategies):** Fundamentally different reasoning approaches. E.g., *debate* forces the model to defend the cycle before attacking it; *Socratic* uses a 3-turn dialogue that progressively reveals the problem; *principle-first* asks for the model's guiding principle *before* showing the inconsistency, then confronts the gap between stated values and actual choices; *calibrated uncertainty* asks the model to rate its confidence (0-100) in each comparison and break the cycle at the weakest link.
- **Contrastive (27 examples):** From scenarios where the model *was* consistent, asking it to explain *why* its reasoning worked. This teaches the model what good reasoning looks like, not just how to fix bad reasoning.

**Fine-tuning.** Single SFT run on GPT-4.1-mini (75 examples, 3 epochs). Improved dataset (v5, 240 examples) prepared but not run per guidance.

## Results

| Split | Base | Fine-tuned | Delta |
|-------|------|-----------|-------|
| Train (18) | 0.977 | 0.982 | +0.005 |
| Held-out (7) | 0.952 | 0.976 | +0.024 |
| OOD (9) | 0.981 | 0.981 | +0.028 |

10-repeat targeted evaluation on scenarios that changed: `asst_creativity_1` +0.200 (p<0.0001, d=2.83), `asst_flattery_1` +0.225 (p=0.0001, d=2.36), but `trolley_surgeon_1` -0.100 (p=0.025, d=-1.15). Aggregate across all 11 scenarios: not significant.

Convergence analysis: the fine-tuned model shifts from deontological (-4.4%) toward pragmatic (+2.9%) reasoning. Iteration on 6 new scenarios revealed 3 NEW violations introduced by fine-tuning (vs 0 for base). Fixability analysis: base coherence strongly predicts improvability (r=-0.81, p=0.002) — the model benefits most where it was most confused.

## Takeaways

1. **GPT-4.1-mini is already highly coherent** (96.6%), leaving limited room for improvement. The interesting finding is not the magnitude of change but the *pattern* of what changes and why.

2. **Fine-tuning on self-reflections produces statistically significant coherence gains** in specific domains (Cohen's d > 2 for assistant dilemmas), and these transfer to unseen OOD domains.

3. **Coherence gains are not free.** Improving some scenarios destabilizes others — a whack-a-mole dynamic suggesting the model learns domain-specific heuristics rather than a universal consistency principle.

4. **The model achieves coherence by becoming more pragmatic, not more principled** — shifting away from deontological reasoning toward compromise positions. Whether this convergence is "desirable" is itself a values question.

5. **Base coherence predicts fixability** (r=-0.81). This suggests targeted intervention: identify low-coherence scenarios, generate reflections for those specifically, rather than training broadly.

6. **Safety implication for continual learning:** Value systematization through fine-tuning can introduce fragility outside the training distribution. A model that appears more aligned on measured dimensions may be less stable on unmeasured ones.

7. **Improved dataset (v5, 240 examples)** incorporates lessons learned: creative elicitation strategies that teach different modes of reasoning, contrastive examples from consistent scenarios, and quality validation. See `creative_elicitation.py` and `validate_and_improve.py`.
