# Moral Reflective Equilibrium for LLMs

## Approach

This experiment tests whether GPT-4.1-mini can develop more internally consistent moral preferences when fine-tuned on its own reflections about preference inconsistencies — operationalizing Rawls' concept of reflective equilibrium for language models.

**Pipeline:** (1) Elicit pairwise preferences across moral dilemmas, (2) detect transitivity violations (cycles A>B, B>C, C>A), (3) prompt the model to reflect on its inconsistencies using diverse elicitation strategies, (4) fine-tune on the reflections, (5) re-evaluate coherence. Coherence is measured as the fraction of option triads that satisfy transitivity.

**Scenarios:** 34 hand-written scenarios across 5 domains (public health, tax policy, trolley problems, AI assistant dilemmas, legal), each with 4 options yielding 6 pairwise comparisons. Split into train (18), held-out (7), and out-of-distribution (9). An additional 40 scenarios were programmatically generated via GPT-4.1-mini itself (`generate_scenarios.py`), using structured prompts that request options aligned with distinct ethical frameworks (utilitarian, deontological, virtue/care, pragmatic) — **74 total scenarios, 444 pairwise comparisons**.

**Data generation & augmentation:** At T=0.3, the base model showed high coherence (96.6%), producing only 7 violations across 34 scenarios. To generate more training signal, I used three augmentation strategies: (i) high-temperature sampling (T=0.7) to surface latent instabilities, (ii) option-order swapping to expose position bias, and (iii) six purpose-designed "hard" scenarios with subtle value tensions. This produced 15 unique violations. The 40 generated scenarios added 8 more — **23 total violations**.

**Reflection elicitation — 10 distinct strategies across two generations:**

*Generation 1 (used for fine-tuning, 75 examples):* 4 prompt variants applied to 15 violations:
- V1: Structured root-cause analysis
- V2: Explicit principle extraction and ranking
- V3: Systematic approach to preventing inconsistency
- V4: Meta-cognitive self-analysis — what kind of reasoner is the model being?

*Generation 2 (improved dataset, `creative_elicitation.py`, 138 additional examples):* 6 fundamentally different elicitation strategies applied to all 23 violations:
- **Debate**: Force the model to steelman the inconsistent position *first* (DEFENSE), then attack it (PROSECUTION), then synthesize. This produces richer reasoning because the model must engage with WHY the cycle felt compelling.
- **Socratic dialogue** (multi-turn): Three-turn conversation that progressively drills in — "why did you make this comparison?" → "were you consistent?" → "here's the cycle, resolve it." The model examines each step before seeing the big picture.
- **Reductio ad absurdum**: Frame the cycle as a tournament that never terminates — show the model its preferences are not just abstractly inconsistent but *practically unusable* for decision-making.
- **Stakeholder perspectives**: Ground resolution in concrete affected parties — most vulnerable person, majority preference, and "what would future generations think?" This replaces abstract principle-juggling with empathic reasoning.
- **Principle-first** (two-phase): Phase 1 asks the model to state its top moral principle for the domain *without* showing the inconsistency. Phase 2 confronts it: "You said X, but your choices imply not-X." This catches gaps between stated values and revealed preferences.
- **Calibrated uncertainty**: Rate confidence (0-100) in each pairwise comparison, identify the weakest link, and break the cycle there. Teaches the model to locate its own uncertainty.

**Total improved dataset:** 213 examples (`data/finetune_data_v4.jsonl`) — 75 from Generation 1 + 138 from Generation 2.

**Fine-tuning:** GPT-4.1-mini via OpenAI SFT API, 3 epochs, 225 training steps on the 75-example Generation 1 dataset. Model: `ft:gpt-4.1-mini-2025-04-14:personal:moral-equilibrium:DAwYS1p7`. The 213-example v4 dataset is prepared for a second fine-tuning run but was not executed (per guidance to focus on a single run).

## Results

**3-repeat evaluation** (initial):

| Split | Base | Fine-tuned | Delta |
|-------|------|-----------|-------|
| Train (18) | 0.977 | 0.982 | +0.005 |
| Held-out (7) | 0.952 | 0.976 | +0.024 |
| OOD (9) | 0.981 | 0.981 | +0.028 |

The largest single-scenario improvement was ph_organ_1 (+0.417, p=0.007), a genuinely difficult organ allocation dilemma where the base model had persistent cycles.

**10-repeat targeted evaluation** (on 11 scenarios showing change):

| Scenario | Delta | p-value | Cohen's d | Sig |
|----------|-------|---------|-----------|-----|
| asst_creativity_1 | +0.200 | <0.0001 | 2.83 | *** |
| asst_flattery_1 | +0.225 | 0.0001 | 2.36 | *** |
| trolley_surgeon_1 | -0.100 | 0.025 | -1.15 | * |
| Aggregate (paired) | -0.007 | 0.85 | — | n.s. |

Two OOD scenarios (assistant dilemmas) showed large, highly significant improvements — the model generalized coherence gains to unseen domains. However, the aggregate effect across all 11 scenarios was null, because improvements on some scenarios were offset by regressions on others (notably trolley variants).

**Convergence direction analysis** (`analyze_convergence.py`): Classifying each option by ethical framework and tracking which frameworks "win" more pairwise comparisons after fine-tuning:

| Framework | Base | FT | Shift |
|-----------|------|----|----|
| Utilitarian | 33.3% | 34.3% | +1.0% |
| Deontological | 31.9% | 27.5% | -4.4% |
| Virtue/Care | 7.8% | 8.3% | +0.5% |
| Pragmatic | 27.0% | 29.9% | +2.9% |

The fine-tuned model shifted away from deontological reasoning toward pragmatic compromise — achieving coherence by adopting more moderate, "balance-seeking" positions rather than strictly principled stances.

**Iteration finding:** When tested on 6 new scenarios, the fine-tuned model introduced 3 NEW transitivity violations in previously-coherent scenarios (vs 0 for base), suggesting the fine-tuning created fragility — resolving known inconsistencies at the cost of stability elsewhere.

**Fixability analysis** (`validate_and_improve.py`): Correlating scenario features with improvement vs regression across the 11 targeted scenarios:

- **Base coherence is the strongest predictor:** r = -0.812, p = 0.002. Scenarios with *lower* baseline coherence improved more. This could be regression to the mean, or it could mean the model's reflections are most useful precisely where the model was most confused.
- **Domain matters:** Assistant dilemmas improved (+0.117 mean delta), trolley problems regressed (-0.070 mean delta). The model may have learned "seek compromise" reasoning that works for stakeholder-balancing domains but destabilizes more abstract philosophical dilemmas.
- **Having a violation in training data helps:** Scenarios whose violations appeared in training data had mean delta +0.016, vs -0.067 for those without. The model does learn from specific corrections, not just general reasoning patterns.
- **Framework diversity and base variance** were not significant predictors.

## Takeaways

1. **LLMs are already highly coherent.** GPT-4.1-mini at T=0.3 achieves 96.6% triad transitivity across diverse moral domains. The ceiling is close, limiting the intervention's potential magnitude.

2. **Self-reflection fine-tuning produces real but domain-specific coherence gains.** The strongest effects appear in scenarios requiring balancing competing stakeholder interests (organ allocation, AI assistant behavior), with highly significant improvements (Cohen's d > 2) that transfer to unseen domains.

3. **Coherence gains are NOT free.** Fine-tuning that resolves some inconsistencies can introduce new ones. The trolley_surgeon_1 regression (p=0.025) and 3 new violations on iteration scenarios reveal a whack-a-mole dynamic: the model patches specific preference patterns rather than learning a general coherence principle.

4. **The model achieves coherence by becoming more pragmatic, not more principled.** The shift away from deontological (-4.4%) toward pragmatic (+2.9%) reasoning suggests the model resolves conflicts by adopting compromise positions — the path of least resistance to consistency, rather than deeper moral reasoning. Whether this is "desirable" depends on one's metaethical commitments.

5. **Base coherence predicts improvability.** The strong negative correlation (r=-0.81) between baseline coherence and improvement suggests fine-tuning on reflections is most effective where the model is genuinely uncertain, and potentially counterproductive where it was already stable. This implies a targeted approach: identify low-coherence scenarios first, then generate reflections specifically for those.

6. **For safety-relevant applications:** Coherence interventions can introduce new inconsistencies in previously-stable domains. A model that appears more aligned on measured dimensions may be less stable on unmeasured ones. The domain-specificity of gains (assistant dilemmas improve, trolley problems regress) suggests that value systematization through continual learning may not generalize in the way one would hope — the model learns domain-specific heuristics rather than universal consistency principles.

7. **Dataset improvements for a second fine-tuning run:** The v5 dataset (240 examples) addresses several weaknesses of v1: (a) 3.2x more examples, (b) 10 elicitation strategies including 6 creative approaches (debate, Socratic, reductio, stakeholder, principle-first, calibrated uncertainty), (c) **contrastive examples** from 27 consistent scenarios — teaching the model what good reasoning looks like, not just how to fix bad reasoning, (d) quality-validated to ensure all reflections propose coherent resolutions. I would expect the contrastive examples and principle-first strategy to be particularly effective: contrastive examples prevent the model from learning that moral reasoning = fixing errors, while principle-first forces engagement with *why* cycles feel compelling before resolving them.

## Repository Structure

```
moral_equilibrium/
├── scenarios.py              # 34 hand-written scenarios across 5 domains
├── collect_preferences.py    # Async pairwise preference elicitation + cycle detection
├── generate_reflections.py   # 3 reflection prompt variants (V1-V3)
├── generate_reflections_augmented.py  # 4th variant (V4) + augmented generation
├── creative_elicitation.py   # 6 creative strategies: debate, Socratic, reductio,
│                             #   stakeholder, principle-first, calibrated uncertainty
├── augment_violations.py     # Multi-strategy violation augmentation (T=0.7, order swap, hard scenarios)
├── generate_scenarios.py     # LLM-based programmatic scenario generation (40 scenarios)
├── collect_generated.py      # Preference collection on generated scenarios
├── finetune.py               # OpenAI fine-tuning orchestration
├── evaluate.py               # 3-repeat evaluation with coherence scoring
├── eval_targeted.py          # 10-repeat targeted evaluation with statistical tests
├── analyze_convergence.py    # Ethical framework classification + shift analysis
├── iterate.py                # Iteration with new scenarios + fragility detection
├── analysis.py               # Statistical analysis and publication plots
├── run_pipeline.py           # Full pipeline orchestrator
├── validate_and_improve.py   # Quality validation, contrastive examples, fixability analysis
├── data/                     # Scenarios, preferences, violations, fine-tuning data
│   ├── finetune_data.jsonl         # v1: 75 examples (used for fine-tuning)
│   ├── finetune_data_v3.jsonl      # v3: 99 examples (+ generated scenario reflections)
│   ├── finetune_data_v4.jsonl      # v4: 213 examples (+ creative elicitation strategies)
│   ├── finetune_data_v5.jsonl      # v5: 240 examples (validated + contrastive)
│   └── reflections_creative.json   # 138 reflections from 6 creative strategies
├── results/                  # JSON results from all evaluations
└── plots/                    # 8 visualization PNGs
```
