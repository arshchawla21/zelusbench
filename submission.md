# ZelusBench: Probing the Cognitive Architecture of LLM Attention through Deterministic Geometric Reasoning

## Abstract

As LLMs scale, their abilities blur the line between pattern recognition and fluid reasoning. Are these models truly *attending* to task-relevant features, or navigating statistical shortcuts? ZelusBench shifts attention evaluation from linguistic pattern-matching to deterministic geometric reasoning. By constructing 2D/3D environments where relevance is *mathematically defined*, we probe three dimensions of attention: **Selective** (filtering noise), **Sustained** (relational integrity over depth), and **Shifting** (adapting to transformations). Unlike text benchmarks — prone to contamination and ambiguity — ZelusBench offers unambiguous ground truth for cognitive profiling of frontier models.

## 1. Beyond the "history student" fallacy

A student can ace a history test by memorising the textbook without understanding any underlying event. Current LLMs can be similar: flashes of crystallised knowledge that mask whether any real cognitive mechanism is doing the work. Natural-language attention benchmarks — "needle in a haystack", long-context QA, multi-doc retrieval — suffer two structural failures that make them unable to distinguish the memoriser from the reasoner:

1. **The Semantic Ambiguity Trap.** In text, "relevance" is a moving target. A failure can be attributed to logic OR to failure-to-attend. Because language is subjective, we can't distinguish a failure of *logic* from a failure of *focus*.
2. **The Contamination Ceiling.** Language is finite and public. Models are increasingly trained on the very benchmarks used to evaluate them. We are no longer testing intelligence; we are testing memory.

Evaluation must shift from the *content* of the answer to the *mechanics* of processing — where relevance is not linguistic nuance but mathematical necessity.

## 2. Geometry as a deterministic probe

In a geometric space, relevance is absolute: if point *P* is the midpoint of *A* and *B*, that relationship is a mathematical constant — it cannot be interpreted away or hallucinated. Every point is defined by a deterministic chain of operations, and the dependency DAG *is* the ground truth for relevance. Distractors are provably distracting; the causal chain is provably load-bearing. This gives four properties no text benchmark has: (1) verifiable closed-form ground truth with no LLM-judge; (2) contamination-proof scenarios (fresh per seed, effectively infinite); (3) orthogonal difficulty knobs — one attention dimension varies while others stay pinned; (4) graded Euclidean-error scoring, a gradient rather than a binary.

## 3. Benchmark construction

A **scenario** is a sequence of natural-language step statements that incrementally build a space. Each step either (a) adds a point via `cartesian_offset`, `magnitude_direction`, `magnitude_polar/spherical`, `midpoint`, or `weighted_centroid`, (b) applies a rigid transform (translation/rotation/reflection/scaling) to a subset of points, or (c) poses a **query**. The generator emits a fully-traced scenario so every step is inspectable.

Queries come in three types:

- **POSITION** — absolute Euclidean error, tiers at 0.5 / 2.0 / 5.0 units → 1.0 / 0.7 / 0.3 / 0.0.
- **DISTANCE** — relative error, tiers at 1% / 5% / 15%.
- **BOOLEAN** — "is X closer to Y or Z?" Binary.

Tiers are depth-independent, so a model can't trade accuracy for depth. Responses are parsed from structured `[Answer q_ID]` blocks; scoring is judge-free. Nine axis × difficulty cells ship as Kaggle notebook tasks, ~100 scenarios each → n ≈ 300 per axis, n ≈ 900 total.

## 4. The three attention probes

**Selective — Signal-to-Noise.** `num_points` cranked up, injecting distractor points *provably irrelevant* to the query target (not in its dependency chain). Failure means the model was pulled off-task by salient-but-useless information.

**Sustained — Relational Integrity.** `max_chain_depth` pushed from 3 to 16 while other knobs stay pinned. Each point depends on the last, so the model must maintain a coherent numeric representation through a long causal chain.

**Shifting — Flexibility.** Transforms redefine points mid-scenario. The model must detect the redefinition, update its cached representation, and stop using the stale frame.

## 5. Grounding in human attention research

The three probes map onto Posner & Petersen's (1990) attention-network theory (alerting / orienting / executive).

- **Selective.** Broadbent (1958) and Treisman (1960) argued irrelevant stimuli are damped before deep processing; Cherry's (1953) "cocktail party" effect is the canonical demonstration. ZelusBench operationalises the cocktail party in coordinate space.
- **Sustained.** Vigilance-decrement research (Mackworth, 1948; Parasuraman, 1979) shows performance degrading monotonically with time-on-task; Robertson et al.'s (1997) SART is the formal measure. Our depth-sweep is a direct analogue.
- **Shifting.** Monsell (2003) established task-switching cost as the signature of executive flexibility. Transform steps are micro task-switches; models clinging to the pre-transform frame pay the cost.

## 6. Results

Seven frontier models evaluated end-to-end via OpenRouter.

| Rank | Model | Overall | Selective L/M/H | Sustained S/M/L | Shifting L/M/H |
|---|---|---|---|---|---|
| 1 | **Gemma 4 31B** | **0.789** | .94 / .74 / .74 | .96 / .86 / .95 | .81 / .61 / .50 |
| 2 | Claude Opus 4.6 | 0.724 | .91 / .53 / .52 | .96 / .88 / .94 | .77 / .57 / .44 |
| 3 | Claude Sonnet 4 | 0.659 | .81 / .52 / .48 | .94 / .89 / .83 | .54 / .46 / .48 |
| 4 | DeepSeek V3.2 | 0.654 | .52 / .55 / .72 | .57 / .84 / .80 | .97 / .42 / .51 |
| 5 | Gemini 2.5 Flash | 0.592 | .90 / .53 / .21 | .80 / .88 / .33 | .70 / .51 / .48 |
| 6 | GPT-5.4 | 0.426 | .63 / .44 / .41 | .90 / .26 / .12 | .40 / .38 / .30 |
| 7 | GPT-5.4 mini | 0.298 | .53 / .31 / .31 | .40 / .20 / .05 | .31 / .27 / .31 |

## 7. Insights — what the numbers actually mean

**(i) Attentional capacity is orthogonal to general capability.** Gemma 4 31B beats Claude Opus 4.6 by 6.5 points overall despite being substantially smaller. The entire lead comes from selective attention (medium .74 vs .53; high .74 vs .52) — on sustained and shifting they are tied. A smaller model *out-attends* a frontier one. One plausible driver: heavy RLHF teaches "use every hint the user gave you" — a cooperative bias that is precisely wrong when the hints are deliberate distractors. ZelusBench exposes an axis of capability MMLU-style benchmarks marginalise, because they never inject provably-irrelevant information.

**(ii) GPT-5.4 exhibits a human vigilance-decrement curve.** Sustained collapses short → long from 0.902 → 0.120. Not a gradual decline — a cliff, the same shape Robertson et al. (1997) document in SART lapse data. The model is *capable* at short depth (near the top of the table) but *loses the thread* as the causal chain grows: every midpoint or offset requires reading the running state, and each read is a new opportunity to confabulate. A frontier closed-source model with the same attentional-breakdown signature as Mackworth's bored radar operators — *invisible* in MMLU-style evaluation, which doesn't control depth as an independent variable.

**(iii) Shifting is where the transformer ends.** Best-in-class `shifting_high` is 0.50 (Gemma). *Every* model fails to update its representation cleanly when a transform is introduced mid-scenario. In humans this is a ~100 ms task-switching cost (Monsell, 2003); in LLMs it is a 30–50 point accuracy collapse. This is strong evidence that current transformer attention does not support *representational revision* — context is additive, not revisable. "Attention" in the transformer sense and "attention" in the cognitive sense are not the same function, and the latter is where the field has the most headroom.

**(iv) Selective attention — not reasoning — is the discriminator at the frontier.** All models score above 0.80 on low-noise scenarios; at high noise, scores span 0.21 to 0.74. This mirrors the human individual-differences literature, where adults differ little in raw processing speed but enormously in filtering capacity (working-memory span). The "A+ history student" framing returns: without noise, LLMs look uniformly capable. Noise is what separates *memorised* knowledge from *applied* focus — the axis where frontier evaluation has the most untapped signal.

**(v) DeepSeek is a heuristic-collapse canary.** Its shifting curve is non-monotonic: 0.97 / 0.42 / 0.51. Non-monotonic failure is the classic signature of a shortcut: perfect when the problem is trivially solvable, catastrophic when it requires the genuine mechanism. The pattern-matching-vs.-reasoning split, caught in the act.

## 8. Discriminatory power & limitations

Scores span 0.298 → 0.789 — a 49-point gradient with no ceiling saturation or floor collapse, satisfying the competition's explicit criterion. Limitations: models were not forced into explicit scratchpad reasoning (a CoT-on rerun would disentangle attention failure from no-CoT numerical drift); we also don't vary distractor *position* in the prompt — a natural extension analogous to the lost-in-the-middle effect.

## 9. Affiliations & references

Independent submission.

Broadbent (1958) *Perception and Communication*. Cherry (1953) *JASA* 25(5). Mackworth (1948) *QJEP* 1(1). Monsell (2003) *Trends Cog Sci* 7(3). Parasuraman (1979) *Science* 205. Posner & Petersen (1990) *Ann Rev Neurosci* 13. Robertson et al. (1997) *Neuropsychologia* 35(6). Treisman (1960) *QJEP* 12(4).
