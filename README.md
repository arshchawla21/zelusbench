# ZelusBench

> A geometric benchmark that isolates **attention** — selective filtering, sustained tracking, and adaptive updating — from raw reasoning ability.

The math at every step is trivial (vector addition, midpoints, rotation matrices). The challenge is maintaining an accurate internal representation across long, noisy, and disrupted sequences. Because every coordinate is defined by a chain of explicit operations, ground truth is **deterministically computable** — no LLM judge, no semantic ambiguity.

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Three Attention Axes](#three-attention-axes)
3. [Scenario Generation](#scenario-generation)
4. [Geometric Primitives](#geometric-primitives)
5. [Transforms](#transforms)
6. [Scoring](#scoring)
7. [Response Parsing](#response-parsing)
8. [Benchmark Tasks](#benchmark-tasks)
9. [Project Structure](#project-structure)
10. [Testing](#testing)
11. [Running on Kaggle](#running-on-kaggle)

---

## Core Idea

A scenario is a sequence of **statements** (point definitions and transforms) followed by **queries**, presented to the model in natural language. Every point is defined by a chain of operations on existing points, so the dependency DAG *is* the ground truth. Distractors are provably distracting; the causal chain is provably load-bearing.

Four properties no text benchmark has:

1. **Verifiable closed-form ground truth** — no LLM judge.
2. **Contamination-proof** — fresh per seed, effectively infinite.
3. **Orthogonal difficulty knobs** — one attention axis varies while others stay pinned.
4. **Graded scoring** — Euclidean error binned into tiers, not binary.

---

## Three Attention Axes

Each axis maps to one parameter that is varied while everything else is randomized or pinned.

| Axis | Knob | What it tests | Cognitive analogue |
|---|---|---|---|
| **Selective** | `num_points` (noise) | Filtering provably-irrelevant points out of the dependency chain | Cherry's cocktail-party effect (1953) |
| **Sustained** | `max_chain_depth` | Maintaining a coherent numeric state through a long causal chain | SART vigilance decrement (Robertson 1997) |
| **Shifting** | `transform_prob` | Detecting mid-scenario redefinitions and updating the cached representation | Task-switching cost (Monsell 2003) |

---

## Scenario Generation

Scenarios are built by a **single generative loop** — no pre-planning, no separate "distractor" concept. At each step the generator either:

1. **Adds a point** (relative to existing points)
2. **Applies a transform** (rotates, translates, reflects, or scales a subset)
3. **Emits a query**

Generation is fully deterministic from `ScenarioConfig` + `seed`. No randomness at evaluation time.

### Configuration

```python
@dataclass
class ScenarioConfig:
    dim: int = 3
    min_chain_depth: int = 3
    max_chain_depth: int = 7
    leaf_bias: float = 0.5         # 1.0 = pure chain, 0.0 = bushy
    num_points: int = 8            # selective-attention knob
    transform_prob: float = 0.1    # shifting-attention knob
    transform_types: list[str]
    query_types: list[QueryType]   # POSITION, DISTANCE, BOOLEAN
    num_queries: int = 3
    point_def_types: list[str]
    coord_min/max, magnitude_min/max
    query_target_depth | query_min_depth
    seed: int | None
```

`ScenarioConfig.randomize_except(rng, pinned={...})` generates diverse backgrounds while pinning only the variable under test. The same seed index produces the same randomized background across difficulty levels — so any score delta is attributable to the pinned axis, not background variance.

### Phases

```
Phase 1 (depth ramp):  Force linear growth until min_chain_depth is reached.
Phase 2 (generative):  Add points (via leaf_bias) and transforms (via transform_prob).
Phase 3 (queries):     Emit queries targeting deep / specific points.
```

### Example Prompt

```
Process the following 3D spatial reasoning scenario. Statements are
chronological — propagate all transformations before answering.
Format: [Answer q_ID] value — e.g. [Answer q_001] (0.0, 0.0, 0.0)

Point A is 1.8 units from Point O at angle 267 degrees.
Point B is at offset (-3.1, -1.9, 0.0) from Point A.
Point C is 5.5 units from Point B in the direction (-0.3, -0.9, 1.4).
Rotate Point B, Point C by 120 degrees around (0,0,0) on axis (1,0,0).
Point D is the midpoint of Point A, Point C.

[Query q_001] Position of D? (x, y, z)
```

---

## Geometric Primitives

All scenarios live in Cartesian 2D or 3D space. Origin `O` is the implicit zero vector. Angles are degrees in prompts, radians internally.

| Type | Example | Form |
|---|---|---|
| `cartesian_offset` | *"B is at offset (3, -1, 2) from A."* | `target = anchor + offset` |
| `magnitude_direction` | *"B is 5.0 units from A in direction (1, 1, 0)."* | `target = anchor + m·normalize(d)` |
| `magnitude_polar` *(2D)* | *"C is 4.0 units from A at angle 60°."* | `target = anchor + m·[cos θ, sin θ]` |
| `magnitude_spherical` *(3D)* | *"C is 3.2 units from A at θ=45°, φ=30°."* | `target = anchor + m·spherical(θ, φ)` |
| `midpoint` | *"D is the midpoint of A, B, C."* | `target = mean(points)` |
| `weighted_centroid` | *"E is the centroid of A (w=2), B (w=1)."* | `target = weighted_mean(points, w)` |
| `projection` | *"F is the projection of C onto line AB."* | `target = A + ((C-A)·(B-A) / |B-A|²)·(B-A)` |

The `Space` object holds a DAG of `PointDefinition`s and resolves any point to an absolute coordinate by topological walk. The engine is the single source of truth for ground truth.

---

## Transforms

Transforms mutate the geometric state mid-scenario. They are the primary mechanism for testing the **shifting** axis.

| Transform | Parameters |
|---|---|
| `rotation` | `points`, `axis`, `center`, `angle` |
| `translation` | `points`, `displacement` |
| `reflection` | `points`, `plane_normal`, `plane_point` |
| `scaling` | `points`, `center`, `factor` |

When a point is transformed, **everything defined relative to it must update**. The engine propagates through the dependency DAG; the model is expected to do the same.

---

## Scoring

Each query has a closed-form deterministic answer. Scoring is judge-free and depth-independent (so a model can't trade accuracy for depth).

### POSITION queries — absolute Euclidean error

| Tier | Threshold | Score |
|---|---|---|
| EXACT | error < 0.5 units | 1.0 |
| CLOSE | error < 2.0 units | 0.7 |
| APPROXIMATE | error < 5.0 units | 0.3 |
| WRONG | otherwise | 0.0 |

Absolute (not relative) error: a reasoning chain that drifts 5 units off should get the same penalty whether the target lives near the origin or far from it.

### DISTANCE queries — relative error

| Tier | Threshold | Score |
|---|---|---|
| EXACT | rel. error < 1% | 1.0 |
| CLOSE | rel. error < 5% | 0.7 |
| APPROXIMATE | rel. error < 15% | 0.3 |
| WRONG | otherwise | 0.0 |

`rel_error = |pred - truth| / max(|truth|, ε)` with `ε = 1.0` to avoid blow-up near zero.

### BOOLEAN queries — binary

`"Is X closer to Y or Z?"` → 1.0 if exact match, 0.0 otherwise.

### REFUSED

Unparseable model output → 0.0. Tracked separately so refusal can be distinguished from incorrect answers in diagnostic profiles.

---

## Response Parsing

Models are instructed to emit answers as:

```
[Answer q_001] (3.0, -1.5, 2.0)
[Answer q_002] 5.385
[Answer q_003] B
```

The parser is robust to chain-of-thought reasoning leaking into the response. Strategy, in order:

1. **Split by `[Answer q_ID]` tags** — extract the line after each tag.
2. **Fall back to `[Query q_ID]` tags** if the answer count doesn't match.
3. **Fall back to full text** for each query.

Within each block, coordinate / distance / boolean parsers use the **last regex match**, not the first. This means intermediate scratch-work like *"the midpoint is at (1, 2, 3) so..."* is skipped and only the final value is captured.

---

## Benchmark Tasks

Nine Kaggle benchmark notebooks in `tasks/` — three attention axes × three difficulty levels. Each notebook evaluates a model across two pinned levels of its target axis with 10 seeds per level → **20 scenarios per task, 180 total**. All tasks use POSITION queries targeting deep points.

### Selective Attention — `num_points` sweep

Background: depth 5, `transform_prob=0.1`, dim=3. Noise multiplier sets `num_points = round(depth × multiplier)`.

| Task | Noise multiplier | num_points | Distractor share |
|---|---|---|---|
| `selective_short` | 1.0 – 1.5× | 5 – 8 | 0% – 33% |
| `selective_medium` | 2.0 – 3.0× | 10 – 15 | 50% – 67% |
| `selective_long` | 4.0 – 5.0× | 20 – 25 | 75% – 80% |

### Sustained Attention — `max_chain_depth` sweep

Background: noise 1.5×, `transform_prob=0.1`, dim=3. Queries are forced to depth ≥ `depth − 2`.

| Task | Depths | num_points |
|---|---|---|
| `sustained_short` | 3 – 6 | round(depth × 1.5) |
| `sustained_medium` | 9 – 12 | round(depth × 1.5) |
| `sustained_long` | 15 – 18 | round(depth × 1.5) |

### Shifting Attention — `transform_prob` sweep

Background: depth 6, `num_points=12`, dim=3. Queries are placed after transforms, `query_min_depth=4`. Transforms cascade: rotating point D invalidates everything downstream.

| Task | Transform density |
|---|---|
| `shifting_short` | 0.0 – 0.1 |
| `shifting_medium` | 0.2 – 0.3 |
| `shifting_long` | 0.4 – 0.5 |

### Coverage

| Axis | Tasks | Scenarios | Queries |
|---|---|---|---|
| Selective | 3 | 60 | ~180 |
| Sustained | 3 | 60 | ~180 |
| Shifting | 3 | 60 | ~180 |
| **Total** | **9** | **180** | **~540** |

---

## Project Structure

```
zelusbench/
├── README.md
├── submission.md                       # Kaggle competition writeup
├── pyproject.toml
│
├── zelusbench/                         # Core library (uploaded as Kaggle dataset)
│   ├── geometry/
│   │   ├── point.py                    # PointDefinition: 7 def types
│   │   ├── space.py                    # DAG, propagation, resolution
│   │   ├── transforms.py               # rotation/translation/reflection/scaling
│   │   └── vectors.py                  # vec math utilities
│   │
│   ├── scenarios/
│   │   ├── config.py                   # ScenarioConfig + randomize_except
│   │   ├── generator.py                # ScenarioGenerator: phased loop
│   │   └── templates.py                # NL templates for statements/queries
│   │
│   └── evaluation/
│       ├── parser.py                   # Robust last-match extraction
│       ├── scorer.py                   # Tiered absolute/relative/binary scoring
│       └── reports.py                  # Diagnostic profiles
│
├── tasks/                              # 9 Kaggle benchmark notebooks
│   ├── zelusbench_selective_{short,medium,long}.ipynb
│   ├── zelusbench_sustained_{short,medium,long}.ipynb
│   └── zelusbench_shifting_{short,medium,long}.ipynb
│
├── tests/                              # 79 unit tests
│   ├── test_point.py                   # 21 tests — all point def types
│   ├── test_space.py                   # 12 tests — DAG, propagation
│   ├── test_transforms.py              # 14 tests — all transforms
│   └── test_scenarios.py               # 32 tests — generation, parsing, scoring
│
└── viz.ipynb                           # Visualisations (animated 3D scenarios)
```

### Key Design Principles

- **`Space` is the single source of truth.** All transforms mutate the `Space`; ground truth comes from resolving the DAG. If the engine is correct, ground truth is correct.
- **Single generative loop.** Each point, transform, and query is generated inline using only points that already exist. No forward references, no separate distractor concept.
- **Continuous knobs.** `leaf_bias`, `num_points`, `transform_prob` are continuous parameters, not discrete categories — enables smooth difficulty curves.
- **Deterministic from seed.** Every scenario is reproducible from `ScenarioConfig + seed`.
- **Background determinism across levels.** Within a task, the same seed index produces the same randomized background regardless of pinned-knob value, so score deltas isolate the causal effect of the pinned axis.
- **Depth-targeted queries.** When testing sustained attention, queries land at the requested chain depth — not wherever the generator happens to place them.

---

## Testing

79 unit tests covering the four engine layers. Run with:

```bash
uv run pytest
```

| File | Tests | Covers |
|---|---|---|
| `test_point.py` | 21 | Every point definition type resolves correctly given anchor positions; serialization round-trips. |
| `test_space.py` | 12 | DAG construction, topological resolution, transform propagation through dependents, JSON serialization. |
| `test_transforms.py` | 14 | Rotation (2D/3D, arbitrary axis), translation, reflection (line/plane), scaling (with center), composition. |
| `test_scenarios.py` | 32 | Scenario generation, depth targeting, prompt rendering, response parsing (incl. CoT-leak resilience), tiered scoring. |

The engine is the only thing that needs to be correct — if `Space` resolves coordinates faithfully, ground truth is correct by construction. Tests focus on that invariant.

---

## Running on Kaggle

Each notebook in `tasks/` is a standalone Kaggle Benchmarks task using the `kaggle_benchmarks` (kbench) library.

### Setup

1. Upload the `zelusbench/` package as a Kaggle dataset.
2. Each notebook imports from the dataset and uses `@kbench.task` + `llm.prompt()`.
3. Each notebook ends with `%choose <task_name>` to register with the benchmark runner.

### Task Pattern

```python
@kbench.task(name="zelusbench_sustained_long")
def zelusbench_sustained_long(llm) -> tuple[float, float]:
    scores = []
    for depth in [15, 18]:
        for i in range(10):
            bg_rng = random.Random(i * 7919)        # deterministic background
            cfg = ScenarioConfig.randomize_except(bg_rng, pinned={
                "min_chain_depth": depth,
                "max_chain_depth": depth,
                "num_points": round(depth * 1.5),
                "transform_prob": 0.1,
                "dim": 3,
                "query_min_depth": depth - 2,
                "query_types": [QueryType.POSITION],
                "seed": depth * 1000 + i,
            })
            scenario = ScenarioGenerator(cfg).generate(scenario_id=f"d{depth}_s{i}")
            response = llm.prompt(scenario.prompt)
            scores.extend(score_response(response, scenario))
    mean = np.mean([s.score for s in scores])
    sem = np.std([s.score for s in scores]) / np.sqrt(len(scores))
    return mean, sem

zelusbench_sustained_long.run(llm=kbench.llm)
```

Scoring is local and deterministic — no judge model, no network round-trip per query.
