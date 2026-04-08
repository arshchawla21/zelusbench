# ZelusBench

> A geometric benchmark for measuring LLM attention through deterministic spatial reasoning tasks.

ZelusBench isolates **attention** — selective focus, sustained tracking, and adaptive updating — from raw reasoning ability. The geometric operations at each step are simple; the challenge is maintaining an accurate internal representation across long, noisy, and disrupted sequences.

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Generative Scenario Construction](#generative-scenario-construction)
3. [Geometric Primitives](#geometric-primitives)
4. [Events (Transformations)](#events-transformations)
5. [Benchmark Tasks](#benchmark-tasks)
6. [Scoring & Metrics](#scoring--metrics)
7. [Project Structure](#project-structure)
8. [Running on Kaggle](#running-on-kaggle)

---

## Core Idea

A scenario is a **sequence of statements and queries** presented to the model in natural language. Statements either define points (relative to other points) or apply transformations to the space. Queries ask the model to report computed positions, distances, or relationships.

Because every point is defined relative to others via explicit geometric operations, the ground-truth answer is always **deterministically computable**. No LLM judge is needed — we compare numerical outputs against exact solutions (within a tolerance).

The key insight: **the math at each step is trivial** (vector addition, rotation matrices, midpoints). What makes the task hard is:

- Tracking a growing graph of dependencies across a long prompt
- Ignoring irrelevant points and relationships
- Updating the entire representation when a transformation occurs
- Maintaining accuracy as dimensionality and chain depth increase

This means the benchmark measures **attention**, not mathematical ability.

---

## Generative Scenario Construction

Scenarios are built by a **single generative loop**, not pre-planned upfront. At each step, the generator either:

1. **Adds a point** — defined relative to an existing point (or set of points via midpoint/centroid)
2. **Applies a transform** — rotates, translates, reflects, or scales existing points
3. **Emits a query** — asks about a position, distance, or boolean relationship

Every point is real — there is no separate "distractor" concept. Points that don't matter for a query are simply points the model must track but doesn't need for the answer.

### Configuration Parameters

Three continuous knobs control difficulty:

| Parameter | Range | Effect |
|---|---|---|
| **`leaf_bias`** | 0.0 – 1.0 | Controls graph topology. `1.0` = always extend leaf nodes (linear chains). `0.0` = branch from any point (bushy/diamond graphs). |
| **`num_points`** | 4 – 40+ | Total points generated. More points = more irrelevant information to filter through. |
| **`transform_prob`** | 0.0 – 1.0 | Probability each step triggers a transform instead of adding a point. `0.0` = static space. `0.2` = ~20% of steps are transforms. |

Additional config: `dim` (2D/3D), `min/max_chain_depth`, `transform_types`, `query_types`, `point_def_types`, coordinate/magnitude ranges.

### Generation Phases

```
Phase 1 (depth ramp):    Force linear growth until min_chain_depth is reached.
Phase 2 (generative):    Add points (via leaf_bias) and transforms (via transform_prob).
Phase 3 (queries):       Emit all remaining queries targeting deep/specific points.
```

### Example Prompt

```
Process the following 3D spatial reasoning scenario. Statements are chronological — propagate all transformations before answering.
Format: [Answer q_ID] value — e.g. [Answer q_001] (0.0, 0.0, 0.0) or [Answer q_002] 5.385 or [Answer q_003] B

---

Point A is 1.8 units from Point O at angle 267 degrees.
Point B is 1.7 units from Point A at angle 267 degrees.
Point C is at offset (-3.1, -1.9, 0.0) from Point B.
Point D is 5.5 units from Point C at angle 196 degrees.
Point E is 3.4 units from Point A in the direction (-0.3, -0.9, 1.4).
Rotate Point C, Point E by 120 degrees around center (0.0, 0.0, 0.0) around the axis (1.0, 0.0, 0.0).
Point F is at offset (0.3, 2.2, 0.8) from Point D.

[Query q_000] Distance from C to B?
Point G is 7.9 units from Point E at angle 308 degrees.
Point H is 2.1 units from Point G at angle 128 degrees.

[Query q_001] Distance from H to F?
Point I is the weighted centroid of Point D (weight 0.3), Point B (weight 0.9), Point H (weight 0.7), Point F (weight 0.3).
Point J is 3.8 units from Point A in the direction (2.1, 0.8, -1.6).

[Query q_002] Position of F? (x, y, z)
```

---

## Geometric Primitives

### Coordinate System

All scenarios operate in **Cartesian space** of configurable dimensionality (2D or 3D). The origin `O` is always implicitly defined at the zero vector. Angles use **degrees** in natural-language prompts; the engine works in radians internally.

### Point Definitions

Every point (except the origin) is defined relative to one or more existing points. A `PointDefinition` is an object that, given the current state of the world, deterministically resolves to an absolute coordinate.

| Type | Natural Language Example | Parametric Form |
|---|---|---|
| **Cartesian offset** | *"Point B is at offset (3, -1, 2) from Point A."* | `target = anchor + offset_vector` |
| **Magnitude + direction vector** | *"Point B is 5.0 units from A in direction [1, 1, 0]."* | `target = anchor + magnitude * normalize(direction)` |
| **Magnitude + spherical angles** *(3D)* | *"Point C is 3.2 units from A at θ=45°, φ=30°."* | `target = anchor + magnitude * spherical_to_cartesian(θ, φ)` |
| **Magnitude + polar angle** *(2D)* | *"Point C is 4.0 units from A at angle 60°."* | `target = anchor + magnitude * [cos(θ), sin(θ)]` |
| **Midpoint** | *"Point D is the midpoint of A, B, and C."* | `target = mean([A, B, C])` |
| **Weighted centroid** | *"Point E is the centroid of A (weight 2), B (weight 1)."* | `target = weighted_mean(points, weights)` |
| **Projection** | *"Point F is the projection of C onto line AB."* | `target = A + dot(C-A, B-A)/dot(B-A, B-A) * (B-A)` |

---

## Events (Transformations)

Events mutate the geometric state mid-scenario. They are the primary mechanism for testing **attention updating** — the model must propagate changes through its internal representation.

| Event | Description | Parameters |
|---|---|---|
| **Rotation** | Rotate a subset of points around an axis through a given center. | `points`, `axis`, `center`, `angle` |
| **Translation** | Shift a subset of points by a displacement vector. | `points`, `displacement` |
| **Reflection** | Reflect a subset of points across a plane (3D) or line (2D). | `points`, `plane_normal`, `plane_point` |
| **Scaling** | Scale distances of a subset of points from a center. | `points`, `center`, `factor` |

Transforms are generated probabilistically during the generative loop (controlled by `transform_prob`). They only target points that already exist — no forward references.

### Propagation Rules

When a point is transformed, all points **defined relative to it** must also update. The engine tracks a dependency DAG and propagates changes correctly. The model is expected to do the same.

---

## Benchmark Tasks

ZelusBench is a suite of **18 Kaggle benchmark tasks**, each a standalone notebook in `tasks/`. Tasks are split into two categories: **isolated** (vary one axis, randomize everything else) and **combined** (all axes at a fixed difficulty tier).

### Design Philosophy

Each isolated benchmark varies **only its target variable** while randomizing all other conditions via `ScenarioConfig.randomize_except()`. This isolates the causal effect of each attention axis across diverse backgrounds.

Queries are **depth-targeted**: when testing sustained attention, every query targets points at exactly the specified chain depth.

Backgrounds are **deterministic across levels**: within a category, the same seed index produces the same randomized background regardless of the target level.

### Isolated Benchmarks

#### 1. Sustained Attention (3 tasks)

Does accuracy degrade as dependency chains grow longer? Pins `leaf_bias=1.0` and `query_target_depth` for exact depth targeting.

| Task | Notebook | Depths | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_sustained_short` | `attn_sustained_short.ipynb` | 2, 3, 4 | 10 each | 30 |
| `attn_sustained_medium` | `attn_sustained_medium.ipynb` | 8, 9, 10 | 10 each | 30 |
| `attn_sustained_long` | `attn_sustained_long.ipynb` | 16, 18, 20 | 10 each | 30 |

#### 2. Selective Attention (3 tasks)

Can the model filter irrelevant points? Pins `num_points` — more points means more noise.

| Task | Notebook | `num_points` | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_selective_clean` | `attn_selective_clean.ipynb` | 4 (minimal) | 15 | 15 |
| `attn_selective_noisy` | `attn_selective_noisy.ipynb` | 15 (moderate) | 15 | 15 |
| `attn_selective_saturated` | `attn_selective_saturated.ipynb` | 40 (dense) | 15 | 15 |

#### 3. Attention Updating (3 tasks)

Can the model propagate transforms through its representation? Pins `transform_prob`.

| Task | Notebook | `transform_prob` | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_updating_static` | `attn_updating_static.ipynb` | 0.0 (no transforms) | 15 | 15 |
| `attn_updating_light` | `attn_updating_light.ipynb` | 0.1 (~10% steps) | 15 | 15 |
| `attn_updating_heavy` | `attn_updating_heavy.ipynb` | 0.25 (~25% steps) | 15 | 15 |

#### 4. Structural Attention (4 tasks)

How does dependency graph topology affect accuracy? Pins `leaf_bias`.

| Task | Notebook | `leaf_bias` | Topology | Seeds | Scenarios |
|---|---|---|---|---|---|
| `attn_structural_linear` | `attn_structural_linear.ipynb` | 1.0 | Pure chain extension | 15 | 15 |
| `attn_structural_branching` | `attn_structural_branching.ipynb` | 0.5 | Even mix | 15 | 15 |
| `attn_structural_merging` | `attn_structural_merging.ipynb` | 0.25 | Mostly internal anchors | 15 | 15 |
| `attn_structural_diamond` | `attn_structural_diamond.ipynb` | 0.0 | Pure random anchor (bushy) | 15 | 15 |

#### 5. Dimensionality (2 tasks)

Can the model maintain higher-dimensional state? Pins `dim`.

| Task | Notebook | Dim | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_dim_2` | `attn_dim_2.ipynb` | 2D | 15 | 15 |
| `attn_dim_3` | `attn_dim_3.ipynb` | 3D | 15 | 15 |

### Combined Benchmarks (3 tasks)

All knobs set to the same difficulty tier simultaneously.

| Task | Notebook | Depth | `leaf_bias` | `num_points` | `transform_prob` | Dim | Seeds | Queries |
|---|---|---|---|---|---|---|---|---|
| `attn_simple` | `attn_simple.ipynb` | 2–3 | 1.0 | 4 | 0.0 | 2D | 15 | 45 |
| `attn_medium` | `attn_medium.ipynb` | 5–8 | 0.5 | 12 | 0.1 | 3D | 15 | 45 |
| `attn_complex` | `attn_complex.ipynb` | 16–32 | 0.25 | 30 | 0.2 | 3D | 15 | 75 |

### Total Coverage

| Category | Tasks | Scenarios | Queries/Scenario | Est. Queries |
|---|---|---|---|---|
| Sustained Attention | 3 | 90 | 3 | ~270 |
| Selective Attention | 3 | 45 | 3 | ~135 |
| Attention Updating | 3 | 45 | 3 | ~135 |
| Structural Attention | 4 | 60 | 3 | ~180 |
| Dimensionality | 2 | 30 | 3 | ~90 |
| Combined (simple/medium/complex) | 3 | 45 | 3–5 | ~165 |
| **Total** | **18** | **315** | — | **~975** |

---

## Scoring & Metrics

### Per-Query Scoring

Each query has a deterministic ground-truth answer (a coordinate vector or scalar distance). Scoring uses **relative error with tiered thresholds**:

| Tier | Condition | Score |
|---|---|---|
| Exact | Relative error < 1% | 1.0 |
| Close | Relative error < 5% | 0.7 |
| Approximate | Relative error < 15% | 0.3 |
| Wrong | Relative error ≥ 15% | 0.0 |
| Refused / Unparseable | Model doesn't produce a coordinate | 0.0 |

Relative error: `‖predicted - truth‖ / max(‖truth‖, ε)` where `ε` avoids division by zero near the origin.

### Response Format

Models are instructed to wrap each answer with a structured tag:

```
[Answer q_000] (3.0, -1.5, 2.0)
[Answer q_001] 5.385
[Answer q_002] B
```

The parser tries `[Answer q_ID]` first, falls back to `[Query q_ID]` splitting, then full-text extraction (using the last match to skip reasoning chain-of-thought).

### Diagnostic Profiles

The benchmark produces **diagnostic profiles**, not a single leaderboard number:

1. **Attention Decay Curve** — accuracy vs. chain depth (sustained attention)
2. **Density Robustness** — accuracy vs. num_points (selective attention)
3. **Transform Adaptation** — accuracy vs. transform_prob (attention updating)
4. **Topology Sensitivity** — accuracy vs. leaf_bias (structural attention)
5. **Dimensionality Gap** — 2D vs. 3D accuracy delta
6. **Combined Difficulty Curve** — simple → medium → complex progression

---

## Project Structure

```
zelusbench/
├── README.md
├── pyproject.toml
│
├── zelusbench/                    # Core library (uploaded as Kaggle dataset)
│   ├── __init__.py
│   │
│   ├── geometry/                  # Geometric engine (source of truth)
│   │   ├── __init__.py
│   │   ├── point.py               # PointDefinition: 7 definition types
│   │   ├── space.py               # Space: dependency DAG, position resolution
│   │   ├── transforms.py          # Rotation, translation, reflection, scaling
│   │   └── vectors.py             # Vector math utilities
│   │
│   ├── scenarios/                 # Generative scenario construction
│   │   ├── __init__.py
│   │   ├── config.py              # ScenarioConfig: leaf_bias, num_points, transform_prob
│   │   ├── generator.py           # ScenarioGenerator: step-by-step generative loop
│   │   └── templates.py           # Natural language templates for statements/queries
│   │
│   └── evaluation/                # Scoring & metrics
│       ├── __init__.py
│       ├── parser.py              # Extract coordinates/distances from model output
│       ├── scorer.py              # Tiered scoring (EXACT/CLOSE/APPROXIMATE/WRONG)
│       └── reports.py             # Diagnostic profiles across dimensions
│
├── tasks/                         # Kaggle benchmark notebooks (18 tasks, one per notebook)
│   │
│   │  # Sustained Attention (chain depth)
│   ├── attn_sustained_short.ipynb     # Depths 2, 3, 4
│   ├── attn_sustained_medium.ipynb    # Depths 8, 9, 10
│   ├── attn_sustained_long.ipynb      # Depths 16, 18, 20
│   │
│   │  # Selective Attention (num_points)
│   ├── attn_selective_clean.ipynb     # 4 points
│   ├── attn_selective_noisy.ipynb     # 15 points
│   ├── attn_selective_saturated.ipynb # 40 points
│   │
│   │  # Attention Updating (transform_prob)
│   ├── attn_updating_static.ipynb     # 0.0 (no transforms)
│   ├── attn_updating_light.ipynb      # 0.1
│   ├── attn_updating_heavy.ipynb      # 0.25
│   │
│   │  # Structural Attention (leaf_bias)
│   ├── attn_structural_linear.ipynb   # 1.0 (linear)
│   ├── attn_structural_branching.ipynb# 0.5 (mixed)
│   ├── attn_structural_merging.ipynb  # 0.25 (bushy)
│   ├── attn_structural_diamond.ipynb  # 0.0 (random anchor)
│   │
│   │  # Dimensionality
│   ├── attn_dim_2.ipynb               # 2D
│   ├── attn_dim_3.ipynb               # 3D
│   │
│   │  # Combined (all axes at one tier)
│   ├── attn_simple.ipynb              # All easy (baseline)
│   ├── attn_medium.ipynb              # All moderate
│   └── attn_complex.ipynb             # All hard (stress test)
│
└── tests/                         # Unit tests (79 tests)
    ├── test_point.py              # 21 tests: all point definition types
    ├── test_space.py              # 12 tests: DAG, propagation, serialization
    ├── test_transforms.py         # 13 tests: all transform types
    └── test_scenarios.py          # 33 tests: generation, parsing, scoring, depth targeting
```

### Key Design Principles

- **`Space` is the single source of truth.** It holds a DAG of `PointDefinition` objects and resolves any point to an absolute coordinate. All transforms mutate the `Space`. Engine correctness is verified with unit tests — if the engine is correct, ground truth is correct.
- **Single generative loop.** Scenarios are built step-by-step — each point, transform, and query is generated inline using only points that already exist. No forward references, no separate "distractor" concept.
- **Continuous difficulty knobs.** Topology (`leaf_bias`), density (`num_points`), and dynamism (`transform_prob`) are continuous parameters, not discrete categories. This enables smooth difficulty curves.
- **Deterministic generation from seed.** Every scenario is fully reproducible from a `ScenarioConfig` + seed. No randomness at evaluation time.
- **Depth-targeted queries.** When `query_target_depth` is set, all queries target points at exactly the specified chain depth.
- **Randomized backgrounds.** `ScenarioConfig.randomize_except()` generates diverse scenarios while pinning only the variable under test.

---

## Running on Kaggle

Each notebook in `tasks/` is a standalone Kaggle Benchmark task using the `kaggle_benchmarks` (kbench) library.

### Setup

1. Upload the `zelusbench/` package as a Kaggle dataset
2. Each task notebook imports from the dataset and uses `@kbench.task` + `llm.prompt()`
3. Each notebook ends with `%choose <task_name>` to register with the benchmark

### Task Pattern

```python
@kbench.task(name="zelusbench_attn_sustained_short")
def zelusbench_attn_sustained_short(llm) -> tuple[float, float]:
    for depth in CHAIN_DEPTHS:       # e.g., [2, 3, 4]
        for i in range(SEEDS):
            bg_rng = random.Random(i * 7919)  # deterministic background
            cfg = ScenarioConfig.randomize_except(bg_rng, pinned={
                "min_chain_depth": depth, "max_chain_depth": depth,
                "leaf_bias": 1.0,
                "query_target_depth": depth, "seed": depth * 1000 + i,
            })
            scenario = ScenarioGenerator(cfg).generate(scenario_id)
            response = llm.prompt(scenario.prompt)
            scores = score_response(response, scenario)
    return overall_accuracy, std_dev

zelusbench_attn_sustained_short.run(llm=kbench.llm)
```

### Evaluation

Scoring is deterministic — no LLM judge needed. Each query is scored by relative error against the geometric ground truth:

| Tier | Relative Error | Score |
|---|---|---|
| Exact | < 1% | 1.0 |
| Close | < 5% | 0.7 |
| Approximate | < 15% | 0.3 |
| Wrong | ≥ 15% | 0.0 |
| Refused | Unparseable | 0.0 |
