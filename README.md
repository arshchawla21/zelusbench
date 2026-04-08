# ZelusBench

> A geometric benchmark for measuring LLM attention through deterministic spatial reasoning tasks.

ZelusBench isolates **attention**, selective focus, sustained tracking, and adaptive updating, from raw reasoning ability. The geometric operations at each step are simple; the challenge is maintaining an accurate internal representation across long, noisy, and disrupted sequences.

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Geometric Primitives](#geometric-primitives)
3. [Events (Transformations)](#events-transformations)
4. [Scenario Structure](#scenario-structure)
5. [Benchmark Tasks](#benchmark-tasks)
6. [Scoring & Metrics](#scoring--metrics)
7. [Project Structure](#project-structure)
8. [Running on Kaggle](#running-on-kaggle)

---

## Core Idea

A scenario is a **sequence of statements and queries** presented to the model in natural language. Statements either define points (relative to other points) or apply transformations to the space. Queries ask the model to report computed positions, distances, or relationships.

Because every point is defined relative to others via explicit geometric operations, the ground-truth answer is always **deterministically computable**. No LLM judge is needed, we compare numerical outputs against exact solutions (within a tolerance).

The key insight: **the math at each step is trivial** (vector addition, rotation matrices, midpoints). What makes the task hard is:

- Tracking a growing graph of dependencies across a long prompt
- Ignoring irrelevant points and relationships (distractors)
- Updating the entire representation when a transformation or invalidation occurs
- Switching between reference frames

This means the benchmark measures **attention**, not mathematical ability.

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

### Distractor Points

Some points exist only to create noise. They may:

- Belong to a **disconnected subgraph** (defined relative to each other but never queried and never connected to queried points)
- Be **on the dependency chain** but irrelevant to the specific query (e.g., query asks about Z, and M branches off from C but Z doesn't depend on M)
- **Re-state known information** in a different form ("Note that A is 5 units from the origin" when A was already defined)

---

## Events (Transformations)

Events mutate the geometric state mid-scenario. They are the primary mechanism for testing **attention updating** - the model must propagate changes through its internal representation.

| Event | Description | Parameters |
|---|---|---|
| **Rotation** | Rotate a subset of points around an axis through a given center. | `points: list`, `axis: vec`, `center: vec`, `angle: float` |
| **Translation** | Shift a subset of points by a displacement vector. | `points: list`, `displacement: vec` |
| **Reflection** | Reflect a subset of points across a plane (3D) or line (2D). | `points: list`, `plane_normal: vec`, `plane_point: vec` |
| **Scaling** | Scale distances of a subset of points from a center. | `points: list`, `center: vec`, `factor: float` |
| **Invalidation** | Redefine a point entirely. All downstream dependents must update. | `point: str`, `new_definition: PointDefinition` |
| **Frame shift** | Change the reference frame: "From now on, describe everything relative to Point B." | `new_origin: str` |

### Propagation Rules

When a point is invalidated or transformed, all points **defined relative to it** must also update. The engine tracks a dependency DAG and propagates changes correctly. The model is expected to do the same.

---

## Scenario Structure

A scenario is not a single prompt-response pair. It is an **interleaved sequence** of statements and queries, simulating a long evolving conversation.

```
┌─────────────────────────────────────┐
│ System: task instructions           │
├─────────────────────────────────────┤
│ Statement Block 1: define A, B, C   │
│ Query 1: "What is B's position?"    │  ← checkpoint (easy)
│ Statement Block 2: define D, E, F   │
│ Distractor Block: define X, Y       │
│ Query 2: "Distance from A to F?"    │  ← sustained tracking
│ Event: rotate all points 90° on Z   │
│ Query 3: "What is B's position now?"│  ← post-transform update
│ Event: invalidate C                 │
│ Query 4: "What is F's position?"    │  ← DAG propagation 
│ ...                                 │
└─────────────────────────────────────┘
```

Each scenario yields **multiple scored queries**, giving a granular performance trace.

---

## Benchmark Tasks

ZelusBench is a suite of **8 Kaggle benchmark tasks**, each a standalone notebook in `tasks/`. Tasks are split into two categories: **isolated** (vary one attention axis, randomize everything else) and **combined** (all axes at a fixed difficulty tier).

### Design Philosophy

Each isolated benchmark varies **only its target variable** while randomizing all other conditions (DAG structure, distractors, transforms, dimensionality, point definition types, coordinate ranges). This isolates the causal effect of each attention axis across diverse backgrounds rather than measuring it in one artificially uniform setup.

Queries are **depth-targeted**: every query in a benchmark must probe the exact difficulty level being tested (e.g., depth=8 queries target points at exactly chain depth 8).

### Isolated Benchmarks

#### 1. Sustained Attention — `sustained_attention.ipynb`

Does accuracy degrade as dependency chains grow longer?

| Depth | Example |
|---|---|
| 2 | A → B (query B) |
| 4 | A → B → C → D (query D) |
| 8 | 8-hop chain |
| 16 | 16-hop chain |
| 32 | 32-hop chain |

Uses LINEAR structure to guarantee exact depth targeting. All other knobs (distractors, transforms, dim, point types) are randomized per scenario. 50 scenarios, ~150 queries.

#### 2. Selective Attention — `selective_attention.ipynb`

Does the model get distracted by irrelevant but salient information?

| Level | Ratio (irrelevant : relevant) |
|---|---|
| Clean | 0:1 (no distractors) |
| Low | 1:1 |
| High | 3:1 |
| Extreme | 10:1 |

Distractors include disconnected subgraphs, irrelevant branches, and restatements. Background (depth, structure, transforms, dim) randomized. 40 scenarios, ~120 queries.

#### 3. Attention Updating — `attention_updating.ipynb`

Can the model update its representation after geometric transforms?

| Level | Transforms |
|---|---|
| Static | 0 |
| Light | 2 (rotation, translation) |
| Heavy | 4 (+ reflection, scaling) |
| Extreme | 7 (all types) |

Background randomized. 40 scenarios, ~120 queries.

#### 4. Structural Attention — `structural_attention.ipynb`

How does dependency graph topology affect accuracy?

| Structure | Description |
|---|---|
| Linear | A → B → C → D |
| Branching | A → B, A → C (two branches from a common root) |
| Merging | Two chains converge via midpoint |
| Diamond | A → B, A → C, D = f(B, C) |

Background randomized. 40 scenarios, ~120 queries.

#### 5. Dimensionality — `dimensionality.ipynb`

Can the model maintain higher-dimensional state?

| Dim | Space |
|---|---|
| 2D | Flat plane (x, y) |
| 3D | Standard spatial (x, y, z) |

Background randomized. 30 scenarios, ~90 queries.

### Combined Benchmarks

These test how multiple difficulty axes interact simultaneously.

#### 6. Attention Simple — `attn_simple.ipynb`

All knobs at minimum difficulty. Shallow chains (2-3), linear structure, no distractors, no transforms, 2D. Models should score near-perfectly. 15 scenarios, 45 queries.

#### 7. Attention Medium — `attn_medium.ipynb`

All knobs at moderate difficulty. Medium chains (5-8), branching/merging structures, low distractors, light transforms, 3D. 15 scenarios, 45 queries.

#### 8. Attention Complex — `attn_complex.ipynb`

All knobs at maximum difficulty. Deep chains (16-32), diamond DAG, extreme distractors, heavy/extreme transforms, 3D, all query and point types. Designed to be very challenging. 15 scenarios, 75 queries.

### Total Coverage

| Task | Scenarios | Queries |
|---|---|---|
| Sustained Attention | 50 | 150 |
| Selective Attention | 40 | 120 |
| Attention Updating | 40 | 120 |
| Structural Attention | 40 | 120 |
| Dimensionality | 30 | 90 |
| Attention Simple | 15 | 45 |
| Attention Medium | 15 | 45 |
| Attention Complex | 15 | 75 |
| **Total** | **245** | **765** |

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

### Diagnostic Profiles

The benchmark produces **diagnostic profiles**, not a single leaderboard number:

1. **Attention Decay Curve** — accuracy vs. chain depth (sustained attention)
2. **Distractor Robustness Score** — accuracy retention ratio noisy/clean (selective attention)
3. **Transform Adaptation Score** — accuracy drop from static baseline (attention updating)
4. **Topology Sensitivity** — accuracy by DAG structure (structural attention)
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
│   ├── scenarios/                 # Scenario generation
│   │   ├── __init__.py
│   │   ├── config.py              # ScenarioConfig: all difficulty knobs + randomize_except()
│   │   ├── generator.py           # ScenarioGenerator: builds scenarios with depth-targeted queries
│   │   ├── distractors.py         # Distractor injection strategies
│   │   └── templates.py           # Natural language templates for statements/queries
│   │
│   └── evaluation/                # Scoring & metrics
│       ├── __init__.py
│       ├── parser.py              # Extract coordinates/distances from model output
│       ├── scorer.py              # Tiered scoring (EXACT/CLOSE/APPROXIMATE/WRONG)
│       └── reports.py             # Diagnostic profiles across dimensions
│
├── tasks/                         # Kaggle benchmark notebooks (one per task)
│   ├── sustained_attention.ipynb  # Chain depth: 2, 4, 8, 16, 32
│   ├── selective_attention.ipynb  # Distractors: 0:1, 1:1, 3:1, 10:1
│   ├── attention_updating.ipynb   # Transforms: 0, 2, 4, 7
│   ├── structural_attention.ipynb # DAG: linear, branching, merging, diamond
│   ├── dimensionality.ipynb       # Dim: 2D, 3D
│   ├── attn_simple.ipynb          # Combined easy (baseline)
│   ├── attn_medium.ipynb          # Combined medium
│   └── attn_complex.ipynb         # Combined hard (stress test)
│
└── tests/                         # Unit tests (77 tests)
    ├── test_point.py              # 21 tests: all point definition types
    ├── test_space.py              # 12 tests: DAG, propagation, serialization
    ├── test_transforms.py         # 13 tests: all transform types
    └── test_scenarios.py          # 31 tests: generation, parsing, scoring, depth targeting
```

### Key Design Principles

- **`Space` is the single source of truth.** It holds a DAG of `PointDefinition` objects and resolves any point to an absolute coordinate. All transforms mutate the `Space`. Engine correctness is verified with unit tests — if the engine is correct, ground truth is correct.
- **Deterministic generation from seed.** Every scenario is fully reproducible from a `ScenarioConfig` + seed. No randomness at evaluation time.
- **Depth-targeted queries.** When `query_target_depth` is set, all queries target points at exactly the specified chain depth. This ensures isolated benchmarks actually measure what they claim.
- **Randomized backgrounds.** `ScenarioConfig.randomize_except()` generates diverse scenarios while pinning only the variable under test. This isolates causal effects across varied conditions.
- **Temporal consistency.** The system prompt enforces chronological ordering — queries only reference points already defined, and transforms propagate to all subsequent references.

---

## Running on Kaggle

Each notebook in `tasks/` is a standalone Kaggle Benchmark task using the `kaggle_benchmarks` (kbench) library.

### Setup

1. Upload the `zelusbench/` package as a Kaggle dataset
2. Each task notebook imports from the dataset and uses `@kbench.task` + `llm.prompt()`
3. Each notebook ends with `%choose <task_name>` to register with the benchmark

### Task Pattern

```python
@kbench.task(name="zelusbench_sustained_attention")
def zelusbench_sustained_attention(llm) -> tuple[float, float]:
    for depth in CHAIN_DEPTHS:
        for seed in range(SEEDS):
            cfg = ScenarioConfig.randomize_except(rng, pinned={...})
            scenario = ScenarioGenerator(cfg).generate(scenario_id)
            response = llm.prompt(scenario.prompt)
            scores = score_response(response, scenario)
    return overall_accuracy, std_dev

zelusbench_sustained_attention.run(llm=kbench.llm)
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