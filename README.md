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

Some points exist only to create noise. They are real points with valid definitions — they just don't matter for any query. They may:

- Belong to a **disconnected subgraph** (defined relative to each other but never queried and never connected to queried points)
- Be **on the dependency chain** but irrelevant to the specific query (e.g., query asks about Z, and M branches off from C but Z doesn't depend on M)

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

ZelusBench is a suite of **18 Kaggle benchmark tasks**, each a standalone notebook in `tasks/`. Tasks are split into two categories: **isolated** (vary one attention axis, randomize everything else) and **combined** (all axes at a fixed difficulty tier). Each level of each axis is its own task, enabling fine-grained leaderboard comparison.

### Design Philosophy

Each isolated benchmark varies **only its target variable** while randomizing all other conditions (DAG structure, distractors, transforms, dimensionality, point definition types, coordinate ranges). This isolates the causal effect of each attention axis across diverse backgrounds rather than measuring it in one artificially uniform setup.

Queries are **depth-targeted**: every query in a benchmark must probe the exact difficulty level being tested (e.g., depth=8 queries target points at exactly chain depth 8).

Backgrounds are **deterministic across levels**: within a category (e.g., sustained attention), the same seed index produces the same randomized background (dim, distractors, transforms, point types) regardless of the target level. This ensures fair comparison — the only thing that changes between short/medium/long is the chain depth.

### Isolated Benchmarks

#### 1. Sustained Attention (3 tasks)

Does accuracy degrade as dependency chains grow longer? Uses LINEAR structure to guarantee exact depth targeting. All other knobs randomized per scenario.

| Task | Notebook | Depths | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_sustained_short` | `attn_sustained_short.ipynb` | 2, 3, 4 | 10 each | 30 |
| `attn_sustained_medium` | `attn_sustained_medium.ipynb` | 8, 9, 10 | 10 each | 30 |
| `attn_sustained_long` | `attn_sustained_long.ipynb` | 16, 18, 20 | 10 each | 30 |

#### 2. Selective Attention (3 tasks)

Does the model get distracted by irrelevant but salient information? Distractors include disconnected subgraphs, irrelevant branches, and restatements.

| Task | Notebook | Distractor Level | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_selective_clean` | `attn_selective_clean.ipynb` | CLEAN (0:1) | 15 | 15 |
| `attn_selective_noisy` | `attn_selective_noisy.ipynb` | HIGH (3:1) | 15 | 15 |
| `attn_selective_saturated` | `attn_selective_saturated.ipynb` | EXTREME (10:1) | 15 | 15 |

#### 3. Attention Updating (3 tasks)

Can the model update its representation after geometric transforms?

| Task | Notebook | Transform Density | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_updating_static` | `attn_updating_static.ipynb` | STATIC (0 transforms) | 15 | 15 |
| `attn_updating_light` | `attn_updating_light.ipynb` | LIGHT (2 transforms) | 15 | 15 |
| `attn_updating_heavy` | `attn_updating_heavy.ipynb` | EXTREME (7 transforms) | 15 | 15 |

#### 4. Structural Attention (4 tasks)

How does dependency graph topology affect accuracy?

| Task | Notebook | DAG Structure | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_structural_linear` | `attn_structural_linear.ipynb` | LINEAR | 15 | 15 |
| `attn_structural_branching` | `attn_structural_branching.ipynb` | BRANCHING | 15 | 15 |
| `attn_structural_merging` | `attn_structural_merging.ipynb` | MERGING | 15 | 15 |
| `attn_structural_diamond` | `attn_structural_diamond.ipynb` | DIAMOND | 15 | 15 |

#### 5. Dimensionality (2 tasks)

Can the model maintain higher-dimensional state?

| Task | Notebook | Dim | Seeds | Scenarios |
|---|---|---|---|---|
| `attn_dim_2` | `attn_dim_2.ipynb` | 2D | 15 | 15 |
| `attn_dim_3` | `attn_dim_3.ipynb` | 3D | 15 | 15 |

### Combined Benchmarks (3 tasks)

These test how multiple difficulty axes interact simultaneously. All knobs are set to the same tier.

| Task | Notebook | Depth | Structure | Distractors | Transforms | Dim | Seeds | Scenarios | Queries |
|---|---|---|---|---|---|---|---|---|---|
| `attn_simple` | `attn_simple.ipynb` | 2–3 | LINEAR | CLEAN | STATIC | 2D | 15 | 15 | 45 |
| `attn_medium` | `attn_medium.ipynb` | 5–8 | BRANCHING/MERGING | LOW | LIGHT | 3D | 15 | 15 | 45 |
| `attn_complex` | `attn_complex.ipynb` | 16–32 | DIAMOND | EXTREME | HEAVY/EXTREME | 3D | 15 | 15 | 75 |

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
├── tasks/                         # Kaggle benchmark notebooks (18 tasks, one per notebook)
│   │
│   │  # Sustained Attention (chain depth)
│   ├── attn_sustained_short.ipynb     # Depths 2, 3, 4
│   ├── attn_sustained_medium.ipynb    # Depths 8, 9, 10
│   ├── attn_sustained_long.ipynb      # Depths 16, 18, 20
│   │
│   │  # Selective Attention (distractors)
│   ├── attn_selective_clean.ipynb     # CLEAN (0:1 ratio)
│   ├── attn_selective_noisy.ipynb     # HIGH (3:1 ratio)
│   ├── attn_selective_saturated.ipynb # EXTREME (10:1 ratio)
│   │
│   │  # Attention Updating (transforms)
│   ├── attn_updating_static.ipynb     # STATIC (0 transforms)
│   ├── attn_updating_light.ipynb      # LIGHT (2 transforms)
│   ├── attn_updating_heavy.ipynb      # EXTREME (7 transforms)
│   │
│   │  # Structural Attention (DAG topology)
│   ├── attn_structural_linear.ipynb   # LINEAR
│   ├── attn_structural_branching.ipynb# BRANCHING
│   ├── attn_structural_merging.ipynb  # MERGING
│   ├── attn_structural_diamond.ipynb  # DIAMOND
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
@kbench.task(name="zelusbench_attn_sustained_short")
def zelusbench_attn_sustained_short(llm) -> tuple[float, float]:
    for depth in CHAIN_DEPTHS:       # e.g., [2, 3, 4]
        for i in range(SEEDS):
            rng = random.Random(i * 7919)  # deterministic background
            cfg = ScenarioConfig.randomize_except(rng, pinned={
                "min_chain_depth": depth, "max_chain_depth": depth,
                "dag_structure": DAGStructure.LINEAR,
                "query_target_depth": depth, "seed": seed,
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