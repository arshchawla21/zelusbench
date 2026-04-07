# ZelusBench

> A geometric benchmark for measuring LLM attention through deterministic spatial reasoning tasks.

ZelusBench isolates **attention**, selective focus, sustained tracking, and adaptive updating, from raw reasoning ability. The geometric operations at each step are simple; the challenge is maintaining an accurate internal representation across long, noisy, and disrupted sequences.

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Geometric Primitives](#geometric-primitives)
3. [Events (Transformations)](#events-transformations)
4. [Scenario Structure](#scenario-structure)
5. [Benchmark Dimensions](#benchmark-dimensions)
6. [Scoring & Metrics](#scoring--metrics)
7. [Project Structure](#project-structure)
8. [Kaggle Submission Format](#kaggle-submission-format)

---

## Core Idea

A scenario is a **sequence of statements and queries** presented to the model in natural language. Statements either define points (relative to other points) or apply transformations to the space. Queries ask the model to report computed positions, distances, or relationships.

Because every point is defined relative to others via explicit geometric operations, the ground-truth answer is always **deterministically computable**. No LLM judge is needed — we compare numerical outputs against exact solutions (within a tolerance).

The key insight: **the math at each step is trivial** (vector addition, rotation matrices, midpoints). What makes the task hard is:

- Tracking a growing graph of dependencies across a long prompt
- Ignoring irrelevant points and relationships (distractors)
- Updating the entire representation when a transformation or invalidation occurs
- Switching between reference frames

This means the benchmark measures **attention**, not mathematical ability.

---

## Geometric Primitives

### Coordinate System

All scenarios operate in **Cartesian space** of configurable dimensionality (2D, 3D, or 4D). The origin `O` is always implicitly defined at the zero vector. Angles use **degrees** in natural-language prompts; the engine works in radians internally.

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

Events mutate the geometric state mid-scenario. They are the primary mechanism for testing **attention updating** — the model must propagate changes through its internal representation.

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

## Benchmark Dimensions

The benchmark is parameterized across several independent axes. Each axis isolates a specific facet of attention. By varying one axis while holding others constant, we produce diagnostic profiles — not just a single aggregate score.

### 1. Chain Depth — *Sustained Attention*

How many dependency hops lie between the origin and the queried point?

| Level | Chain Depth | Example |
|---|---|---|
| Shallow | 1–3 | A → B → C (query C) |
| Medium | 4–7 | A → B → C → D → E → F (query F) |
| Deep | 8–15 | Long transitive chains |

**Metric:** accuracy as a function of chain depth (expect degradation curve).

### 2. Distractor Density — *Selective Attention*

How many irrelevant points and relationships are present relative to the relevant ones?

| Level | Ratio (irrelevant : relevant) |
|---|---|
| Clean | 0:1 (no distractors) |
| Low noise | 1:1 |
| High noise | 3:1 |
| Extreme noise | 10:1 |

**Metric:** accuracy drop from clean → noisy at fixed chain depth (the "distractor tax").

### 3. Transformation Count — *Attention Updating*

How many events (rotations, translations, invalidations) occur before the query?

| Level | Events |
|---|---|
| Static | 0 |
| Light | 1–2 |
| Heavy | 3–5 |
| Extreme | 6+ |

**Metric:** accuracy as a function of transformation count. Separately measured for each event type (rotations vs. invalidations may have very different costs).

### 4. Dimensionality — *Representational Load*

| Level | Space |
|---|---|
| 2D | Flat plane |
| 3D | Standard spatial |
| 4D | Abstract high-dimensional |

**Metric:** accuracy by dimensionality at fixed chain depth. Tests whether the model can maintain higher-dimensional state.

### 5. Query Position — *Recency & Primacy Bias*

Where in the sequence is the relevant information defined?

| Position | Description |
|---|---|
| Recent | Queried point defined in the last block |
| Early | Queried point defined in the first block |
| Scattered | Dependencies spread across early, middle, and late blocks |

**Metric:** accuracy by information position. Reveals primacy/recency bias in attention.

### 6. DAG Complexity — *Structural Attention*

| Level | Structure |
|---|---|
| Linear chain | A → B → C → D |
| Branching | A → B, A → C, query involves both |
| Merging | D = midpoint(B, C), then E from D |
| Diamond | A → B, A → C, D = f(B, C) |

**Metric:** accuracy by graph topology at fixed total node count.

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

### Aggregate Reports

The benchmark produces **diagnostic profiles**, not a single leaderboard number:

1. **Attention Decay Curve** — accuracy vs. chain depth (per dimensionality)
2. **Distractor Robustness Score** — accuracy retention ratio (noisy / clean)
3. **Transform Adaptation Score** — accuracy post-event vs. pre-event (per event type)
4. **Positional Bias Map** — accuracy heatmap by where information appeared in the sequence
5. **Topology Sensitivity** — accuracy by DAG structure
6. **Overall ZelusBench Score** — weighted harmonic mean across all dimensions (for leaderboard)

---

## Project Structure

```
zelusbench/
├── README.md
├── pyproject.toml
│
├── zelusbench/
│   ├── __init__.py
│   │
│   ├── geometry/                  # Core geometric engine (source of truth)
│   │   ├── __init__.py
│   │   ├── point.py               # Point class, PointDefinition variants
│   │   ├── space.py               # World state: resolves all points, tracks DAG
│   │   ├── transforms.py          # Rotation, translation, reflection, scaling
│   │   └── vectors.py             # Vector math utilities (normalize, angle conversion, etc.)
│   │
│   ├── scenarios/                 # Scenario generation
│   │   ├── __init__.py
│   │   ├── generator.py           # ScenarioGenerator: builds randomized scenarios
│   │   ├── config.py              # ScenarioConfig: all difficulty knobs
│   │   ├── distractors.py         # Distractor injection strategies
│   │   └── templates.py           # Natural language templates for statements/queries
│   │
│   ├── evaluation/                # Scoring & metrics
│   │   ├── __init__.py
│   │   ├── parser.py              # Extract coordinates from model output
│   │   ├── scorer.py              # Compare predicted vs. ground truth
│   │   └── reports.py             # Generate diagnostic profiles & plots
│   │
│   └── runner/                    # Benchmark execution
│       ├── __init__.py
│       ├── runner.py              # Orchestrates scenario → prompt → model → score
│       └── kaggle.py              # Kaggle-format dataset & submission generation
│
├── tests/                         # Unit tests for geometry engine correctness
│   ├── test_point.py
│   ├── test_space.py
│   ├── test_transforms.py
│   └── test_scenarios.py
│
├── notebooks/
│   └── analysis.ipynb             # Visualize benchmark results
│
└── data/                          # Generated benchmark datasets
    ├── scenarios/                 # JSON scenario files
    └── solutions/                 # Ground truth answers
```

### Key Design Principles

- **`Space` is the single source of truth.** It holds a DAG of `PointDefinition` objects and can resolve any point to an absolute coordinate at any time. All transforms mutate the `Space`. The engine's correctness is verified with unit tests — if the engine is correct, ground truth is correct.
- **Scenarios are serializable.** A scenario is a JSON object containing the sequence of statements, events, and queries plus ground-truth answers. This is what gets published to Kaggle.
- **Templates are swappable.** The same geometric scenario can be rendered into different natural language phrasings (formal vs. casual, verbose vs. terse) to control for prompt sensitivity.

---

## Kaggle Submission Format

### Published Dataset

Each row in `scenarios.csv`:

| Column | Description |
|---|---|
| `scenario_id` | Unique scenario identifier |
| `prompt` | Full multi-turn prompt text (all statements, events, and queries) |
| `query_id` | Identifier for each query within the scenario |
| `query_index` | Position of this query in the sequence (0-indexed) |
| `dimension` | 2, 3, or 4 |
| `chain_depth` | Dependency depth for this query |
| `distractor_ratio` | Ratio of irrelevant to relevant points |
| `num_transforms` | Number of events before this query |
| `dag_structure` | `linear`, `branching`, `merging`, or `diamond` |

Ground truth answers are held back on a private test set.

### Submission Format

```csv
query_id,x,y,z,w
q_001,3.0,4.0,0.0,
q_002,1.5,-2.3,7.1,
...
```

Coordinates are floats. Unused dimensions (e.g., `w` in 3D, `z` and `w` in 2D) are left blank.

### Evaluation Metric

Public leaderboard uses the **Overall ZelusBench Score** (weighted harmonic mean across benchmark dimensions). The detailed diagnostic breakdown is available to participants for their own analysis.