# Analyzing Customer Journeys in e-commerce through Markov Chains

This repository provides a concise, end-to-end pipeline to transform raw e-commerce clickstream into session-level state transitions, estimate an absorbing Markov chain, and produce actionable metrics: purchase vs drop absorption probabilities, expected steps/time to conversion, and segment/item insights.

## Table of Contents

1. [Overview](#overview)
2. [Articles / Publications](#articles--publications)
3. [Project Workflow](#project-workflow)
4. [File Structure](#file-structure)
5. [Data Directory](#data-directory)
6. [Outputs](#outputs)
7. [Key Concepts / Variables](#key-concepts--variables)
8. [Installation and Setup](#installation-and-setup)
9. [Usage](#usage)
10. [Results / Interpretation](#results--interpretation)
11. [Technical Details](#technical-details)
12. [Dependencies](#dependencies)
13. [Notes / Limitations](#notes--limitations)
14. [Contributing](#contributing)

---

## Overview

* **Goal**: Model customer browsing-to-purchase journeys on an e‑commerce site as an absorbing Markov chain to measure transition dynamics, purchase/drop absorption probabilities, expected steps/time to conversion, and segment/item insights.
* **Approach**: Stream-clean clickstream events, sessionize with inactivity gaps, derive state transitions with START/DROP, then compute a row-stochastic transition matrix and perform absorbing Markov analysis. Segment users and items for actionable metrics.
* **Highlights**:
  - Scalable CSV/Parquet streaming with chunking for large datasets.
  - Rigorous QC and validation of transitions and transition matrix.
  - End-to-end pipeline with segmentation and item-level analyses.

---

## Articles / Publications

* TBD

---

## Project Workflow

1. **Data Collection / Extraction**: Load provided e‑commerce event logs and item property files from `data/`.
2. **Data Preprocessing / Cleaning**: Build stable item→category map, normalize events, and sessionize with START/DROP transitions; write tidy outputs under `data_clean/`.
3. **Modeling / Analysis**: Build transition matrix; validate; compute absorption probabilities and expected steps/time; segment by behavior and analyze top items.
4. **Evaluation / Validation**: Automated checks on transitions and transition matrix; reports under `reports/`.
5. **Reporting**: CSV reports for absorption probabilities, expected steps/time, segmentation summaries, and item metrics.

---

## File Structure

### Core Scripts

Each script/module documents purpose, inputs, outputs, and key steps.

#### `run_pipeline.py`

* **Purpose**: Orchestrate the full cleaning + modeling pipeline with consistent CLI.
* **Input**: Uses files produced/consumed by modules listed below.
* **Output**: Chained outputs across `data_clean/` and `reports/`.
* **Key Features**: Pass-through of chunking/session parameters, logging, sequential execution of steps 1–9.

#### `cleaning_scripts/01_build_item_category.py`

* **Purpose**: Build stable item→category mapping using latest timestamp per item.
* **Input**: `data/item_properties_part1.csv`, `data/item_properties_part2.csv`
* **Output**: `data_clean/item_to_category.csv`
* **Key Features**: Chunked CSV streaming; keeps latest `categoryid` by `timestamp`.

#### `cleaning_scripts/02_enrich_events.py`

* **Purpose**: Normalize raw events, map to canonical states, join item categories; write tidy Parquet.
* **Input**: `data/events.csv`, `data_clean/item_to_category.csv`
* **Output**: `data_clean/events_enriched.parquet`
* **Key Features**: Event normalization, purchase consistency checks, datetime features; PyArrow ParquetWriter.

#### `cleaning_scripts/03_build_transitions.py`

* **Purpose**: Sessionize by inactivity gap, collapse repeated states, add START/DROP, and derive transitions.
* **Input**: `data_clean/events_enriched.parquet`
* **Output**: `data_clean/journeys_markov_ready.csv`
* **Key Features**: Ensures absorbing behavior at first PURCHASE; computes `delta_seconds`; adds segment fields.

#### `cleaning_scripts/04_qc_and_finalize.py`

* **Purpose**: QC checks and write preparation summary.
* **Input**: `data_clean/journeys_markov_ready.csv`, `data_clean/item_to_category.csv`
* **Output**: `reports/prep_summary.json`
* **Key Features**: Structural checks: columns, non-negative deltas, START first, proper terminal state.

#### `modelling_scripts/01_build_transition_matrix.py`

* **Purpose**: Build row-stochastic transition matrix and counts from transitions.
* **Input**: `data_clean/journeys_markov_ready.csv`
* **Output**: `data_clean/transition_matrix.csv`, `data_clean/transition_counts.csv`
* **Key Features**: Supports configurable absorbing states; enforces absorbing rows.

#### `modelling_scripts/02_validation.py`

* **Purpose**: Validate transition matrix soundness.
* **Input**: `data_clean/transition_matrix.csv`
* **Output**: `reports/validation_report.json`
* **Key Features**: Checks row sums, non-negativity, absorbing self-loops, (I−Q) invertibility.

#### `modelling_scripts/03_markov_model.py`

* **Purpose**: Absorbing chain analysis; fundamental matrix N, absorption probabilities B, expected steps.
* **Input**: `data_clean/transition_matrix.csv`
* **Output**: `reports/absorption_probabilities.csv`, `reports/expected_steps_to_absorption.csv`, `reports/markov_model_meta.json`
* **Key Features**: Outputs labeled matrices/series for transparency.

#### `modelling_scripts/04_segmentation.py`

* **Purpose**: Segment-specific Markov analysis (new vs repeat, top categories, purchase-rate buckets, single vs multiple category buyers).
* **Input**: `data_clean/journeys_markov_ready.csv`
* **Output**: CSVs under `reports/segments/*/*`
* **Key Features**: Builds per-user metrics; computes absorption probabilities, expected time and steps by segment.

#### `modelling_scripts/05_item_analysis.py`

* **Purpose**: Item-level absorption analysis for top items by purchase/drop probability.
* **Input**: `data_clean/journeys_markov_ready.csv`
* **Output**: CSVs under `reports/items/*.csv`
* **Key Features**: Maps items to sessions; computes metrics from START; thresholds by minimum sessions.

---

## Data Directory

Contents under `data/` and `data_clean/`:

* **Raw Data (`data/`)**: `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`.
* **Processed Data (`data_clean/`)**:
  - `item_to_category.csv`
  - `events_enriched.parquet`
  - `journeys_markov_ready.csv`
  - `transition_counts.csv`
  - `transition_matrix.csv`

---

## Outputs

Reports under `reports/`:

* **QC / Prep**: `prep_summary.json`
* **Validation**: `validation_report.json`
* **Core Markov**: `absorption_probabilities.csv`, `expected_steps_to_absorption.csv`, `markov_model_meta.json`
* **Segmentation**: CSVs under `reports/segments/{new_repeat,category,purchase_rate,single_vs_multiple}/`
* **Item Analysis**: CSVs under `reports/items/`

---

## Key Concepts / Variables

* **States**: `START`, `VIEW`, `ADD_TO_CART`, `PURCHASE`, `DROP` (absorbing: `PURCHASE`, `DROP`).
* **Sessionization**: New session after inactivity gap (default 30 minutes).
* **Transitions**: Derived per session, collapsing consecutive duplicates, stopping at first `PURCHASE`.
* **Segments**: `is_repeat`, `dominant_session_categoryid`, `event_count_in_session`, `has_purchase_in_session`.

---

## Installation and Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd markov_chain_ecom
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare data**

* Place raw files in `data/` as named above.

---

## Usage

### Run Complete Pipeline

```bash
python -m run_pipeline \
  --chunk-rows 1000000 \
  --session-gap-minutes 30 \
  --absorbing PURCHASE,DROP \
  --segments-top-k 10 \
  --segments-min-count 1000 \
  --items-top-k 10 \
  --items-min-sessions 50
```

### Run Individual Components

```bash
# 1) Item → Category mapping
python -m cleaning_scripts.01_build_item_category --chunk-rows 1000000

# 2) Enrich events
python -m cleaning_scripts.02_enrich_events --chunk-rows 1000000

# 3) Build transitions
python -m cleaning_scripts.03_build_transitions --session-gap-minutes 30

# 4) QC & summary
python -m cleaning_scripts.04_qc_and_finalize

# 5) Build transition matrix
python -m modelling_scripts.01_build_transition_matrix --absorbing PURCHASE,DROP

# 6) Validate matrix
python -m modelling_scripts.02_validation --absorbing PURCHASE,DROP

# 7) Core Markov analysis
python -m modelling_scripts.03_markov_model --absorbing PURCHASE,DROP

# 8) Segmentation
python -m modelling_scripts.04_segmentation

# 9) Item analysis
python -m modelling_scripts.05_item_analysis --top-k 10 --min-sessions 50
```

---

## Results / Interpretation

* **Absorption Probabilities (B)**: Probability of eventually `PURCHASE` vs `DROP` from transient states (notably `START`).
* **Expected Steps (t)**: Expected number of steps to absorption from each transient state.
* **Expected Time**: Via semi‑Markov approximation using observed holding times per state (reported in segmentation/item modules as CSVs).
* **Segments & Items**: Compare purchase/drop propensity and time-to-absorption across behavioral segments and top items.

---

## Technical Details

* **Algorithms / Models**: Absorbing Markov chains; fundamental matrix N=(I−Q)⁻¹; absorption matrix B=N·R; expected steps t=N·1; expected time via N·τ.
* **Frameworks / Tools**: pandas, numpy, pyarrow, matplotlib/seaborn/plotly (for downstream plotting), scipy (utilities), pytest (tests).
* **Implementation Notes**: Chunked CSV reading; row-stochastic enforcement and absorbing rows; validations for QA; outputs are filesystem-friendly CSV/Parquet.

---

## Dependencies

See `requirements.txt`:

* `pandas>=2.0.0`
* `numpy>=1.24.0`
* `pyarrow>=15.0.0`
* `tqdm>=4.66.0`
* `scipy>=1.10.0`
* `matplotlib>=3.7.0`
* `seaborn>=0.12.0`
* `plotly>=5.20.0`
* `pytest>=7.4.0`

---

## Notes / Limitations

* Assumes standardized raw files under `data/` with expected column names.
* Holding time estimates use average dwell per state; results reflect that approximation.
* Very rare states or items may be filtered by thresholds (e.g., min sessions in item analysis).

---

## Contributing

Pull requests welcome. Please add/adjust tests in `test_scripts/` for new functionality.
