Action plan for data cleaning & preprocessing
0) Inputs & outputs

Inputs (already in data/):

events.csv

item_properties_part1.csv

item_properties_part2.csv

category_tree.csv

Output (single file):

data_clean/journeys_markov_ready.csv — one row per state transition within a session, with enriched category info and segments. This file lets you directly:

estimate transition matrices,

compute absorbing-chain metrics (drop vs. purchase),

build segment-specific models.

1) Read, schema, and basic hygiene

Define explicit dtypes to avoid inference errors

events.csv

timestamp: int64 (milliseconds since epoch; convert later)

visitorid: string

event: categorical ({view, addtocart, transaction} expected)

itemid: string

transactionid: string (nullable; mostly null except purchases)

item_properties_part*.csv

timestamp: int64

itemid: string

property: string

value: string

category_tree.csv

2 integer columns: categoryid, parentid (nullable; ~25 NAs are known)
(Fields/values per mirrors of the Kaggle page & community notebooks. 
O'Reilly Media
GitHub
RPubs
)

Concatenate item properties
item_properties = concat([part1, part2], axis=0, ignore_index=True) (split into 2 files on Kaggle). 
Kaggle

Drop obvious bad rows

In events: drop rows missing any of timestamp, visitorid, event; keep itemid missing only if event type never requires item (but RetailRocket events are item-level; so drop rows with null itemid).

In properties: drop rows with null in any of the 4 columns.

In category tree: keep rows even if parentid is null (it indicates a root or data gap, which we’ll handle).

Deduplicate

Remove exact duplicate rows in all three inputs (full-row duplicates).

For events, also dedupe same visitorid, same itemid, same event, same timestamp (if present).

Convert timestamps

All timestamp fields are milliseconds since Unix epoch. Convert to UTC datetime64[ns, UTC], plus derive:

date, hour, dow (useful later for QA/EDA and potential time-based segmentation).
(Examples showing ms-epoch and column layout appear across open notebooks/docs. 
GitHub
Cnblogs
)

2) Build a stable item → category mapping

RetailRocket stores item attributes in a temporal, key–value table; properties can repeat over time. We need a single, current category per item for journey states & segments.

Filter to category property only

Keep rows where property denotes category (commonly named categoryid or similar). (Known from community Notebook patterns and documentation that item properties include category as a key. 
Packt
)

Choose latest category per item (global latest)

Sort by itemid, timestamp and take the last row per itemid → itemid → categoryid.

If you want time-accurate enrichment (optional, heavier), you can later do as-of joins at event time, but for journey modeling a single stable category per item keeps states compact and reproducible.

Validate coverage

Report % of itemid in events that get a category. If some items lack categories, set categoryid = -1 (or Unknown) and keep them (don’t lose transitions).

Attach hierarchical labels (optional but recommended)

Use category_tree.csv to build parent chains (category → parent → … until root), allowing:

category_l1 (root), category_l2, … for coarse vs. fine segmentation.

Keep parentid nulls (roots) as is; the RPubs note confirms NA rows exist. 
RPubs

3) Enrich events with item category & tidy the event space

Left-join events on the item→category mapping: add categoryid (and derived category_l1 etc. if built).

Normalize event values

Canonicalize to lowercase: view, addtocart, transaction. (Matches descriptions seen in mirrors/docs. 
Packt
)

Sanity filters

Remove bot-like spam: visitors with extreme event rates (e.g., > 1 event/sec sustained; configurable), or sessions with > N item views with zero unique items, etc. (Keep counts/flags rather than dropping aggressively in first pass.)

Transaction consistency: retain rows where event == 'transaction' and transactionid non-null; for other events, transactionid must be null (if not, null it).

4) Sessionization (critical for journeys)

We’ll model journeys within sessions (absorbing states make most sense within a contiguous intent window).

Sort and group by visitor: events.sort_values(['visitorid','timestamp']).

Define sessions by inactivity gap: start a new session if gap > 30 minutes (standard web analytics rule of thumb).

Create session_id = cumulative_session_counter_per_visitor.

Keep only sessions with at least two events (so transitions exist). Retain singletons separately if needed for drop analysis from start.

5) Map raw events to Markov states

We want a compact, interpretable state space aligned to RetailRocket’s three core actions while supporting category-level segmentation.

Base states

START (synthetic)

VIEW (page/product view)

ADD_TO_CART

PURCHASE (absorbing)

DROP (absorbing; synthetic end for sessions that don’t purchase)

Category-aware state variants (optional but powerful)

VIEW:{category_l1} or VIEW:{categoryid}

ADD:{category_l1}

PURCHASE:{category_l1}

This supports segment-specific matrices by product category without exploding states too much.

State assignment

For each event row:

VIEW if event == 'view'

ADD_TO_CART if event == 'addtocart'

PURCHASE if event == 'transaction'

Also retain itemid and categoryid for category-specific variants.

6) Build transitions (one row per edge)

Within each (visitorid, session_id):

Prepend a synthetic START at the session’s first timestamp (t_start - ε).

Order by timestamp; for ties, use a stable sort by (event priority: view → addtocart → transaction).

Create consecutive pairs: (prev_state → next_state) with prev_ts, next_ts, delta_seconds.

Session termination

If no PURCHASE in session → append … → DROP at t_end + ε.

If one or more purchases occur, treat the first PURCHASE as absorbing and stop adding further transitions in that session (optional: keep post-purchase browsing for separate analyses, but exclude from Markov chain used for time-to-purchase).

De-noise impossible hops (optional)

Collapse repeated identical consecutive states (e.g., multiple VIEW in a row on different items) into a single VIEW with updated timestamp, or keep them and let the chain learn self-loops. For cleaner chains, collapsing is recommended.

Enforce START can only transition to {VIEW, ADD_TO_CART, PURCHASE, DROP}.

7) Customer segmentation fields

Add columns that let you filter/build segment-specific chains:

New vs. Repeat

For each visitorid, compute first_purchase_ts (from any session).

For each session, set is_repeat = session_start_ts > first_purchase_ts (True/False/Unknown if never purchased).

Category segment

Use the session’s dominant category (e.g., mode of VIEW categories) or keep per-transition categories and subset at modeling time.

(Optional) Traffic-free temporal segments: dow, hour_bin if you plan time-of-day comparisons.

8) Final output schema (single CSV)

Write one tidy transitions table to data_clean/journeys_markov_ready.csv with:

Visitor & session: visitorid, session_id, session_start_ts, session_end_ts

Transition info:

from_state, to_state (e.g., START → VIEW, VIEW:Electronics → ADD:Electronics, …)

from_ts, to_ts, delta_seconds

from_categoryid, to_categoryid (nullable for synthetic states), plus category_l1 columns if built

Item context (optional): from_itemid, to_itemid (often null except for raw VIEW/ADD/PURCHASE)

Segments: is_repeat (bool), dominant_session_categoryid (optional)

QC helpers: event_count_in_session, has_purchase_in_session (bool)

This file supports:

Drop-off points: estimate transition probabilities to DROP from each from_state (and category/segment filters).

Funnel optimization & expected time-to-purchase: build absorbing Markov chain with absorbing states {PURCHASE, DROP}; compute fundamental matrix to get expected steps/time to absorption per starting state.

Segment-specific models: filter by is_repeat or by category_l1 to fit separate transition matrices.

9) Use-case-specific cleaning nuances

Identify drop-off points

Keep DROP appended for non-purchase sessions so the chain has a true absorbing non-conversion.

Remove bot-like sessions (Section 3) to avoid inflating DROP.

Expected time-to-purchase

Ensure delta_seconds is non-negative and reasonable (cap extreme gaps or split sessions earlier).

Optionally collapse same-state streaks to avoid biasing step counts with repeated VIEWs.

Segmented models

For new vs repeat: ensure first_purchase_ts is computed over the full timeline (not just train period).

For category segments: prefer level-1 categories to keep matrices dense and stable.

10) QA checks (fail-fast assertions)

Columns present with correct dtypes after each major step.

events unique keys sanity: fraction of null transactionid by event type is as expected (≈ all null except transaction).

Sessionization:

No session has START in the middle.

Every session ends with either PURCHASE or DROP.

Transitions:

No impossible transitions (e.g., from DROP or PURCHASE to anything).

No negative delta_seconds.

Coverage:

% of events with attached category; log uncovered %.

Size expectations:

item_properties ~20M rows split into two parts; join should not explode row counts. (Matches dataset description. 
O'Reilly Media
)

11) Deliverables & file layout

data_clean/journeys_markov_ready.csv (main)

data_clean/item_to_category.csv (cache for reproducibility)

data_clean/category_tree_enriched.csv (optional with hierarchy levels)

reports/prep_summary.json (row counts, coverage %, filter thresholds used)