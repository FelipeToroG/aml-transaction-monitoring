# Claude case-narrative showcase (v1 alerts)

Worked examples of the adjudication layer: real alerts from the v1 model, each
turned into an evidence-bound case narrative (or a structured refusal) by
Claude. These are the actual outputs of `src/triage/narrator.py`, not
hand-written samples.

## How these were generated

- **Alerts.** The held-out test fold (761,752 transactions) was scored with the
  v1 ensemble (calibrated LightGBM with the Isolation-Forest anomaly score
  stacked in as a feature) at the operational threshold (0.0417, k = 384/day).
  Two top-ranked true positives and one borderline alert near the threshold
  were selected to show range.
- **Evidence.** Each alert's evidence bundle was assembled by the batch path,
  `src/triage/evidence.py::build_batch_evidence_snapshot`, which loads the
  entity's real recent activity from the surrounding batch. The narrator
  therefore reasons over the entity's actual prior transactions, not a bare
  single record. (The single-transaction API path deliberately leaves recent
  activity empty as a documented cold-start; see that module.)
- **Model.** Claude Haiku 4.5 (the repo's offline-eval model), temperature 0.
- **Validation.** Every output is parsed against the Pydantic v2 schemas in
  `src/triage/schemas.py` (`extra="forbid"`, discriminated narrative/refusal
  union) and citation-grounded: each cited `feature_name` must appear in the
  bundle's `triggered_features` and each cited `transaction_id` must exist in
  the bundle. Outputs that fail are retried, then degraded to a structured
  refusal rather than emitted.
- **A note on identifiers.** The IBM HI-Small corpus has no native transaction
  id, so the ids below are synthetic labels assigned for this showcase:
  `TXN-TEST-<id>` for the alert transaction under review and `TXN-HIST-<id>` for
  the entity's prior activity. In a real deployment the batch path uses the
  source system's transaction-id column. All cited ids resolve to records
  present in each alert's bundle.

| # | Alert | Ground truth | Risk | Tier | Output | Citations grounded |
|---|---|---|---|---|---|---|
| 1 | High-ranked alert | Laundering (TP) | 0.8667 | tier_3_critical | Narrative (integration) | 7/7 |
| 2 | High-ranked alert | Laundering (TP) | 0.7659 | tier_2_high | Narrative (integration) | 8/8 |
| 3 | Borderline near threshold | Licit (FP) | 0.0418 | below display tiers | Refusal (`insufficient_evidence`) | n/a |

---

## Example 1: high-confidence true positive

**Alert context**

| Field | Value |
|---|---|
| Source account | `80154AAF0` (bank 21749) |
| Destination account | `80010C0B0` (bank 10) |
| Amount | $16,896.52 USD, ACH |
| Risk score | 0.8667 (tier_3_critical) |
| Ground truth | `Is Laundering = 1` (true positive) |

**Evidence the narrator saw.** Top engineered contributors
`src_7d_amount_sum`, `src_7d_amount_max`, `src_7d_amount_std`,
`dst_dormancy_seconds`, `src_7d_amount_mean`, plus the source entity's **5
recent outbound transfers** from the trailing window (e.g. $91,564.18 and
$11,667.00 ACH transfers). The destination had **0 recent inbound transfers**
in the window, consistent with the dormancy the narrative flags.

**Generated narrative** (`suspected_typology: integration`, `confidence: high`)

> The source entity exhibits elevated transaction volume and volatility over a
> 7-day window, with the current transaction of $16,896.52 to a dormant
> destination account occurring amid a pattern of large outbound transfers. The
> destination account shows no recent activity prior to this inbound transfer,
> suggesting potential integration or layering activity. The source entity's
> 7-day transaction profile—characterized by high mean amounts, extreme
> variance, and a maximum single transaction of $8.9M—is inconsistent with
> typical commercial activity.

Risk indicators (each citation-bound):

1. **[high]** Exceptionally high 7-day cumulative volume ($16.9M) with extreme
   variance (std $2.39M), indicating either high-volume legitimate commerce or
   orchestrated fund movement across counterparties. *cites features
   `src_7d_amount_sum`, `src_7d_amount_std`*
2. **[high]** Destination dormant 918,960 s (~10.6 days) before this transfer,
   consistent with funds introduced into a previously inactive account. *cites
   feature `dst_dormancy_seconds`*
3. **[medium]** Transfer routed from a high-velocity source to a dormant
   destination with no inbound activity to that destination in the trailing 24h.
   *cites transaction `TXN-TEST-760995`*
4. **[medium]** Rapid, sequential outbound transfers to multiple distinct
   counterparties within 24h, consistent with distribution/layering. *cites
   transactions `TXN-HIST-2313440`, `TXN-HIST-2313441`, `TXN-HIST-2313439`*

**Recommended action:** priority `immediate`, SAR consideration `true`. Obtain
beneficial-ownership and business-purpose documentation for both accounts,
request transaction narratives for the 7-day window, investigate the
destination's dormancy and any recent reactivation, and cross-reference the
source entity's counterparties for common beneficial owners or layering chains.

Regulatory references cited: 31 CFR §1010.320; FATF integration stage; FinCEN
SAR Narrative Guidance §III.D (layering).

---

## Example 2: high-confidence true positive

**Alert context**

| Field | Value |
|---|---|
| Source account | `80CEC5700` (bank 235509) |
| Destination account | `8081AFAE0` (bank 9371) |
| Amount | $19,537.97 USD, ACH |
| Risk score | 0.7659 (tier_2_high) |
| Ground truth | `Is Laundering = 1` (true positive) |

**Evidence the narrator saw.** Top contributors `src_24h_amount_sum`,
`src_7d_amount_sum`, `dst_dormancy_seconds`, `src_24h_amount_max`,
`src_7d_amount_max`, plus **8 recent source outbound transfers** (e.g.
$94,625.49 and $56,592.63 ACH) and **8 recent destination inbound transfers**.
The destination's history is striking: small repeating micro-transactions
($84.29 Credit Card, $252.32 Cheque) from a single counterparty `100428660`,
exactly the contrast the narrator picks up. This example is the clearest
demonstration of the batch evidence assembly: in the first pass, with empty
recent activity, the model refused this same alert for `no_baseline`; with the
real history loaded it produces a grounded narrative.

**Generated narrative** (`suspected_typology: integration`, `confidence: high`)

> The source entity (account 80CEC5700) exhibits elevated outbound activity
> totaling $277,611.33 over 24 hours, with the current transaction of
> $19,537.97 to a dormant destination account (8081AFAE0) that has received
> only small, repetitive inbound transfers from a single counterparty over the
> preceding 72 hours. The destination account's historical pattern of
> micro-transactions ($84.29 and $252.32 recurring from counterparty
> 100428660) contrasts sharply with the incoming $19,537.97 ACH transfer,
> suggesting potential integration or layering activity.

Risk indicators (each citation-bound):

1. **[high]** Unusually high outbound volume in a compressed timeframe, multiple
   large transfers within hours of the alert. *cites feature
   `src_24h_amount_sum` and transactions `TXN-HIST-5018235`, `TXN-HIST-5018221`*
2. **[high]** Destination dormant before the alert, with inbound history limited
   to small repetitive micro-transactions from a single source. *cites feature
   `dst_dormancy_seconds` and transactions `TXN-HIST-4512749`, `TXN-HIST-3393417`*
3. **[medium]** Large disparity between the destination's historical profile
   (micro-transactions under $4,000) and the incoming $19,537.97 ACH transfer,
   suggesting funds staged for reintroduction. *cites transaction
   `TXN-TEST-760745` and feature `src_24h_amount_max`*

**Recommended action:** priority `immediate`, SAR consideration `true`. Obtain
documentation for both accounts, request invoices/contracts for the $19,537.97
transfer and the preceding $277,611.33 of outbound activity, investigate the
destination's relationship with the recurring micro-transaction counterparty,
and check for outbound movement after deposit (integration signal).

Regulatory references cited: 31 CFR §1010.320; FinCEN SAR Narrative Guidance
§III.D; FATF Typologies 2021 (dormant-account reactivation with
disproportionate deposits).

---

## Example 3: refusal on low confidence (working as intended)

**Alert context**

| Field | Value |
|---|---|
| Source account | `811B90FA0` (bank 231706) |
| Destination account | `811B90FA0` (same account, intra-bank conversion) |
| Amount | $16,211.65 USD, ACH |
| Risk score | 0.0418 (just above the 0.0417 threshold; below all display tiers) |
| Ground truth | `Is Laundering = 0` (false positive) |

**Evidence the narrator saw.** Top contributors centered on the destination
account (`dst_7d_amount_sum`, `dst_7d_amount_max`, `dst_dormancy_seconds`) and
`src_dormancy_seconds`. Critically, the **source entity had only 1 prior
transaction** in all available history (a $42,698.35 reinvestment 12 days
earlier), a genuine near-cold-start. The destination side had 8 recent inbound
transfers, but the source baseline needed to judge this transaction as
anomalous simply does not exist.

This is the refusal path doing its job. The alert barely cleared the
operational threshold, the entity has no real baseline, and the pattern is
ambiguous. Rather than fabricate a plausible-sounding narrative for what is in
fact a licit transaction, the narrator declines and routes to a human. A
refusal is a first-class output, not an error.

**Generated refusal** (`code: insufficient_evidence`)

> **Reason.** Alert score of 0.0418 falls below tier-1 display thresholds.
> Triggered features reflect destination account dormancy and 7-day volume
> aggregation, but source entity shows only one historical transaction
> (`TXN-HIST-106488`) from 12 days prior, insufficient to establish baseline
> activity pattern or detect anomalous deviation. Current transaction
> (`TXN-TEST-761416`) is a single intra-bank currency conversion; no
> convergent-source, rapid in-out, or structuring pattern is evident from the
> bundle.
>
> **Recommended action.** Route to investigator queue for manual review of full
> entity history beyond the 24-hour window and counterparty risk assessment. If
> the destination account activity pattern persists, escalate on next cycle.

The refusal validated cleanly against the `RefusalReason` schema (`reason` a
plain string under the 500-character cap; `recommended_action` a plain string).
An earlier prompt revision that documented only the narrative shape caused the
model to malform refusals; that gap is fixed in
`src/triage/prompts.py::render_output_schema_doc`.

---

## What these examples demonstrate

- **Citation grounding is enforced, not requested.** Across the two narratives,
  all 15 citations resolve to a feature or transaction present in the bundle.
  Uncited risk claims fail schema validation by construction.
- **Refusal is a first-class output.** The borderline, baseline-less alert
  produces a structured refusal instead of a fabricated story, the
  hallucination guard the system is built around.
- **Evidence quality drives output quality.** Example 2 flips from refusal to a
  grounded narrative once the entity's real recent activity is loaded, which is
  exactly why the batch path assembles `recent_transactions` from history.

Generation cost for all three examples was approximately $0.06 on Claude Haiku
4.5 ($1 / $5 per million input / output tokens).
