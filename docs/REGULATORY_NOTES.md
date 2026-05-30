# Regulatory mapping

This document maps the system's components to the bank-regulatory and AML-compliance requirements they address. It is a reference for compliance officers, model-risk-management reviewers, and auditors.

## US Bank Secrecy Act / FinCEN

The service is designed for an institution operating under the US Bank Secrecy Act and FinCEN reporting requirements.

| Requirement | System component |
|---|---|
| Currency Transaction Report (CTR) threshold awareness (31 CFR §1010.311, $10,000) | `STRUCTURING_THRESHOLD_USD` constant in `src.features.entity_features`; drives the sub-threshold-share feature family |
| Suspicious Activity Report (SAR) narrative input | `CaseNarrative` schema is designed for direct incorporation into FinCEN's SAR-narrative free-text field; the prompt explicitly references SAR formatting |
| SAR justification capture | Required by the API on `disposition="sar_filed"` (HTTP 400 if missing) |
| Audit trail retention | `audit_retention_days: 1825` (5 years) in `configs/api_config.yaml` matches the regulatory minimum |

## FATF (Financial Action Task Force) typologies

The typology catalog in `src.data.typologies` cites FATF guidance for each named pattern:

- **Structuring** — 31 CFR §1010.311; FinCEN SAR Narrative Guidance §III.B
- **Smurfing** — FATF Typologies Report 2018: Professional Money Laundering, §3.2
- **Layering** — FATF Three Stages of Money Laundering; FinCEN SAR Narrative Guidance §III.D
- **Integration** — FATF Three Stages of Money Laundering
- **Rapid in-out movement (mule)** — FinCEN Advisory FIN-2020-A003
- **Round-amount anomaly** — FATF Typologies Report 2021: Indicators of Suspicious Activity
- **High-risk corridor** — FATF Public Statement on High-Risk Jurisdictions; OFAC SDN List

When the narrator surfaces a typology, the regulatory reference flows through to the case narrative's `regulatory_references` field. SAR drafters can cite the reference directly.

## Model risk management (US SR 11-7)

The Federal Reserve's SR 11-7 supervisory guidance establishes the framework for managing risk in models used by US banks. The service addresses each component:

### Conceptual soundness

- **Cost-weighted Precision@k objective** is documented in `docs/EVALUATION.md` with the derivation and the cost matrix.
- **Hybrid scoring rationale** is documented in `README.md` § Engineering decisions worth noting.
- **Threshold calibration methodology** is documented in `EVALUATION.md` § Threshold tuning.

### Ongoing monitoring

- **Drift detection** via Population Stability Index in `src.monitoring.drift`. Severity bands map to operational responses (`monitor` / `warning` / `regulator-relevant`).
- **Fairness audit** via segment-level demographic parity, equal opportunity, and FPR parity in `src.monitoring.fairness`.
- **Score-distribution drift** is monitored separately from feature drift — a shift at constant input distribution indicates a model or pipeline regression and is the first thing to break under upstream changes.

### Outcomes analysis

- Every alert's full feature vector and model decision are persisted in the `alerts` table.
- Every investigator disposition is persisted in the `feedback` table.
- The pair supports the outcomes analysis SR 11-7 calls for: comparing model predictions against post-hoc investigator labels.

### Model documentation

- Architecture in `docs/ARCHITECTURE.md`
- Evaluation in `docs/EVALUATION.md`
- Engineering decisions in `README.md`
- Per-alert audit snapshot via `src.evaluation.reports.build_audit_snapshot`

## EU and UK equivalents

The same components map to the EU and UK supervisory expectations:

- **EBA Guidelines on ML models** — covered by the conceptual soundness, ongoing monitoring, and outcomes analysis above.
- **GDPR data minimisation** — the schema in `src.data.loader` carries only the columns the model requires; PII is not persisted in the audit log.
- **UK PRA SS1/23 (model risk management)** — the same SR 11-7 mapping applies; the documentation is structured the same way.

## Disparate impact and fair lending principles

While AML monitoring is not directly governed by fair-lending regulations, the same disparate-impact concerns apply because biased AML alerts disproportionately route specific customer cohorts into compliance-investigator review. The fairness module thresholds (`DEMOGRAPHIC_PARITY_THRESHOLDS = 0.02 / 0.05`) reference the US fair-lending 4/5-rule convention.

When a parity gap above the `regulator-relevant` threshold appears in the fairness snapshot, the operator response should be:

1. Verify the gap is real (not driven by a sparse segment).
2. Quantify the impact (which segments are over-/under-served).
3. Determine whether the gap reflects underlying risk differences (legitimate) or model artifact (action required).
4. If model artifact: retrain with segment-aware sampling or fairness-aware regularisation; rollback the current model in the interim.

## Limitations

The system is designed to satisfy the engineering and documentation expectations a model-risk-management function would have. It does not replace:

- Independent model validation by a function organisationally separate from the development team.
- Periodic regulator examination.
- Legal-team review of SAR filings before submission.

The narrator produces *draft* narratives for investigator review. A compliance officer's signature is what makes a SAR-narrative legally binding, not the model's output.
