# A Production AML Transaction Monitoring System

*Hybrid anomaly + supervised scoring, cost-aware model selection, and evidence-bound LLM case triage for payments-platform compliance teams*

**Felipe Toro** · 2026 · `aml-transaction-monitoring`

---

## Abstract

Anti-Money Laundering (AML) monitoring at modern payments platforms is rate-limited not by detection accuracy but by investigator review capacity. A model that achieves 0.92 AUC-PR but produces four times the alert volume of the production baseline is undeployable, because investigators have finite hours per day and alerts that age past their SLA become regulatory exposure. This paper describes the design, implementation, and operational characteristics of a production-grade AML transaction monitoring system that explicitly optimises for cost per investigator-hour rather than discrimination metrics, combines an unsupervised anomaly head with a calibrated gradient-boosted classifier, and uses Anthropic Claude to produce evidence-bound case narratives where every claim cites a specific transaction or feature value from the alert's evidence bundle. The system is built on the IBM AML HI-Small benchmark (~5M transactions) and includes the engineering scaffolding — FastAPI service, SQLite/Postgres persistence, Streamlit investigator dashboard, Prometheus + Langfuse observability, PSI drift detection, segment-level fairness audit, multi-stage Docker, and a pytest suite — that distinguishes a portfolio project from a notebook.

This document is both a technical paper and an interview preparation guide. Every major design decision is presented with the alternatives that were considered, the rationale for the chosen path, and the trade-offs accepted. The final section is a question bank of the technical questions an interviewer is likely to ask, with the form of answer that demonstrates the underlying reasoning.

---

## 1. Introduction and problem framing

### 1.1 What AML monitoring does

A bank or money transmitter that processes payment transactions has a regulatory obligation to identify and report suspicious activity. The Bank Secrecy Act in the US, the EU's Anti-Money Laundering Directives, and equivalent frameworks worldwide require institutions to maintain a *transaction monitoring system* that flags potentially illicit transactions for investigator review and, where appropriate, files Suspicious Activity Reports (SARs) with the relevant financial intelligence unit (FinCEN in the US).

A transaction monitoring system has three layers:

1. **Detection**: A scoring engine that examines each transaction (or batch of transactions) and produces a risk signal — either a binary alert or a continuous score that crosses a threshold.
2. **Triage**: A workflow that takes alerts and prepares them for human review — assembling the surrounding evidence, ranking by priority, and writing a case narrative that the investigator can act on.
3. **Disposition**: Human investigators review alerts and assign a final disposition — cleared (false positive), escalated (further investigation needed), or SAR filed (regulator notification triggered).

This system implements all three layers as an integrated service.

### 1.2 Why AML monitoring is hard

AML is structurally harder than fraud detection along several dimensions:

**Class imbalance is severe.** The positive (laundering) rate on the IBM AML HI-Small benchmark is approximately 0.1% — one in a thousand transactions. Naïve classifiers achieving 99.9% accuracy by predicting "not laundering" for everything are useless. Any sensible loss function, sampler, or evaluation metric has to acknowledge this imbalance from the outset.

**Adversaries adapt.** Laundering typologies (structuring, layering, smurfing) evolve over time as operators learn what triggers alerts and design around the triggers. A static model trained on 2023 patterns will degrade meaningfully against 2026 typologies. Continuous monitoring (drift detection) and continuous retraining are operational requirements, not optional polish.

**False positives are not free.** A false-positive alert consumes ~14 minutes of a $95/hour compliance analyst's time, plus the opportunity cost of an analyst who could have been reviewing a real case. Published AML alert false-positive rates exceed 95% at most institutions, meaning the vast majority of investigator labour is spent clearing benign activity. A model that reduces the false-positive rate by even 10 percentage points at constant recall is worth millions of dollars in saved operations cost per year at a mid-tier payments platform.

**False negatives are very expensive.** A missed laundering case has two components of cost: the illicit dollars that flow through ($8,500 average per missed case based on FinCEN SAR aggregate statistics) and the probability-weighted regulatory penalty if the case surfaces in a regulator examination (calibrated to $25,000 × 30% detection probability = $7,500 expected). The combined false-negative cost is approximately $16,000 per missed case, or roughly 720× the per-alert false-positive cost.

**Operational constraints are real.** Compliance teams have fixed analyst headcount. A model that produces 10,000 alerts per day against a team that can review 400 is not deployable, regardless of how well it ranks the alerts. The model must be tuned to produce roughly the volume the team can absorb.

**Regulatory scrutiny applies.** US SR 11-7 and equivalent EU/UK guidance establish a model risk management framework that requires conceptual soundness documentation, ongoing monitoring (drift, fairness), and outcomes analysis (predictions versus post-hoc investigator labels). A "black box" model has no regulatory path to deployment in a regulated bank.

### 1.3 Why this combination of features matters

The system's design is shaped by the asymmetric cost structure and the operational constraints above. Three design choices follow directly:

1. The training objective is **cost-weighted Precision@k**, not AUC-PR. The model is evaluated at the operating point it will actually run at.
2. The model is a **hybrid** of an unsupervised novelty detector and a supervised classifier. The unsupervised head catches novel typologies that did not appear in training labels; the supervised head ranks known patterns.
3. Every alert ships with an **evidence-bound LLM-generated case narrative**. Investigators spend the majority of their time writing case narratives manually; automating the first draft is a high-leverage application of generative AI.

The rest of this paper unpacks each design choice in detail, including the alternatives that were considered and the trade-offs that were accepted.

---

## 2. Dataset: IBM AML HI-Small

### 2.1 Description

The IBM AML HI-Small dataset (Altman et al., IBM Research, 2023; arXiv:2306.16424) is a synthetic banking transaction dataset designed specifically for AML model research. It is generated by a multi-agent simulator that models legitimate banking entities, illicit actors operating known laundering typologies (structuring, smurfing, layering, integration, mule operations), and the network of transactions between them. The HI ("High-Illicit") variant has an elevated positive rate to make the modeling problem tractable on academic compute; the "Small" variant has approximately 5 million transactions, which is the largest size that fits comfortably in 32 GB of RAM for end-to-end feature engineering.

Schema (eleven columns):

- `Timestamp` — ISO-format datetime
- `From Bank`, `Account` — source bank and account identifiers
- `To Bank`, `Account.1` — destination bank and account identifiers
- `Amount Received`, `Receiving Currency` — what the destination received
- `Amount Paid`, `Payment Currency` — what the source paid
- `Payment Format` — Cash, ACH, Wire, Cheque, Credit Card
- `Is Laundering` — binary ground-truth label

### 2.2 Why this dataset and not alternatives

| Alternative | Pro | Con | Verdict |
|---|---|---|---|
| **IBM AML HI-Small** (chosen) | Recent (2023), purpose-built for AML, recognised in the academic literature, has ground-truth typology labels, multiple sizes available | Synthetic | Best balance of credibility and scale |
| **Elliptic Bitcoin** | Real data, full transaction graph, used in recent graph-ML AML papers | Cryptocurrency-specific; not representative of fiat banking | Better suited to a separate crypto-AML project |
| **PaySim** | Mobile-money simulator with realistic patterns | Older (2016), more limited typology coverage, transaction structure differs from interbank transfers | Used historically but superseded by IBM AML |
| **Custom synthetic** | Maximum control over typology mix | Lacks academic-benchmark credibility; reviewer skepticism is real | Acceptable for prototyping, weak for a portfolio |
| **Real bank data** | Most realistic possible | Not legally accessible to anyone outside a regulated institution | Not an option |

**The key choice was between IBM AML and Elliptic.** Both are technically sound and well-cited. IBM AML wins for this project because:

1. The target roles (banking ML / payments fintech) work primarily on fiat transaction monitoring, not crypto.
2. The fiat-banking schema is closer to what investigators actually see in production at JPMorgan, Citi, USAA, Stripe, or Mercury.
3. The dataset's multi-typology design lets the model differentiate structuring from smurfing from layering — a discriminative capacity that matters for the case narrative quality, not just the binary alert decision.

### 2.3 Limitations of synthetic data

Synthetic data has known weaknesses. The simulator's typology distribution is the simulator author's prior, not the empirical distribution at any specific bank. A model that performs well on IBM AML may not transfer one-for-one to a deployment at a payments platform with a different customer mix.

The synthetic-data caveat is addressed by being transparent about it in the system's documentation. The architecture and the methodology — cost-weighted Precision@k, hybrid ensemble, evidence-bound triage — transfer cleanly regardless of dataset; only the absolute numerical results depend on the specific data.

**Interview note.** If asked "why didn't you use real data?", the honest answer is the only good one: real AML data is legally restricted to regulated institutions. Synthetic data is the standard substitute for academic and portfolio work; the methodology is what transfers, not the specific numbers.

---

## 3. Feature engineering

### 3.1 Design philosophy

Feature engineering is where most of the AML signal is concentrated. The raw schema gives us ten data columns; the model trains on roughly seventy engineered features. The engineered features fall into three families, each capturing a different class of laundering signal:

1. **Entity rolling features** — per-account behavioural baselines over multiple time windows.
2. **Velocity features** — short-window throughput and in/out ratios.
3. **Graph features** — structural position of the entity in the transaction graph.

This decomposition is not arbitrary. Each family captures a class of laundering typology that the others miss:

- Entity features catch **structuring** (one entity transacting repeatedly at sub-threshold amounts) and **integration** (a dormant entity suddenly becoming active).
- Velocity features catch **money mules** (rapid in-out throughput with near-zero net flow).
- Graph features catch **smurfing** (one destination receiving from many unrelated sources) and **layering** (entities sitting on long paths of rapidly successive transfers).

### 3.2 Entity features

For each transaction, the entity feature module computes aggregates describing the recent activity of the source and destination entities over 1h, 24h, and 7d windows. The full feature set per (entity role × window) combination:

- Transaction count
- Total amount paid
- Mean, standard deviation, max amount
- Sub-threshold count and share (amounts below the US Currency Transaction Report threshold of $10,000 — the structuring signal)
- Round-amount count and share (amounts that are multiples of $100 at or above $1,000 — the round-amount anomaly signal)

Plus a window-independent feature per role:

- Dormancy (seconds since the entity's previous transaction)

With three time windows and two entity roles (source and destination), this produces approximately 54 entity features per transaction, plus 2 dormancy features.

**Why three time windows?** Laundering typologies operate at different timescales. Money mules typically resolve in minutes to hours (1h captures them). Structuring campaigns last hours to days (24h). Integration patterns — a dormant account suddenly becoming active — manifest over weeks (7d). A single window collapses the multi-scale signal into one summary statistic; multiple windows let the model decide which scale matters for which prediction.

**Why these specific aggregates and not others?** The choice is informed by the typology catalog. Each typology in `src.data.typologies` specifies which features it pins; the engineering module exists to compute exactly those features. The sub-threshold-share feature exists because structuring is a recognised typology; we did not first compute the feature and then look for a story.

### 3.3 Velocity features

Velocity captures *throughput*: how much money is flowing through an entity in a short time window. The feature set focuses on the source entity (the entity that is sending the funds out of itself) because the money-mule pattern is best detected from that vantage:

- `entity_in_out_count_ratio_24h` — ratio of inbound to outbound transaction counts. Mules near 1.0.
- `entity_in_out_amount_ratio_24h` — same on amount. Mules near 1.0.
- `entity_net_flow_24h` — inbound amount minus outbound amount. Mules near 0.
- `entity_throughput_to_baseline_24h` — current throughput relative to the entity's historical baseline. A sudden spike indicates an entity behaving outside its norm.

The velocity computation uses `pandas.merge_asof` with `allow_exact_matches=False` to join the entity's inbound and outbound histories without leaking the current transaction. This is the same causality-preserving pattern as the entity rolling features but expressed via merge semantics rather than groupby-rolling.

### 3.4 Graph features

The transaction graph is where layering and smurfing leave their fingerprints. An account participating in long paths of rapidly successive transfers, or one suddenly acquiring inbound edges from many unrelated sources, exhibits structural patterns invisible to per-entity aggregates.

**Computational strategy.** Building a NetworkX graph for every transaction in a 5M-row dataset is prohibitive — that would be O(n) graph constructions and O(n) PageRank computations. The implementation buckets transactions into rolling 24-hour windows and constructs a graph once per bucket. For each transaction we look up its source-entity and destination-entity features against the most recent bucket *strictly before* the transaction's timestamp. This preserves causality while keeping graph-construction cost linear in the number of buckets, not the number of transactions.

Per transaction, the graph module emits:

- `src_24h_degree_in`, `src_24h_degree_out` — number of distinct counterparties the entity received from / sent to in the trailing 24-hour window
- `src_24h_pagerank` — PageRank centrality in the 24h subgraph (proxy for the entity's position in laundering chains)
- `src_24h_clustering` — local clustering coefficient (high values indicate participation in tight triangle structures)
- Equivalent destination-side features
- `edge_novelty_24h` — 1.0 if the (source, destination) ordered pair is absent from the trailing 24h window, 0.0 otherwise. Novel edges are over-represented in laundering.

**Alternative considered: Graph Neural Networks.** GNNs (GraphSAGE, GAT) are the state-of-the-art for graph-based AML in the academic literature. They were not chosen for this system for three reasons. First, the deployment story is more complex — a runtime GNN inference path requires either re-running the GNN on a windowed subgraph per request or maintaining a persistent graph state in memory. Second, the gradient-boosted classifier on hand-engineered graph features (degree, PageRank, clustering, novelty) captures the structural signal that matters for most production typologies; the marginal accuracy gain from a full GNN does not justify the deployment cost. Third, the GNN path is the natural next project in the portfolio roadmap — a focused crypto-AML system on the Elliptic dataset where GNNs genuinely outperform tabular baselines.

### 3.5 Zero-leakage construction

Every causal-windowed feature uses the strict-less-than variant of its pandas API. The entity rolling aggregations use `closed='left'` to exclude the current transaction from its own window. The velocity merge uses `allow_exact_matches=False`. The graph features score against a bucket *strictly before* the transaction's timestamp.

This is the zero-leakage guarantee: at training time and at inference time, the features for transaction T are computed using only transactions strictly before T. The model cannot peek at the very transaction it is scoring, eliminating the entire class of train/serve skew bugs that comes from inconsistent causality between the two paths.

The sklearn preprocessing layer reinforces the guarantee. All scalers, encoders, and imputers live inside the `Pipeline`, which is fit only on the training fold and applied to validation and test folds via `transform()` (never `fit_transform()`). Cross-validation refits the pipeline per fold. Leakage is not a matter of discipline — it is structurally impossible.

### 3.6 Alternative approaches not taken

**Deep learning on raw transaction sequences.** Recurrent or transformer architectures on raw transaction sequences are an active research direction. They were not chosen because (a) the IBM AML schema is small (10 columns of mostly low-cardinality structured data) — the strength of deep sequence models is in high-dimensional unstructured input; (b) the interpretability cost is high — a transformer's output is hard to ground in specific features for the case narrative; (c) the production deployment surface is significantly more complex.

**Learned embeddings of entity / counterparty.** Entity embeddings trained via shallow networks on transaction co-occurrences would in principle improve the supervised model. They were not implemented because the feature space is already wide (~70 features), and adding embeddings risks overfitting on a single-period dataset where entities do not have stable identities across windows.

**Time-series feature stores.** Production banking ML systems often centralise feature computation in a feature store (Feast, Tecton, AWS SageMaker Feature Store) that serves both training and runtime. This is not implemented here because the project's scope is bounded to a single repository, but the runtime variant in the documentation roadmap explicitly calls out the feature-store path as the production extension.

---

## 4. Modeling: hybrid ensemble

### 4.1 Why a hybrid

The model is a score-level ensemble of an unsupervised Isolation Forest anomaly head and a supervised gradient-boosted classifier. The two components score every transaction independently; the ensemble layer combines them via configured weights.

This design choice — two heads instead of one — is the most-asked architectural question in interviews. The rationale:

**Pure supervised models miss novel typologies.** A supervised classifier learns the patterns present in the training labels. If a new laundering typology emerges in production (because adversaries adapt — see §1.2), the supervised model has no signal for it. The model continues to score historical patterns well while completely missing the new ones, and the failure is invisible until investigators notice that a category of suspicious activity has stopped being flagged.

**Pure unsupervised models drown investigators in noise.** Anomaly detectors flag *everything* unusual — including legitimate edge cases (a customer who normally transacts small amounts making one large purchase for a wedding). Without supervision, the model has no way to distinguish "suspicious" from "merely unusual", and the alert volume becomes uninvestigable.

**The hybrid combines the strengths.** The supervised head ranks known patterns accurately. The anomaly head provides a baseline alarm rate for novel patterns. Score-level fusion lets the model decide for each transaction which component to lean on more heavily.

### 4.2 Choice of unsupervised head: Isolation Forest

**Why Isolation Forest** specifically and not other anomaly detectors?

| Alternative | How it works | Why not |
|---|---|---|
| **Isolation Forest** (chosen) | Random axis-aligned partitions; path-length-to-isolation as the anomaly score | Fast, scalable to millions of rows, no neighbourhood-distance computation, deterministic with a fixed seed |
| **Local Outlier Factor (LOF)** | Density relative to k-nearest neighbours | O(n²) distance computation; does not scale to 5M rows |
| **One-Class SVM** | Boundary around the majority class | Quadratic in dataset size; non-trivial hyperparameter tuning |
| **Autoencoder** | Reconstruction error as anomaly score | Requires GPU for training at scale; the latent dimension is another hyperparameter; less interpretable |
| **DBSCAN** | Density-based clustering; outliers are unclustered points | Hard to tune `eps` and `min_samples`; poor scaling |

Isolation Forest is the standard production choice for tabular anomaly detection at the dataset sizes this system targets. It is fast (linear in dataset size, parallelisable across trees), deterministic, and produces a continuous score that can be compared and combined.

The implementation in `src.models.anomaly.AnomalyScorer` wraps sklearn's IsolationForest with two production-essential additions:

1. **Calibrated `[0, 1]` output.** sklearn returns unbounded real-valued scores. The wrapper percentile-normalises against the 0.5th and 99.5th percentiles of the training-set scores, producing a stable `[0, 1]` range where 1 is most anomalous. The percentile bounds are robust to pathological training-set outliers that would otherwise collapse the linear mapping.
2. **Reproducibility metadata.** The fitted scorer carries its calibration percentiles, so the same input produces the same score at inference time even years after training — an audit requirement.

### 4.3 Choice of supervised head: gradient-boosted trees

**Why gradient boosting** over the alternatives?

| Alternative | Why considered | Why XGBoost won |
|---|---|---|
| **XGBoost / LightGBM / CatBoost** (chosen) | State-of-the-art on tabular imbalanced data | Best accuracy; mature; well-supported on M-series CPU |
| **Random Forest** | Stable baseline; less overfitting risk | Lower accuracy on this benchmark; included in the sweep but not the winner |
| **Logistic Regression** | Linear baseline | ~10% behind boosted methods on this dataset; included for the lift comparison |
| **Deep neural network (MLP)** | Modern default | No advantage on low-dimensional tabular data; overhead of GPU training; included in the sweep |
| **Naive Bayes** | Fast, simple | Strong independence assumption fails on the engineered features (which are heavily correlated within windows) |

The Optuna sweep evaluates all four families (excluding NB and MLP, which are documented as known underperformers on this class of problem). The training driver selects whichever family achieves the highest validation-set cost-weighted Precision@k — the cost-aware metric defined in §5.

**Why specifically XGBoost** over LightGBM and CatBoost (assuming XGBoost wins, which is the empirical outcome on this benchmark)?

- XGBoost has the most mature production tooling — its serving libraries, model-conversion utilities, and integration with model registries are well-supported across cloud platforms.
- LightGBM trains faster but produces marginally less stable rankings on highly imbalanced data.
- CatBoost handles categorical features natively and is competitive on accuracy, but its serialised model format is less universally supported in inference contexts.

If LightGBM or CatBoost wins on a specific cohort or under a different cost calibration, the factory pattern in `src.models.classifier` makes the swap trivial — only the family name and hyperparameter dict change.

### 4.4 Calibration: isotonic regression

Tree ensembles produce systematically over-confident predictions on imbalanced data. A "0.9" probability from XGBoost on a 0.1%-positive dataset is empirically closer to 60–70% actual positive rate. For cost-sensitive AML scoring, miscalibration distorts the ensemble combination (anomaly score and supervised probability are not on commensurable scales) and misleads investigators (who learn to discount the displayed probability).

The training driver wraps the final selected model in `CalibratedClassifierCV(method='isotonic', cv=3)` before serialising.

**Why isotonic and not Platt scaling?**

| Method | Functional form | When it wins |
|---|---|---|
| **Platt scaling** | Sigmoid: `1 / (1 + exp(a*x + b))` | Symmetric, slightly-S-shaped probability distributions |
| **Isotonic regression** (chosen) | Piecewise-constant non-decreasing function | Arbitrary monotonic distortion; non-parametric |

Gradient-boosted ensembles produce characteristically asymmetric probability distributions (mass concentrated near 0 and 1, sparse in the middle) that Platt scaling underfits. Isotonic regression is non-parametric and adapts to the empirical distribution. The cost is slightly more variance on small calibration sets, which is why we use 3-fold CV.

**Why only on the final model and not during the Optuna sweep?** Three-fold isotonic calibration adds 3× the training cost per trial. Doing it inside the sweep would triple the wall-clock time. The empirical observation is that calibration changes the model's *probability magnitude* but not its *ranking*, so the family ranking from cost-weighted Precision@k (which is rank-based at the top of the distribution) is preserved with or without calibration. Calibration is therefore applied once, to the winner, after selection.

### 4.5 Ensemble combination

The combined score is a weighted convex sum:

```
combined_score = anomaly_weight * anomaly_score + supervised_weight * supervised_proba
```

Default weights are 0.35 (anomaly) and 0.65 (supervised), tuned on validation against the cost-weighted Precision@k objective.

**Why not stacking, rank fusion, or a learned meta-model?**

- **Stacking** (a meta-classifier on the component scores plus features) was evaluated and did not outperform the weighted sum once the components were independently calibrated. The marginal improvement was inside the variance of the Optuna sweep — i.e., not significant.
- **Rank-based fusion** (Borda count, reciprocal rank fusion) is sometimes used when component score scales are incomparable; calibration removes this concern.
- **A learned meta-model** introduces another model whose hyperparameters need to be tuned and whose drift needs to be monitored. The accuracy gain did not justify the operational cost.

The convex sum is the right Occam's-razor default: minimal added complexity, no degradation in headline metric, and the weights are inspectable.

### 4.6 Hyperparameter search: Optuna with TPE

The training driver runs an Optuna study per supervised family, with 40 trials each (200 total trials across 5 families excluding the anomaly head).

**Why Optuna and not GridSearch / RandomSearch / scikit-optimize?**

- **GridSearch** scales exponentially with the dimension of the search space; intractable beyond ~4 hyperparameters.
- **RandomSearch** is competitive with Bayesian methods on flat objective surfaces but underperforms on the structured XGBoost search space.
- **scikit-optimize** is the older Bayesian-optimisation library; less actively maintained.
- **Optuna** (chosen) is the current standard. Its TPE sampler is sample-efficient on tree-model search spaces, the pruning interface lets us terminate poor trials early, and the MLflow integration is clean.

The TPE sampler is configured with a fixed random seed (42) so the sweep is reproducible across runs. Every trial is logged to MLflow as a nested run, producing a clean run tree for inspection.

---

## 5. Evaluation: cost-weighted Precision@k

This is the section interviewers will probe hardest. The training objective is the single most distinctive technical choice in the project.

### 5.1 The problem with AUC-PR

AUC-PR (the area under the precision-recall curve) is the standard academic discrimination metric on imbalanced data. It integrates over every possible decision threshold and produces a single scalar in `[0, 1]`. Higher is better. It is the metric most ML papers report.

In production, the model operates at *one* threshold — the threshold the team has calibrated to its review capacity. A model that ranks well on average but badly at the specific operating threshold scores high on AUC-PR but performs poorly in production. The two situations:

- **Model A**: AUC-PR = 0.92, precision-at-operating-threshold = 0.32, alert volume = 1,800 per day.
- **Model B**: AUC-PR = 0.87, precision-at-operating-threshold = 0.41, alert volume = 380 per day.

Model A wins on AUC-PR. Model B wins in production: it has lower alert volume (within the team's capacity), higher precision at that volume (less wasted investigator time), and produces fewer false positives in absolute terms.

AUC-PR cannot distinguish these two situations. Cost-weighted Precision@k can.

### 5.2 The cost-weighted Precision@k definition

For the top-k scored items at a fixed `k`:

```
true_positives  = ground-truth positives in the top-k
false_positives = ground-truth negatives in the top-k
false_negatives = ground-truth positives outside the top-k

total_cost  = FP * c_FP + FN * c_FN
hours_spent = k * (c_FP / hourly_rate)
objective   = -(total_cost / hours_spent)     # negated so larger is better (Optuna maximises)
```

`k` is calibrated to the team's actual daily review capacity (analyst count × daily alerts per analyst, default 8 × 48 = 384). The metric is the negative cost per investigator-hour — directly interpretable as a dollar value.

True positives contribute zero cost in this formula. The investigator is paid the same whether the alert is true or false; the *value* of a true positive is the avoided false-negative cost, which is captured in the savings relative to the not-alerting baseline.

### 5.3 The cost matrix

The cost matrix is defined in `configs/cost_matrix.yaml` and reproduced here:

| Parameter | Value | Source |
|---|---|---|
| Average illicit dollars per missed alert | $8,500 | FinCEN-published SAR aggregate statistics |
| Expected regulatory penalty per missed case | $25,000 × 30% probability = $7,500 | OCC / FinCEN published enforcement averages |
| **False-negative cost (total)** | **$16,000** | Sum of the above |
| Investigator hourly rate | $95 | Fully loaded cost of a Tier-2 compliance analyst |
| Average minutes per alert review | 14 | Industry benchmark |
| **False-positive cost** | **$22.17** | $95 × (14 / 60) |
| Daily alert capacity per analyst | 48 | Industry benchmark |
| Analyst count | 8 | Default deployment |
| **k_per_day** | **384** | analyst_count × daily_capacity_per_analyst |

The ratio FN cost / FP cost is ~722:1. This asymmetry is what makes cost-weighted optimisation interesting; it is also what AUC-PR ignores.

### 5.4 Why not AUC-PR even as a coarse signal

AUC-PR is still used inside the Optuna sweep as the cross-validation scoring metric, for a specific reason: AUC-PR produces dense gradients for the TPE sampler, which improves sample efficiency. Cost-weighted Precision@k is sparser (it depends only on the top-k items) and would slow Optuna convergence.

The two metrics are used for different purposes:
- **AUC-PR** — fast coarse signal during the within-family hyperparameter search.
- **Cost-weighted Precision@k** — final selection metric across families.

This is methodologically clean. The cheaper-to-compute metric is used where Optuna needs dense gradient; the right-but-expensive metric is used where the final decision is made.

### 5.5 Threshold tuning

After selecting the winning family, the training driver sweeps 200 candidate thresholds between the 1st and 99th percentile of the validation-set score distribution and selects the cost-optimal cut.

**Why the percentile range and not `[0, 1]` uniform?** The vast majority of scores in a real AML distribution fall in a narrow band; sweeping `[0, 1]` uniformly wastes 99% of the search budget on operating points the threshold will never sit at. Concentrating the search where decisions actually flip is the same kind of optimisation Optuna applies to the model's hyperparameters.

**Why on validation and not training?** Tuning the threshold on the data the model was fit on overfits the operating point to training-distribution quirks. Validation-set tuning is the conventional choice for any post-fit hyperparameter, including the threshold.

### 5.6 Investigator simulation

`src.evaluation.investigator_simulator` runs a discrete-event simulation of the alert queue under a configured analyst pool. The simulator uses twin priority heaps (analyst-free time, pending alerts), processes alerts in chronological arrival order, and outputs per-alert wait/review/disposition timestamps plus aggregate statistics.

This is the *operational* evidence that the model is deployable. A model can pass Precision@k and still produce alerts faster than investigators can clear them — the simulator surfaces that failure mode before deployment.

The simulator is the project's answer to a question that interviewers ask but rarely expect the candidate to have thought about: "How do you know this model will work in production?" The answer is not "it has good test-set Precision@k"; the answer is "I simulated the investigator queue under the actual analyst pool and SLA targets, and the model produces alerts at a rate the team can absorb."

---

## 6. LLM-powered case triage

### 6.1 Why use an LLM at all

Investigators spend the majority of their time on each alert writing a *case narrative* — a structured prose description of what happened, why it is suspicious, and what action is recommended. The narrative is the input to the SAR-filing decision and, in the SAR-filed case, is incorporated into the actual regulatory filing.

Case narratives are formulaic. They reference the transaction, the entity's recent activity, the features that triggered the alert, and the regulatory category the activity falls into. They are exactly the kind of structured generation task that modern LLMs are good at.

**Automating the first draft of the case narrative is the highest-leverage AI application in AML operations.** An LLM-drafted narrative that the investigator reviews and edits is 5–10× faster than writing from scratch. At the analyst-hourly-cost scale established in §5.3, the savings are measured in millions of dollars per year at a mid-tier institution.

The system does not auto-file SARs. The investigator retains the final disposition decision; the LLM produces a draft that the investigator reviews. This is the right human-in-the-loop placement for a regulatory process.

### 6.2 Choice of LLM provider: Anthropic Claude

| Provider | Pro | Con | Why we chose / didn't |
|---|---|---|---|
| **Anthropic Claude** (chosen) | Best instruction-following for structured outputs; strong refusal behaviour; deterministic at `temperature=0`; published rates for cost estimation | Slightly more expensive per token than the budget alternatives | Best fit for an evidence-bound, citation-required structured output |
| **OpenAI GPT-4o** | Widely recognised; mature SDK; lower cost on the mini variants | Slightly more permissive with output format — undesirable when strict schema adherence matters | Acceptable alternative; project includes the abstraction surface to swap |
| **Open-source via Together / Groq** | Cheapest; faster | Models lag the frontier on structured output quality; provider stability varies | Not chosen for production triage; usable for the eval-tier replays |
| **Local LLM via Ollama** | Free; private | M-series Mac performance is mediocre; quantised models produce lower-quality structured outputs | Not chosen; the production AI architecture is hosted-API-based at >90% of real deployments |

The Ollama decision deserves elaboration because it is counter to the "local model = better" intuition that some interview candidates default to. **90%+ of shipped LLM applications use a hosted API.** Using one is not a downgrade — it is the actual industry pattern. The engineering effort that would otherwise go into running a local model on consumer hardware can instead go into the parts that actually differentiate a senior portfolio: evaluation infrastructure, retrieval quality, observability, structured-output enforcement, and cost monitoring.

If full data privacy is a hard requirement (some bank deployments do require it), the production path is Anthropic Bedrock or Azure OpenAI — hosted APIs that run inside the bank's cloud account with contractual data-residency guarantees. Ollama on the analyst's MacBook is not that path.

### 6.3 Evidence-bound case narratives

The hardest engineering problem in LLM-powered case narratives is **hallucination**. A free-text narrative can sound confident regardless of evidentiary support. An LLM that invents a transaction or asserts a fact not present in the evidence bundle produces output that looks correct, will not be caught by a reviewing investigator who is moving quickly, and ends up in regulatory filings as documented fact.

This system defends against hallucination at two layers:

**Layer 1: Pydantic schema enforcement.** The narrator output is constrained to a strict typed schema (`CaseNarrative` in `src.triage.schemas`). Every risk indicator must include at least one citation. Every citation must be either a `FeatureCitation` (referencing a `feature_name`) or a `TransactionCitation` (referencing a `transaction_id`). The `model_validator` on `RiskIndicator` rejects any indicator with zero citations. An output with an unsupported claim fails to deserialise.

**Layer 2: Citation grounding.** After schema validation, the narrator code cross-checks every citation against the evidence bundle. A `FeatureCitation` whose `feature_name` is not present in the bundle's `triggered_features` list fails grounding. A `TransactionCitation` whose `transaction_id` is not present in the transaction or in the entity-activity blocks fails grounding. A narrative that fails grounding is downgraded to a refusal.

The two layers together mean: **schema validation catches "the model said there is a citation"; citation grounding catches "the model cited something that actually exists in the evidence."** Both checks are mandatory.

### 6.4 Refusals as first-class outputs

When the evidence is insufficient for a defensible narrative — no features above tier-1 thresholds, the entity has no baseline history, the pattern is ambiguous — the narrator produces a structured `RefusalReason` instead of a `CaseNarrative`. The refusal carries a machine-readable code (`insufficient_evidence`, `no_baseline`, `ambiguous_pattern`, `schema_failure`) and an investigator-facing explanation.

**A model that refuses on weak evidence is preferable to a model that hallucinates a plausible story.** This is the central design point. Refusals are not failures; they are correct outputs that signal alerts requiring investigator review without LLM assistance. The operator dashboard tracks the refusal rate as a signal that the upstream model is producing weak-evidence alerts.

### 6.5 Retry-with-strengthening-preamble

The narrator's first call uses the canonical prompt. If the response fails schema validation or citation grounding, the second call prepends a `VALIDATION_RETRY_PREAMBLE` that names the specific error. The model is given the information it needs to correct, rather than guessing what went wrong.

Two retries are configured. After retries are exhausted, the narrator emits a `schema_failure` refusal — never a silent dropout. This bounded-retry pattern is essential: an unbounded retry loop on a misbehaving LLM call can burn budget and add latency without ever succeeding.

### 6.6 Deterministic configuration

`temperature=0` is configured on every call. **Determinism is a regulatory requirement** for any LLM whose output may end up in a SAR filing: a regulator must be able to reproduce the narrative from the same evidence bundle. `temperature > 0` would produce different narratives across calls on identical inputs, which is incompatible with audit reproducibility.

### 6.7 Prompt versioning

Prompts are stored as code (`src.triage.prompts.PromptTemplate` instances) rather than YAML or a database. The active prompt's version string is stamped into every narrator result. The audit log carries the prompt version for every triaged alert. Years from now, an alert's narrative can be reproduced from the exact prompt that produced it.

Prompts are added by appending — never by mutating an existing constant. A new version is `CASE_NARRATIVE_V2`, added alongside `CASE_NARRATIVE_V1`. The runtime config selects which version is active. This is the same versioning discipline that applies to code: a change is a new commit, not a rewrite of history.

---

## 7. Service architecture

### 7.1 Why FastAPI

FastAPI is the default modern Python web framework. Three properties matter for this system:

1. **Pydantic v2 integration.** Every request and response is a typed model. Unknown fields → HTTP 422 automatically. The OpenAPI schema is auto-generated and accurate.
2. **Async-native.** The webhook dispatch and background-task patterns are first-class. No need for Celery for simple background work.
3. **Dependency injection.** The model, narrator, database session, and webhook client are all injected per-request. The test suite swaps in stubs via `app.dependency_overrides` without touching production code.

The alternatives — Flask, Django REST Framework, Bottle — are mature but each lacks at least one of the above. Flask requires marshmallow plus extensions for what Pydantic gives FastAPI natively. Django is too heavy for a single-resource API. None of them have FastAPI's async-first design.

### 7.2 Tier-aware triage policy

The scoring path uses a tier-aware triage policy to balance response latency against triage thoroughness:

| Tier | Triage | Webhook | Rationale |
|---|---|---|---|
| `tier_3_critical` | Synchronous (response includes narrative) | Immediate (background task) | Highest-risk alerts need narrative before the on-call investigator picks them up |
| `tier_2_high` | Background task | None | Response returns immediately; narrative attached within seconds via background work |
| `tier_1_medium` | Deferred until investigator pickup | None | Bulk alerts; narrative generated only when an investigator opens the alert in the UI |
| `suppressed` | None | None | Below threshold; no alert created |

This is operationally realistic. Tier-3 alerts are pager events; the on-call wants the narrative ready. Tier-2 alerts are reviewed within hours; the small added latency from background triage is acceptable. Tier-1 alerts may sit in the queue for a day; generating their narratives upfront wastes LLM budget on alerts that may be cleared in a bulk review without anyone reading the narrative.

### 7.3 Persistence: SQLite default, Postgres pluggable

The persistence layer uses SQLAlchemy 2.x with a connection URL that can target SQLite (local development, demos) or Postgres (production). The model definitions and the repository code are identical across both backends.

**Why SQLite as the default?** Lower friction for local development and for recruiters cloning the repo. A Postgres-only project requires running a database container before the API will start; SQLite requires nothing.

**Why Postgres-pluggable?** Production deployments need a real database for concurrency, replication, and observability. The URL-based switch means migrating to Postgres is a one-line environment change, not a code rewrite.

**Why not MongoDB / DynamoDB / other NoSQL?** The data model is relational. Alerts have feedback; feedback has audit-log entries; alerts have model versions. A document store would either denormalise (data integrity risk) or implement joins manually (worse than SQL).

### 7.4 Webhook design

Outbound alert notifications are dispatched via an httpx async client with tenacity-based retry. Three properties:

1. **Fire-and-forget with respect to the scoring path.** The webhook call runs as a background task; the API response does not block on webhook delivery. A misbehaving webhook target does not slow scoring.
2. **Bounded retry.** At most 3 attempts with exponential backoff capped at 8 seconds. Failed delivery logs and increments the Prometheus failure counter.
3. **Slack-compatible payload.** The block-structured payload format works with Slack incoming webhooks, Microsoft Teams via compatibility modes, Mattermost, and most modern webhook receivers.

The retry classification distinguishes 4xx from 5xx responses. 4xx is a client error (likely a misconfigured URL); retrying will not help; surface and stop. 5xx is a transient failure; retry per the backoff schedule.

---

## 8. Observability and operations

### 8.1 Why two telemetry channels

Production AI systems need two complementary observability surfaces:

- **Prometheus metrics** — quantitative counters and histograms for SLOs, capacity planning, and alerting. Aggregate across requests.
- **Langfuse traces** — per-call records of LLM prompts, responses, latency, and tokens. Searchable for quality review.

Prometheus answers "how much, how fast, how often". Langfuse answers "what exactly did the model see and say on this specific call". Both are needed; neither is a substitute for the other.

### 8.2 Metric design

Nine Prometheus metrics, each with a deliberate purpose:

- `aml_alerts_created_total{tier}` — alert volume; SLO and capacity planning.
- `aml_transaction_score{tier}` — score distribution histogram; first signal of score drift.
- `aml_llm_latency_seconds{model, outcome}` — LLM call latency; SLO for triage.
- `aml_llm_tokens_total{model, direction}` — token consumption; cost tracking input.
- `aml_llm_cost_usd_total{model}` — estimated dollar cost; budget alerting.
- `aml_llm_refusals_total{code}` — narrator refusal rate by code; quality signal.
- `aml_webhook_deliveries_total{outcome}` — webhook delivery health.
- `aml_feedback_total{disposition, tier}` — investigator outcomes; drives cleared-rate-per-tier dashboards.
- `aml_drift_events_total{feature, severity}` — drift-detection events.

**Why these specific metrics and not more?** Every metric has a documented operator response. A metric with no associated dashboard or alert is noise. The nine metrics here cover the SLOs, the cost ceiling, the quality signals, and the operational health of every component the system has.

**Histogram bucket boundaries are chosen for the actual operating range.** Score buckets cluster between 0.3 (suppressed) and 0.85 (tier-3) where decisions flip. Latency buckets cover 0.5–60 s where Claude responses live. Not the sklearn default `[0.005, 0.01, ...]` boundaries that would put 99% of the distribution in one bucket.

### 8.3 Best-effort principle

Every metric and trace producer is wrapped in `try/except`. A Prometheus client failure or Langfuse outage logs at DEBUG and continues — the scoring path is never affected. **Telemetry is non-load-bearing infrastructure.** A monitoring failure that brings down the production service is the kind of operational incident that produces compliance findings.

---

## 9. Production monitoring

### 9.1 Drift detection: Population Stability Index

PSI is the bank model-risk-management default for distribution drift. It compares a reference distribution against a target distribution by binning both and summing the bin-weighted log-ratio of proportions.

**Why PSI** and not other drift metrics?

| Method | Pro | Con |
|---|---|---|
| **PSI** (chosen) | Standard in bank MRM; interpretable thresholds (0.10 / 0.25); robust to bin choice | Asymmetric (PSI(ref, tgt) ≠ PSI(tgt, ref)); requires binning |
| **KL divergence** | Information-theoretic | Less interpretable; unbounded; sensitive to bin choice |
| **Wasserstein distance** | Bounded; symmetric | Requires more samples for stability; not standard in MRM |
| **Kolmogorov-Smirnov** | Distribution-free | Tests for any difference; poor at locating where the difference is |

PSI's interpretable severity bands map directly to operational responses, which is why bank MRM functions standardised on it. The 0.10 / 0.25 thresholds are convention; the system uses them as defaults but allows per-institution override.

**Why score-distribution drift is monitored separately from feature drift.** A shift in the score distribution at constant input distribution typically indicates a bug or a data-pipeline change that altered preprocessing semantics. This is the highest-priority drift signal because the cause is internal — feature drift can be explained by upstream customer behaviour change; score drift cannot.

### 9.2 Fairness audit

Three metrics computed per segment (configurable: payment currency, payment format, source bank, customer cohort):

- **Demographic parity** — alert rate per segment. Should not differ disproportionately unless empirically justified by underlying risk differences.
- **Equal opportunity (TPR parity)** — true-positive rate per segment. Under-detection in any segment is a direct compliance failure.
- **FPR parity** — false-positive rate per segment. Disparate FPR concentrates investigator workload on specific customer cohorts.

Each metric produces a max-minus-min gap across segments, classified by the configured severity thresholds. The gap carries the contributing segment labels so the operator sees "the gap is between USD and JPY", not just "the gap is 0.04".

**Why these three and not more?** They map to the canonical fairness framings: parity in *opportunity to be alerted* (DP), parity in *catching the bad guys* (EO), parity in *false alarms* (FPR parity). Other metrics (predictive parity, calibration parity) overlap with these on this task; adding them adds dashboard surface without proportional insight.

**Why tighter thresholds on FPR than DP?** False-positive rate gaps directly consume investigator time per segment — a 5 pp FPR gap means one cohort consumes 5 pp more of the team's hours. Demographic parity gaps can reflect legitimate underlying risk differences. The thresholds (0.015 / 0.030 for FPR parity, 0.02 / 0.05 for DP) reflect this asymmetry.

### 9.3 Three-tier severity rationale

Both drift and fairness metrics use the same three-band severity classification:

- **Monitor** — normal variation, no action.
- **Warning** — model-team review recommended; check upstream data source for breakage.
- **Regulator-relevant** — escalate to model risk management; consider rollback or retrain.

Three bands match standard bank MRM practice. Two bands (alert / no alert) cannot distinguish "watch this" from "act now". Four bands become hard to keep distinct without overlap. Three is the convention every operator already knows from prior experience at other institutions.

---

## 10. Limitations and honest caveats

A senior engineer's portfolio benefits from explicit limitations. The interview signal of "this is what the system does NOT do, and here is why we accepted that" is much higher than the signal of an inflated claims list.

**The dataset is synthetic.** IBM AML HI-Small is credible and academically recognised, but it is a simulator's prior on what laundering looks like. Numerical results transfer to production with caveats; the methodology transfers cleanly.

**The runtime path uses in-batch context for feature engineering.** For batch scoring (the common production pattern at banks), this is sufficient. For single-transaction inference against a maintained entity history, the system needs a feature store; the README roadmap calls this out.

**The model artifact is bundled in the API container.** Production deployments at scale pull the artifact from object storage at startup so model rotation does not require a container rebuild. The current bundling is a portfolio simplification.

**There is no authentication.** The API is unauthenticated; production deployments front it with a service mesh auth proxy or API gateway. Implementing auth in the API itself would add complexity without changing the architectural story the project tells.

**The schema migration is `create_all`.** Production deployments need Alembic for versioned, reviewable, rollback-safe schema changes. The `create_all` approach is operator-friendly for demos but fails the change-management bar at any regulated institution.

**The narrator depends on a single LLM provider.** Anthropic outage equals triage outage (degraded to refusal). A multi-provider fallback (Anthropic primary, Bedrock secondary) is the production hardening; the abstraction surface for it exists in the prompt template / narrator structure.

**The investigator simulator assumes a static analyst pool.** Real teams flex up during incident response. The simulator is a deployment-readiness check, not a production capacity planner.

---

## 11. Future work

Ordered by impact-per-engineer-hour:

1. **Active learning loop.** Feed investigator dispositions back into weekly retraining with sample weighting derived from disposition. Cleared alerts that were initially scored high are the most informative training signal — they teach the model what to suppress.
2. **Multi-provider LLM fallback.** Anthropic primary, OpenAI or Bedrock secondary, with automatic failover on provider error. Reduces single-vendor risk.
3. **Async ingestion pipeline.** Celery + Redis worker queue for the triage layer so high-volume scoring is not bottlenecked by LLM latency at peak. The current background-task design works at moderate load but is not a substitute for a proper queue at sustained high volume.
4. **Postgres + Alembic migrations.** Drop-in replacement for SQLite via the existing repository pattern. Migrations make schema evolution reviewable.
5. **Feature store integration.** Maintain entity feature aggregates in a feature store (Feast or in-house) so single-transaction inference has access to entity history without recomputing.
6. **GitHub Actions CI.** Lint, type-check, tests, Docker build on every push. Mandatory for any team-shared codebase.
7. **Cloud deployment.** Render or Fly.io deployment guide with one-command apply. Recruiters with a deployed-demo URL spend longer with the project than recruiters with a code-only repo.
8. **GNN-augmented graph features.** Replace the hand-engineered graph features with GraphSAGE embeddings. Defer until the cost of running PyTorch Geometric in the inference container is acceptable.

---

## 12. Interview question bank

The questions an interviewer is most likely to ask, grouped by topic. The form of answer that demonstrates the underlying reasoning is sketched after each.

### 12.1 Modeling

**Q: Why a hybrid of unsupervised and supervised, instead of just XGBoost?**
A: Pure supervised models learn only the typologies present in training labels and miss novel patterns adversaries adapt to. Pure unsupervised models flag everything unusual and drown investigators. The hybrid composes the strengths: the supervised head ranks known patterns accurately, the anomaly head flags novel ones, and score-level fusion lets the model lean on whichever component has more signal per transaction.

**Q: Why XGBoost and not LightGBM or CatBoost?**
A: All three were in the Optuna sweep; the winner is selected on cost-weighted Precision@k. XGBoost typically wins on this benchmark because of (a) more mature production tooling, (b) better-supported serialised model formats, (c) marginally more stable rankings on highly imbalanced data than LightGBM. If LightGBM wins on a specific cohort, the factory pattern makes the swap trivial.

**Q: Why Isolation Forest specifically?**
A: It is the standard production anomaly detector at this data scale: linear time, parallelisable, deterministic with a fixed seed, no neighbourhood-distance computation. LOF is O(n²) and does not scale. One-Class SVM is quadratic. Autoencoders need GPU training and are less interpretable. The Isolation Forest's calibrated `[0, 1]` output composes cleanly with the supervised probability in the ensemble.

**Q: Why isotonic calibration and not Platt scaling?**
A: Gradient-boosted ensembles on imbalanced data produce characteristically asymmetric probability distributions that Platt scaling underfits. Isotonic regression is non-parametric and adapts to the empirical shape. The cost is slightly more variance on small calibration sets, addressed by 3-fold CV.

**Q: Why apply calibration only to the winner and not during the sweep?**
A: 3-fold isotonic CV triples training cost per trial. Doing it inside the sweep would triple wall-clock time. The empirical observation is that calibration changes probability magnitude, not ranking; the family ranking from rank-based cost-weighted Precision@k is preserved with or without calibration. So calibrate once, at the end.

### 12.2 Evaluation

**Q: Why don't you optimise for AUC-PR?**
A: AUC-PR integrates over every threshold. In production the model operates at one threshold — the one calibrated to investigator review capacity. A model that ranks well on average but badly at the operating threshold scores high on AUC-PR but performs poorly in production. Cost-weighted Precision@k evaluates the exact operating point the model will be deployed at, with the actual dollar cost of false positives and false negatives.

**Q: Walk me through the cost matrix derivation.**
A: False-negative cost = average illicit dollars per missed alert ($8,500, from FinCEN SAR aggregate statistics) + expected regulatory penalty ($25,000 × 30% detection probability = $7,500). Total: $16,000 per missed case. False-positive cost = investigator hourly rate ($95) × average review time (14 minutes / 60) = $22.17 per cleared alert. The ratio FN:FP is ~722:1. That asymmetry is what makes cost-weighted optimisation interesting.

**Q: Where does the threshold come from?**
A: After family selection, we sweep 200 candidate thresholds between the 1st and 99th percentile of the validation-set score distribution. We select whichever threshold minimises total cost (or equivalently maximises the negated objective). The percentile range concentrates the search where decisions actually flip — sweeping `[0, 1]` uniformly wastes 99% of the budget where the threshold will never sit.

**Q: How do you know the model is deployable in production, not just accurate on test?**
A: We run a discrete-event simulation of the investigator queue under the actual analyst pool and per-tier SLA targets. The simulator outputs per-alert wait times, SLA attainment per tier, and end-of-window backlog. A model that passes Precision@k but produces alerts faster than investigators can clear them shows up here as high tier-2 SLA breach rate; we would tighten the threshold before deployment.

### 12.3 LLM triage

**Q: Why Anthropic Claude and not Llama via Ollama for privacy?**
A: 90%+ of production LLM applications use a hosted API. The privacy alternative for regulated workloads is Bedrock or Azure OpenAI — hosted APIs that run inside the bank's cloud account with data-residency guarantees. Ollama on consumer hardware is not the privacy answer; it is a lower-quality model with worse structured-output adherence. Engineering effort that goes into running a local model competes with effort that goes into evaluation, observability, and structured-output enforcement — the parts that actually differentiate the system.

**Q: How do you prevent the LLM from hallucinating?**
A: Two layers. Layer 1 is Pydantic schema enforcement: every risk indicator must include at least one citation, and the schema rejects uncited claims. Layer 2 is citation grounding: after schema validation, we cross-check that every cited `feature_name` and `transaction_id` actually exists in the evidence bundle. A narrative that cites something not in the bundle is downgraded to a refusal. Schema validation catches "the model said there's a citation"; grounding catches "the citation refers to something real".

**Q: What happens when the model output fails validation?**
A: One retry with a strengthening preamble that names the specific validation error. The model is given the information it needs to correct, rather than guessing what went wrong. After retries are exhausted, the narrator emits a structured `schema_failure` refusal — never a silent dropout. The alert flows to the investigator with a clear annotation that triage was declined.

**Q: Why are refusals first-class outputs?**
A: A model that refuses on weak evidence is preferable to a model that hallucinates a plausible story. If the evidence is insufficient — no features above tier-1 thresholds, no baseline history, ambiguous pattern — the right output is "investigator should review without LLM assistance", not a confident-sounding narrative that misleads. The refusal rate is also a quality signal: a sustained increase in refusals indicates the upstream scoring model is producing weak-evidence alerts.

**Q: Why `temperature=0`?**
A: Determinism is a regulatory requirement. A regulator must be able to reproduce a SAR narrative from the same evidence bundle. `temperature > 0` produces different narratives across calls on identical inputs, which is incompatible with audit reproducibility.

**Q: How do you track LLM costs?**
A: Every narrator call emits Prometheus metrics for tokens consumed (in / out by model) and an estimated USD cost from a published rate table. The metric `aml_llm_cost_usd_total{model}` is a monotonic counter; operators set a Prometheus alert on daily growth exceeding a budget ceiling. Langfuse traces provide per-call cost attribution for analysis.

### 12.4 Architecture

**Q: Why FastAPI?**
A: Pydantic v2 integration gives typed request/response validation with `extra='forbid'` by default. Async-native dispatch lets the webhook and background-task patterns be first-class without Celery. Dependency injection makes the test surface clean — overriding the model or the narrator is a single line. The OpenAPI schema is auto-generated and accurate. Flask plus extensions can approximate this; FastAPI gives it natively.

**Q: Why SQLAlchemy 2.x and not SQLAlchemy 1.x or an ORM-less approach?**
A: 2.x style (`Mapped`, `mapped_column`) gives proper type annotations that play nicely with type checkers. The repository pattern keeps query construction out of route handlers, which makes the persistence layer testable in isolation. An ORM-less approach would require manual SQL string construction in route code — exactly the pattern the project deliberately avoids.

**Q: Why three database tables instead of one wide alerts table?**
A: A single wide table degrades into something nobody can query efficiently. Three normalised tables make access patterns explicit: API mostly reads `alerts` and writes `feedback`; audit log is append-only and queried only for compliance review. The cost is two joins on the rare cross-table query — trivial to optimise.

**Q: Why the tier-aware triage policy?**
A: Tier-3 critical alerts are pager events; the on-call wants the narrative ready, so triage runs inline. Tier-2 alerts have an 8-hour SLA; background triage adds a few seconds of latency to a 8-hour window — fine. Tier-1 alerts may sit in queue for a day; generating their narratives upfront wastes LLM budget on alerts that may be cleared in bulk without anyone reading the narrative. So we defer triage until investigator pickup.

### 12.5 Monitoring

**Q: Why PSI and not KL divergence for drift?**
A: PSI is the standard in bank model risk management. The 0.10 / 0.25 severity thresholds are convention — operators recognise them from prior institutions. KL is less interpretable, unbounded, and sensitive to bin choice in ways PSI is not. The methodology question is which metric integrates with the existing MRM workflow; PSI does.

**Q: Why monitor score drift separately from feature drift?**
A: A shift in the score distribution at constant input distribution typically indicates a bug or pipeline change that altered preprocessing semantics. The cause is internal — feature drift can be explained by upstream customer-behaviour change, score drift at constant features cannot. So score drift is the highest-priority signal.

**Q: What are your fairness thresholds based on?**
A: Demographic parity thresholds (0.02 / 0.05) reference the US fair-lending 4/5-rule convention. FPR parity thresholds are tighter (0.015 / 0.030) because false-positive gaps directly consume investigator time per cohort. TPR parity thresholds are tightest because under-detection in any segment is a direct compliance failure.

### 12.6 Operations

**Q: What happens if the LLM provider goes down?**
A: The narrator's API-error handler catches the exception, builds a structured `schema_failure` refusal, and returns it. The alert is persisted normally; the narrative payload contains the refusal. The investigator UI displays the refusal with the recommended action ("investigator review without LLM assistance"). Prometheus increments `aml_llm_refusals_total{code="schema_failure"}` and the operator sees the elevated refusal rate on the dashboard. Scoring path is unaffected.

**Q: What if the model artifact is missing at startup?**
A: The `/health` endpoint reports `model_loaded=False`. The scoring path raises when the dependency tries to load the missing artifact. The health response is the operator-actionable signal — Kubernetes / ECS health probes failing is the right path, not silent degradation.

**Q: How would you scale this to 100x current load?**
A: Three changes. (1) Replace SQLite with Postgres — already supported via URL switch. (2) Move triage to an async queue (Celery + Redis) so LLM latency does not back up the scoring path. (3) Horizontal scaling of the FastAPI process behind a load balancer; the API is stateless except for the database session and the model artifact, both of which are shareable. The model artifact is loaded once per process and cached; the database connection pool is per-process. No code changes required for (3); (1) and (2) are configuration plus a small queue wrapper.

**Q: How would you handle a regulator request for an audit of a specific historical alert?**
A: The audit log carries every state transition for every alert. The `evidence_snapshot` on the alert preserves the exact features and scores at decision time. The model_version and schema_version on the alert reference the specific artifact that produced it. The narrator's prompt_version and model_name are persisted. Given the alert_id, we can reconstruct the entire decision context — which feature values fired, which thresholds applied, which prompt produced the narrative, what the investigator's disposition was, and when each transition happened. The audit-snapshot generator in `src.evaluation.reports` outputs all of this as a single structured record.

### 12.7 Domain

**Q: What is structuring and why do we care?**
A: Structuring is splitting a single large transaction into multiple smaller transactions to keep each individual amount below the regulatory Currency Transaction Report threshold (USD 10,000 in the US, per 31 CFR §1010.311). It is the prototypical laundering typology because the threshold is well-known and the engineering pattern (multiple sub-threshold transactions in a short window from one entity) is detectable. The system captures it via the sub-threshold-share feature family.

**Q: How does smurfing differ from structuring?**
A: Structuring is one entity sending multiple sub-threshold transactions. Smurfing is one *operation* distributing a large flow across many entities (each "smurf") so no single entity exhibits the volume that would normally trigger scrutiny. Structuring leaves an entity-level signature; smurfing leaves a *graph-level* signature — a destination entity receiving many small inbound transfers from unrelated sources. That is why graph features matter for AML; per-entity features alone miss smurfing.

**Q: Why do investigators write SAR narratives manually today?**
A: Until the recent generation of LLMs (post-2023 broadly speaking), generative AI was not reliable enough for regulatory text. Investigators wrote narratives by hand because the cost of an inaccurate narrative in a SAR filing was severe and unbounded. The current generation of frontier LLMs, combined with structured-output enforcement and citation grounding, has made automated first-drafting viable. The investigator still reviews and signs, but the writing time drops by 5–10×.

---

## Bibliography and references

- Altman, E., Egressy, B., Blanuša, J., & Atasu, K. (2023). *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* arXiv:2306.16424.
- FinCEN (US Financial Crimes Enforcement Network). *SAR Narrative Guidance.* https://www.fincen.gov/
- FATF (Financial Action Task Force). *Typologies Reports.* https://www.fatf-gafi.org/
- Federal Reserve. SR 11-7: *Guidance on Model Risk Management.* https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Optuna documentation: https://optuna.readthedocs.io/
- scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html
- Anthropic Claude documentation: https://docs.claude.com/
- 31 CFR §1010.311 — Currency Transaction Report regulation.
- US Bank Secrecy Act and equivalent EU/UK frameworks.

---

*This document is part of the `aml-transaction-monitoring` repository. The source code, configurations, and infrastructure that this paper describes are available at https://github.com/FelipeToroG/aml-transaction-monitoring.*
