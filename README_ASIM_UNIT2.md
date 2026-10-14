# Credit Card Churn: Activity, Evidence and Classification

Unit 2 Data Analytics with AI project for Muhammad Asim (`chasimarshad-bot`).

A historical analysis of 10,127 credit card customer records, with statistical tests, a classification prototype and an interactive Streamlit dashboard for a retention analyst. **1,627 customers (16.07%) are recorded as attrited.** Activity measures distinguish the recorded groups, but the dataset has no dates to establish advance warning or intervention effectiveness.

This project extends [the Unit 1 submission](https://github.com/chasimarshad-bot/Credit_Card_Churn-Analysis). The original four notebooks and figures are retained. Their original narrative is archived in [docs/unit1_README.md](docs/unit1_README.md); the narrower interpretations below supersede its causal and timing suggestions. Imported work is credited in the Credits section.

## Review and submission

Start with the [assessment evidence map](docs/ASSESSMENT_MAP.md), [personal reflection](docs/LEARNING_REFLECTION.md), [submission checklist](docs/SUBMISSION_CHECKLIST.md) and [project walkthrough](docs/WALKTHROUGH.md). The [readiness review](docs/READINESS_REVIEW.md) and [technical quality review](docs/TECHNICAL_REVIEW.md) provide optional supporting checks. The deployed technical project is complete. Asim's walkthrough and reflection were confirmed complete on 14 October 2026; final assessment submission remains outstanding. The release Git bundle preserves the complete Unit 2 commit history; the linked source repositories retain the original Unit 1 history.

## Run locally

Use Python 3.12.8. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-unit2.txt
python run_analysis.py
python -m streamlit run app.py
```

The app normally serves at http://localhost:8501. All paths resolve relative to the project files. No API key, database, Kaggle login or paid service is needed to run the supplied snapshot. The host installs only `requirements.txt`; use `requirements-unit2.txt` locally for analysis and tests. Dependencies for the retained Unit 1 notebooks are in `requirements-unit1.txt`; those optional profiling packages are not required by Unit 2.

For the new notebooks, select this virtual environment as the notebook kernel and run `05_unit2_statistics.ipynb`, then `06_unit2_machine_learning.ipynb`. Both include saved execution outputs and start from the raw data. The model notebook takes longer because it repeats cross-validation, importance and bootstrap diagnostics.

## Dataset content and handling

Source: [Credit Card customers, Sakshi Goyal](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers), with original attribution to Analyttica LEAPS. The supplied raw CSV has 10,127 rows and 23 columns. One row is one customer. `Attrition_Flag` is the recorded status; `Churn` maps attrited to 1 and retained to 0. Dataset licensing/redistribution terms should be confirmed at the source before broader reuse.

The raw file is never modified. The pipeline validates it, applies deterministic transformations and exports `outputs/unit2/customers.csv`. A SHA-256 digest and quality summary are saved in `quality.json`.

- Reject duplicate/missing customer identifiers, invalid labels, missing fields and invalid numeric domains.
- Exclude identifiers, stored Naive Bayes classifier outputs and redundant `Avg_Open_To_Buy`.
- Convert `Unknown` demographic entries to missing: 1,519 education, 1,112 income and 749 marital status entries. Preserve customer rows. No demographic facts are imputed.
- Exclude demographics, card category and tenure from model features. The concentration of tenure at 36 months is unexplained; it is not proof of a default value.
- Use ten numerical activity/account measures: product count, inactive months, contacts, credit limit, revolving balance, amount-change ratio, transaction amount/count, count-change ratio and utilisation.

A missingness test failing to reject independence does not prove random missingness. Valid unusual values remain; small groups and unknown measurement timing are interpretation limits.

## Business requirements

| Requirement | Evidence and use |
|---|---|
| BR1: Quantify recorded attrition and uncertainty | Portfolio metrics, conditional rates, Wilson intervals |
| BR2: Investigate activity and portfolio associations | Three hypothesis tests, effect sizes, product chart and activity density plots |
| BR3: Support historical segment exploration | Filterable dashboard with labelled charts, data table and aggregate CSV export |
| BR4: Evaluate an attrition classification prototype | Training CV comparison, validation threshold, test errors and majority baseline |

These support identifying questions for a retention study. They do not establish campaign profit or identify preventable departures.

## Hypotheses and validation

Unit 1 already explored the snapshot. These are exploratory follow-up tests, not independent confirmation. All three tests are two-sided, with Holm correction across this family at alpha 0.05.

| Hypothesis | Method | Result |
|---|---|---|
| H1: Transaction-count distributions differ by status | Mann-Whitney U; signed rank-biserial | Median 43 attrited vs 71 retained; effect -0.591; adjusted p < 0.001 |
| H2: Q4/Q1 count-change ratio distributions differ | Mann-Whitney U; signed rank-biserial | Median 0.531 vs 0.721; effect -0.491; adjusted p < 0.001 |
| H3: Product count and attrition are associated | Chi-square; Cramer's V | V = 0.167; adjusted p < 0.001; minimum expected count > 146 |

Mann-Whitney compares distributions; interpreting it strictly as a median test requires comparable shapes. The product test does not show that every adjacent group differs. A reported machine p-value of zero is floating-point underflow; it should be read as p < 0.001. Full statistics are exported to `hypotheses.csv`.

## Project plan and methodology

The [GitHub project board](https://github.com/users/chasimarshad-bot/projects/3) tracks delivery. The [task board and MoSCoW priorities](docs/PROJECT_PLAN.md) cover validation, ETL, statistical explanation, tests, modelling, dashboard, evaluation and delivery. The [criterion map](docs/ASSESSMENT_MAP.md) identifies evidence for every Unit 2 pass criterion. It distinguishes implemented evidence from the received personal reflection, remaining walkthrough and LMS submission.

Descriptive statistics and nonparametric comparisons suit this observational snapshot. Logistic regression offers a linear benchmark; a random forest captures nonlinear interactions. More elaborate tuning is deferred because an external dated dataset is more valuable than further optimisation of this known snapshot.

## Model results and decision trade-off

The reproducible seed-42 split uses 6,076 training, 2,025 validation and 2,026 test customers. Imputation/scaling fit only within training folds. Three-fold stratified training CV selects the model by mean average precision (AP). Validation chooses the threshold to maximise recall with an illustrative precision floor of 60%. No outreach costs are supplied, so this is not a profit-optimal rule. No test metric chooses the model or threshold.

Training-only comparison (three stratified folds; SD describes variation across these folds):

| Candidate | Mean CV average precision | CV AP standard deviation | Mean CV ROC AUC |
|---|---:|---:|---:|
| Logistic regression | 0.717 | 0.0183 | 0.920 |
| Random forest | 0.914 | 0.0013 | 0.984 |

Random forest was selected by CV average precision. This is the selection evidence; the following held-out test table evaluates the selected model after its threshold is fixed.

| Internal test measure | Selected random forest | Majority baseline |
|---|---:|---:|
| Average precision | 0.922 | 0.161 |
| ROC AUC | 0.983 | 0.500 |
| Recall | 96.6% | 0.0% |
| Precision | 61.4% | 0.0% |
| Accuracy | 89.7% | 83.9% |

At the validation-selected threshold of 0.25, the test outcomes are **315 true positives, 198 false positives, 11 false negatives and 1,502 true negatives**. The 300-resample conditional bootstrap AP interval is approximately 0.899-0.942. It excludes retraining uncertainty, selection bias and drift.

Removing contact count changes validation AP from 0.92565 to 0.92576. Contact timing therefore appears avoidable in a future study, but the timing of other fields still needs verification. Feature importance is validation permutation importance; it measures reliance, not causation. Class-weighted scores are not calibrated future churn probabilities.

![Product rates](outputs/unit2/product_rates.png)
![Internal test outcomes](outputs/unit2/confusion_matrix.png)

## Dashboard design and visualisation rationale

The Streamlit app has four tabs:

1. **Portfolio overview:** segment size, attrition, activity median, outcome counts and product rates with Wilson intervals. Aggregate table and CSV provide a second way to access the values.
2. **Activity & relationships:** normalised histograms, two-dimensional density heatmaps and contact-rate trends. Notebook hexbin plots provide a static alternative for dense observations.
3. **Model review:** fixed test metrics, error counts and validation feature reliance. Sidebar filters do not recompute model results.
4. **About the data:** source, exclusions and practical limits.

Card category, product count and transaction-range filters apply only to descriptive views. Missing-demographic inclusion is explicit; empty and small samples have clear messages. A Reset filters button restores the population. Chart labels identify groups in addition to colour. Histograms compare distribution shapes; product bars compare rates; density plots reveal concentration without scatter overplotting. Bin counts are not churn probabilities.

Expandable data tables provide transaction frequencies, contact rates, classification outcomes and feature reliance without relying on chart colour or hover. Feature names use plain-language labels.

The hosted overview was visually checked from the supplied screenshot. On 14 October 2026, actual hosted browser checks passed for all tabs, filtering, reset, empty/small segments and aggregate CSV download; see the [browser audit](docs/HOSTED_BROWSER_AUDIT.json). Formal accessibility auditing has not been completed; automated interaction tests do not establish every assistive-technology behaviour.

![Hosted dashboard overview](outputs/unit2/dashboard_hosted.png)

## Conclusions and recommendations

Lower transaction activity and count-change ratios accompany recorded attrition. Customers with fewer products also show greater attrition. A nonlinear classifier separates observed statuses substantially better than the majority baseline within this dataset.

Investigate activity-led retention with human review. Obtain dated features before a defined churn horizon, validate on another period/bank, review subgroup errors and calibration, agree the costs of false alarms/misses and run a controlled outreach trial. The data does not show that adding products prevents churn, that contact causes departure or that frequency falls before spend.

## Limitations and unfixed issues

- Full-snapshot Unit 1 exploration means this internal test is not external validation.
- No timestamps establish prediction lead time; some activity may overlap closure.
- Sampling representativeness, data-generation process and future drift are unknown.
- Excluding demographics does not remove behavioural proxies or establish fairness.
- Small segments have wide intervals; default dashboard exports contain aggregates only.
- Asim's personal reflection and walkthrough are confirmed complete; final assessment submission remains outstanding.
- No known application logic defect remains from the completed automated checks. Missing or invalid source data receives a clear message; unavailable model evidence leaves descriptive exploration usable. Hosted rendering and operation are supported by the user screenshot and confirmation; formal accessibility review remains a future improvement.

## Deployment

[Open the live dashboard](https://creditcardchurn-unit2-iaagzfaqp5pcffhj3yicmh.streamlit.app/). It is hosted on **Streamlit Community Cloud** using this repository, branch `main`, entry point `app.py` and Python 3.12.

The user supplied a screenshot of the rendered dashboard and confirmed it works on 12 October 2026. See [deployment details](docs/DEPLOYMENT.md). No second hosting service, paid server, database or application secrets are required.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src app.py run_analysis.py
```

Thirteen automated tests pass. Report checks reject inconsistent metrics, confusion counts and feature evidence; failure-path dashboard tests verify clear messages and continued descriptive access. Data tests cover rejection of corrupt input, row preservation, leakage exclusions, disjoint/stratified splits, Wilson intervals and multiple-test correction. Dashboard tests cover initial rendering, empty selections, reset, missing-data exclusion and narrow segments. Both new notebooks execute end to end. A fresh Windows Python 3.12 environment also passed the seven tests and dependency checks after a PyArrow compatibility pin. The [verification record](docs/VERIFICATION.md) records completed checks and limits.

## Main libraries and structure

pandas/numpy handle ETL; scipy supplies statistical tests; scikit-learn supplies pipelines and classification; matplotlib produces notebook figures; Plotly/Streamlit provide interactive exploration; nbformat/nbclient support reproducible notebook execution.

```text
app.py                         Interactive dashboard
run_analysis.py                Reproduce Unit 2 reports
src/                           ETL, statistics and modelling functions
tests/                         Data and dashboard guardrails
jupyter_notebooks/01-04*        Retained Unit 1 work
jupyter_notebooks/05-06*        Executed Unit 2 analysis
inputs/datasets/raw/            Unmodified snapshot
outputs/unit2/                 Cleaned data, metrics, splits and figures
docs/                          Plan, criteria, reflection and delivery records
requirements.txt               Lean hosted dashboard dependencies
requirements-unit2.txt         Offline analysis and test dependencies
requirements-unit1.txt         Optional legacy notebook environment
```

## Development roadmap and learning

The implementation adds functional ETL, automated validation, training-only pipelines, uncertainty reporting and density visualisations. Technical changes and their rationale are recorded in [docs/LEARNING_REFLECTION.md](docs/LEARNING_REFLECTION.md). Asim's supplied first-person reflection is recorded; his independent walkthrough was confirmed complete on 14 October 2026. Future priorities are dated external validation, subgroup evaluation, calibration and intervention-cost analysis.

## Use of AI and other assistance

OpenAI Codex assisted with Unit 2 planning, Python functions, statistical/model evaluation, Streamlit implementation, documentation and automated checks. Sanif coordinated this assistance for Asim. The [assistance record](docs/AI_USAGE.md) documents the actual tasks, validation and remaining review. No claim is made that Asim authored or verified this work unaided. Asim should review it, disclose assistance in the submission and explain the methods himself.

## Credits

- Dataset: Sakshi Goyal, *Credit Card customers*, Kaggle; original attribution Analyttica LEAPS.
- Existing Unit 1 material: [Asim's submitted repository](https://github.com/chasimarshad-bot/Credit_Card_Churn-Analysis); local reference checkout from [Sanif's repository](https://github.com/sanifmujawar/credit-card-customer-churn-analysis). Imported material is retained, not represented as newly developed Unit 2 code.
- Structure/documentation: [Code Institute data analytics template](https://github.com/Code-Institute-Org/data-analytics-template) and [README template](https://github.com/Code-Institute-Solutions/da-README-template).
- Assessment requirements: Code Institute Unit 2 handbook and supplied learning-outcome checklist. Private assessor feedback is not included.
- Method references: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [SciPy Mann-Whitney reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).
- Development assistance: OpenAI Codex and Sanif, as described above. Commits identify the publishing account; they are not proof of unaided authorship.
