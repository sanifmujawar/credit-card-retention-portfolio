# Credit Card Retention Insights

A mentor portfolio adaptation maintained by Sanif Mujawar. This project explores recorded credit-card attrition in a historical snapshot of 10,127 customers, compares Logistic Regression and Random Forest, and presents results in a filterable Streamlit dashboard.

This is a credited adaptation of [Muhammad Asim's Unit 2 project](https://github.com/chasimarshad-bot/Credit_Card_Churn-Unit2), preserved at source commit `d041c72bfe082090e5cd5a789e057d8f21088061`. It is a portfolio example, not a separate course submission or a claim of independent authorship.

## My role

I coordinated mentoring and development assistance for Asim, including AI-assisted implementation and review. This portfolio copy records that role, preserves the original development history and credits the contributors. The [original README](README_ASIM_UNIT2.md) records the source project's analysis and assessment context. I have not represented the imported code or Asim's reflection as work I completed unaided.

## Results and practical limits

- 1,627 customers (16.07%) are recorded as attrited.
- Lower transaction activity and smaller product portfolios accompany recorded attrition; these are associations, not established causes.
- Three-fold training cross-validation average precision: Logistic Regression 0.717; Random Forest 0.914.
- Selected Random Forest internal test: AP 0.922, precision 61.4%, recall 96.6%; 198 false positives and 11 false negatives.
- This historical snapshot has no dates establishing advance warning. Prior exploration limits holdout independence. It is an educational classifier, not a live banking decision system.

![Dashboard overview from the original deployment](outputs/unit2/dashboard_hosted.png)

## Explore the dashboard

The four tabs cover portfolio outcomes, account activity, fixed model evaluation and dataset limitations. Sidebar filters support card category, products held, transaction count and missing-demographic inclusion. Empty/small groups have clear messages; CSV downloads contain aggregates.

The [original hosted demonstration](https://creditcardchurn-unit2-iaagzfaqp5pcffhj3yicmh.streamlit.app/) belongs to the source project. This portfolio copy has not been separately deployed. Community Cloud applications may sleep; use the wake-up button if needed.

## Run locally

Use Python 3.12. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-unit2.txt
python -m streamlit run app.py
```

For dashboard-only hosting, install `requirements.txt`. To reproduce reports and validate the analysis:

```powershell
python run_analysis.py
python -m unittest discover -s tests -v
python scripts/audit_submission.py
```

The source contains the two executed Unit 2 notebooks, reusable ETL/statistics/modelling modules, saved metrics and plots, and 13 automated tests. No application secrets are required.

## Provenance and credits

- Dataset: [Credit Card customers by Sakshi Goyal on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers), with original attribution to Analyttica LEAPS. Confirm source redistribution terms before broader reuse.
- Source project: Muhammad Asim / `chasimarshad-bot`; original Unit 1 references and subsequent Unit 2 implementation are retained and credited in the original README.
- Mentoring and coordination: Sanif Mujawar / `sanifmujawar`.
- Development assistance: OpenAI Codex, coordinated by Sanif. See the imported [assistance record](docs/AI_USAGE.md).
- Project documentation structure: Code Institute templates, linked in the original README.

Imported documents in `docs/` retain the original project's historical assessment records. Their learner status, form drafts, release links and deployment evidence refer to Asim's source project, not a submission by Sanif. Git commits preserve original contributor attribution; new portfolio changes are attributed to Sanif. No private submission form or LMS email is included.
