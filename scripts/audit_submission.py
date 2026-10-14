"""Audit submitted notebooks, exported results, links and Git attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import pandas as pd
from src.data import RAW_PATH, OUTPUT, load_data
from src.modelling import metrics, split_data
from src.statistics import test_hypotheses
from src.reports import load_model_reports


def audit():
    checks = []
    df = load_data()
    exported = pd.read_csv(OUTPUT / 'customers.csv')
    pd.testing.assert_frame_equal(df, exported, check_dtype=False, rtol=1e-10)
    quality = json.loads((OUTPUT / 'quality.json').read_text(encoding='utf-8'))
    assert quality['rows_clean'] == len(df)
    assert quality['churn_count'] == int(df.Churn.sum())
    source_bytes = RAW_PATH.read_bytes()
    lf_bytes = source_bytes.replace(b'\r\n', b'\n')
    # Git checkout/archive line endings can differ without changing CSV content.
    digest_variants = {hashlib.sha256(content).hexdigest()
                       for content in (source_bytes, lf_bytes, lf_bytes.replace(b'\n', b'\r\n'))}
    assert quality['source_sha256'] in digest_variants, 'Source content differs beyond line endings; rerun ETL'
    checks.append('Raw source digest (allowing checkout line endings), validated rows and cleaned export agree')

    results, _ = load_model_reports(OUTPUT)
    checks.append('Saved model report schema, feature evidence and classification counts are consistent')
    predictions = pd.read_csv(OUTPUT / 'test_predictions.csv')
    assignments = pd.read_csv(OUTPUT / 'split_assignments.csv')
    assert assignments.snapshot_row.is_unique
    assert set(assignments.snapshot_row) == set(df.index)
    for label, partition in zip(['train','validation','test'],split_data(df)):
        assert set(assignments.loc[assignments.split.eq(label),'snapshot_row']) == set(partition.index)
    assert set(predictions.snapshot_row) == set(assignments.loc[assignments.split.eq('test'),'snapshot_row'])
    assert predictions.snapshot_row.is_unique
    assert np.array_equal(predictions.actual_churn, df.loc[predictions.snapshot_row,'Churn'])
    assert np.array_equal(predictions.predicted_churn, predictions.score.ge(results['test']['threshold']).astype(int))
    recomputed = metrics(predictions.actual_churn,predictions.score,results['test']['threshold'])
    for key,value in recomputed.items():
        assert np.isclose(value,results['test'][key],rtol=1e-10,atol=1e-12), key
    checks.append('Split assignments, held-out labels, predictions and reported metrics agree')

    expected = test_hypotheses(df)
    saved = pd.read_csv(OUTPUT / 'hypotheses.csv')
    pd.testing.assert_frame_equal(expected,saved,check_dtype=False,rtol=1e-10)
    checks.append('Saved hypothesis statistics agree with the current validated data')

    for name in ['05_unit2_statistics.ipynb','06_unit2_machine_learning.ipynb']:
        notebook = nbformat.read(ROOT/'jupyter_notebooks'/name,as_version=4)
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type=='code']
        assert code and all(c.execution_count is not None for c in code), name
        assert not any(o.output_type=='error' for c in code for o in c.outputs), name
    checks.append('Both Unit 2 notebooks are valid, executed and contain no error outputs')

    for path in [ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md'))]:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if link.startswith(('https://','http://','mailto:','#')):
                continue
            assert (path.parent/link.split('#')[0]).exists(), f'{path.name}: missing {link}'
    checks.append('Local README and documentation links resolve')

    history = subprocess.check_output(['git','log','--format=%an|%ae|%cn|%ce'],cwd=ROOT,text=True).splitlines()
    expected_identity = 'chasimarshad-bot|257852985+chasimarshad-bot@users.noreply.github.com'
    portfolio_identity = 'sanifmujawar|43861628+sanifmujawar@users.noreply.github.com'
    allowed = {identity+'|'+identity for identity in [expected_identity,portfolio_identity]}
    assert history and all(line in allowed for line in history)
    checks.append(f'All {len(history)} commits preserve identified source or portfolio contributor attribution')
    status = json.loads((ROOT/'docs/submission_status.json').read_text(encoding='utf-8'))
    pending = []
    if not status['live_dashboard_url'] or not status['hosted_app_verified']:
        pending.append('Deploy the dashboard and verify the returned live URL')
    if not status['learner_review_complete']:
        pending.append("Asim must confirm his independent project walkthrough" if status.get("personal_reflection_received") else "Asim must review the work and complete his personal reflection")
    return {'checks_passed':checks, 'pending_submission_items':pending,
            'submission_ready':not pending}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-submission-ready',action='store_true')
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result,indent=2))
    if args.require_submission_ready and not result['submission_ready']:
        raise SystemExit(1)
