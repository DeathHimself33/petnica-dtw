"""Check saved OOF arithmetic, identities, coverage and source hashes."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from kimore_dataset import read_manifest
def _experiment_inputs_sha256(samples):
    digest = hashlib.sha256()
    for s in samples:
        payload = dict(sample_id=s.sample_id, subject_id=s.subject_id, cohort=s.cohort, exercise=s.exercise, score=format(s.score, '.17g'), frames=s.frames, position_sha256=hashlib.sha256(s.position_path.read_bytes()).hexdigest())
        digest.update(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8') + b'\n')
    return digest.hexdigest()


def metrics(frame):
    a, p = frame.actual_ts, frame.predicted_ts
    return dict(mae=float((a-p).abs().mean()), rmse=float(np.sqrt(((a-p)**2).mean())), pearson=float(a.corr(p)), spearman=float(a.rank().corr(p.rank())))


def main():
    manifest = pd.read_csv(ROOT/'kimore_audit_output/kimore_manifest.csv')
    targets = manifest.set_index('sample_id').clinical_ts
    evidence = []
    cached_hash = {}
    for summary_path in sorted((ROOT/'results').rglob('evaluation_summary.json')):
        if 'full_project_audit' in summary_path.parts:
            continue
        summary = json.loads(summary_path.read_text())
        path = summary_path.parent/'oof_predictions.csv'
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        assert frame.sample_id.is_unique
        assert np.isfinite(frame[['actual_ts','predicted_ts']]).all().all()
        assert np.allclose(frame.actual_ts, targets.loc[frame.sample_id].values, rtol=0, atol=1e-10)
        assert (frame.groupby('subject_id').fold.nunique()==1).all()
        assert (frame.subject_id != frame.reference_subject_id).all()
        ref_fold = frame.set_index('subject_id').fold.to_dict()
        assert all(ref_fold.get(r.reference_subject_id) != r.fold for r in frame.itertuples())
        key = next(k for k in ['overall_frame_qc_yu_xiong_dtw','overall_yu_xiong_dtw','overall_plain_dtw'] if k in summary)
        calculated = metrics(frame)
        assert all(np.isclose(v, summary[key][k], atol=1e-10, equal_nan=True) for k,v in calculated.items())
        for variant in ['raw','qc']:
            if variant+'_predicted_ts' in frame:
                assert np.allclose(frame[variant+'_predicted_ts'], frame[variant+'_calibration_intercept'] + frame[variant+'_calibration_slope']*frame[variant+'_paper_score_0_100'])
                assert np.allclose(frame[variant+'_paper_score_0_100'], np.clip(100*(1-frame[variant+'_mean_aligned_vector_angle_degrees']/90), 0, 100))
        ex = summary['exercise']
        if ex not in cached_hash:
            samples, _ = read_manifest(ROOT/'kimore_audit_output/kimore_manifest.csv', ex)
            cached_hash[ex] = _experiment_inputs_sha256(samples)
        evidence.append(dict(path=str(path.relative_to(ROOT)), samples=len(frame), metrics=calculated, source_hash_matches=summary.get('experiment_inputs_sha256')==cached_hash[ex]))
    qc = pd.read_csv(ROOT/'results/interpretable_dtw/all_exercises/oof_predictions_all_exercises.csv')
    ml = pd.read_csv(ROOT/'results/ml_baseline/multiseed_analysis/ensemble_oof_predictions.csv')
    assert set(qc.sample_id)==set(ml.sample_id)
    qc_folds = qc.set_index('sample_id').fold
    assert all(qc_folds[r.sample_id] == r.fold for r in ml.itertuples())
    seed_columns = [c for c in ml if c.startswith('prediction_seed_') and c != 'prediction_seed_std']
    assert np.allclose(ml[seed_columns].mean(axis=1), ml.ensemble_predicted_ts, rtol=0, atol=1e-10)
    assert np.allclose(ml[seed_columns].std(axis=1, ddof=1), ml.prediction_seed_std, rtol=0, atol=1e-10)
    assert np.allclose(ml.ensemble_absolute_error, abs(ml.ensemble_predicted_ts-ml.actual_ts), rtol=0, atol=1e-10)
    target_rows = manifest.dropna(subset=['clinical_ts','clinical_po','clinical_cf'])
    target_sum_mismatch = target_rows[~np.isclose(target_rows.clinical_ts, target_rows.clinical_po+target_rows.clinical_cf)].sample_id.tolist()
    included = set(qc.sample_id)
    coverage = []
    for (ex, cohort), group in manifest.groupby(['exercise','cohort']):
        subset = group[group.sample_id.isin(included)]
        coverage.append(dict(exercise=ex, cohort=cohort, total=len(group), retained=len(subset), excluded=len(group)-len(subset), all_target_mean=float(group.clinical_ts.mean()), retained_target_mean=float(subset.clinical_ts.mean())))
    out = ROOT/'results/full_project_audit'
    out.mkdir(parents=True, exist_ok=True)
    (out/'evaluation_artifacts.json').write_text(json.dumps(dict(runs=evidence, shared_ml_dtw_population=len(ml), ensemble_arithmetic_matches=True, ts_equals_po_plus_cf_mismatches=target_sum_mismatch, cohort_coverage=coverage), indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
