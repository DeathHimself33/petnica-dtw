"""Post-hoc descriptive Es4 diagnostics; never fits or changes a predictor.

Run from the repository root with Python, numpy, pandas and Pillow.
Outputs contain local dataset identifiers and stay in ignored results/.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/es4_diagnostics'
sys.path.insert(0, str(ROOT/'src'))
from kimore_dataset import load_joint_positions, JointSequence
from kimore_yu_xiong_dtw import yu_xiong_vectors
from kimore_interpretable_quality import apply_frame_quality_control


def corr(a, b):
    a, b = pd.Series(np.asarray(a)), pd.Series(np.asarray(b))
    return float(a.corr(b)) if a.std() > 0 and b.std() > 0 else None


def metrics(g):
    y, p = g.actual_ts, g.ensemble_predicted_ts
    return dict(n=len(g), mae=float(abs(p-y).mean()),
                rmse=float(np.sqrt(np.mean((p-y)**2))), pearson=corr(y,p),
                bias=float((p-y).mean()), target_mean=float(y.mean()),
                target_sd=float(y.std()), prediction_sd=float(p.std()),
                dtw_mae=float(abs(g.qc_dtw_predicted_ts-y).mean()),
                mean_baseline_mae=float(abs(g.training_exercise_mean_ts-y).mean()),
                seed_sd=float(g.prediction_seed_std.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT/'results/ml_baseline/multiseed_analysis/ensemble_oof_predictions.csv')
    assert d.sample_id.is_unique and len(d) == 338
    q = pd.concat([pd.read_csv(ROOT/f'results/interpretable_dtw/all_exercises/Es{i}/component_quality.csv') for i in range(1,6)])
    qs = q.groupby('sample_id').agg(total_frames=('total_frames','first'),
        dropped_frames=('dropped_frames','first'), interpolated_frames=('interpolated_frames','first'),
        tracked_mean=('tracked_fraction','mean'), tracked_min=('tracked_fraction','min'),
        invalid_mean=('invalid_fraction','mean'), temporal_outliers=('temporal_outlier_frames','sum'))
    d = d.merge(qs, on='sample_id', validate='one_to_one')
    assert len(d) == 338, 'QC merge lost held-out samples'
    seed_columns = [c for c in d if c.startswith('prediction_seed_') and c != 'prediction_seed_std']
    assert len(seed_columns) == 5
    assert np.allclose(d[seed_columns].mean(axis=1), d.ensemble_predicted_ts)
    assert np.allclose(abs(d.ensemble_predicted_ts-d.actual_ts), d.ensemble_absolute_error)
    d['drop_fraction'] = d.dropped_frames/d.total_frames
    d['interpolation_fraction'] = d.interpolated_frames/d.total_frames
    d['residual'] = d.ensemble_predicted_ts-d.actual_ts
    d['dtw_residual'] = d.qc_dtw_predicted_ts-d.actual_ts
    d['dtw_absolute_error'] = abs(d.dtw_residual)
    d['score_band'] = pd.cut(d.actual_ts, [-1,20,30,40,50], labels=['0-20','20-30','30-40','40-50'])
    e = d[d.exercise == 'Es4'].copy()
    assert len(e) == 66 and e.subject_id.is_unique
    e.sort_values('ensemble_absolute_error', ascending=False).to_csv(OUT/'es4_samples.csv',index=False)
    tables = {}
    for name, frame, group in [('exercise',d,'exercise'),('cohort',e,'cohort'),('fold',e,'fold'),('quality',e,'quality_status'),('score_band',e,'score_band')]:
        records = [{group:str(k), **metrics(g)} for k,g in frame.groupby(group,observed=True)]
        pd.DataFrame(records).to_csv(OUT/f'{name}_metrics.csv',index=False)
        tables[name] = records
    diagnostic_cols = ['tracked_mean','tracked_min','invalid_mean','drop_fraction','interpolation_fraction','total_frames','prediction_seed_std']
    qc_corr = {c:dict(pearson=corr(e[c],e.ensemble_absolute_error),spearman=corr(e[c].rank(),e.ensemble_absolute_error.rank())) for c in diagnostic_cols}
    worst = e.nlargest(10,'ensemble_absolute_error')
    seeds = [c for c in e if c.startswith('prediction_seed_') and c != 'prediction_seed_std']
    fold_seed = []
    for c in seeds:
        for fold,g in e.groupby('fold'):
            fold_seed.append(dict(seed=c,fold=int(fold),mae=float(abs(g[c]-g.actual_ts).mean()),bias=float((g[c]-g.actual_ts).mean())))
    pd.DataFrame(fold_seed).to_csv(OUT/'fold_seed_metrics.csv',index=False)
    # Descriptive exclusion sensitivity, not a revised performance estimate.
    sensitivity = {str(k):metrics(e.drop(e.nlargest(k,'ensemble_absolute_error').index)) for k in [1,3,5,10]}
    qc_counts = q.drop_duplicates('sample_id').groupby(['exercise','sample_quality_status']).size().unstack(fill_value=0)
    qc_counts.to_csv(OUT/'coverage.csv')
    same = set(worst.sample_id) & set(e.nlargest(10,'dtw_absolute_error').sample_id)
    summary = dict(tables=tables, qc_correlations=qc_corr,
        residual_correlation=corr(e.residual,e.dtw_residual), absolute_error_correlation=corr(e.ensemble_absolute_error,e.dtw_absolute_error),
        top10_overlap=sorted(same),top10_absolute_error_share=float(worst.ensemble_absolute_error.sum()/e.ensemble_absolute_error.sum()),
        exclusion_sensitivity=sensitivity, target_prediction_slope=float(np.polyfit(e.actual_ts,e.ensemble_predicted_ts,1)[0]))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Fixed-progress skeleton contact sheets: three worst and two best examples.
    manifest = pd.read_csv(ROOT/'kimore_audit_output/kimore_manifest.csv').set_index('sample_id')
    selected = pd.concat([e.nlargest(3,'ensemble_absolute_error'), e.nsmallest(2,'ensemble_absolute_error')])
    bones = [(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(20,8),(8,9),(9,10),(0,12),(12,13),(13,14),(0,16),(16,17),(17,18)]
    im = Image.new('RGB',(1500,len(selected)*260),'white'); draw = ImageDraw.Draw(im)
    feature_rows=[]
    for row_no,(_,r) in enumerate(selected.iterrows()):
        raw = np.genfromtxt(manifest.loc[r.sample_id,'position_path'],delimiter=',')
        raw = raw[:,:100].reshape(-1,25,4); xyz=raw[:,:,:3]; centered=xyz-xyz[:,0:1,:]
        y0=row_no*260
        draw.text((10,y0+5),f'{r.sample_id}  TS {r.actual_ts:.1f}  ML {r.ensemble_predicted_ts:.1f}  DTW {r.qc_dtw_predicted_ts:.1f}  QC {r.quality_status}', fill='black')
        for col,idx in enumerate(np.rint(np.linspace(0,len(raw)-1,7)).astype(int)):
            for a,b in bones:
                pts=[(col*210+105+float(centered[idx,j,0])*90,y0+140-float(centered[idx,j,1])*90) for j in (a,b)]
                draw.line(pts,fill='#265c92' if min(raw[idx,a,3],raw[idx,b,3])==2 else '#db7b28',width=3)
            draw.text((col*210+10,y0+235),f'{idx}/{len(raw)-1}',fill='black')
        torso=xyz[:,20]-xyz[:,0]
        tilt=np.degrees(np.arccos(np.clip(torso[:,1]/np.maximum(np.linalg.norm(torso,axis=1),1e-9),-1,1)))
        feature_rows.append(dict(sample_id=r.sample_id,torso_tilt_p05=float(np.percentile(tilt,5)),torso_tilt_p95=float(np.percentile(tilt,95)),frames=len(raw)))
    im.save(OUT/'skeleton_examples.png')
    pd.DataFrame(feature_rows).to_csv(OUT/'visual_example_measurements.csv',index=False)
    # Quantify 128-grid reconstruction distortion on QC-valid frames only.
    # Linear reconstruction is a diagnostic, not the model's input algorithm.
    sampling=[]
    for _,r in e.iterrows():
        seq=load_joint_positions(Path(manifest.loc[r.sample_id,'position_path']))
        v=yu_xiong_vectors(seq,allow_degenerate_frames=True)
        quality=apply_frame_quality_control(seq,v,'Es4')
        v=quality.repaired_vectors
        idx=np.rint(np.linspace(0,len(v)-1,128)).astype(int)
        idx=idx[~quality.dropped_frame_mask[idx]]
        recon=np.stack([np.interp(np.arange(len(v)),idx,v[idx,j,k]) for j in range(9) for k in range(3)],axis=1).reshape(-1,9,3)
        recon/=np.maximum(np.linalg.norm(recon,axis=2,keepdims=True),1e-12)
        angles=np.degrees(np.arccos(np.clip(np.sum(v*recon,axis=2),-1,1)))[~quality.dropped_frame_mask]
        sampling.append(dict(sample_id=r.sample_id,mean_reconstruction_degrees=float(angles.mean()),p95_reconstruction_degrees=float(np.percentile(angles,95))))
    sampling=pd.DataFrame(sampling).merge(e[['sample_id','ensemble_absolute_error']],validate='one_to_one')
    sampling.to_csv(OUT/'resampling_diagnostics.csv',index=False)
    # Exact representational invariance under a rigid roll about body-forward.
    seq=load_joint_positions(Path(manifest.loc[e.iloc[0].sample_id,'position_path']))
    original=JointSequence(seq.positions[:1],seq.tracking_states[:1])
    v=yu_xiong_vectors(original,allow_degenerate_frames=True)
    axis=v[:,8:9,:]; centered=original.positions-original.positions[:,0:1,:]
    theta=np.deg2rad(20)
    rotated=centered*np.cos(theta)+np.cross(axis,centered)*np.sin(theta)+axis*np.sum(axis*centered,axis=2,keepdims=True)*(1-np.cos(theta))
    changed=yu_xiong_vectors(JointSequence(rotated,original.tracking_states),allow_degenerate_frames=True)
    assert np.allclose(v,changed,atol=1e-10)
    summary['representation_check']=dict(rotation_degrees=20,max_feature_difference=float(np.max(abs(v-changed))))
    summary['resampling']=dict(median_mean_angle=float(sampling.mean_reconstruction_degrees.median()),
        median_p95_angle=float(sampling.p95_reconstruction_degrees.median()),
        error_correlation=corr(sampling.mean_reconstruction_degrees,sampling.ensemble_absolute_error))
    splits=[]
    for fold in range(1,6):
        train=e[~e.fold.isin([fold,fold%5+1])]
        assert np.isclose(train.actual_ts.mean(),e.loc[e.fold==fold,'training_exercise_mean_ts'].iloc[0],atol=1e-4)
        splits.append(dict(test_fold=fold,train_n=len(train),low_score_n=int((train.actual_ts<=20).sum()),train_mean=float(train.actual_ts.mean())))
    summary['training_splits']=splits
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['representation_check','resampling','training_splits']},indent=2))


if __name__ == '__main__':
    main()
