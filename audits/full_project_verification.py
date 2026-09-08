"""Independent audit checks and defect reproducers; production inputs stay read-only.

Run with the project's Python environment. Synthetic mutations are confined to
temporary directories. Evidence is written to results/full_project_audit/.
"""
from pathlib import Path
import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/full_project_audit'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tests')]
from kimore_ml_data import build_ml_dataset, apply_feature_standardizer, FeatureStandardizer, fit_feature_standardizer
from kimore_ml_model import TemporalScoreModel
from kimore_dtw import exact_dtw
from kimore_yu_xiong_dtw import yu_xiong_dtw
from kimore_dataset import read_manifest, load_joint_positions, JOINT_INDEX
from kimore_pilot_review import render_skeleton, video_path, PILOT_SAMPLE_IDS
from kimore_apply_pilot_labels import merge_labels
from kimore_expert_review import create_app
from kimore_expert_review.model import validate_adjudication, save_adjudication
from test_expert_review_app import ExpertReviewAppTests
import train_ml_baseline as trainer
import analyze_ml_results as aggregation


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def dtw_oracle():
    rng=np.random.default_rng(92731)
    def brute(cost):
        def search(i,j):
            if i==0 and j==0: return cost[0,0]
            options=[]
            if i: options.append(search(i-1,j))
            if j: options.append(search(i,j-1))
            if i and j: options.append(search(i-1,j-1))
            return cost[i,j]+min(options)
        return search(len(cost)-1,cost.shape[1]-1)
    for _ in range(150):
        n,m=rng.integers(1,5,size=2)
        a,b=rng.integers(-3,4,size=(n,2)),rng.integers(-3,4,size=(m,2))
        cost=((a[:,None]-b[None,:])**2).sum(axis=2)
        alignment=exact_dtw(a,b)
        assert np.isclose(alignment.total_squared_cost,brute(cost))
        assert np.isclose(alignment.total_squared_cost,exact_dtw(b,a).total_squared_cost)
        steps=np.diff(alignment.path,axis=0)
        assert ((steps>=0)&(steps<=1)).all() and (steps.sum(axis=1)>0).all()
        a,b=rng.normal(size=(n,9,3)),rng.normal(size=(m,9,3))
        a/=np.linalg.norm(a,axis=2,keepdims=True); b/=np.linalg.norm(b,axis=2,keepdims=True)
        cost=np.degrees(np.arccos(np.clip(np.einsum('nkd,mkd->nmk',a,b),-1,1))).sum(axis=2)
        alignment=yu_xiong_dtw(a,b)
        assert np.isclose(alignment.total_angular_cost_degrees,brute(cost),atol=1e-8)
    return dict(squared_cases=150,angular_cases=150,passed=True)


def ml_artifacts():
    print('Rebuilding all ML tensors from source recordings...',flush=True)
    dataset,exclusions=build_ml_dataset(ROOT/'kimore_audit_output/kimore_manifest.csv',progress=lambda _:None)
    with np.load(ROOT/'results/ml_data/kimore_all_exercises_128.npz',allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    comparisons={}
    for name in arrays:
        value=np.asarray(getattr(dataset,name))
        comparisons[name]=bool(np.array_equal(value,arrays[name]))
    assert all(comparisons.values()),comparisons
    torch.set_num_threads(2)
    result=[]
    runs=[ROOT/'results/ml_baseline/tcn_bigru_attention']+[ROOT/f'results/ml_baseline/tcn_bigru_attention_seed_{s}' for s in range(20260904,20260908)]
    for run in runs:
        print('Checking checkpoints:',run.name,flush=True)
        oof=read_csv(run/'oof_predictions.csv')
        for fold in range(1,6):
            # These are trusted locally generated project checkpoints.
            ckpt=torch.load(run/f'fold_{fold}/checkpoint.pt',map_location='cpu',weights_only=False)
            train=np.flatnonzero(~np.isin(arrays['fold_numbers'],[fold,fold%5+1]))
            test=np.flatnonzero(arrays['fold_numbers']==fold)
            valid=np.flatnonzero(arrays['fold_numbers']==fold%5+1)
            assert not (set(arrays['subject_ids'][train])&set(arrays['subject_ids'][valid]))
            assert not (set(arrays['subject_ids'][train])&set(arrays['subject_ids'][test]))
            scaler=fit_feature_standardizer(arrays['features'],arrays['frame_mask'],train)
            assert np.array_equal(scaler.mean,ckpt['feature_mean']) and np.array_equal(scaler.scale,ckpt['feature_scale'])
            ts=trainer.fit_target_standardizer(arrays['targets'],arrays['exercise_indices'],train)
            assert np.array_equal(ts.mean,ckpt['target_mean']) and np.array_equal(ts.scale,ckpt['target_scale'])
            model=TemporalScoreModel(**ckpt['model']).eval(); model.load_state_dict(ckpt['model_state'])
            x=apply_feature_standardizer(arrays['features'][test],arrays['frame_mask'][test],scaler)
            with torch.inference_mode():
                y=model(torch.from_numpy(x),torch.from_numpy(arrays['frame_mask'][test]),torch.from_numpy(arrays['component_observed_mask'][test]),torch.from_numpy(arrays['exercise_indices'][test])).numpy()
            y=y*ts.scale[arrays['exercise_indices'][test]]+ts.mean[arrays['exercise_indices'][test]]
            saved={r['sample_id']:float(r['predicted_ts']) for r in oof if int(r['fold'])==fold}
            assert set(saved)==set(arrays['sample_ids'][test])
            delta=max(abs(float(p)-saved[s]) for p,s in zip(y,arrays['sample_ids'][test]))
            result.append(dict(run=run.name,fold=fold,prediction_max_abs_delta_cpu=delta))
    return dict(tensor_exact_matches=comparisons,samples=len(dataset.targets),exclusions=len(exclusions),checkpoints=result)


def resume_reproducer():
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)
        original=ROOT/'results/ml_baseline/tcn_bigru_attention'
        for fold in range(1,6):
            (out/f'fold_{fold}').mkdir()
            shutil.copyfile(original/f'fold_{fold}/predictions.csv',out/f'fold_{fold}/predictions.csv')
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.main(['--data',str(ROOT/'results/ml_data/kimore_all_exercises_128.npz'),'--output-dir',str(out),'--resume','--seed','999123','--channels','32','--device','cpu'])
        seed,summary,rows=aggregation.load_run(out)
        source=read_csv(original/'oof_predictions.csv')
        assert seed==999123 and rows==source
        assert not list(out.glob('fold_*/checkpoint.pt'))
        return dict(new_seed=seed,claimed_channels=summary['configuration']['channels'],unchanged_old_predictions=len(rows),accepted_as_complete_run=True,checkpoints_required=False)


def app_reproducers():
    case=ExpertReviewAppTests(); case.setUp()
    try:
        for reviewer in ['audit_second_A','audit_second_B']:
            case._start(reviewer)
            for pos in [1,2]: case._save_label(pos,'error',error_type='posture',severity='moderate')
            case.client.post('/finish',data={'csrf_token':case._csrf()})
        candidate=case.app.extensions['review_candidates'][0]
        verdict=dict(review_status='reviewed',execution_label='correct',error_type='',severity='',reviewer_confidence='high',review_notes='Resolved only for A')
        save_adjudication(case.database,'adjudicator_A',candidate,verdict)
        # Current session is reviewer B. A's global adjudication is reused.
        response=case.client.get('/export/adjudicated.csv')
        rows=list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        leaked=next(r for r in rows if r['sample_id']==candidate.key[0])
        assert leaked['final_annotator']=='adjudicator_A' and leaked['adjudication_complete']=='true'
        # A fresh browser can claim an existing sealed ID without a secret.
        other=case.app.test_client(); other.get('/')
        with other.session_transaction() as s: token=s['csrf_token']
        other.post('/session',data={'csrf_token':token,'reviewer_id':'audit_second_A'})
        unauthenticated=other.get('/agreement').status_code
        assert unauthenticated==200
        before=case.client.get('/export/agreement.json').json
        agreement_html=case.client.get('/agreement').get_data(as_text=True)
        count_renders_method='built-in method items' in agreement_html
        assert count_renders_method
        primary=read_csv(case.primary)
        for r in primary:
            r.update(execution_label='error',error_type='posture',severity='moderate')
        case._write_csv(case.primary,list(primary[0]),primary)
        new=create_app(queue_path=case.queue,primary_labels_path=case.primary,sheets_dir=case.sheets,database_path=case.database,testing=True,secret_key='test-secret')
        client=new.test_client()
        with client.session_transaction() as s: s['reviewer_id']='audit_second_B'
        after=client.get('/export/agreement.json').json
        assert before['exact_agreement']==0 and after['exact_agreement']==1
        try:
            validate_adjudication(dict(execution_label='ungradable',reviewer_confidence='high',review_notes='No visible evidence'))
        except ValueError: forced_binary=True
        else: forced_binary=False
        assert forced_binary
        return dict(cross_reviewer_adjudication_reused=True,sealed_identity_reentry_without_auth=unauthenticated,
                    mutable_primary_changes_sealed_agreement=[before['exact_agreement'],after['exact_agreement']],ungradable_final_disallowed=forced_binary,
                    agreement_item_count_renders_dict_method=count_renders_method)
    finally: case.tearDown()


def pilot_checks():
    folder=ROOT/'results/interpretable_dtw/pilot_review'
    labels=read_csv(ROOT/'annotations/kimore_es3_pilot_labels.csv')
    queue=read_csv(folder/'pilot_annotation_queue.csv')
    second=read_csv(folder/'second_review_queue.csv')
    keys={(r['sample_id'],r['candidate_rank']) for r in second}
    uncertain=[r for r in labels if r['execution_label'] in ['uncertain','ungradable']]
    missed=[(r['sample_id'],r['candidate_rank']) for r in uncertain if (r['sample_id'],r['candidate_rank']) not in keys]
    changed=[dict(r) for r in queue]; changed[0]['component_name']='different_component'; changed[0]['original_frame_start']='999999'
    merged=merge_labels(changed,labels)
    assert merged[0]['component_name']=='different_component' and merged[0]['execution_label']
    samples,_=read_manifest(ROOT/'kimore_audit_output/kimore_manifest.csv','Es3')
    samples={s.sample_id:s for s in samples}
    refs=sorted({r['reference_sample_id'] for r in queue})
    fallbacks=[]
    for sid in refs:
        sample=samples[sid]
        if video_path(sample) is not None: continue
        seq=load_joint_positions(sample.position_path); pose=seq.positions[0]
        h=max(float(pose[JOINT_INDEX['Head'],1]-min(pose[JOINT_INDEX['AnkleLeft'],1],pose[JOINT_INDEX['AnkleRight'],1])),.5)
        y=470-(pose[:,1]-pose[JOINT_INDEX['SpineBase'],1])*360/h
        clipped=[name for name,i in JOINT_INDEX.items() if y[i]<0 or y[i]>=540]
        fallbacks.append(dict(reference=sid,clipped_joints_first_frame=clipped))
        import cv2
        cv2.imwrite(str(OUT/f'fallback_{sid}.png'),render_skeleton(seq,0))
    return dict(uncertain_pilot_total=len(uncertain),uncertain_outside_app_queue=missed,
        labels_attach_after_interval_change=True,missing_rgb_references=fallbacks,
        pilot_default_queue='results/interpretable_dtw/annotation_queue.csv',current_default_experiment_queue='results/interpretable_dtw/Es3/annotation_queue.csv')


def main():
    OUT.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    checks={}
    for name,fn in [('dtw_oracle',dtw_oracle),('resume',resume_reproducer),('expert_app',app_reproducers),('pilot',pilot_checks),('ml_artifacts',ml_artifacts)]:
        print('Running',name,flush=True)
        try: checks[name]=fn()
        except Exception as error:
            checks[name]={'audit_error':repr(error)}
            import traceback; traceback.print_exc()
        (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    checks['elapsed_seconds']=time.perf_counter()-start
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    if any(isinstance(value,dict) and 'audit_error' in value for value in checks.values()):
        raise SystemExit('One or more audit checks failed; inspect verification.json')


if __name__=='__main__': main()
