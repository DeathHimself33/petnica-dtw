"""Read-only source-data audit. No labels, features or training inputs are changed.

Run from any directory: python audit_es2_es4_data.py
Dependencies: numpy, pandas, openpyxl, Pillow. Outputs: results/es2_es4_audit/.
Motion thresholds are exploratory flags, never execution-quality labels.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import warnings
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/es2_es4_audit'
sys.path.insert(0,str(ROOT/'src'))
from kimore_dataset import load_joint_positions, JointSequence
from kimore_yu_xiong_dtw import yu_xiong_vectors
from kimore_interpretable_quality import apply_frame_quality_control


def numeric(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=[]
        for row in csv.reader(f):
            while row and not row[-1].strip(): row.pop()
            if row: rows.append([float(v) for v in row])
    return np.asarray(rows)


def prefix(mask):
    return int(np.flatnonzero(~mask)[0]) if (~mask).any() else len(mask)


def association(a,b):
    frame=pd.DataFrame({'a':a,'b':b}).dropna()
    if len(frame)<3 or frame.a.nunique()<2 or frame.b.nunique()<2: return None
    return float(frame.a.rank().corr(frame.b.rank()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(ROOT/'kimore_audit_output/kimore_manifest.csv').fillna('')
    pred=pd.read_csv(ROOT/'results/ml_baseline/multiseed_analysis/ensemble_oof_predictions.csv')
    selected=manifest[manifest.exercise.isin(['Es2','Es4'])]
    assert selected.sample_id.is_unique and len(selected)==156
    # Independent workbook parser checks both filename and the internal ID.
    labels=[]; copies={}; subjects={}
    for r in manifest.itertuples():
        if r.position_path: subjects[r.subject_id]=Path(r.position_path).parents[2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for sid,folder in sorted(subjects.items()):
            copies[sid]=[]
            for path in sorted(folder.glob('Es*/Label/ClinicalAssessment*.xlsx')):
                w=load_workbook(path,read_only=True,data_only=True)
                rows=list(w.active.values); w.close()
                headers,values=rows[0],rows[1]
                mapped={str(h).strip().lower():v for h,v in zip(headers,values) if h is not None}
                internal=str(mapped.get('subject id','')).strip()
                correct_name=bool(re.fullmatch(r'ClinicalAssessment_'+re.escape(sid)+r'(?:\(\d+\))?\.xlsx',path.name,re.I))
                scores={}
                for h,v in mapped.items():
                    match=re.fullmatch(r'clinical\s+(ts|po|cf)\s+ex\s*#?\s*([1-5])',h)
                    if match: scores[f'{match[1]}{match[2]}']=v
                entry=dict(subject_id=sid,path=str(path.relative_to(ROOT)),internal_id=internal,
                    filename_matches=correct_name,internal_id_matches=internal.casefold()==sid.casefold())
                labels.append(entry)
                if correct_name: copies[sid].append(scores)
    pd.DataFrame(labels).to_csv(OUT/'workbook_identity.csv',index=False)
    rows=[]; signals={}; anomalies=[]; hashes={}
    for r in selected.itertuples():
        row=dict(sample_id=r.sample_id,subject_id=r.subject_id,exercise=r.exercise,
                 manifest_usable=str(r.position_target_usable).lower()=='true',manifest_issues=r.issues)
        score_sources=copies.get(r.subject_id,[])
        row['score_copy_conflict']=any(x!=score_sources[0] for x in score_sources[1:])
        for kind in ['ts','po','cf']:
            observed=getattr(r,'clinical_'+kind)
            source=score_sources[0].get(kind+r.exercise[-1]) if score_sources else None
            row[kind+'_source_matches']=(observed=='' and source is None) or (observed!='' and source is not None and np.isclose(float(observed),float(source),atol=1e-8,rtol=0))
        if not r.position_path:
            rows.append(row); continue
        path=Path(r.position_path)
        row['path_identity_matches']=path.parents[2].name==r.subject_id and path.parents[1].name==r.exercise
        digest=hashlib.sha256(path.read_bytes()).hexdigest(); hashes.setdefault(digest,[]).append(r.sample_id)
        row['position_sha256']=digest
        try:
            seq=load_joint_positions(path)
        except ValueError as error:
            row['load_error']=str(error); rows.append(row); continue
        p,t=seq.positions,seq.tracking_states; n=len(p)
        row.update(frames=n,manifest_frames_match=n==int(r.position_frames),
            adjacent_duplicate_positions=int(np.all(p[1:]==p[:-1],axis=(1,2)).sum()),
            inferred_fraction=float((t==1).mean()),untracked_fraction=float((t==0).mean()))
        regular_step=np.zeros(n-1,dtype=bool)
        if r.timestamp_path:
            ts=numeric(r.timestamp_path).reshape(-1); dt=np.diff(ts); positive=dt[dt>0]
            med=float(np.median(positive)) if len(positive) else np.nan
            row.update(timestamp_frames=len(ts),timestamp_frame_delta=len(ts)-n,
                nonincreasing_timestamps=int((dt<=0).sum()),timestamp_median_step=med,
                timestamp_gap_over_2x=int((dt>2*med).sum()),timestamp_max_step_ratio=float(dt.max()/med),
                timestamp_gap_over_5x=int((dt>5*med).sum()),timestamp_large_gap_fraction=float((dt>2*med).mean()),
                raw_timestamp_span=float(ts[-1]-ts[0]))
            if len(ts)==n: regular_step=(dt>0)&(dt<=1.5*med)
        if r.orientation_path:
            row['orientation_frame_delta']=len(numeric(r.orientation_path))-n
        scale=float(np.median(np.linalg.norm(p[:,20]-p[:,0],axis=1)))
        centered=(p-p[:,0:1])/max(scale,1e-9)
        joints=[4,5,6,8,9,10,12,13,14,16,17,18,20]
        displacement=np.linalg.norm(np.diff(centered[:,joints],axis=0),axis=2)
        tracked=(t[:-1,joints]==2)&(t[1:,joints]==2)
        valid=tracked.sum(axis=1)>=6
        energy=np.full(n-1,np.nan)
        for i in np.flatnonzero(valid): energy[i]=np.percentile(displacement[i,tracked[i]],75)
        smooth=pd.Series(energy).rolling(15,center=True,min_periods=8).median().to_numpy()
        row['motion_valid_fraction']=float(np.isfinite(smooth).mean())
        edge=max(1,int(.1*len(smooth)))
        for threshold in [.001,.002,.005]:
            low=np.isfinite(smooth)&(smooth<threshold)
            name=str(threshold).replace('.','p')
            lead,tail=prefix(low),prefix(low[::-1])
            row['low_motion_fraction_'+name]=float(low.sum()/max(np.isfinite(smooth).sum(),1))
            row['terminal_fraction_'+name]=min(lead+tail,len(low))/len(low)
            row['edge_low_fraction_'+name]=float(np.r_[low[:edge],low[-edge:]].mean())
        row['motion_p90']=float(np.nanpercentile(smooth,90)) if np.isfinite(smooth).any() else np.nan
        torso=p[:,20]-p[:,0]
        roll=np.degrees(np.arctan2(torso[:,0],torso[:,1]))
        row['torso_roll_range_p05_p95']=float(np.diff(np.percentile(roll,[5,95]))[0])
        for axis,name in [(0,'x'),(2,'z')]:
            row['root_'+name+'_range_torso_units']=float(np.diff(np.percentile(p[:,0,axis],[5,95]))[0]/max(scale,1e-9))
        # Large jumps are geometric flags, not clinical errors. Fully tracked
        # endpoints are required; retain step locations for inspection.
        jump=np.any((displacement>.25)&tracked,axis=1)
        v=yu_xiong_vectors(seq,allow_degenerate_frames=True)
        qc=apply_frame_quality_control(seq,v,r.exercise)
        usable=~qc.dropped_frame_mask
        kept=jump&usable[:-1]&usable[1:]
        qc_angles=np.degrees(np.arccos(np.clip(np.sum(qc.repaired_vectors[:-1]*qc.repaired_vectors[1:],axis=2),-1,1)))
        surviving=kept&regular_step&(qc_angles.max(axis=1)>45)
        row.update(qc_status=qc.quality_status,jump_steps=int(jump.sum()),
                   retained_jump_steps=int(kept.sum()),retained_jump_fraction=float(kept.sum()/max((usable[:-1]&usable[1:]).sum(),1)),
                   regular_retained_jump_steps=int((kept&regular_step).sum()),
                   regular_jump_with_qc_angle_over45=int(surviving.sum()),
                   regular_retained_jump_fraction=float((kept&regular_step).sum()/max((usable[:-1]&usable[1:]&regular_step).sum(),1)))
        for i in np.flatnonzero(kept):
            anomalies.append(dict(sample_id=r.sample_id,from_frame=int(i),to_frame=int(i+1),
                regular_timestep=bool(regular_step[i]),
                either_frame_interpolated=bool(qc.interpolated_component_mask[i:i+2].any()),
                max_qc_angular_step=float(qc_angles[i].max()),
                max_displacement_torso_units=float(displacement[i,tracked[i]].max())))
        signals[r.sample_id]=(p,t,smooth,roll,scale)
        rows.append(row)
    audit=pd.DataFrame(rows)
    audit.to_csv(OUT/'all_samples.csv',index=False)
    jump_table=pd.DataFrame(anomalies,columns=['sample_id','from_frame','to_frame','regular_timestep','either_frame_interpolated','max_qc_angular_step','max_displacement_torso_units'])
    jump_table.to_csv(OUT/'retained_jump_candidates.csv',index=False)
    pp=pred[pred.exercise.isin(['Es2','Es4'])].copy()
    pp['dtw_absolute_error']=abs(pp.qc_dtw_predicted_ts-pp.actual_ts)
    merged=pp.merge(audit,on=['sample_id','subject_id','exercise'],validate='one_to_one')
    assert len(merged)==136 and merged.manifest_usable.all()
    targets=pd.to_numeric(selected.set_index('sample_id').clinical_ts,errors='coerce')
    assert np.allclose(merged.actual_ts,targets.loc[merged.sample_id],atol=1e-5,rtol=0)
    merged.to_csv(OUT/'evaluated_samples.csv',index=False)
    comparisons=[]; correlations=[]; selections=[]
    cols=['inferred_fraction','untracked_fraction','timestamp_max_step_ratio','retained_jump_fraction','regular_retained_jump_fraction','low_motion_fraction_0p002','terminal_fraction_0p002','edge_low_fraction_0p002','torso_roll_range_p05_p95','root_x_range_torso_units','root_z_range_torso_units']
    for exercise,g in merged.groupby('exercise'):
        for method in ['ensemble_absolute_error','dtw_absolute_error']:
            order=g.sort_values([method,'sample_id'])
            for band,part in [('best_quartile',order.head(len(g)//4)),('worst_quartile',order.tail(len(g)//4))]:
                comparisons.append(dict(exercise=exercise,method=method,band=band,n=len(part),
                    error_mean=float(part[method].mean()),**{c:float(part[c].median()) for c in cols}))
            for c in cols: correlations.append(dict(exercise=exercise,method=method,indicator=c,spearman=association(g[c],g[method])))
        # Two largest ML errors with distinct low-error controls, matched by
        # cohort first and score proximity second. Not causal matching.
        worst=g.nlargest(2,'ensemble_absolute_error'); used=set(worst.sample_id)
        chosen=[]
        for _,bad in worst.iterrows():
            chosen.append((bad,'high ML error'))
            pool=g[(g.ensemble_absolute_error<=g.ensemble_absolute_error.median())&~g.sample_id.isin(used)].copy()
            pool['cohort_mismatch']=pool.cohort!=bad.cohort
            pool['score_gap']=abs(pool.actual_ts-bad.actual_ts)
            good=pool.sort_values(['cohort_mismatch','score_gap','sample_id']).iloc[0]
            chosen.append((good,'low-error comparison')); used.add(good.sample_id)
        im=Image.new('RGB',(1500,340*len(chosen)),'white'); draw=ImageDraw.Draw(im)
        bones=[(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(20,8),(8,9),(9,10),(0,12),(12,13),(13,14),(0,16),(16,17),(17,18)]
        for rn,(r,label) in enumerate(chosen):
            selections.append(dict(exercise=exercise,sample_id=r.sample_id,role=label,actual_ts=r.actual_ts,ml_error=r.ensemble_absolute_error,dtw_error=r.dtw_absolute_error))
            p,t,smooth,roll,scale=signals[r.sample_id]; y0=rn*340
            draw.text((10,y0+5),f'{r.sample_id} | {label} | TS {r.actual_ts:.2f} ML error {r.ensemble_absolute_error:.2f} DTW error {r.dtw_absolute_error:.2f}',fill='black')
            for col,idx in enumerate(np.rint(np.linspace(0,len(p)-1,7)).astype(int)):
                for a,b in bones:
                    pts=[(105+col*210+float(p[idx,j,0]-p[idx,0,0])*80,y0+130-float(p[idx,j,1]-p[idx,0,1])*80) for j in (a,b)]
                    draw.line(pts,fill='#265c92' if min(t[idx,a],t[idx,b])==2 else '#c57216',width=3)
                draw.text((col*210+12,y0+218),f'frame {idx}',fill='black')
            # Shared motion scale, clipping disclosed on the chart.
            draw.text((10,y0+245),'Motion: torso lengths/frame; chart clipped at .03; red = .002; gaps = insufficient tracking',fill='black')
            draw.line([(20,y0+315-.002/.03*45),(1480,y0+315-.002/.03*45)],fill='#c33',width=1)
            for i in range(1,len(smooth)):
                if np.isfinite(smooth[i-1:i+1]).all():
                    draw.line([(20+(i-1)/(len(smooth)-1)*1460,y0+315-min(smooth[i-1],.03)/.03*45),(20+i/(len(smooth)-1)*1460,y0+315-min(smooth[i],.03)/.03*45)],fill='#265c92',width=2)
        im.save(OUT/f'{exercise}_matched_examples.png')
    pd.DataFrame(selections).to_csv(OUT/'visual_selection.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'error_quartile_comparison.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'indicator_correlations.csv',index=False)
    top_jumps=jump_table[jump_table.regular_timestep&jump_table.sample_id.isin(merged.sample_id)].sort_values('max_displacement_torso_units',ascending=False).drop_duplicates('sample_id').head(4)
    top_jumps.to_csv(OUT/'visual_jump_selection.csv',index=False)
    im=Image.new('RGB',(1200,300*len(top_jumps)),'white'); draw=ImageDraw.Draw(im)
    for rn,r in enumerate(top_jumps.itertuples()):
        p,t,*_=signals[r.sample_id]; start=max(0,r.from_frame-1); stop=min(len(p)-1,r.to_frame+1)
        draw.text((10,rn*300+5),f'{r.sample_id}: normal timestamp step; jump {r.max_displacement_torso_units:.2f} torso lengths',fill='black')
        for col,idx in enumerate(range(start,stop+1)):
            for a,b in bones:
                pts=[(150+col*300+float(p[idx,j,0]-p[idx,0,0])*80,rn*300+145-float(p[idx,j,1]-p[idx,0,1])*80) for j in (a,b)]
                draw.line(pts,fill='#265c92' if min(t[idx,a],t[idx,b])==2 else '#c57216',width=3)
            draw.text((col*300+80,rn*300+260),f'frame {idx}',fill='black')
    im.save(OUT/'jump_examples.png')
    # Demonstrate exact invariance to a time-varying root translation.
    p,t,*_=next(iter(signals.values()))
    shift=np.zeros((len(p),1,3)); shift[:,0,0]=.2*np.sin(np.linspace(0,4*np.pi,len(p)))
    delta=float(np.max(abs(yu_xiong_vectors(JointSequence(p,t),True)-yu_xiong_vectors(JointSequence(p+shift,t),True))))
    assert delta<1e-10
    summary=dict(manifest_rows=len(audit),evaluated_rows=len(merged),workbooks=len(labels),
        workbook_identity_anomalies=[r for r in labels if not r['filename_matches'] or not r['internal_id_matches']],
        score_mismatch_samples=audit.loc[~audit[['ts_source_matches','po_source_matches','cf_source_matches']].all(axis=1),'sample_id'].tolist(),
        duplicate_recordings=[v for v in hashes.values() if len(v)>1],translation_feature_max_difference=delta,
        exercises={})
    for ex,g in merged.groupby('exercise'):
        allg=audit[audit.exercise==ex]
        summary['exercises'][ex]=dict(manifest_n=len(allg),loaded_n=int(allg.frames.notna().sum()),evaluated_n=len(g),
            timestamp_mismatches=int((g.timestamp_frame_delta!=0).sum()),nonincreasing_timestamps=int(g.nonincreasing_timestamps.sum()),
            recordings_with_large_timestamp_gaps=int((g.timestamp_gap_over_2x>0).sum()),
            recordings_with_over5x_gaps=int((g.timestamp_gap_over_5x>0).sum()),
            median_max_timestamp_ratio=float(g.timestamp_max_step_ratio.median()),
            adjacent_duplicate_positions=int(g.adjacent_duplicate_positions.sum()),
            recordings_with_retained_jumps=int((g.retained_jump_steps>0).sum()),retained_jump_steps=int(g.retained_jump_steps.sum()),
            regular_retained_jump_steps=int(g.regular_retained_jump_steps.sum()),
            regular_jumps_with_qc_angle_over45=int(g.regular_jump_with_qc_angle_over45.sum()),
            median_low_motion_fraction=float(g.low_motion_fraction_0p002.median()),
            terminal_over_10percent={c:int((g[c]>.1).sum()) for c in ['terminal_fraction_0p001','terminal_fraction_0p002','terminal_fraction_0p005']})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
