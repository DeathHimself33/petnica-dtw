"""Exercise a supported six-fold export against trainer orchestration, no training."""
from full_project_verification import *
from unittest.mock import patch


def main():
    dataset, _ = build_ml_dataset(ROOT/'kimore_audit_output/kimore_manifest.csv', n_splits=6, progress=lambda _:None)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        arrays={key:np.asarray(value) for key,value in vars(dataset).items()}
        np.savez(folder/'six.npz', **arrays)
        requested=[]
        def fake_train(a, fold, output, device, config):
            requested.append(fold)
            return [dict(fold=fold, validation_fold=fold%5+1, sample_id=str(a['sample_ids'][i]), subject_id=str(a['subject_ids'][i]), cohort=str(a['cohorts'][i]), exercise=trainer.EXERCISES[int(a['exercise_indices'][i])], actual_ts=float(a['targets'][i]), predicted_ts=30., training_exercise_mean_ts=30.) for i in np.flatnonzero(a['fold_numbers']==fold)]
        with patch.object(trainer, 'train_one_fold', side_effect=fake_train), contextlib.redirect_stdout(io.StringIO()):
            trainer.main(['--data',str(folder/'six.npz'),'--output-dir',str(folder/'run'),'--device','cpu'])
        summary=json.loads((folder/'run/summary.json').read_text())
        result=dict(exported_samples=len(dataset.targets), exported_folds=sorted(set(int(x) for x in dataset.fold_numbers)), requested_test_folds=requested, reported_completed_folds=summary['completed_folds'], reported_oof_samples=summary['oof_samples'], missing_test_samples=int((dataset.fold_numbers==6).sum()), training_mocked=True)
        assert result['exported_samples']-result['reported_oof_samples']==result['missing_test_samples']>0
        (OUT/'six_fold_contract.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__=='__main__': main()
