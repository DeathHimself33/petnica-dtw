"""Reproduce saved test predictions on the original GPU without training."""
from full_project_verification import *


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Original CUDA backend unavailable')
    trainer.set_seed(20260903)
    device = torch.device('cuda')
    with np.load(ROOT/'results/ml_data/kimore_all_exercises_128.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    runs = [ROOT/'results/ml_baseline/tcn_bigru_attention'] + [ROOT/f'results/ml_baseline/tcn_bigru_attention_seed_{s}' for s in range(20260904, 20260908)]
    results = []
    for run in runs:
        config = json.loads((run/'summary.json').read_text())['configuration']
        saved = {r['sample_id']: float(r['predicted_ts']) for r in read_csv(run/'oof_predictions.csv')}
        for fold in range(1, 6):
            ckpt = torch.load(run/f'fold_{fold}/checkpoint.pt', map_location='cpu', weights_only=False)
            scaler = FeatureStandardizer(ckpt['feature_mean'], ckpt['feature_scale'])
            x = apply_feature_standardizer(arrays['features'], arrays['frame_mask'], scaler)
            tensors = tuple(torch.from_numpy(v) for v in [x, arrays['frame_mask'], arrays['component_observed_mask'], arrays['exercise_indices'], arrays['targets']])
            indices = np.flatnonzero(arrays['fold_numbers'] == fold)
            loader = trainer.make_loader(tensors, indices, config['batch_size'], False, config['seed'])
            model = TemporalScoreModel(**ckpt['model']).to(device)
            model.load_state_dict(ckpt['model_state'])
            y, _, ex = trainer.predict_normalized(model, loader, device)
            y = y * ckpt['target_scale'][ex] + ckpt['target_mean'][ex]
            delta = max(abs(float(p)-saved[s]) for p, s in zip(y, arrays['sample_ids'][indices]))
            results.append(dict(run=run.name, fold=fold, max_abs_delta=delta))
        print(run.name, max(r['max_abs_delta'] for r in results if r['run']==run.name), flush=True)
    (OUT/'checkpoint_gpu_inference.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), results=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
