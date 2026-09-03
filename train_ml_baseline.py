"""Train the compact multi-exercise TCN-BiGRU-attention KIMORE baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - exercised only without ML extras
    raise SystemExit(
        "PyTorch is required. Install it with: "
        ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-ml.txt"
    ) from error

from kimore_evaluation import regression_metrics  # noqa: E402
from kimore_ml_data import (  # noqa: E402
    EXERCISES,
    apply_feature_standardizer,
    fit_feature_standardizer,
    fold_train_test_indices,
)
from kimore_ml_model import TemporalScoreModel, trainable_parameter_count  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    channels: int
    gru_hidden: int
    dropout: float
    seed: int
    amp: bool


@dataclass(frozen=True)
class TargetStandardizer:
    mean: np.ndarray
    scale: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/ml_data/kimore_all_exercises_128.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ml_baseline/tcn_bigru_attention"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--fold",
        default="all",
        choices=("all", "1", "2", "3", "4", "5"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "cpu")
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds that already contain predictions.csv",
    )
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access it")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_target_standardizer(
    targets: np.ndarray,
    exercise_indices: np.ndarray,
    train_indices: np.ndarray,
) -> TargetStandardizer:
    means = np.zeros(len(EXERCISES), dtype=np.float64)
    scales = np.ones(len(EXERCISES), dtype=np.float64)
    for exercise_index in range(len(EXERCISES)):
        matching = train_indices[
            exercise_indices[train_indices] == exercise_index
        ]
        if len(matching) == 0:
            raise ValueError(
                f"Training split has no {EXERCISES[exercise_index]} samples"
            )
        values = targets[matching].astype(np.float64)
        means[exercise_index] = float(np.mean(values))
        scale = float(np.std(values))
        scales[exercise_index] = scale if scale > np.finfo(float).eps else 1.0
    return TargetStandardizer(mean=means, scale=scales)


def normalize_targets(
    targets: np.ndarray,
    exercise_indices: np.ndarray,
    standardizer: TargetStandardizer,
) -> np.ndarray:
    return (
        (targets - standardizer.mean[exercise_indices])
        / standardizer.scale[exercise_indices]
    ).astype(np.float32)


def denormalize_targets(
    targets: np.ndarray,
    exercise_indices: np.ndarray,
    standardizer: TargetStandardizer,
) -> np.ndarray:
    return (
        targets * standardizer.scale[exercise_indices]
        + standardizer.mean[exercise_indices]
    )


def make_loader(
    tensors: tuple[torch.Tensor, ...],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(*(tensor[index_tensor] for tensor in tensors))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def predict_normalized(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    exercises: list[np.ndarray] = []
    with torch.inference_mode():
        for vectors, frame_mask, observed_mask, exercise, target in loader:
            output = model(
                vectors.to(device, non_blocking=True),
                frame_mask.to(device, non_blocking=True),
                observed_mask.to(device, non_blocking=True),
                exercise.to(device, non_blocking=True),
            )
            predictions.append(output.detach().cpu().numpy())
            targets.append(target.numpy())
            exercises.append(exercise.numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(exercises),
    )


def train_one_fold(
    arrays: dict[str, np.ndarray],
    test_fold: int,
    output_dir: Path,
    device: torch.device,
    config: TrainingConfig,
) -> list[dict[str, object]]:
    fold_numbers = arrays["fold_numbers"]
    subject_ids = tuple(str(value) for value in arrays["subject_ids"])
    outer_train, test_indices = fold_train_test_indices(
        subject_ids, fold_numbers, test_fold
    )
    validation_fold = test_fold % 5 + 1
    validation_indices = outer_train[fold_numbers[outer_train] == validation_fold]
    train_indices = outer_train[fold_numbers[outer_train] != validation_fold]
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("Training and validation partitions must be non-empty")

    feature_standardizer = fit_feature_standardizer(
        arrays["features"], arrays["frame_mask"], train_indices
    )
    standardized_features = apply_feature_standardizer(
        arrays["features"], arrays["frame_mask"], feature_standardizer
    )
    target_standardizer = fit_target_standardizer(
        arrays["targets"], arrays["exercise_indices"], train_indices
    )
    normalized_targets = normalize_targets(
        arrays["targets"], arrays["exercise_indices"], target_standardizer
    )

    tensors = (
        torch.from_numpy(standardized_features),
        torch.from_numpy(arrays["frame_mask"]),
        torch.from_numpy(arrays["component_observed_mask"]),
        torch.from_numpy(arrays["exercise_indices"]),
        torch.from_numpy(normalized_targets),
    )
    train_loader = make_loader(
        tensors, train_indices, config.batch_size, True, config.seed + test_fold
    )
    validation_loader = make_loader(
        tensors, validation_indices, config.batch_size, False, config.seed
    )
    test_loader = make_loader(
        tensors, test_indices, config.batch_size, False, config.seed
    )

    set_seed(config.seed + test_fold)
    model = TemporalScoreModel(
        channels=config.channels,
        gru_hidden=config.gru_hidden,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 100
    )
    loss_function = nn.SmoothL1Loss()
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_validation_mae = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        item_count = 0
        for vectors, frame_mask, observed_mask, exercise, target in train_loader:
            vectors = vectors.to(device, non_blocking=True)
            frame_mask = frame_mask.to(device, non_blocking=True)
            observed_mask = observed_mask.to(device, non_blocking=True)
            exercise = exercise.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(vectors, frame_mask, observed_mask, exercise)
                loss = loss_function(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(target)
            item_count += len(target)
        scheduler.step()

        validation_normalized, _, validation_exercises = predict_normalized(
            model, validation_loader, device
        )
        validation_predictions = denormalize_targets(
            validation_normalized, validation_exercises, target_standardizer
        )
        validation_actual = arrays["targets"][validation_indices]
        validation_mae = float(
            np.mean(np.abs(validation_actual - validation_predictions))
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / item_count,
                "validation_mae": validation_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_mae < best_validation_mae - 1e-4:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Fold {test_fold} epoch {epoch:03d}: "
                f"loss={loss_sum / item_count:.4f}, "
                f"val_MAE={validation_mae:.3f}, "
                f"best={best_validation_mae:.3f}@{best_epoch}",
                flush=True,
            )
        if epochs_without_improvement >= config.patience:
            print(f"Fold {test_fold}: early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    fold_dir = output_dir / f"fold_{test_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model": {
                "channels": config.channels,
                "gru_hidden": config.gru_hidden,
                "dropout": config.dropout,
            },
            "feature_mean": feature_standardizer.mean,
            "feature_scale": feature_standardizer.scale,
            "target_mean": target_standardizer.mean,
            "target_scale": target_standardizer.scale,
            "test_fold": test_fold,
            "validation_fold": validation_fold,
            "best_epoch": best_epoch,
        },
        fold_dir / "checkpoint.pt",
    )
    write_csv(history, fold_dir / "history.csv")

    test_normalized, _, test_exercises = predict_normalized(
        model, test_loader, device
    )
    test_predictions = denormalize_targets(
        test_normalized, test_exercises, target_standardizer
    )
    rows: list[dict[str, object]] = []
    for index, prediction in zip(test_indices, test_predictions, strict=True):
        exercise_index = int(arrays["exercise_indices"][index])
        training_exercise = train_indices[
            arrays["exercise_indices"][train_indices] == exercise_index
        ]
        rows.append(
            {
                "fold": test_fold,
                "validation_fold": validation_fold,
                "sample_id": str(arrays["sample_ids"][index]),
                "subject_id": str(arrays["subject_ids"][index]),
                "cohort": str(arrays["cohorts"][index]),
                "exercise": EXERCISES[exercise_index],
                "actual_ts": float(arrays["targets"][index]),
                "predicted_ts": float(prediction),
                "training_exercise_mean_ts": float(
                    np.mean(arrays["targets"][training_exercise])
                ),
                "absolute_error": float(
                    abs(float(arrays["targets"][index]) - float(prediction))
                ),
                "quality_status": str(arrays["quality_statuses"][index]),
                "retained_fraction": float(arrays["retained_fractions"][index]),
            }
        )
    write_csv(rows, fold_dir / "predictions.csv")
    write_json(
        {
            "fold": test_fold,
            "validation_fold": validation_fold,
            "training_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "best_epoch": best_epoch,
            "best_validation_mae": best_validation_mae,
            "elapsed_seconds": time.perf_counter() - started,
            "test_metrics": safe_metrics(rows, "predicted_ts"),
        },
        fold_dir / "metrics.json",
    )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(document: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_metrics(
    rows: list[dict[str, object]], prediction_field: str
) -> dict[str, float | None]:
    actual = np.asarray([float(row["actual_ts"]) for row in rows])
    predicted = np.asarray([float(row[prediction_field]) for row in rows])
    metrics = regression_metrics(actual, predicted)
    return {
        name: float(value) if np.isfinite(value) else None
        for name, value in metrics.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        channels=args.channels,
        gru_hidden=args.gru_hidden,
        dropout=args.dropout,
        seed=args.seed,
        amp=not args.no_amp,
    )
    with np.load(args.data.expanduser().resolve(), allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = {
        "features",
        "frame_mask",
        "component_observed_mask",
        "targets",
        "exercise_indices",
        "fold_numbers",
        "sample_ids",
        "subject_ids",
        "cohorts",
        "quality_statuses",
        "retained_fractions",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"ML dataset is missing arrays: {sorted(missing)}")

    set_seed(config.seed)
    probe_model = TemporalScoreModel(
        channels=config.channels,
        gru_hidden=config.gru_hidden,
        dropout=config.dropout,
    )
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Samples: {len(arrays['targets'])}", flush=True)
    print(f"Trainable parameters: {trainable_parameter_count(probe_model):,}", flush=True)
    print(f"Configuration: {asdict(config)}", flush=True)

    requested_folds = range(1, 6) if args.fold == "all" else (int(args.fold),)
    all_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for fold in requested_folds:
        predictions_path = output_dir / f"fold_{fold}" / "predictions.csv"
        if args.resume and predictions_path.is_file():
            print(f"Fold {fold}: loading completed predictions", flush=True)
            all_rows.extend(read_csv(predictions_path))
            continue
        all_rows.extend(
            train_one_fold(arrays, fold, output_dir, device, config)
        )

    write_csv(all_rows, output_dir / "oof_predictions.csv")
    summary: dict[str, object] = {
        "architecture": "three-block TCN + bidirectional GRU + masked attention",
        "multi_exercise_strategy": "exercise embedding and separate regression heads",
        "evaluation": (
            "subject-disjoint outer test fold; next fold used for early stopping"
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "trainable_parameters": trainable_parameter_count(probe_model),
        "configuration": asdict(config),
        "completed_folds": list(requested_folds),
        "oof_samples": len(all_rows),
        "overall_model_metrics": safe_metrics(all_rows, "predicted_ts"),
        "overall_training_exercise_mean_metrics": safe_metrics(
            all_rows, "training_exercise_mean_ts"
        ),
        "per_exercise_model_metrics": {
            exercise: safe_metrics(
                [row for row in all_rows if row["exercise"] == exercise],
                "predicted_ts",
            )
            for exercise in EXERCISES
            if any(row["exercise"] == exercise for row in all_rows)
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(summary, output_dir / "summary.json")
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
