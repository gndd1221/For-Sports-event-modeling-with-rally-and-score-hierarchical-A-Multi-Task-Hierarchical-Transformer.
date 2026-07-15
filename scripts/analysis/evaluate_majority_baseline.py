import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.default_config import DEFAULT_GRID_OFFSET, load_sport_config


SPORT_DATA_DIRS = {
    "table_tennis": PROJECT_ROOT / "data" / "table_tennis" / "processed_data",
    "table_tennis_all": PROJECT_ROOT / "data" / "table_tennis" / "processed_data_all",
    "badminton_all": PROJECT_ROOT / "data" / "badminton" / "processed_data_badminton_all",
    "tennis": PROJECT_ROOT / "data" / "tennis" / "Tennis_processed_data",
}


def resolve_data_dir(sport: str) -> Path:
    sport_cfg = load_sport_config(sport)
    data_dir = sport_cfg.get("data_dir")
    if data_dir:
        candidate = Path(data_dir)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
    return SPORT_DATA_DIRS[sport]


def load_full_config(sport: str, data_dir: Path) -> dict:
    with open(data_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sport_cfg = load_sport_config(sport)
    config["sport"] = sport
    config["targets"] = sport_cfg["targets"]
    config["loss_weights"] = sport_cfg.get("loss_weights", {})
    config["training_args"] = sport_cfg.get("training_args", {}).copy()
    config["data_args"] = {"data_dir": str(data_dir)}
    return config


def load_targets(sport: str, data_dir: Path, split: str, targets: list[str]) -> dict[str, np.ndarray]:
    import pickle

    with open(data_dir / f"{split}_data.pkl", "rb") as f:
        matches = pickle.load(f)
    with open(data_dir / "config.json", "r", encoding="utf-8") as f:
        preprocess_config = json.load(f)

    feature_indices = {
        name: idx for idx, name in enumerate(preprocess_config["features_to_extract"])
    }
    values = {task: [] for task in targets}

    for match in matches:
        for set_data in match["sets"]:
            rally_iter = []
            if "games" in set_data:
                for game_data in set_data["games"]:
                    rally_iter.extend(game_data["rallies"])
            else:
                rally_iter.extend(set_data["rallies"])

            for rally_data in rally_iter:
                shots = rally_data["shots"]
                if len(shots) < 2:
                    continue
                for target_shot_idx in range(1, len(shots)):
                    target_shot = shots[target_shot_idx]
                    for task in targets:
                        label = int(target_shot[feature_indices[task]])
                        if label != 0:
                            values[task].append(label)

    return {task: np.asarray(task_values, dtype=np.int64) for task, task_values in values.items()}


def majority_label(y_train: np.ndarray) -> int:
    counts = Counter(int(v) for v in y_train if int(v) != 0)
    if not counts:
        raise ValueError("No non-padding labels found in the training split.")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def class_prior(y_train: np.ndarray, num_classes: int, eps: float) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for label in y_train:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1.0

    # Keep padding index effectively impossible while still avoiding log(0).
    probs = counts + eps
    probs[0] = eps
    probs = probs / probs.sum()
    return probs


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> tuple[float, float]:
    targets_one_hot = np.eye(num_classes)[y_true]

    try:
        valid_classes = [i for i in range(num_classes) if np.sum(targets_one_hot[:, i]) > 0]
        if len(valid_classes) > 1 and y_prob.shape[1] == num_classes:
            targets_one_hot_valid = targets_one_hot[:, valid_classes]
            y_prob_valid = y_prob[:, valid_classes]
            roc_w = roc_auc_score(
                targets_one_hot_valid,
                y_prob_valid,
                multi_class="ovr",
                average="weighted",
            )
        else:
            roc_w = 0.0
    except (ValueError, IndexError):
        roc_w = 0.0

    try:
        fpr_m, tpr_m, _ = roc_curve(targets_one_hot.ravel(), y_prob.ravel())
        roc_m = auc(fpr_m, tpr_m)
    except (ValueError, IndexError):
        roc_m = 0.0

    return float(roc_w), float(roc_m)


def compute_loss(
    config: dict,
    task: str,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> float:
    training_args = config.get("training_args", {})
    label_smoothing = float(training_args.get("label_smoothing", 0.1))
    safe_probs = np.clip(probs, 1e-12, 1.0)
    safe_probs = safe_probs / safe_probs.sum()
    log_probs = np.log(safe_probs)

    nll = -log_probs[y_true]
    smooth_loss = -float(np.mean(log_probs))
    ce = float(np.mean((1.0 - label_smoothing) * nll + label_smoothing * smooth_loss))

    if task != "location" or not training_args.get("use_distance_loss", False):
        return ce

    grid_offset = int(training_args.get("grid_offset", DEFAULT_GRID_OFFSET))
    grid_width = 3
    grid_size = grid_width * grid_width
    grid_end = grid_offset + grid_size
    is_grid = (y_true >= grid_offset) & (y_true < grid_end)
    if not np.any(is_grid):
        return ce

    coords = np.asarray(
        [[row, col] for row in range(grid_width) for col in range(grid_width)],
        dtype=np.float64,
    )
    grid_probs = safe_probs[grid_offset:grid_end]
    grid_probs = grid_probs / grid_probs.sum()
    target_idx = y_true[is_grid] - grid_offset
    distances = np.linalg.norm(coords[target_idx][:, None, :] - coords[None, :, :], axis=2)
    expected_distance = float(np.mean(np.sum(grid_probs[None, :] * distances, axis=1)))
    lambda_dist = float(training_args.get("lambda_dist", 1.0))
    return ce + lambda_dist * expected_distance


def compute_avg_distance(config: dict, task: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    training_args = config.get("training_args", {})
    if task != "location" or not training_args.get("use_distance_loss", False):
        return ""

    grid_width = 3
    grid_offset = int(training_args.get("grid_offset", DEFAULT_GRID_OFFSET))
    grid_size = grid_width * grid_width
    grid_end = grid_offset + grid_size
    mask_grid = (
        (y_true >= grid_offset)
        & (y_true < grid_end)
        & (y_pred >= grid_offset)
        & (y_pred < grid_end)
    )
    if not np.any(mask_grid):
        return ""

    coords = np.asarray(
        [[row, col] for row in range(grid_width) for col in range(grid_width)],
        dtype=np.float64,
    )
    target_coords = coords[y_true[mask_grid] - grid_offset]
    pred_coords = coords[y_pred[mask_grid] - grid_offset]
    distances = np.linalg.norm(target_coords - pred_coords, axis=1)
    return f"{float(np.mean(distances)):.4f}"


def evaluate_sport(sport: str, output_dir: Path, eps: float) -> list[dict[str, str]]:
    data_dir = resolve_data_dir(sport)
    config = load_full_config(sport, data_dir)
    targets = config["targets"]

    train_targets = load_targets(sport, data_dir, "train", targets)
    test_targets = load_targets(sport, data_dir, "test", targets)

    rows = []
    for task in targets:
        y_train = train_targets[task]
        y_true = test_targets[task]
        num_classes = int(config.get(f"num_{task}"))
        majority = majority_label(y_train)
        probs = class_prior(y_train, num_classes, eps)

        y_pred = np.full_like(y_true, fill_value=majority)
        y_prob = np.tile(probs, (len(y_true), 1))

        roc_w, roc_m = compute_auc(y_true, y_prob, num_classes)
        loss = compute_loss(config, task, y_true, probs)
        avg_distance = compute_avg_distance(config, task, y_true, y_pred)

        train_count = int(np.sum(y_train == majority))
        train_total = int(len(y_train))
        test_count = int(np.sum(y_true == majority))
        test_total = int(len(y_true))

        rows.append(
            {
                "model_type": "majority_class_train_prior",
                "sport": sport,
                "task": task,
                "num_classes": str(num_classes),
                "majority_label_processed": str(majority),
                "majority_label_semantic": str(majority - 1),
                "train_majority_count": str(train_count),
                "train_samples": str(train_total),
                "train_majority_ratio": f"{train_count / train_total:.4f}",
                "test_majority_count": str(test_count),
                "test_samples": str(test_total),
                "test_majority_ratio": f"{test_count / test_total:.4f}",
                "accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
                "precision": f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}",
                "recall": f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}",
                "f1": f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}",
                "roc_auc_weighted": f"{roc_w:.4f}",
                "roc_auc_micro": f"{roc_m:.4f}",
                "loss": f"{loss:.4f}",
                "avg_distance": avg_distance,
            }
        )

    sport_csv = output_dir / f"{sport}_majority_baseline.csv"
    write_csv(sport_csv, rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "sport",
        "task",
        "majority_label_semantic",
        "train_majority_ratio",
        "test_majority_ratio",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc_weighted",
        "roc_auc_micro",
        "loss",
        "avg_distance",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Majority-Class Baseline Results\n\n")
        f.write(
            "Prediction uses the most frequent non-padding training label for each task. "
            "Probability scores use the full training-label prior distribution, so loss and AUC remain finite and comparable across tasks.\n\n"
        )
        f.write(
            "`majority_label_semantic` is `majority_label_processed - 1`, because processed labels reserve index 0 for padding / ignored labels.\n\n"
        )
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(col, "") for col in columns) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a training-set majority-class baseline on processed racket-sport datasets."
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        default=["table_tennis", "table_tennis_all", "badminton_all", "tennis"],
        choices=sorted(SPORT_DATA_DIRS.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "results" / "majority_baseline"),
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_rows = []
    for sport in args.sports:
        rows = evaluate_sport(sport, output_dir, args.eps)
        all_rows.extend(rows)

    write_csv(output_dir / "majority_baseline_all.csv", all_rows)
    write_markdown(output_dir / "majority_baseline_summary.md", all_rows)

    print(f"Wrote {len(all_rows)} task rows to {output_dir}")
    print(output_dir / "majority_baseline_all.csv")
    print(output_dir / "majority_baseline_summary.md")


if __name__ == "__main__":
    main()
