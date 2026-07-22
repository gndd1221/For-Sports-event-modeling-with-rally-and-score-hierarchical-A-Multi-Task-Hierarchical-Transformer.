import argparse
import copy
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_TMP = _PROJECT_ROOT / ".tmp" / "matplotlib"
_MPL_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_TMP))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataloader import get_dataloader
from src.default_config import get_model_class


TASKS = ("type", "backhand", "location", "strength", "spin")
STAGE_ORDER = (
    "opening_l1",
    "middle_l1_l2",
    "set_boundary_l1_l3",
    "late_l1_l2_l3",
)
STAGE_LABELS = {
    "opening_l1": "Opening: L1 only",
    "middle_l1_l2": "Middle: L1+L2",
    "set_boundary_l1_l3": "Set boundary: L1+L3",
    "late_l1_l2_l3": "Late: L1+L2+L3",
}

GROUP_ORDER = ("L1 only (TA)", "L1+L2 (TA)", "MT-HTA Core", "Final MT-HTA")
EXPECTED_SEEDS = (42, 123, 2024)


def classify_run(config):
    training_args = config.get("training_args", {})
    model_args = config.get("model_args", {})
    model_type = training_args.get("model_type")
    if model_type == "task_attention_L1":
        return "L1 only (TA)"
    if model_type == "task_attention_L1_L2":
        return "L1+L2 (TA)"
    if model_type != "task_attention_itransformer_feature_token":
        return None
    if training_args.get("use_rlw", False):
        return None

    skip_window_size = int(model_args.get("skip_window_size", 0) or 0)
    advanced_flags = (
        bool(model_args.get("use_gated_fusion", False)),
        bool(model_args.get("use_shot_aware_pe", False)),
        bool(model_args.get("use_top_down_attention", False)),
        bool(model_args.get("use_turn_based_gating", False)),
    )
    optional_flags_off = not model_args.get("use_temporal_scale_gating", False) and not model_args.get(
        "use_task_decoder", False
    )
    if skip_window_size == 0 and advanced_flags == (False, False, False, False) and optional_flags_off:
        return "MT-HTA Core"
    if skip_window_size == 1 and advanced_flags == (True, True, True, True) and optional_flags_off:
        return "Final MT-HTA"
    return None


def discover_run_groups(results_root, checkpoint):
    results_root = Path(results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    candidates = defaultdict(lambda: defaultdict(list))
    for run_dir in results_root.iterdir():
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "train" / "weights" / f"{checkpoint}.pth"
        if not run_dir.is_dir() or not config_path.is_file() or not checkpoint_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            group = classify_run(config)
            seed = int(config.get("training_args", {}).get("seed"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if group in GROUP_ORDER and seed in EXPECTED_SEEDS:
            candidates[group][seed].append(run_dir)

    missing = [
        f"{group}, seed={seed}"
        for group in GROUP_ORDER
        for seed in EXPECTED_SEEDS
        if not candidates[group][seed]
    ]
    if missing:
        raise RuntimeError(
            "Could not discover all required completed runs under "
            f"{results_root}. Missing: {', '.join(missing)}"
        )

    return {
        group: {
            seed: max(candidates[group][seed], key=lambda path: path.name)
            for seed in EXPECTED_SEEDS
        }
        for group in GROUP_ORDER
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze hierarchy-context availability and a fixed-rule table-tennis case study."
    )
    parser.add_argument("--sport", default="table_tennis", choices=["table_tennis"])
    parser.add_argument("--prefix_len", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", default="best_model", choices=["best_model", "last_model"])
    parser.add_argument("--device", default=None, help="Examples: cuda, cuda:0, cpu")
    parser.add_argument("--data_dir", default="data/table_tennis/processed_data")
    parser.add_argument("--results_root", default="outputs/results/table_tennis")
    parser.add_argument(
        "--output_dir",
        default="outputs/results/hierarchy_progress_case_study",
    )
    return parser.parse_args()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_for_sample(sample_info):
    has_l2 = sample_info["rally_idx"] > 0
    has_l3 = sample_info["set_idx"] > 0
    if not has_l2 and not has_l3:
        return "opening_l1"
    if has_l2 and not has_l3:
        return "middle_l1_l2"
    if not has_l2 and has_l3:
        return "set_boundary_l1_l3"
    return "late_l1_l2_l3"


def select_fixed_prefix_indices(dataset, prefix_len):
    return [
        idx
        for idx, info in enumerate(dataset.samples)
        if info["target_shot_idx"] == prefix_len
    ]


def select_case(dataset):
    candidates = []
    for match_idx, match in enumerate(dataset.all_matches):
        sets = match["sets"]
        if len(sets) < 2 or not sets[0]["rallies"]:
            continue
        if len(sets[0]["rallies"][0]["shots"]) < 4:
            continue

        middle = next(
            (
                (0, rally_idx)
                for rally_idx, rally in enumerate(sets[0]["rallies"])
                if rally_idx >= 3 and len(rally["shots"]) >= 4
            ),
            None,
        )
        late = None
        for set_idx in range(1, len(sets)):
            late = next(
                (
                    (set_idx, rally_idx)
                    for rally_idx, rally in enumerate(sets[set_idx]["rallies"])
                    if rally_idx >= 1 and len(rally["shots"]) >= 4
                ),
                None,
            )
            if late is not None:
                break
        if middle is None or late is None:
            continue

        total_rallies = sum(len(set_data["rallies"]) for set_data in sets)
        candidates.append(
            {
                "match_idx": match_idx,
                "match_id": match.get("match_id", match_idx),
                "total_rallies": total_rallies,
                "total_sets": len(sets),
                "opening": (0, 0),
                "middle": middle,
                "late": late,
            }
        )

    if not candidates:
        raise RuntimeError("No test match satisfies the fixed case-selection rules.")

    median_rallies = statistics.median(c["total_rallies"] for c in candidates)

    def tie_key(candidate):
        match_id = candidate["match_id"]
        match_sort = (0, int(match_id)) if str(match_id).isdigit() else (1, str(match_id))
        return (abs(candidate["total_rallies"] - median_rallies), match_sort, candidate["match_idx"])

    selected = min(candidates, key=tie_key)
    selected["candidate_count"] = len(candidates)
    selected["candidate_median_rallies"] = median_rallies

    sample_lookup = {
        (
            info["match_idx"],
            info["set_idx"],
            info["rally_idx"],
            info["target_shot_idx"],
        ): idx
        for idx, info in enumerate(dataset.samples)
    }
    case_indices = {}
    for checkpoint in ("opening", "middle", "late"):
        set_idx, rally_idx = selected[checkpoint]
        key = (selected["match_idx"], set_idx, rally_idx, 3)
        if key not in sample_lookup:
            raise RuntimeError(f"Missing fixed-prefix case sample: {key}")
        case_indices[checkpoint] = sample_lookup[key]
    return selected, case_indices


def load_model(run_dir, checkpoint, device):
    config_path = Path(run_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    model_type = config["training_args"]["model_type"]
    model_class, overrides = get_model_class(model_type)
    saved_model_args = copy.deepcopy(config.get("model_args", {}))
    saved_hierarchy = saved_model_args.get("hierarchy_levels") or config.get("hierarchy_levels")
    saved_fusion = saved_model_args.get("fusion_type")
    config.setdefault("model_args", {}).update(overrides)
    if saved_hierarchy:
        config["model_args"]["hierarchy_levels"] = saved_hierarchy
    if saved_fusion:
        config["model_args"]["fusion_type"] = saved_fusion

    model = model_class(config).to(device)
    checkpoint_path = Path(run_dir) / "train" / "weights" / f"{checkpoint}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, config, checkpoint_path


def extract_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def sample_metadata(dataset, sample_idx):
    info = dataset.samples[sample_idx]
    match = dataset.all_matches[info["match_idx"]]
    return {
        "sample_idx": sample_idx,
        "match_idx": info["match_idx"],
        "match_id": match.get("match_id", info["match_idx"]),
        "set_idx": info["set_idx"],
        "rally_idx": info["rally_idx"],
        "target_shot_idx": info["target_shot_idx"],
        "stage": stage_for_sample(info),
    }


def evaluate_run(model, dataset, collate_fn, sample_indices, case_index_set, batch_size):
    predictions = {stage: {task: [] for task in TASKS} for stage in STAGE_ORDER}
    targets = {stage: {task: [] for task in TASKS} for stage in STAGE_ORDER}
    stage_match_ids = {stage: set() for stage in STAGE_ORDER}
    case_records = {}

    with torch.no_grad():
        for start in range(0, len(sample_indices), batch_size):
            chunk = sample_indices[start : start + batch_size]
            batch = collate_fn([dataset[idx] for idx in chunk])
            logits = extract_logits(model(batch))
            probs = {task: torch.softmax(logits[task], dim=1).cpu().numpy() for task in TASKS}
            preds = {task: np.argmax(probs[task], axis=1) for task in TASKS}
            batch_targets = {task: batch["targets"][task].cpu().numpy() for task in TASKS}

            for row_idx, sample_idx in enumerate(chunk):
                meta = sample_metadata(dataset, sample_idx)
                stage = meta["stage"]
                stage_match_ids[stage].add(meta["match_id"])
                for task in TASKS:
                    predictions[stage][task].append(int(preds[task][row_idx]))
                    targets[stage][task].append(int(batch_targets[task][row_idx]))
                if sample_idx in case_index_set:
                    case_records[sample_idx] = {
                        "metadata": meta,
                        "targets": {task: int(batch_targets[task][row_idx]) for task in TASKS},
                        "probabilities": {task: probs[task][row_idx].copy() for task in TASKS},
                    }
    return predictions, targets, stage_match_ids, case_records


def location_distance(predictions, targets, grid_offset=2, grid_width=3):
    total = 0.0
    count = 0
    grid_end = grid_offset + grid_width * grid_width
    for pred, target in zip(predictions, targets):
        if grid_offset <= pred < grid_end and grid_offset <= target < grid_end:
            pred_idx = pred - grid_offset
            target_idx = target - grid_offset
            pred_coord = np.array(divmod(pred_idx, grid_width), dtype=float)
            target_coord = np.array(divmod(target_idx, grid_width), dtype=float)
            total += float(np.linalg.norm(pred_coord - target_coord))
            count += 1
    return (total / count if count else float("nan")), count


def compute_stage_rows(model_name, seed, predictions, targets, stage_match_ids):
    rows = []
    for stage in STAGE_ORDER:
        task_f1 = {}
        task_accuracy = {}
        for task in TASKS:
            y_true = targets[stage][task]
            y_pred = predictions[stage][task]
            task_f1[task] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            task_accuracy[task] = accuracy_score(y_true, y_pred)
        distance, distance_count = location_distance(
            predictions[stage]["location"], targets[stage]["location"]
        )
        row = {
            "model": model_name,
            "seed": seed,
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            "num_samples": len(targets[stage][TASKS[0]]),
            "num_matches": len(stage_match_ids[stage]),
            "macro_accuracy": np.mean(list(task_accuracy.values())),
            "macro_f1": np.mean(list(task_f1.values())),
            "location_avg_distance": distance,
            "location_distance_count": distance_count,
        }
        for task in TASKS:
            row[f"{task}_accuracy"] = task_accuracy[task]
            row[f"{task}_f1"] = task_f1[task]
        rows.append(row)
    return rows


def summarize_stage_rows(seed_rows):
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["model"], row["stage"])].append(row)

    id_columns = {
        "model",
        "seed",
        "stage",
        "stage_label",
        "num_samples",
        "num_matches",
        "location_distance_count",
    }
    metric_columns = [key for key in seed_rows[0] if key not in id_columns]
    summaries = []
    for model_name in GROUP_ORDER:
        for stage in STAGE_ORDER:
            rows = groups[(model_name, stage)]
            if len(rows) != 3:
                raise RuntimeError(f"Expected three seeds for {model_name}, {stage}; found {len(rows)}")
            summary = {
                "model": model_name,
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                "num_samples": rows[0]["num_samples"],
                "num_matches": rows[0]["num_matches"],
                "location_distance_count": rows[0]["location_distance_count"],
            }
            for metric in metric_columns:
                values = [float(row[metric]) for row in rows]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
            summaries.append(summary)
    return summaries


def aggregate_case_predictions(case_indices, all_case_records):
    rows = []
    for model_name in GROUP_ORDER:
        for checkpoint, sample_idx in case_indices.items():
            records = [
                all_case_records[(model_name, seed)][sample_idx]
                for seed in EXPECTED_SEEDS
            ]
            meta = records[0]["metadata"]
            for task in TASKS:
                probabilities = np.stack([record["probabilities"][task] for record in records])
                mean_probabilities = probabilities.mean(axis=0)
                prediction = int(np.argmax(mean_probabilities))
                target = int(records[0]["targets"][task])
                top_k = min(3, len(mean_probabilities))
                top3 = np.argsort(mean_probabilities)[-top_k:][::-1].tolist()
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "model": model_name,
                        "match_id": meta["match_id"],
                        "set_number": meta["set_idx"] + 1,
                        "rally_number": meta["rally_idx"] + 1,
                        "observed_prefix_length": meta["target_shot_idx"],
                        "completed_rallies_l2": meta["rally_idx"],
                        "completed_sets_l3": meta["set_idx"],
                        "task": task,
                        "target_processed": target,
                        "target_semantic": target - 1,
                        "prediction_processed": prediction,
                        "prediction_semantic": prediction - 1,
                        "top1_correct": int(prediction == target),
                        "confidence": float(mean_probabilities[prediction]),
                        "true_class_probability": float(mean_probabilities[target]),
                        "top3_processed": ";".join(map(str, top3)),
                        "top3_semantic": ";".join(str(value - 1) for value in top3),
                        "top3_correct": int(target in top3),
                    }
                )
    return rows


def make_context_variant(item, condition):
    variant = copy.deepcopy(item)
    if condition in {"remove_l2", "remove_l2_l3"}:
        variant["rally_history"] = []
    if condition in {"remove_l3", "remove_l2_l3"}:
        variant["set_history"] = []
    return variant


def evaluate_single_item(model, item, collate_fn):
    with torch.no_grad():
        output = model(collate_fn([item]))
    logits = extract_logits(output)
    return {
        task: torch.softmax(logits[task], dim=1)[0].detach().cpu().numpy()
        for task in TASKS
    }, output[1] if isinstance(output, tuple) else None


def analyze_context_removal(dataset, collate_fn, case_indices, final_run_dirs, checkpoint, device):
    conditions = {
        "opening": ("natural",),
        "middle": ("natural", "remove_l2"),
        "late": ("natural", "remove_l3", "remove_l2_l3"),
    }
    probability_store = defaultdict(list)
    target_store = {}

    for seed, run_dir in final_run_dirs.items():
        model, _, _ = load_model(run_dir, checkpoint, device)
        for checkpoint_name, sample_idx in case_indices.items():
            base_item = dataset[sample_idx]
            target_store[checkpoint_name] = {task: int(base_item["targets"][task]) for task in TASKS}
            for condition in conditions[checkpoint_name]:
                item = make_context_variant(base_item, condition)
                probabilities, _ = evaluate_single_item(model, item, collate_fn)
                for task in TASKS:
                    probability_store[(checkpoint_name, condition, task)].append(probabilities[task])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for checkpoint_name, checkpoint_conditions in conditions.items():
        natural_means = {
            task: np.stack(probability_store[(checkpoint_name, "natural", task)]).mean(axis=0)
            for task in TASKS
        }
        for condition in checkpoint_conditions:
            for task in TASKS:
                mean_probabilities = np.stack(
                    probability_store[(checkpoint_name, condition, task)]
                ).mean(axis=0)
                natural = natural_means[task]
                target = target_store[checkpoint_name][task]
                prediction = int(np.argmax(mean_probabilities))
                natural_prediction = int(np.argmax(natural))
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "condition": condition,
                        "task": task,
                        "target_processed": target,
                        "target_semantic": target - 1,
                        "prediction_processed": prediction,
                        "prediction_semantic": prediction - 1,
                        "confidence": float(mean_probabilities[prediction]),
                        "true_class_probability": float(mean_probabilities[target]),
                        "delta_true_class_probability_vs_natural": float(
                            mean_probabilities[target] - natural[target]
                        ),
                        "label_flip_vs_natural": int(prediction != natural_prediction),
                    }
                )
    return rows


def collect_attention(dataset, collate_fn, case_indices, final_run_dirs, checkpoint, device):
    store = defaultdict(list)
    token_labels = ["L1", "L2", "L3", "Hitter", "Receiver"]
    availability = {
        "opening": ["L1", "L2 (empty)", "L3 (empty)", "Hitter", "Receiver"],
        "middle": ["L1", "L2", "L3 (empty)", "Hitter", "Receiver"],
        "late": token_labels,
    }

    for seed, run_dir in final_run_dirs.items():
        model, config, _ = load_model(run_dir, checkpoint, device)
        if config["model_args"].get("use_sequence_fusion", False):
            raise RuntimeError("This heatmap assumes summary-token Task Attention, not sequence fusion.")
        for checkpoint_name, sample_idx in case_indices.items():
            _, debug = evaluate_single_item(model, dataset[sample_idx], collate_fn)
            if not debug or not debug.get("task_attn_weights"):
                raise RuntimeError("Final MT-HTA did not return task attention weights in eval mode.")
            for task in TASKS:
                weights = debug["task_attn_weights"][task]
                mean_heads = weights.mean(dim=1).squeeze(0).squeeze(0).cpu().numpy()
                if len(mean_heads) != len(token_labels):
                    raise RuntimeError(
                        f"Unexpected Task Attention token count: {len(mean_heads)}; expected 5."
                    )
                store[(checkpoint_name, task)].append(mean_heads)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    matrices = {}
    for checkpoint_name in ("opening", "middle", "late"):
        matrix = []
        for task in TASKS:
            seed_weights = np.stack(store[(checkpoint_name, task)])
            mean_weights = seed_weights.mean(axis=0)
            matrix.append(mean_weights)
            for token_idx, token in enumerate(token_labels):
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "task": task,
                        "token": token,
                        "display_token": availability[checkpoint_name][token_idx],
                        "attention_mean": float(mean_weights[token_idx]),
                        "attention_std_across_seeds": float(
                            seed_weights[:, token_idx].std(ddof=1)
                        ),
                    }
                )
        matrices[checkpoint_name] = np.stack(matrix)
    return rows, matrices, availability


def plot_attention_heatmap(output_path, matrices, display_labels):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    vmax = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    titles = {"opening": "Opening", "middle": "Middle", "late": "Late"}
    for axis, checkpoint in zip(axes, ("opening", "middle", "late")):
        matrix = matrices[checkpoint]
        axis.pcolormesh(
            np.arange(matrix.shape[1] + 1),
            np.arange(matrix.shape[0] + 1),
            matrix,
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            shading="flat",
        )
        axis.set_title(titles[checkpoint])
        axis.set_xticks(
            np.arange(matrix.shape[1]) + 0.5,
            display_labels[checkpoint],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(
            np.arange(matrix.shape[0]) + 0.5,
            [task.title() for task in TASKS],
        )
        axis.invert_yaxis()
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                color = "white" if value > vmax * 0.62 else "black"
                axis.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.23, top=0.82, wspace=0.28)
    colorbar_axis = fig.add_axes([0.92, 0.25, 0.014, 0.52])
    color_steps = 64
    for step in range(color_steps):
        y0 = vmax * step / color_steps
        colorbar_axis.add_patch(
            Rectangle(
                (0.0, y0),
                1.0,
                vmax / color_steps,
                facecolor=plt.cm.Blues((step + 0.5) / color_steps),
                edgecolor="none",
            )
        )
    colorbar_axis.set_xlim(0.0, 1.0)
    colorbar_axis.set_ylim(0.0, vmax)
    colorbar_axis.set_xticks([])
    colorbar_axis.set_ylabel("Mean attention weight")
    fig.suptitle("Task Attention across hierarchy-context stages (three-seed mean)", fontsize=11)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_case_selection(output_dir, selected, case_indices, dataset):
    rows = []
    for checkpoint in ("opening", "middle", "late"):
        sample_idx = case_indices[checkpoint]
        meta = sample_metadata(dataset, sample_idx)
        rows.append(
            {
                "checkpoint": checkpoint,
                "selection_rule": {
                    "opening": "Set 1 Rally 1; rally length >= 4",
                    "middle": "Set 1 earliest rally after >=3 completed rallies; rally length >= 4",
                    "late": "Earliest later-set rally after >=1 completed rally; rally length >= 4",
                }[checkpoint],
                "candidate_count": selected["candidate_count"],
                "candidate_median_rallies": selected["candidate_median_rallies"],
                "match_id": selected["match_id"],
                "total_match_rallies": selected["total_rallies"],
                "total_match_sets": selected["total_sets"],
                "sample_idx": sample_idx,
                "set_number": meta["set_idx"] + 1,
                "rally_number": meta["rally_idx"] + 1,
                "observed_prefix_length": meta["target_shot_idx"],
                "completed_rallies_l2": meta["rally_idx"],
                "completed_sets_l3": meta["set_idx"],
            }
        )
    write_csv(Path(output_dir) / "hierarchy_case_selection.csv", rows)


def write_summary(output_dir, stage_summary, selected, case_rows, removal_rows):
    final_rows = {
        row["stage"]: row
        for row in stage_summary
        if row["model"] == "Final MT-HTA"
    }
    lines = [
        "# Hierarchy Progress Case Study Summary",
        "",
        "## Fixed-prefix context availability",
        "",
        "All samples observe three shots and predict the fourth shot.",
        "",
        "| Stage | Samples | Matches | Macro F1 | Location F1 | Location distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGE_ORDER:
        row = final_rows[stage]
        lines.append(
            f"| {STAGE_LABELS[stage]} | {row['num_samples']} | {row['num_matches']} | "
            f"{row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f} | "
            f"{row['location_f1_mean']:.4f} +/- {row['location_f1_std']:.4f} | "
            f"{row['location_avg_distance_mean']:.4f} +/- {row['location_avg_distance_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These groups differ in match stage and target distribution. The table is descriptive and does not identify a causal context effect.",
            "",
            "## Fixed-rule case",
            "",
            f"- Eligible matches: {selected['candidate_count']}",
            f"- Median total rallies: {selected['candidate_median_rallies']}",
            f"- Selected match_id: {selected['match_id']}",
            f"- Selected match: {selected['total_rallies']} rallies, {selected['total_sets']} sets",
            f"- Prediction rows: {len(case_rows)}",
            f"- Context-removal rows: {len(removal_rows)}",
            "",
            "Attention weights are descriptive model diagnostics and are not interpreted as causal feature importance.",
        ]
    )
    (Path(output_dir) / "hierarchy_case_study_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.prefix_len != 3:
        raise ValueError("The preregistered analysis requires --prefix_len 3.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_dir) / "test_data.pkl"
    config_path = Path(args.data_dir) / "config.json"
    dataset, collate_fn = get_dataloader(args.sport, str(data_path), str(config_path))
    sample_indices = select_fixed_prefix_indices(dataset, args.prefix_len)
    selected, case_indices = select_case(dataset)
    case_index_set = set(case_indices.values())

    expected_counts = {
        "opening_l1": 20,
        "middle_l1_l2": 279,
        "set_boundary_l1_l3": 64,
        "late_l1_l2_l3": 989,
    }
    actual_counts = defaultdict(int)
    for sample_idx in sample_indices:
        actual_counts[stage_for_sample(dataset.samples[sample_idx])] += 1
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(f"Unexpected fixed-prefix stage counts: {dict(actual_counts)}")
    if selected["match_id"] != 25:
        raise RuntimeError(f"Fixed case rule selected match_id={selected['match_id']}, expected 25.")

    resolved_runs = discover_run_groups(args.results_root, args.checkpoint)

    stage_seed_rows = []
    all_case_records = {}
    print(f"Device: {device}")
    print(f"Fixed-prefix samples: {len(sample_indices)}")
    print(f"Selected case match_id: {selected['match_id']}")

    for model_name, seed_runs in resolved_runs.items():
        for seed, run_dir in seed_runs.items():
            print(f"Evaluating {model_name}, seed {seed}: {run_dir.name}")
            model, _, checkpoint_path = load_model(run_dir, args.checkpoint, device)
            predictions, targets, match_ids, case_records = evaluate_run(
                model,
                dataset,
                collate_fn,
                sample_indices,
                case_index_set,
                args.batch_size,
            )
            stage_seed_rows.extend(
                compute_stage_rows(model_name, seed, predictions, targets, match_ids)
            )
            all_case_records[(model_name, seed)] = case_records
            print(f"Loaded checkpoint: {checkpoint_path}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    stage_summary = summarize_stage_rows(stage_seed_rows)
    case_rows = aggregate_case_predictions(case_indices, all_case_records)
    final_runs = resolved_runs["Final MT-HTA"]
    removal_rows = analyze_context_removal(
        dataset, collate_fn, case_indices, final_runs, args.checkpoint, device
    )
    attention_rows, attention_matrices, attention_labels = collect_attention(
        dataset, collate_fn, case_indices, final_runs, args.checkpoint, device
    )

    write_csv(output_dir / "hierarchy_context_stage_metrics_by_seed.csv", stage_seed_rows)
    write_csv(output_dir / "hierarchy_context_stage_summary.csv", stage_summary)
    write_case_selection(output_dir, selected, case_indices, dataset)
    write_csv(output_dir / "hierarchy_case_predictions.csv", case_rows)
    write_csv(output_dir / "hierarchy_case_context_removal.csv", removal_rows)
    write_csv(output_dir / "hierarchy_case_attention_weights.csv", attention_rows)
    plot_attention_heatmap(
        output_dir / "hierarchy_case_attention_heatmap.pdf",
        attention_matrices,
        attention_labels,
    )
    write_summary(output_dir, stage_summary, selected, case_rows, removal_rows)
    print(f"Analysis complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
