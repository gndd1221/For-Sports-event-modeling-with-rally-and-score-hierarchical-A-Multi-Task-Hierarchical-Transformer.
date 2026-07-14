import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_matplotlib_tmp = os.path.join(_project_root, ".tmp", "matplotlib")
os.makedirs(_matplotlib_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _matplotlib_tmp)
os.environ.setdefault("TEMP", _matplotlib_tmp)
os.environ.setdefault("TMP", _matplotlib_tmp)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from tqdm import tqdm

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 3,
})

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Autoregressive rally rollout analysis for table tennis."
    )
    parser.add_argument("--run_dir", default=None, help="Training run directory.")
    parser.add_argument(
        "--aggregate_csvs",
        nargs="+",
        default=None,
        help="Aggregate completed rollout_type_accuracy_surface.csv files across seeds.",
    )
    parser.add_argument(
        "--plot_surface_csv",
        default=None,
        help="Re-render rollout_type_top1_surface and rollout_type_top3_surface from an existing rollout_type_accuracy_surface.csv.",
    )
    parser.add_argument(
        "--sport",
        default="table_tennis",
        choices=["table_tennis"],
        help="Only original table_tennis is supported for this analysis.",
    )
    parser.add_argument(
        "--checkpoint",
        default="last_model",
        choices=["last_model", "best_model"],
        help="Checkpoint filename stem to load.",
    )
    parser.add_argument("--max_horizon", type=int, default=6)
    parser.add_argument("--max_future_k", type=int, default=5)
    parser.add_argument("--max_prefix_len", type=int, default=5)
    parser.add_argument("--min_cell_count", type=int, default=20)
    parser.add_argument(
        "--figure_formats",
        nargs="+",
        default=["png", "pdf", "eps"],
        choices=["png", "pdf", "eps", "svg"],
        help="Figure formats to export. PDF/EPS/SVG are generated directly as vector graphics.",
    )
    parser.add_argument(
        "--heatmap_scale",
        default="auto",
        choices=["auto", "full"],
        help="Use auto for a narrower data-driven color range, or full for fixed 0-1 accuracy range.",
    )
    parser.add_argument(
        "--heatmap_padding",
        type=float,
        default=0.03,
        help="Padding added around the observed accuracy range when --heatmap_scale=auto.",
    )
    parser.add_argument(
        "--skip_formal_figure",
        action="store_true",
        help="Skip the publication-style two-panel rollout figure.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()
    if not args.aggregate_csvs and not args.plot_surface_csv and not args.run_dir:
        parser.error("--run_dir is required unless --aggregate_csvs or --plot_surface_csv is provided.")
    return args


def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(config, run_dir, checkpoint, device):
    import torch
    from src.default_config import get_model_class

    model_type = config.get("training_args", {}).get("model_type", "task_attention")
    ModelClass, config_overrides = get_model_class(model_type)

    saved_model_args = config.get("model_args", {})
    saved_hierarchy = saved_model_args.get("hierarchy_levels") or config.get("hierarchy_levels")
    saved_fusion_type = saved_model_args.get("fusion_type")

    config.setdefault("model_args", {}).update(config_overrides)
    if saved_hierarchy:
        config["model_args"]["hierarchy_levels"] = saved_hierarchy
    if saved_fusion_type:
        config["model_args"]["fusion_type"] = saved_fusion_type

    model = ModelClass(config).to(device)
    model_path = os.path.join(run_dir, "train", "weights", f"{checkpoint}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, model_type, model_path


def resolve_data_dir(args, config):
    from src.default_config import load_sport_config

    sport_cfg = load_sport_config(args.sport)
    return (
        args.data_dir
        or config.get("data_args", {}).get("data_dir")
        or sport_cfg.get("data_dir", "data/table_tennis/processed_data")
    )


def load_test_matches(data_dir):
    test_path = os.path.join(data_dir, "test_data.pkl")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test data: {test_path}")
    with open(test_path, "rb") as f:
        return pickle.load(f)


def make_rollout_item(prefix_shots, target_shot, rally_history, set_history, feature_indices, targets):
    return {
        "shot_seq_current": np.array(prefix_shots, dtype=np.int64),
        "rally_history": rally_history,
        "set_history": set_history,
        "player_id": int(target_shot[feature_indices["player_id"]]),
        "opponent_id": int(target_shot[feature_indices["opponent_id"]]),
        "targets": {
            task: int(target_shot[feature_indices[task]])
            for task in targets
        },
    }


def logits_dict(output):
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict):
                return item
    return output


def task_num_classes(config, task):
    return config.get("model_args", {}).get(f"num_{task}", config.get(f"num_{task}", 2))


def empty_surface_bucket():
    return {
        "type_top1_correct": 0,
        "type_top3_correct": 0,
        "total": 0,
    }


def update_surface_bucket(bucket, target, pred, top3_hit):
    if target == 0:
        return
    bucket["type_top1_correct"] += 1 if pred == target else 0
    bucket["type_top3_correct"] += 1 if top3_hit else 0
    bucket["total"] += 1


def write_csv(path, rows, preferred_fields):
    if not rows:
        return
    fieldnames = list(preferred_fields)
    extra = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extra)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return float(np.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1)))


def cleanup_output_dir(output_dir):
    legacy_outputs = [
        "rollout_metrics_by_horizon.csv",
        "rollout_accuracy_by_rally_length.csv",
        "rollout_macro_f1_by_horizon.png",
        "rollout_top1_by_rally_length.png",
        "rollout_top3_by_rally_length.png",
    ]
    for filename in legacy_outputs:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def surface_matrix(rows, metric, max_future_k, max_prefix_len, min_cell_count, with_samples=False):
    matrix = np.full((max_future_k, max_prefix_len), np.nan, dtype=float)
    samples = np.zeros((max_future_k, max_prefix_len), dtype=int)

    for row in rows:
        future_k = int(row["future_k"])
        observed_n = int(row["observed_n"])
        sample_count = int(row["num_samples"])
        if sample_count < min_cell_count:
            continue
        matrix[future_k - 1, observed_n - 1] = float(row[metric])
        samples[future_k - 1, observed_n - 1] = sample_count

    if with_samples:
        return matrix, samples
    return matrix


def color_limits(matrix, scale="auto", padding=0.03, min_span=0.18):
    valid = matrix[np.isfinite(matrix)]
    if scale == "full" or valid.size == 0:
        return 0.0, 1.0

    low = float(np.min(valid))
    high = float(np.max(valid))
    center = (low + high) / 2.0
    span = max(high - low + (2.0 * padding), min_span)
    vmin = max(0.0, center - span / 2.0)
    vmax = min(1.0, center + span / 2.0)

    if vmax - vmin < min_span:
        if vmin <= 0.0:
            vmax = min(1.0, vmin + min_span)
        elif vmax >= 1.0:
            vmin = max(0.0, vmax - min_span)

    return vmin, vmax


def save_figure(fig, output_dir, stem, figure_formats):
    requested_formats = list(dict.fromkeys(figure_formats or ["png"]))
    for fmt in requested_formats:
        output_path = os.path.join(output_dir, f"{stem}.{fmt}")
        if fmt == "png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")


def plot_type_surface(
    rows,
    output_dir,
    metric,
    title,
    filename,
    max_future_k,
    max_prefix_len,
    min_cell_count,
    figure_formats,
    heatmap_scale,
    heatmap_padding,
):
    matrix, samples = surface_matrix(
        rows,
        metric,
        max_future_k,
        max_prefix_len,
        min_cell_count,
        with_samples=True,
    )
    annotations = [["" for _ in range(max_prefix_len)] for _ in range(max_future_k)]

    for y in range(max_future_k):
        for x in range(max_prefix_len):
            if np.isfinite(matrix[y, x]):
                annotations[y][x] = f"{matrix[y, x]:.2f}\nn={samples[y, x]}"

    cmap = plt.colormaps.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#f2f2f2")
    vmin, vmax = color_limits(matrix, heatmap_scale, heatmap_padding)

    fig, ax = plt.subplots(figsize=(max(7, max_prefix_len * 1.35), max(5, max_future_k * 0.95)))
    ax.set_facecolor("#f2f2f2")
    x_edges = np.arange(0.5, max_prefix_len + 1.5, 1.0)
    y_edges = np.arange(0.5, max_future_k + 1.5, 1.0)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(matrix),
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    mesh.set_rasterized(False)
    cbar = fig.colorbar(mesh, ax=ax, label=title)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xticks(np.arange(1, max_prefix_len + 1))
    ax.set_yticks(np.arange(1, max_future_k + 1))
    ax.set_yticklabels([f"+{i}" for i in range(1, max_future_k + 1)])
    ax.set_xlabel("Observed prefix length n")
    ax.set_ylabel("Future rollout step K")
    ax.set_title(title)

    for y in range(max_future_k):
        for x in range(max_prefix_len):
            if annotations[y][x]:
                color = "white" if matrix[y, x] >= (vmin + vmax) / 2.0 else "black"
                ax.text(x + 1, y + 1, annotations[y][x], ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    stem, original_ext = os.path.splitext(filename)
    save_figure(fig, output_dir, stem or original_ext.lstrip(".") or "rollout_type_surface", figure_formats)
    plt.close(fig)


def plot_formal_type_surface_panel(
    rows,
    output_dir,
    max_future_k,
    max_prefix_len,
    min_cell_count,
    figure_formats,
    heatmap_scale,
    heatmap_padding,
    filename_stem="rollout_type_accuracy_surface_formal",
):
    metrics = [
        ("type_top1_accuracy", "(a) Type Top-1 Accuracy"),
        ("type_top3_accuracy", "(b) Type Top-3 Accuracy"),
    ]
    cmap = plt.colormaps.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#f4f4f4")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.6),
        sharey=True,
        constrained_layout=True,
    )
    x_edges = np.arange(0.5, max_prefix_len + 1.5, 1.0)
    y_edges = np.arange(0.5, max_future_k + 1.5, 1.0)

    for ax, (metric, panel_title) in zip(axes, metrics):
        matrix, samples = surface_matrix(
            rows,
            metric,
            max_future_k,
            max_prefix_len,
            min_cell_count,
            with_samples=True,
        )
        vmin, vmax = color_limits(matrix, heatmap_scale, heatmap_padding)
        ax.set_facecolor("#f4f4f4")
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(matrix),
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        mesh.set_rasterized(False)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Accuracy")
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.set_title(panel_title, pad=8)
        ax.set_xlabel("Observed prefix length n")
        ax.set_xticks(np.arange(1, max_prefix_len + 1))
        ax.set_yticks(np.arange(1, max_future_k + 1))
        ax.set_yticklabels([f"+{i}" for i in range(1, max_future_k + 1)])

        for y in range(max_future_k):
            for x in range(max_prefix_len):
                value = matrix[y, x]
                if not np.isfinite(value):
                    continue
                color = "white" if value >= (vmin + vmax) / 2.0 else "black"
                ax.text(
                    x + 1,
                    y + 1,
                    f"{value:.2f}\nn={samples[y, x]}",
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    linespacing=0.95,
                    color=color,
                )

    axes[0].set_ylabel("Prediction horizon K")
    save_figure(fig, output_dir, filename_stem, figure_formats)
    plt.close(fig)


def read_surface_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_surface_csvs(csv_paths):
    per_seed_rows = [read_surface_csv(path) for path in csv_paths]
    if not per_seed_rows:
        raise ValueError("No rollout CSV files were provided.")

    cell_values = defaultdict(lambda: {
        "num_samples": [],
        "type_top1_accuracy": [],
        "type_top3_accuracy": [],
    })

    future_values = defaultdict(lambda: {
        "num_samples": [],
        "type_top1_accuracy": [],
        "type_top3_accuracy": [],
    })
    overall_values = {
        "num_samples": [],
        "type_top1_accuracy": [],
        "type_top3_accuracy": [],
    }

    for rows in per_seed_rows:
        future_accumulators = defaultdict(lambda: {
            "num_samples": 0,
            "top1_correct_est": 0.0,
            "top3_correct_est": 0.0,
        })
        overall_accumulator = {
            "num_samples": 0,
            "top1_correct_est": 0.0,
            "top3_correct_est": 0.0,
        }

        for row in rows:
            future_k = int(row["future_k"])
            observed_n = int(row["observed_n"])
            num_samples = int(row["num_samples"])
            top1 = float(row["type_top1_accuracy"])
            top3 = float(row["type_top3_accuracy"])
            key = (future_k, observed_n)

            cell_values[key]["num_samples"].append(num_samples)
            cell_values[key]["type_top1_accuracy"].append(top1)
            cell_values[key]["type_top3_accuracy"].append(top3)

            future_accumulators[future_k]["num_samples"] += num_samples
            future_accumulators[future_k]["top1_correct_est"] += top1 * num_samples
            future_accumulators[future_k]["top3_correct_est"] += top3 * num_samples
            overall_accumulator["num_samples"] += num_samples
            overall_accumulator["top1_correct_est"] += top1 * num_samples
            overall_accumulator["top3_correct_est"] += top3 * num_samples

        for future_k, acc in future_accumulators.items():
            num_samples = acc["num_samples"]
            future_values[future_k]["num_samples"].append(num_samples)
            future_values[future_k]["type_top1_accuracy"].append(
                acc["top1_correct_est"] / num_samples if num_samples else 0.0
            )
            future_values[future_k]["type_top3_accuracy"].append(
                acc["top3_correct_est"] / num_samples if num_samples else 0.0
            )

        num_samples = overall_accumulator["num_samples"]
        overall_values["num_samples"].append(num_samples)
        overall_values["type_top1_accuracy"].append(
            overall_accumulator["top1_correct_est"] / num_samples if num_samples else 0.0
        )
        overall_values["type_top3_accuracy"].append(
            overall_accumulator["top3_correct_est"] / num_samples if num_samples else 0.0
        )

    cell_summary_rows = []
    plot_rows = []
    for future_k, observed_n in sorted(cell_values):
        values = cell_values[(future_k, observed_n)]
        num_samples = round(mean(values["num_samples"]))
        top1_mean = mean(values["type_top1_accuracy"])
        top3_mean = mean(values["type_top3_accuracy"])
        cell_summary_rows.append({
            "future_k": future_k,
            "observed_n": observed_n,
            "num_samples_per_seed": num_samples,
            "type_top1_mean": top1_mean,
            "type_top1_std": sample_std(values["type_top1_accuracy"]),
            "type_top3_mean": top3_mean,
            "type_top3_std": sample_std(values["type_top3_accuracy"]),
        })
        plot_rows.append({
            "future_k": future_k,
            "observed_n": observed_n,
            "num_samples": num_samples,
            "type_top1_accuracy": top1_mean,
            "type_top3_accuracy": top3_mean,
        })

    future_summary_rows = []
    for future_k in sorted(future_values):
        values = future_values[future_k]
        future_summary_rows.append({
            "future_step": f"+{future_k}",
            "avg_num_pairs_per_seed": round(mean(values["num_samples"])),
            "type_top1_mean": mean(values["type_top1_accuracy"]),
            "type_top1_std": sample_std(values["type_top1_accuracy"]),
            "type_top3_mean": mean(values["type_top3_accuracy"]),
            "type_top3_std": sample_std(values["type_top3_accuracy"]),
        })

    future_summary_rows.append({
        "future_step": "Overall",
        "avg_num_pairs_per_seed": round(mean(overall_values["num_samples"])),
        "type_top1_mean": mean(overall_values["type_top1_accuracy"]),
        "type_top1_std": sample_std(overall_values["type_top1_accuracy"]),
        "type_top3_mean": mean(overall_values["type_top3_accuracy"]),
        "type_top3_std": sample_std(overall_values["type_top3_accuracy"]),
    })

    return cell_summary_rows, future_summary_rows, plot_rows


def run_aggregate(args):
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.commonpath(args.aggregate_csvs)),
        "rollout_3seed_summary",
    )
    os.makedirs(output_dir, exist_ok=True)

    cell_summary_rows, future_summary_rows, plot_rows = aggregate_surface_csvs(args.aggregate_csvs)

    write_csv(
        os.path.join(output_dir, "rollout_type_accuracy_surface_3seed_summary.csv"),
        cell_summary_rows,
        [
            "future_k",
            "observed_n",
            "num_samples_per_seed",
            "type_top1_mean",
            "type_top1_std",
            "type_top3_mean",
            "type_top3_std",
        ],
    )
    write_csv(
        os.path.join(output_dir, "rollout_type_accuracy_by_future_step_3seed_summary.csv"),
        future_summary_rows,
        [
            "future_step",
            "avg_num_pairs_per_seed",
            "type_top1_mean",
            "type_top1_std",
            "type_top3_mean",
            "type_top3_std",
        ],
    )
    plot_formal_type_surface_panel(
        plot_rows,
        output_dir,
        args.max_future_k,
        args.max_prefix_len,
        args.min_cell_count,
        args.figure_formats,
        args.heatmap_scale,
        args.heatmap_padding,
        filename_stem="rollout_type_accuracy_surface_formal_3seed_mean",
    )

    print(f"Aggregate rollout analysis saved to: {output_dir}")


def run_plot_surface_csv(args):
    csv_path = args.plot_surface_csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing surface CSV: {csv_path}")

    output_dir = args.output_dir or os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    rows = read_surface_csv(csv_path)

    plot_type_surface(
        rows,
        output_dir,
        "type_top1_accuracy",
        "Autoregressive Type Top-1 Accuracy",
        "rollout_type_top1_surface.png",
        args.max_future_k,
        args.max_prefix_len,
        args.min_cell_count,
        args.figure_formats,
        args.heatmap_scale,
        args.heatmap_padding,
    )
    plot_type_surface(
        rows,
        output_dir,
        "type_top3_accuracy",
        "Autoregressive Type Top-3 Accuracy",
        "rollout_type_top3_surface.png",
        args.max_future_k,
        args.max_prefix_len,
        args.min_cell_count,
        args.figure_formats,
        args.heatmap_scale,
        args.heatmap_padding,
    )
    if not args.skip_formal_figure:
        plot_formal_type_surface_panel(
            rows,
            output_dir,
            args.max_future_k,
            args.max_prefix_len,
            args.min_cell_count,
            args.figure_formats,
            args.heatmap_scale,
            args.heatmap_padding,
        )

    print(f"Rollout surface figures saved to: {output_dir}")


def summarize_and_save(output_dir, model_type, model_path, max_future_k, max_prefix_len, total_rallies, used_rallies):
    summary_path = os.path.join(output_dir, "rollout_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Autoregressive Rally Prediction Analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"model_type: {model_type}\n")
        f.write(f"checkpoint: {model_path}\n")
        f.write(f"max_future_k: {max_future_k}\n")
        f.write(f"max_prefix_len: {max_prefix_len}\n")
        f.write(f"total_test_rallies: {total_rallies}\n")
        f.write(f"evaluated_rallies_len_ge_2: {used_rallies}\n")


def run_rollout(args):
    import torch
    from src.dataloader import collate_fn_3L

    config = load_config(args.run_dir)
    targets = config.get("targets", [])
    feature_indices = {name: i for i, name in enumerate(config["features_to_extract"])}
    if "type" not in targets or "type" not in feature_indices:
        raise ValueError("This rollout surface analysis requires the 'type' target/feature.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or os.path.join(args.run_dir, "test_rollout_closed_loop")
    os.makedirs(output_dir, exist_ok=True)
    cleanup_output_dir(output_dir)

    model, model_type, model_path = load_model(config, args.run_dir, args.checkpoint, device)
    data_dir = resolve_data_dir(args, config)
    matches = load_test_matches(data_dir)

    max_future_k = max(1, args.max_future_k)
    max_prefix_len = max(1, args.max_prefix_len)
    min_cell_count = max(1, args.min_cell_count)
    surface_buckets = defaultdict(empty_surface_bucket)

    total_rallies = 0
    used_rallies = 0

    with torch.no_grad():
        for match in tqdm(matches, desc="Rollout matches"):
            for set_idx, set_data in enumerate(match["sets"]):
                set_history = match["sets"][:set_idx]
                for rally_idx, rally_data in enumerate(set_data["rallies"]):
                    total_rallies += 1
                    shots = rally_data["shots"]
                    rally_len = int(shots.shape[0])
                    if rally_len < 2:
                        continue
                    used_rallies += 1

                    rally_history = set_data["rallies"][:rally_idx]
                    max_observed_n = min(rally_len - 1, max_prefix_len)

                    for observed_n in range(1, max_observed_n + 1):
                        prefix = [shots[i].copy() for i in range(observed_n)]
                        max_future_for_prefix = min(max_future_k, rally_len - observed_n)

                        for future_k in range(1, max_future_for_prefix + 1):
                            target_idx = observed_n + future_k - 1
                            target_shot = shots[target_idx]
                            item = make_rollout_item(
                                prefix,
                                target_shot,
                                rally_history,
                                set_history,
                                feature_indices,
                                targets,
                            )
                            batch = collate_fn_3L([item])
                            output = model(batch)
                            logits = logits_dict(output)

                            synthetic_shot = target_shot.copy()

                            for task in targets:
                                task_logits = logits[task]
                                pred = int(torch.argmax(task_logits, dim=1).item())
                                synthetic_shot[feature_indices[task]] = pred

                                if task != "type":
                                    continue

                                target = int(target_shot[feature_indices["type"]])
                                k = min(3, task_num_classes(config, "type"))
                                topk = torch.topk(task_logits, k=k, dim=1).indices.squeeze(0).tolist()
                                update_surface_bucket(
                                    surface_buckets[(future_k, observed_n)],
                                    target,
                                    pred,
                                    target in topk,
                                )

                            prefix.append(synthetic_shot)

    surface_rows = []
    for future_k in range(1, max_future_k + 1):
        for observed_n in range(1, max_prefix_len + 1):
            bucket = surface_buckets[(future_k, observed_n)]
            total = bucket["total"]
            surface_rows.append({
                "future_k": future_k,
                "observed_n": observed_n,
                "num_samples": total,
                "type_top1_accuracy": bucket["type_top1_correct"] / total if total else 0.0,
                "type_top3_accuracy": bucket["type_top3_correct"] / total if total else 0.0,
            })

    write_csv(
        os.path.join(output_dir, "rollout_type_accuracy_surface.csv"),
        surface_rows,
        [
            "future_k",
            "observed_n",
            "num_samples",
            "type_top1_accuracy",
            "type_top3_accuracy",
        ],
    )

    plot_type_surface(
        surface_rows,
        output_dir,
        "type_top1_accuracy",
        "Autoregressive Type Top-1 Accuracy",
        "rollout_type_top1_surface.png",
        max_future_k,
        max_prefix_len,
        min_cell_count,
        args.figure_formats,
        args.heatmap_scale,
        args.heatmap_padding,
    )
    plot_type_surface(
        surface_rows,
        output_dir,
        "type_top3_accuracy",
        "Autoregressive Type Top-3 Accuracy",
        "rollout_type_top3_surface.png",
        max_future_k,
        max_prefix_len,
        min_cell_count,
        args.figure_formats,
        args.heatmap_scale,
        args.heatmap_padding,
    )
    if not args.skip_formal_figure:
        plot_formal_type_surface_panel(
            surface_rows,
            output_dir,
            max_future_k,
            max_prefix_len,
            min_cell_count,
            args.figure_formats,
            args.heatmap_scale,
            args.heatmap_padding,
        )
    summarize_and_save(output_dir, model_type, model_path, max_future_k, max_prefix_len, total_rallies, used_rallies)

    print(f"Rollout analysis saved to: {output_dir}")
    print(f"Evaluated rallies: {used_rallies}/{total_rallies}")


def main():
    args = parse_args()
    if args.aggregate_csvs:
        run_aggregate(args)
    elif args.plot_surface_csv:
        run_plot_surface_csv(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
