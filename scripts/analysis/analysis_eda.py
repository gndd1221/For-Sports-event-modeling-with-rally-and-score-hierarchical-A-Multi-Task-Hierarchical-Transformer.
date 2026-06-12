import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_matplotlib_tmp = PROJECT_ROOT / ".tmp" / "matplotlib"
_matplotlib_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_matplotlib_tmp))
os.environ.setdefault("TEMP", str(_matplotlib_tmp))
os.environ.setdefault("TMP", str(_matplotlib_tmp))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "ps.useafm": True,
    "svg.fonttype": "none",
})


# Raw table-tennis locations use 0 as non-grid / unavailable and 1-9 as the
# tactical 3x3 landing grid.
LOCATION_TO_GRID = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}
LOCATION_ROW_LABELS = ["Short", "Half-long", "Long"]
LOCATION_COL_LABELS = ["Forehand", "Middle", "Backhand"]

SHOT_TYPE_MAP = {
    0: "Zero",
    1: "Drive",
    2: "Counter",
    3: "Smash",
    4: "Twist",
    5: "FastDrive",
    6: "FastPush(A)",
    7: "Flip",
    8: "LongPush(P)",
    9: "FastPush(P)",
    10: "LongPush",
    11: "Drop",
    12: "Chop",
    13: "Block",
    14: "Lob",
    15: "Serve(Trad)",
    16: "Serve(Hook)",
    17: "Serve(Rev)",
    18: "Serve(Squat)",
}


def save_figure(fig, out_dir, stem, formats):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in dict.fromkeys(formats):
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)


def load_processed_data(sport, split):
    if sport == "badminton":
        data_dir = PROJECT_ROOT / "data" / sport / "processed_data_badminton"
    elif sport == "tennis":
        data_dir = PROJECT_ROOT / "data" / sport / "Tennis_processed_data"
    else:
        data_dir = PROJECT_ROOT / "data" / sport / "processed_data"

    data_path = data_dir / f"{split}_data.pkl"
    config_path = data_dir / "config.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}")

    with data_path.open("rb") as f:
        matches = pickle.load(f)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    features = config["features_to_extract"]
    feat_idx = {name: i for i, name in enumerate(features)}
    return matches, feat_idx


def shots_from_processed(matches, feat_idx):
    type_idx = feat_idx.get("type")
    loc_idx = feat_idx.get("location")
    spin_idx = feat_idx.get("spin")
    str_idx = feat_idx.get("strength")
    if type_idx is None or loc_idx is None:
        raise ValueError("Processed config must contain type and location features.")

    rows = []
    rally_counter = 0
    for match in matches:
        for set_data in match.get("sets", []):
            rallies = []
            if "games" in set_data:
                for game in set_data["games"]:
                    rallies.extend(game.get("rallies", []))
            else:
                rallies = set_data.get("rallies", [])

            for rally in rallies:
                rally_counter += 1
                shots = rally.get("shots", [])
                for i, shot in enumerate(shots):
                    rows.append({
                        "match_id": match.get("match_id"),
                        "rally_key": rally.get("rally_id", rally_counter),
                        "ball_round": i + 1,
                        # Processed categorical values are shifted by +1.
                        "type": int(shot[type_idx]) - 1,
                        "location": int(shot[loc_idx]) - 1,
                        "spin": int(shot[spin_idx]) - 1 if spin_idx is not None else np.nan,
                        "strength": int(shot[str_idx]) - 1 if str_idx is not None else np.nan,
                    })
    return pd.DataFrame(rows)


def load_raw_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = {"match_id", "rally_id", "ball_round", "type", "location"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    keep_cols = ["match_id", "rally_id", "ball_round", "type", "location"]
    for optional in ["spin", "strength"]:
        if optional in df.columns:
            keep_cols.append(optional)

    shots = df[keep_cols].copy()
    shots["rally_key"] = shots["rally_id"]
    for col in ["type", "location", "ball_round"]:
        shots[col] = shots[col].astype(int)
    for col in ["spin", "strength"]:
        if col in shots.columns:
            shots[col] = shots[col].astype(int)
        else:
            shots[col] = np.nan

    return shots


def load_shots(args):
    if args.source == "raw_csv":
        return load_raw_csv(args.csv_path)

    matches, feat_idx = load_processed_data(args.sport, args.split)
    return shots_from_processed(matches, feat_idx)


def compute_transition_matrix(shots, num_types):
    matrix = np.zeros((num_types, num_types), dtype=np.int64)
    ordered = shots.sort_values(["match_id", "rally_key", "ball_round"])

    for _, rally in ordered.groupby(["match_id", "rally_key"], sort=False):
        types = rally["type"].to_numpy(dtype=int)
        if len(types) < 2:
            continue
        for prev_type, next_type in zip(types[:-1], types[1:]):
            if 0 <= prev_type < num_types and 0 <= next_type < num_types:
                matrix[prev_type, next_type] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    probs = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    return matrix, probs


def compute_location_grid(shots):
    counts = np.zeros((3, 3), dtype=np.int64)
    locations = shots["location"].astype(int)
    for loc, count in locations.value_counts().items():
        loc = int(loc)
        if loc in LOCATION_TO_GRID:
            row, col = LOCATION_TO_GRID[loc]
            counts[row, col] = int(count)

    total_grid = counts.sum()
    pct = np.divide(counts, total_grid, out=np.zeros_like(counts, dtype=float), where=total_grid != 0)
    excluded = int((~locations.isin(LOCATION_TO_GRID.keys())).sum())
    return counts, pct, excluded


def transition_labels(num_types):
    return [SHOT_TYPE_MAP.get(i, f"Type-{i}") for i in range(num_types)]


def color_limit(values, minimum=0.05):
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return minimum
    return max(float(nonzero.max()), minimum)


def annotate_transition_cells(ax, probs, mode="top", threshold=0.30):
    if mode == "none":
        return

    for row in range(probs.shape[0]):
        for col in range(probs.shape[1]):
            value = probs[row, col]
            if value <= 0:
                continue
            if mode == "top" and value < threshold:
                continue
            text_color = "white" if value >= 0.45 else "black"
            ax.text(
                col + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.4 if mode == "all" else 6.2,
                color=text_color,
            )


def plot_transition_matrix(probs, labels, out_dir, formats, annotation_mode, annotation_threshold):
    fig, ax = plt.subplots(figsize=(8.3, 7.2))
    n = len(labels)
    im = ax.pcolormesh(
        np.arange(n + 1),
        np.arange(n + 1),
        probs,
        cmap="Blues",
        vmin=0,
        vmax=color_limit(probs),
        edgecolors="white",
        linewidth=0.35,
    )
    ax.invert_yaxis()
    ax.set_title("(a) Shot-type transition matrix")
    ax.set_xlabel("Next shot type")
    ax.set_ylabel("Previous shot type")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    annotate_transition_cells(ax, probs, annotation_mode, annotation_threshold)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Row-normalized probability")
    fig.tight_layout()
    save_figure(fig, out_dir, "dataset_analysis_transition_matrix", formats)
    plt.close(fig)


def plot_location_grid(pct, counts, excluded, out_dir, formats):
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.pcolormesh(
        np.arange(4),
        np.arange(4),
        pct,
        cmap="Reds",
        vmin=0,
        vmax=color_limit(pct),
        edgecolors="white",
        linewidth=1.0,
    )
    ax.set_aspect("equal")
    ax.set_title("(b) Landing-location distribution")
    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_yticks(np.arange(3) + 0.5)
    ax.set_xticklabels(LOCATION_COL_LABELS)
    ax.set_yticklabels(LOCATION_ROW_LABELS)
    ax.set_xlabel("Side zone")
    ax.set_ylabel("Depth zone")

    for row in range(3):
        for col in range(3):
            ax.text(
                col + 0.5,
                row + 0.5,
                f"{pct[row, col] * 100:.1f}%\n(n={counts[row, col]:,})",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("")
    if excluded:
        fig.text(
            0.5,
            -0.02,
            f"Location label 0 is non-grid/unavailable and excluded from this 3x3 grid (n={excluded:,}).",
            ha="center",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, out_dir, "dataset_analysis_location_grid", formats)
    plt.close(fig)


def plot_composite(probs, labels, loc_pct, loc_counts, excluded, out_dir, formats, annotation_mode, annotation_threshold):
    fig = plt.figure(figsize=(12.8, 6.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.35], wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    n = len(labels)
    im0 = ax0.pcolormesh(
        np.arange(n + 1),
        np.arange(n + 1),
        probs,
        cmap="Blues",
        vmin=0,
        vmax=color_limit(probs),
        edgecolors="white",
        linewidth=0.3,
    )
    ax0.invert_yaxis()
    ax0.set_title("(a) Shot-type transition matrix")
    ax0.set_xlabel("Next shot type")
    ax0.set_ylabel("Previous shot type")
    ax0.set_xticks(np.arange(n) + 0.5)
    ax0.set_yticks(np.arange(n) + 0.5)
    ax0.set_xticklabels(labels, rotation=55, ha="right")
    ax0.set_yticklabels(labels)
    ax0.set_xlim(0, n)
    ax0.set_ylim(n, 0)
    annotate_transition_cells(ax0, probs, annotation_mode, annotation_threshold)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.015)
    cbar0.set_label("Row-normalized probability")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.pcolormesh(
        np.arange(4),
        np.arange(4),
        loc_pct,
        cmap="Reds",
        vmin=0,
        vmax=color_limit(loc_pct),
        edgecolors="white",
        linewidth=1.0,
    )
    ax1.set_aspect("equal")
    ax1.set_title("(b) Landing-location distribution")
    ax1.set_xticks(np.arange(3) + 0.5)
    ax1.set_yticks(np.arange(3) + 0.5)
    ax1.set_xticklabels(LOCATION_COL_LABELS, rotation=25, ha="right")
    ax1.set_yticklabels(LOCATION_ROW_LABELS)
    ax1.set_xlabel("Side zone")
    ax1.set_ylabel("Depth zone")
    for row in range(3):
        for col in range(3):
            ax1.text(
                col + 0.5,
                row + 0.5,
                f"{loc_pct[row, col] * 100:.1f}%\n{loc_counts[row, col]:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.08, pad=0.035)
    cbar1.set_label("")

    note = "Panel (a) is row-normalized within each previous shot type."
    if excluded:
        note += f" Panel (b) excludes non-grid/unavailable location label 0 (n={excluded:,})."
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(bottom=0.22, top=0.92)
    save_figure(fig, out_dir, "dataset_analysis_composite", formats)
    plt.close(fig)


def write_csv_outputs(out_dir, shots, counts, probs, loc_counts, loc_pct, excluded):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = transition_labels(counts.shape[0])

    pd.DataFrame(counts, index=labels, columns=labels).to_csv(
        out_dir / "dataset_analysis_transition_counts.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(probs, index=labels, columns=labels).to_csv(
        out_dir / "dataset_analysis_transition_probabilities.csv", encoding="utf-8-sig"
    )

    top_rows = []
    row_totals = counts.sum(axis=1)
    for i, prev_label in enumerate(labels):
        for j, next_label in enumerate(labels):
            count = int(counts[i, j])
            if count == 0:
                continue
            top_rows.append({
                "previous_type": prev_label,
                "next_type": next_label,
                "count": count,
                "row_probability": float(probs[i, j]),
                "previous_type_total": int(row_totals[i]),
            })
    top_df = pd.DataFrame(top_rows).sort_values(
        ["row_probability", "count"], ascending=[False, False]
    )
    top_df.to_csv(out_dir / "dataset_analysis_top_transitions.csv", index=False, encoding="utf-8-sig")

    loc_rows = []
    for row_idx, row_label in enumerate(LOCATION_ROW_LABELS):
        for col_idx, col_label in enumerate(LOCATION_COL_LABELS):
            loc_rows.append({
                "depth_zone": row_label,
                "lateral_zone": col_label,
                "count": int(loc_counts[row_idx, col_idx]),
                "share_among_grid_locations": float(loc_pct[row_idx, col_idx]),
            })
    pd.DataFrame(loc_rows).to_csv(out_dir / "dataset_analysis_location_grid.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "num_shots": int(len(shots)),
        "num_rallies": int(shots[["match_id", "rally_key"]].drop_duplicates().shape[0]),
        "num_transitions": int(counts.sum()),
        "grid_location_shots": int(loc_counts.sum()),
        "non_grid_or_unavailable_location_shots": int(excluded),
        "grid_location_share": float(loc_counts.sum() / len(shots)) if len(shots) else 0.0,
    }])
    summary.to_csv(out_dir / "dataset_analysis_summary.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready EDA figures for the Results Dataset Analysis section.")
    parser.add_argument("--source", choices=["raw_csv", "processed"], default="raw_csv")
    parser.add_argument("--csv_path", default=str(PROJECT_ROOT / "data" / "table_tennis" / "TableTennis_dataset.csv"))
    parser.add_argument("--sport", default="table_tennis", help="Used only when --source processed.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Used only when --source processed.")
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "outputs" / "eda"))
    parser.add_argument("--num_types", type=int, default=19)
    parser.add_argument("--figure_formats", nargs="+", default=["pdf"], choices=["pdf", "eps", "svg", "png"])
    parser.add_argument(
        "--transition_annotation",
        choices=["none", "top", "all"],
        default="top",
        help="Annotate transition probabilities in the transition heatmap. 'top' labels only salient cells.",
    )
    parser.add_argument(
        "--transition_annotation_threshold",
        type=float,
        default=0.30,
        help="Probability threshold used when --transition_annotation=top.",
    )
    args = parser.parse_args()

    shots = load_shots(args)
    if shots.empty:
        raise ValueError("No shots were loaded; cannot generate Dataset Analysis figures.")

    counts, probs = compute_transition_matrix(shots, args.num_types)
    loc_counts, loc_pct, excluded = compute_location_grid(shots)
    labels = transition_labels(args.num_types)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv_outputs(out_dir, shots, counts, probs, loc_counts, loc_pct, excluded)
    plot_transition_matrix(
        probs,
        labels,
        out_dir,
        args.figure_formats,
        args.transition_annotation,
        args.transition_annotation_threshold,
    )
    plot_location_grid(loc_pct, loc_counts, excluded, out_dir, args.figure_formats)
    plot_composite(
        probs,
        labels,
        loc_pct,
        loc_counts,
        excluded,
        out_dir,
        args.figure_formats,
        args.transition_annotation,
        args.transition_annotation_threshold,
    )

    print(f"Loaded shots: {len(shots):,}")
    print(f"Rallies: {shots[['match_id', 'rally_key']].drop_duplicates().shape[0]:,}")
    print(f"Transitions: {int(counts.sum()):,}")
    print(f"Grid-location shots: {int(loc_counts.sum()):,}")
    print(f"Non-grid/unavailable location shots: {excluded:,}")
    print(f"Saved Dataset Analysis figures and CSV summaries to: {out_dir}")


if __name__ == "__main__":
    main()
