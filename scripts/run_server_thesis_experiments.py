import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024)

FINAL_MT_HTA_FLAGS = [
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

FINAL_MT_HTA_FLAGS_WO_SHOT_AWARE_PE = [
    "--skip_window_size", "1",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

FINAL_MT_HTA_FLAGS_WO_GATED_FUSION = [
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

FINAL_MT_HTA_FLAGS_WO_TOP_DOWN_ATTENTION = [
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_turn_based_gating",
]

FINAL_MT_HTA_FLAGS_WO_TURN_BASED_GATING = [
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
]

FINAL_MT_HTA_FLAGS_WO_SKIP_WINDOW = [
    "--skip_window_size", "0",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]


def task_attention_feature_token_experiment(sport, seed, flags=None, extra_flags=None):
    return [
        "--sport", sport,
        "--models", "task_attention_itransformer_feature_token",
        "--d_model", "256",
        "--seed", str(seed),
        *(flags or []),
        *(extra_flags or []),
    ]


EXPERIMENTS = [
    # Tennis: complete the latest Final MT-HTA 3-seed set.
    # task_attention_feature_token_experiment("tennis", 2024, FINAL_MT_HTA_FLAGS),

    # Original table tennis: single-seed component ablations for the latest
    # feature-token Final MT-HTA. These isolate one component at a time.
    task_attention_feature_token_experiment(
        "table_tennis", 42, FINAL_MT_HTA_FLAGS_WO_SHOT_AWARE_PE
    ),
    task_attention_feature_token_experiment(
        "table_tennis", 42, FINAL_MT_HTA_FLAGS_WO_GATED_FUSION
    ),
    task_attention_feature_token_experiment(
        "table_tennis", 42, FINAL_MT_HTA_FLAGS_WO_TOP_DOWN_ATTENTION
    ),
    task_attention_feature_token_experiment(
        "table_tennis", 42, FINAL_MT_HTA_FLAGS_WO_TURN_BASED_GATING
    ),
    task_attention_feature_token_experiment(
        "table_tennis", 42, FINAL_MT_HTA_FLAGS_WO_SKIP_WINDOW
    ),

    # Original table tennis: Bare Task Attention + feature-token iTransformer.
    task_attention_feature_token_experiment("table_tennis", 42),

    # Original table tennis: RLW on the latest Final MT-HTA.
    *[
        task_attention_feature_token_experiment(
            "table_tennis", seed, FINAL_MT_HTA_FLAGS, ["--use_rlw"]
        )
        for seed in SEEDS
    ],
]


def summary_filename(args):
    args_list = list(args)

    def value_after(flag, default):
        if flag not in args_list:
            return default
        idx = args_list.index(flag)
        return args_list[idx + 1] if idx + 1 < len(args_list) else default

    sport = value_after("--sport", "table_tennis")
    model = value_after("--models", "unknown_model")
    seed = value_after("--seed", "default")
    digest = hashlib.sha1(" ".join(args_list).encode("utf-8")).hexdigest()[:8]
    return f"results_summary_{sport}_{model}_seed{seed}_{digest}.csv"


def get_sport_from_args(args):
    if "--sport" in args:
        idx = args.index("--sport")
        if idx + 1 < len(args):
            return args[idx + 1]
    return "table_tennis"


def run_experiment(index, total, args):
    sport = get_sport_from_args(args)
    results_dir = ROOT / "outputs" / "results" / sport
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "results_summary.csv"

    args_list = list(args)
    if "--num_workers" not in args_list:
        args_list.extend(["--num_workers", "4"])

    dest = results_dir / summary_filename(args_list)
    if dest.exists():
        print(f"\n[{index}/{total}] SKIP existing run: {' '.join(args_list)}")
        print(f"         summary: {dest}")
        return True, 0.0

    command = [sys.executable, "scripts/run_all_models.py", *args_list]
    print(f"\n{'=' * 80}")
    print(f"[{index}/{total}] RUN")
    print(f"  command: {' '.join(command)}")
    print(f"{'=' * 80}\n")

    start = time.time()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(
            f"\n[{index}/{total}] FAILED "
            f"(exit code {result.returncode}, elapsed {elapsed / 60:.1f} min)"
        )
        return False, elapsed

    if summary_csv.exists():
        shutil.copy(summary_csv, dest)
        print(f"\n[{index}/{total}] Saved summary: {dest}")
    else:
        print(f"\n[{index}/{total}] WARNING: summary CSV not found: {summary_csv}")

    return True, elapsed


def main():
    total = len(EXPERIMENTS)
    success_count = 0
    total_elapsed = 0.0
    failed_runs = []

    print(f"[TWCC Server] Starting thesis experiments: {total} runs")

    for i, args in enumerate(EXPERIMENTS, 1):
        success, elapsed = run_experiment(i, total, args)
        total_elapsed += elapsed
        if success:
            success_count += 1
        else:
            failed_runs.append(" ".join(args))

    print(f"\n{'#' * 80}")
    print("Experiment summary")
    print(f"  success: {success_count}/{total} (elapsed {total_elapsed / 3600:.2f} hours)")
    if failed_runs:
        print("  failed runs:")
        for run in failed_runs:
            print(f"    - {run}")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
