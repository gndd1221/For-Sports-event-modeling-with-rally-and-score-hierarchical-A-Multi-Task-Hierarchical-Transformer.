import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_BASELINE_MODELS = [
    "baseline_lstm_flat",
    "baseline_lstm_context",
    "baseline_h_lstm",
    "baseline_shuttlenet_full",
    "baseline_transformer_flat",
]

_FINAL_MT_HTA_FLAGS = [
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

# ─── 伺服器專用實驗定義 ──────────────────────────────────────────────
# Design rule:
# - Main comparison models use multi-seed evaluation.
# - Exploratory ablations can remain single-seed unless they support a key claim.
EXPERIMENTS = [
    # =========================================================================
    # Main dataset: table_tennis
    # =========================================================================
    # Bare Task Attention: seed 42 already exists; add 123 and 2024.
    # *[
    #     ["--sport", "table_tennis", "--models", "task_attention", "--d_model", "256", "--seed", str(seed)]
    #     for seed in (123, 2024)
    # ],

    # Final MT-HTA w/o iTransformer supports the iTransformer contribution claim,
    # so it follows the same 3-seed protocol as main comparison models.
    # *[
    #     [
    #         "--sport", "table_tennis",
    #         "--models", "task_attention_final_wo_itransformer",
    #         "--d_model", "256",
    #         "--seed", str(seed),
    #         *_FINAL_MT_HTA_FLAGS,
    #     ]
    #     for seed in (42, 123, 2024)
    # ],

    # iTransformer Flat: seed 42 already exists; add 123 and 2024 so that
    # rows reported in the same iTransformer / dual-path table use 3 seeds.
    *[
        [
            "--sport", "table_tennis",
            "--models", "baseline_itransformer_flat",
            "--d_model", "256",
            "--seed", str(seed),
        ]
        for seed in (123, 2024)
    ],

    # Bare Task Attention w/o iTransformer: seed 42 already exists; add 123
    # and 2024 for the same 3-seed reporting protocol.
    *[
        [
            "--sport", "table_tennis",
            "--models", "task_attention_wo_itransformer",
            "--d_model", "256",
            "--seed", str(seed),
        ]
        for seed in (123, 2024)
    ],

    # =========================================================================
    # Expanded dataset: table_tennis_all
    # =========================================================================
    # Final MT-HTA: seed 42 already exists; add 123 and 2024.
    # *[
    #     [
    #         "--sport", "table_tennis_all",
    #         "--models", "task_attention",
    #         "--d_model", "256",
    #         "--seed", str(seed),
    #         *_FINAL_MT_HTA_FLAGS,
    #     ]
    #     for seed in (123, 2024)
    # ],

    # Main baseline on the expanded dataset.
    # *[
    #     [
    #         "--sport", "table_tennis_all",
    #         "--models", "baseline_transformer_flat",
    #         "--d_model", "256",
    #         "--seed", str(seed),
    #     ]
    #     for seed in (123, 2024)
    # ],

    # LSTM with Context: seed 42 already exists; add 123 and 2024 so the
    # expanded-dataset comparison table can report all rows with 3 seeds.
    *[
        [
            "--sport", "table_tennis_all",
            "--models", "baseline_lstm_context",
            "--d_model", "256",
            "--seed", str(seed),
        ]
        for seed in (123, 2024)
    ],

    # Bare Task Attention separates the task-attention mechanism from final modules.
    # *[
    #     ["--sport", "table_tennis_all", "--models", "task_attention", "--d_model", "256", "--seed", str(seed)]
    #     for seed in (42, 123, 2024)
    # ],

    # # Existing cross-dataset / cross-sport runs retained.
    # ["--sport", "badminton_all", "--models", "task_attention", "--d_model", "256"],
    # ["--sport", "tennis", "--models", "task_attention", "--d_model", "256"],
]


def summary_filename(args):
    raw_name = "results_summary " + " ".join(args) + ".csv"
    invalid_chars = '<>:"/\\|?*'
    return "".join("_" if ch in invalid_chars else ch for ch in raw_name)


def get_sport_from_args(args):
    """從參數列表中提取 --sport 的值"""
    if "--sport" in args:
        idx = args.index("--sport")
        if idx + 1 < len(args):
            return args[idx + 1]
    return "table_tennis" # fallback


def run_experiment(index, total, args):
    """執行單一實驗，回傳 (成功與否, 耗時秒數)"""
    sport = get_sport_from_args(args)
    results_dir = ROOT / "outputs" / "results" / sport
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "results_summary.csv"
    
    # 對伺服器端自動添加 --num_workers 4 以加速讀取 (若未指定)
    args_list = list(args)
    if "--num_workers" not in args_list:
        args_list.extend(["--num_workers", "4"])
        
    dest = results_dir / summary_filename(args_list)

    # ─── 已完成檢查 ─────────────────────────────────────────────────────
    if dest.exists():
        print(f"\n[{index}/{total}] ⏭  已完成，跳過: {' '.join(args_list)}")
        print(f"         檔案: {dest}")
        return True, 0.0

    # ─── 執行 ───────────────────────────────────────────────────────────
    command = [sys.executable, "scripts/run_all_models.py", *args_list]
    print(f"\n{'=' * 80}")
    print(f"[{index}/{total}] ▶  伺服器端開始執行:")
    print(f"  命令: {' '.join(command)}")
    print(f"{'=' * 80}\n")

    start = time.time()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[{index}/{total}] ❌ 失敗 (exit code {result.returncode}, 耗時 {elapsed/60:.1f} min)")
        return False, elapsed

    # ─── 複製 summary ──────────────────────────────────────────────────
    if summary_csv.exists():
        shutil.copy(summary_csv, dest)
        print(f"\n[{index}/{total}] 💾 結果已存檔: {dest}")
        
    return True, elapsed


def main():
    total = len(EXPERIMENTS)
    success_count = 0
    total_elapsed = 0.0
    failed_runs = []

    print(f"🚀 [TWCC Server] 啟動伺服器端批次實驗，共 {total} 個任務...")
    
    for i, args in enumerate(EXPERIMENTS, 1):
        success, elapsed = run_experiment(i, total, args)
        total_elapsed += elapsed
        if success:
            success_count += 1
        else:
            failed_runs.append(" ".join(args))
            
    print(f"\n{'#' * 80}")
    print(f"🎉 伺服器端全部實驗執行完畢！")
    print(f"  成功率: {success_count}/{total} (耗時 {total_elapsed/3600:.2f} hours)")
    if failed_runs:
        print("  ❌ 失敗的任務:")
        for r in failed_runs:
            print(f"    - {r}")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
