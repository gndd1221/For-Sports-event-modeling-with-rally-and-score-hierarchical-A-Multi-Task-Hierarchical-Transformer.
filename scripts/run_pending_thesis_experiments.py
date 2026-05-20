"""
run_pending_thesis_experiments.py
=================================
逐一執行論文所需的剩餘實驗。
- 每個實驗完成後，會將 results_summary.csv 複製成帶有實驗標籤的副本。
- 如果該副本已經存在，代表這個實驗之前已經跑過，會自動跳過。
- 如果某個實驗失敗，會記錄錯誤並繼續執行下一個，不會中斷整個流程。
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Final model 共用參數 (舊版桌球)
_FINAL_TT = [
    "--sport", "table_tennis",
    "--models", "task_attention",
    "--d_model", "256",
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

# Final model 共用參數 (新版桌球：包含大量女選手與標註)
_FINAL_TT_ALL = [
    "--sport", "table_tennis_all",
    "--models", "task_attention",
    "--d_model", "256",
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

# Final model 共用參數 (新版羽球)
_FINAL_BD_ALL = [
    "--sport", "badminton_all",
    "--models", "task_attention",
    "--d_model", "256",
    "--skip_window_size", "1",
    "--use_shot_aware_pe",
    "--use_gated_fusion",
    "--use_top_down_attention",
    "--use_turn_based_gating",
]

# ─── 實驗定義：參數列表 ──────────────────────────────────────────────
EXPERIMENTS = [
    # =========================================================================
    # [已完成] 之前的舊資料集實驗 (已註解)
    # =========================================================================
    # [*_FINAL_TT, "--seed", "123"],
    # [*_FINAL_TT, "--seed", "2024"],
    # [*_FINAL_TT, "--seed", "3407"],
    # [*_FINAL_TT, "--seed", "777"],
    # ... 其他已完成的 baselines 與 RLW 實驗 ...

    # =========================================================================
    # 階段一：桌球新資料集驗證 (Table Tennis All)
    # 目的：證明在加入女選手與更多標註後，Final Model 依然統治標準架構
    # =========================================================================
    [*_FINAL_TT_ALL], 
    ["--sport", "table_tennis_all", "--models", "baseline_transformer_flat", "--d_model", "256"],
    ["--sport", "table_tennis_all", "--models", "baseline_lstm_context", "--d_model", "256"],
    # =========================================================================
    # 階段二：羽球跨領域驗證 (Badminton All)
    # 目的：證明框架泛用性，並正面擊敗羽球領域 SOTA (ShuttleNet)
    # =========================================================================
    [*_FINAL_BD_ALL], 
    ["--sport", "badminton_all", "--models", "baseline_shuttlenet_full", "--d_model", "256"], # 羽球必測 SOTA
    ["--sport", "badminton_all", "--models", "baseline_transformer_flat", "--d_model", "256"],
    ["--sport", "badminton_all", "--models", "baseline_lstm_context", "--d_model", "256"],
]


def summary_filename(args):
    raw_name = "results_summary " + " ".join(args) + ".csv"
    invalid_chars = '<>:"/\\|?*'
    return "".join("_" if ch in invalid_chars else ch for ch in raw_name)


def windows_safe_path(path):
    if os.name != "nt":
        return str(path)

    resolved = Path(path).resolve()
    path_str = str(resolved)
    if path_str.startswith("\\\\?\\"):
        return path_str
    return "\\\\?\\" + path_str


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
    
    dest = results_dir / summary_filename(args)

    # ─── 已完成檢查 ─────────────────────────────────────────────────────
    if dest.exists():
        print(f"\n[{index}/{total}] ⏭  已完成，跳過: {' '.join(args)}")
        print(f"         檔案: {dest}")
        return True, 0.0

    # ─── 執行 ───────────────────────────────────────────────────────────
    command = [sys.executable, "scripts/run_all_models.py", *args]
    print(f"\n{'=' * 80}")
    print(f"[{index}/{total}] ▶  開始執行:")
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
        shutil.copy2(windows_safe_path(summary_csv), windows_safe_path(dest))
        print(f"\n[{index}/{total}] ✅ 完成 (耗時 {elapsed/60:.1f} min)")
        print(f"         已儲存: {dest}")
    else:
        print(f"\n[{index}/{total}] ⚠️  完成但未找到 summary CSV")

    return True, elapsed


def main():
    total = len(EXPERIMENTS)
    results = []
    total_start = time.time()

    print(f"\n{'#' * 80}")
    print(f"#  論文實驗批次執行器 (擴展驗證版)")
    print(f"#  共 {total} 個實驗待處理")
    print(f"{'#' * 80}")

    for index, args in enumerate(EXPERIMENTS, start=1):
        success, elapsed = run_experiment(index, total, args)
        results.append((args, success, elapsed))

    # ─── 最終報告 ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n\n{'=' * 80}")
    print(f"  實驗執行報告  (總耗時: {total_elapsed/60:.1f} 分鐘)")
    print(f"{'=' * 80}")
    print(f"{'#':>3}  {'狀態':<4}  {'耗時':>10}")
    print(f"{'-' * 60}")
    for i, (args, success, elapsed) in enumerate(results, 1):
        status = "✅" if success else "❌"
        time_str = f"{elapsed/60:.1f} min" if elapsed > 0 else "skipped"
        print(f"{i:>3}  {status:<4}  {time_str:>10}")
    print(f"{'=' * 80}")

    failed = [args for args, success, _ in results if not success]
    if failed:
        print(f"\n⚠️  以下實驗失敗，需要手動檢查:")
        for args in failed:
            print(f"   - {' '.join(args)}")
    else:
        print(f"\n🎉 所有實驗皆已成功完成！")


if __name__ == "__main__":
    main()
