# 分析工具

本文件整理論文使用的補充分析流程。以下命令皆由 repository 根目錄執行，並可直接貼入 Windows `cmd.exe`。

## Majority-class Baseline

使用各資料集 training split 中最常出現的非 padding 類別作為固定預測：

```bat
python scripts/analysis/evaluate_majority_baseline.py --sports table_tennis table_tennis_all badminton_all tennis
```

結果預設輸出至 `outputs\results\majority_baseline\`。

## Closed-loop Rollout

此分析目前只支援 original table tennis。先完成 Final MT-HTA 訓練，再將 `FINAL_RUN_DIRECTORY` 換成實際的 run 資料夾名稱：

```bat
python scripts/analysis/analyze_rollout_prediction.py --run_dir outputs\results\table_tennis\FINAL_RUN_DIRECTORY --sport table_tennis --checkpoint best_model --figure_formats png pdf svg
```

分析會將模型前一步的預測回填至後續輸入，用於觀察多步預測的誤差累積。預設結果會寫入指定 run 目錄下的 `test_rollout_closed_loop\`。

## Exploratory Data Analysis

由 original table-tennis CSV 產生資料分布與擊球型態轉移圖：

```bat
python scripts/analysis/analysis_eda.py --source raw_csv --csv_path data\table_tennis\TableTennis_dataset.csv --out_dir outputs\eda --figure_formats pdf svg
```

如要分析 processed split，可改用 `--source processed`，並透過 `--sport` 與 `--split` 指定資料集及 split。完整參數可執行：

```bat
python scripts/analysis/analysis_eda.py --help
```

## Hierarchy Progress Case Study

此分析目前只支援 original table tennis，且需要 L1、L1+L2、MT-HTA Core 與 Final MT-HTA 各三個 seeds 的 completed runs。程式會掃描 `results_root` 中的 `config.json` 與 checkpoint，自動辨識符合設定的 runs：

```bat
python scripts/analysis/analyze_hierarchy_progress_case_study.py --results_root outputs\results\table_tennis --data_dir data\table_tennis\processed_data --checkpoint best_model
```

預設輸出至 `outputs\results\hierarchy_progress_case_study\`。若缺少必要的模型或 seed，程式會列出缺少的 run 設定。
