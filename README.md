# MT-HTA Racket Sports Next-Shot Prediction

本專案實作論文中的 Multi-Task Hierarchical Task-Attention Transformer
(MT-HTA)，用於持拍運動的多任務下一拍預測。模型根據歷史擊球序列與比賽脈絡，同時預測下一拍的擊球型態、正反手、落點，以及資料集中可用的力量或旋轉等任務。

目前專案以論文 final model `task_attention_itransformer_feature_token` 為主，並保留 baseline、ablation、EDA 與 closed-loop rollout analysis 所需的程式。

## 支援資料集

| Sport key | 資料來源 | Hierarchy | 主要任務 |
| :--- | :--- | :--- | :--- |
| `table_tennis` | 原始桌球逐拍資料 | Shot -> Rally -> Set | Type, Backhand, Location, Strength, Spin |
| `table_tennis_all` | 擴充桌球逐拍資料 | Shot -> Rally -> Set | Type, Backhand, Location, Strength, Spin |
| `badminton` | ShuttleSet 子集 | Shot -> Rally -> Set | Type, Backhand, Landing Area |
| `badminton_all` | ShuttleSet 擴充資料 | Shot -> Rally -> Set | Type, Backhand, Landing Area |
| `tennis` | Tennis Match Charting Project | Shot -> Rally -> Game -> Set | Type, Backhand, Location, Spin |

各資料集的欄位、任務、資料路徑與訓練預設值定義於 `configs/*.yaml`。

## Final MT-HTA 架構

Final MT-HTA 使用雙路徑階層式編碼器：

- Hierarchical Transformer path：沿著 shot、rally、set 或 game 的時間軸建模比賽進程。
- Feature-token iTransformer path：將每個語意欄位作為一個 feature token，建模欄位之間的交互關係。
- Gated fusion：在每個階層動態融合 Hierarchical Transformer path 與 feature-token iTransformer path。
- Task Attention Fusion：使用任務專屬 query，讓不同預測任務各自選擇相關的階層表示、球員表示與對手表示。

論文 final model 使用下列設定：

```text
model_type = task_attention_itransformer_feature_token
d_model = 256
skip_window_size = 1
use_shot_aware_pe = true
use_gated_fusion = true
use_top_down_attention = true
use_turn_based_gating = true
```

### Feature-token iTransformer

本專案目前只保留 feature-token iTransformer。它的 token 定義不是 embedding dimension，也不是時間點，而是語意欄位本身。

以桌球資料為例，一筆擊球序列可能包含：

```text
roundscore_A, roundscore_B, type, backhand, location, strength, spin, set
```

Feature-token iTransformer 會建立類似下列 tokens：

```text
[roundscore_A token]
[roundscore_B token]
[type token]
[backhand token]
[location token]
[strength token]
[spin token]
[set token]
```

每個 token 會先沿著時間維度做 padding-aware temporal projection。也就是說，模型會根據有效長度遮掉 padding，將同一個 feature 在歷史時間軸上的資訊壓縮成一個 `d_model` 向量。接著，Transformer Encoder 在 feature tokens 之間做 self-attention，讓模型學習例如 type、location、strength、spin 與比分狀態之間的交互關係。

Player ID 與 opponent ID 不作為 feature-token iTransformer 的 token，而是透過 player-style encoder 另外建模，並在任務融合階段使用。

舊版 embedding/channel iTransformer 與 `baseline_itransformer_flat` 已移除。若舊 config 明確指定 `itransformer_tokenization="embedding_dim"`，目前程式會直接報錯，避免 silent mismatch。

## 專案結構

```text
antigravity/
├── configs/                         # sport-specific YAML settings
├── data/                            # raw CSV and processed data
├── preprocessing/                   # raw CSV -> processed PKL/config
├── scripts/
│   ├── train_location_loss.py       # single-model training
│   ├── test_location_loss.py        # single-run evaluation
│   ├── run_all_models.py            # batch train/test runner
│   ├── analyze_rollout_prediction.py
│   └── analysis/
│       └── analysis_eda.py
├── src/
│   ├── default_config.py            # model registry and config builders
│   ├── dataloader.py
│   ├── losses.py
│   ├── model_components.py
│   └── models/
│       ├── model_fuse.py            # Final MT-HTA and framework variants
│       ├── baseline_itransformer_feature_token.py
│       ├── baseline_transformer_flat.py
│       ├── baseline_lstm.py
│       ├── baseline_lstm_flat.py
│       ├── baseline_lstm_context.py
│       ├── baseline_h_lstm.py
│       └── baseline_shuttlenet_full.py
├── utils/
│   ├── evaluator.py
│   └── base_model.py
├── Thesis_data/
│   ├── content/
│   └── figures/
├── outputs/
│   ├── results/
│   └── eda/
├── requirements.txt
└── README.md
```

## 目前可用模型

| Model type | 用途 |
| :--- | :--- |
| `task_attention_itransformer_feature_token` | 論文 Final MT-HTA 的明確模型名稱 |
| `task_attention` | dual-path task-attention model；目前同樣使用 feature-token iTransformer path |
| `task_attention_wo_itransformer` | 移除 iTransformer path 的 bare ablation |
| `task_attention_final_wo_itransformer` | 保留 final modules、移除 iTransformer path 的 key ablation |
| `parallel` | CLS-token fusion variant |
| `task_project` | task projection fusion variant |
| `sequence_attention` | sequence-level task-attention variant |
| `L1_L2` | 只使用 shot + rally hierarchy |
| `L1` | 只使用 shot-level context |
| `baseline_lstm` | current sequence BiLSTM baseline |
| `baseline_lstm_flat` | flat LSTM baseline |
| `baseline_lstm_context` | LSTM with rally-context baseline |
| `baseline_h_lstm` | hierarchical LSTM baseline |
| `baseline_transformer_flat` | flat Transformer baseline |
| `baseline_itransformer_feature_token` | standalone feature-token iTransformer baseline |
| `baseline_shuttlenet_full` | adapted ShuttleNet Seq2Seq baseline |

可用模型清單由 `src/default_config.py` 的 `MODEL_REGISTRY` 與 `scripts/run_all_models.py` 的 `SUPPORTED_MODELS` 控制。

## 環境

建議使用專案目前的 conda 環境：

```powershell
conda activate gpu_pyt
```

若需要重新安裝 Python dependencies：

```powershell
pip install -r requirements.txt
```

`torch` 請依照本機 CUDA 版本安裝；本專案目前檢查使用的環境為 `gpu_pyt`。

## 資料流程

訓練與測試腳本讀取 processed data，而不是直接讀 raw CSV。每個 processed data directory 需要包含：

```text
train_data.pkl
val_data.pkl
test_data.pkl
config.json
player_map.json
```

目前常用 processed data 路徑如下：

| Sport key | Processed data |
| :--- | :--- |
| `table_tennis` | `data/table_tennis/processed_data` |
| `table_tennis_all` | `data/table_tennis/processed_data_all` |
| `badminton` | `data/badminton/processed_data_badminton` |
| `badminton_all` | `data/badminton/processed_data_badminton_all` |
| `tennis` | `data/tennis/Tennis_processed_data` |

若 processed data 已存在，可以直接訓練。若要從 raw CSV 重建 processed data，可執行：

```powershell
python preprocessing/TableTennis_preprocess.py
python preprocessing/Badminton_preprocess.py
python preprocessing/Tennis_preprocess.py
```

注意：目前 preprocessing 腳本有部分 hardcoded dataset 與 output 設定。刪除 processed data 前，應先確認對應腳本會重建到你需要的路徑與資料版本。

## 訓練 Final MT-HTA

單次訓練論文 final model：

```powershell
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating
```

指定 seed：

```powershell
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --seed 42 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating
```

啟用 RLW：

```powershell
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --seed 42 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating --use_rlw
```

訓練完成後，輸出會存到對應 sport config 的 `results_dir`，例如：

```text
outputs/results/table_tennis/run_table_tennis_task_attention_itransformer_feature_token_<timestamp>/
```

每個 run directory 會包含完整 resolved config、training logs、weights 與測試輸出。

## 批次訓練與測試

使用 `run_all_models.py` 可以依序訓練並測試多個模型：

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token baseline_transformer_flat baseline_itransformer_feature_token --d_model 256
```

只訓練不測試：

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token --d_model 256 --skip_test
```

只測試最新 run：

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token --skip_train
```

執行完成後，`run_all_models.py` 會在 sport 的 results directory 中輸出或更新：

```text
results_summary.csv
```

## 測試既有 run

若已經有訓練完成的 run directory，可以直接測試：

```powershell
python scripts/test_location_loss.py --sport table_tennis --run_dir outputs/results/table_tennis/<run_dir>
```

測試結果會輸出到：

```text
<run_dir>/test/evaluation_results.csv
<run_dir>/test/evaluation_results.txt
<run_dir>/test/plots/
```

`evaluation_results.csv` 包含 accuracy、precision、recall、F1、loss，以及 location task 的 average distance。

## EDA 圖表

產生 table tennis dataset analysis 圖與 CSV：

```powershell
python scripts/analysis/analysis_eda.py --source raw_csv --csv_path data/table_tennis/TableTennis_dataset.csv --out_dir outputs/eda --figure_formats pdf
```

主要輸出包含：

```text
outputs/eda/dataset_analysis_composite.pdf
outputs/eda/dataset_analysis_summary.csv
outputs/eda/dataset_analysis_location_grid.csv
outputs/eda/dataset_analysis_transition_matrix.pdf
```

若要編譯 thesis，`Thesis_data/content/results.md` 目前引用：

```text
figures/dataset_analysis_composite.pdf
```

因此需要將 `outputs/eda/dataset_analysis_composite.pdf` 複製到 `Thesis_data/figures/`，或調整 thesis 的 figure path。

## Closed-loop Rollout Analysis

對 Final MT-HTA run 執行 rollout analysis：

```powershell
python scripts/analyze_rollout_prediction.py --run_dir outputs/results/table_tennis/<final_model_run_dir> --sport table_tennis
```

若已完成多個 seed 的 rollout CSV，可聚合：

```powershell
python scripts/analyze_rollout_prediction.py --aggregate_csvs outputs/results/table_tennis/<run1>/rollout_type_accuracy_surface.csv outputs/results/table_tennis/<run2>/rollout_type_accuracy_surface.csv outputs/results/table_tennis/<run3>/rollout_type_accuracy_surface.csv
```

此分析目前只支援 `table_tennis`，並聚焦於 Type prediction 的 closed-loop rollout。

## 論文結果與重現性

`outputs/results/` 中的既有 run 可用來核對論文表格。若刪除這些結果，仍可用目前 code 重跑模型，但 mean +/- std 表格需要重新彙整。

目前 README 只描述可重現的訓練、測試、EDA 與 rollout analysis，不承諾一鍵重建所有 thesis 表格。論文中的部分表格需要手動彙整多 seed 的 `evaluation_results.csv` 或 `results_summary.csv`。

建議保留到論文定稿後再清理：

```text
outputs/results/
outputs/eda/
Thesis_data/content/
Thesis_data/figures/
```

已移除或不作為目前正式流程：

- `baseline_itransformer_flat`
- 舊版 embedding/channel iTransformer
- 已刪除的 thesis helper scripts，例如 `run_server_thesis_experiments.py`、`run_pending_thesis_experiments.py`、`evaluate_stratified.py`、`plot_final_mtha_architecture.py`

## 常用指令摘要

```powershell
# 啟用環境
conda activate gpu_pyt

# 訓練 final model
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating

# 批次訓練/測試
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token baseline_transformer_flat baseline_itransformer_feature_token --d_model 256

# 測試既有 run
python scripts/test_location_loss.py --sport table_tennis --run_dir outputs/results/table_tennis/<run_dir>

# EDA
python scripts/analysis/analysis_eda.py --source raw_csv --csv_path data/table_tennis/TableTennis_dataset.csv --out_dir outputs/eda --figure_formats pdf

# Rollout analysis
python scripts/analyze_rollout_prediction.py --run_dir outputs/results/table_tennis/<final_model_run_dir> --sport table_tennis
```
