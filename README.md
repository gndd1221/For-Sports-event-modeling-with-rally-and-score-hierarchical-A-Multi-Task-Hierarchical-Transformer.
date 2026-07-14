# MT-HTA

**Multi-Task Hierarchical Task-Attention Transformer for racket-sport next-shot prediction**

MT-HTA 是一個面向持拍運動逐拍事件資料的多任務下一拍預測專案。給定目前 rally prefix 與比賽階層脈絡，模型同時預測下一拍的擊球型態、正反手、落點、力量、旋轉等任務，並支援桌球、羽球與網球資料。

本專案的主模型是：

```text
task_attention_itransformer_feature_token
```

它也是論文中的 **Final MT-HTA**：Hierarchical Transformer path + feature-token iTransformer path + gated fusion + Task Attention Fusion。

架構圖：[Final MT-HTA Architecture.pdf](<Thesis_data/figures/Final MT-HTA Architecture.pdf>)

## Highlights

- **Multi-task next-shot prediction**：同時輸出多個下一拍屬性，而不是只預測單一 label。
- **Hierarchy-aware sequence modeling**：依據持拍運動自然結構建模 shot、rally、set/game 等層級。
- **Feature-token iTransformer**：每個語意欄位是一個 token，讓模型學習球種、落點、比分、旋轉等欄位間的交互。
- **Task Attention Fusion**：每個任務有自己的 task query，可從階層脈絡與球員表示中提取不同資訊。
- **Cross-racket support**：目前支援 `table_tennis`、`table_tennis_all`、`badminton`、`badminton_all`、`tennis`。

## Results Snapshot

以下結果來自論文實驗章節；有 `±` 的數值為多個 random seeds 的 mean ± std。

### Original Table Tennis

Final MT-HTA 在原始桌球資料集上取得最高整體 Macro F1。

| Model | Macro F1 | Type F1 | Backhand F1 | Location F1 | Strength F1 | Spin F1 | Loc. Dist. ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Current BiLSTM | 0.5154 ± 0.0034 | 0.4578 ± 0.0054 | 0.7417 ± 0.0028 | 0.2550 ± 0.0123 | 0.5101 ± 0.0018 | 0.6124 ± 0.0021 | 0.8919 ± 0.0080 |
| Rally-context LSTM | 0.5177 ± 0.0030 | 0.4644 ± 0.0140 | 0.7439 ± 0.0012 | 0.2666 ± 0.0057 | 0.5036 ± 0.0040 | 0.6101 ± 0.0109 | 0.9041 ± 0.0144 |
| Hierarchical LSTM | 0.5206 ± 0.0009 | 0.4715 ± 0.0045 | 0.7443 ± 0.0023 | 0.2627 ± 0.0098 | 0.5063 ± 0.0045 | 0.6180 ± 0.0081 | 0.9001 ± 0.0145 |
| Flat Transformer | 0.5191 ± 0.0059 | 0.4598 ± 0.0090 | 0.7355 ± 0.0024 | 0.2667 ± 0.0122 | 0.5148 ± 0.0041 | 0.6185 ± 0.0060 | **0.8791 ± 0.0012** |
| FT-iTransformer | 0.5098 ± 0.0037 | 0.4557 ± 0.0057 | 0.7356 ± 0.0034 | 0.2457 ± 0.0088 | 0.5068 ± 0.0079 | 0.6054 ± 0.0073 | 0.8995 ± 0.0077 |
| Bare MT-HTA | 0.5278 ± 0.0033 | 0.4761 ± 0.0077 | 0.7414 ± 0.0029 | 0.2696 ± 0.0073 | 0.5249 ± 0.0023 | **0.6270 ± 0.0096** | 0.9044 ± 0.0145 |
| MT-HTA w/o iTrans path | 0.5280 ± 0.0051 | 0.4829 ± 0.0079 | **0.7446 ± 0.0047** | **0.2712 ± 0.0083** | 0.5184 ± 0.0020 | 0.6228 ± 0.0057 | 0.9153 ± 0.0124 |
| **Final MT-HTA** | **0.5315 ± 0.0047** | **0.4891 ± 0.0069** | 0.7429 ± 0.0061 | 0.2686 ± 0.0060 | **0.5310 ± 0.0049** | 0.6259 ± 0.0041 | 0.8874 ± 0.0101 |

### Final MT-HTA Task Metrics

| Task | Accuracy | F1-score | Extra metric |
| :--- | :---: | :---: | :---: |
| Type | 0.5143 ± 0.0064 | 0.4891 ± 0.0069 | - |
| Backhand | 0.7482 ± 0.0060 | 0.7429 ± 0.0061 | - |
| Location | 0.3043 ± 0.0057 | 0.2686 ± 0.0060 | Avg. Dist. = 0.8874 ± 0.0101 |
| Strength | 0.5499 ± 0.0023 | 0.5310 ± 0.0049 | - |
| Spin | 0.6606 ± 0.0034 | 0.6259 ± 0.0041 | - |
| **Macro Average** | **0.5555 ± 0.0015** | **0.5315 ± 0.0047** | - |

### Cross-Dataset and Cross-Sport Results

| Dataset | Final MT-HTA Macro F1 | Main observation |
| :--- | :---: | :--- |
| Original table tennis | **0.5315 ± 0.0047** | Best overall Macro F1 among compared baselines |
| Expanded table tennis | **0.5525 ± 0.0007** | Best Macro F1; Flat Transformer remains strongest on location distance |
| Badminton ShuttleSet | **0.5309 ± 0.0028** | Best Macro F1, Type F1, and Landing Area F1 |
| Tennis MCP | **0.7095 ± 0.0064** | Best Macro F1 and extends naturally to L1-L2-L3-L4 hierarchy |

### Closed-Loop Rollout

Closed-loop rollout uses model predictions as future inputs. It is harder than standard single-step evaluation and exposes error accumulation.

![Closed-loop rollout type accuracy](Thesis_data/figures/rollout_type_accuracy_surface_formal_3seed_mean.svg)

| Future step | Avg. #Pairs / seed | Type Top-1 Acc. | Type Top-3 Acc. |
| :---: | :---: | :---: | :---: |
| +1 | 6,602 | 0.5146 ± 0.0061 | 0.8501 ± 0.0049 |
| +2 | 5,070 | 0.3142 ± 0.0075 | 0.6151 ± 0.0024 |
| +3 | 3,638 | 0.2623 ± 0.0126 | 0.5456 ± 0.0122 |
| +4 | 2,479 | 0.2641 ± 0.0096 | 0.5676 ± 0.0185 |
| +5 | 1,697 | 0.2709 ± 0.0076 | 0.5728 ± 0.0176 |
| Overall | 19,486 | 0.3623 ± 0.0075 | 0.6720 ± 0.0077 |

## Architecture

Final MT-HTA has two complementary encoder paths.

**Hierarchical Transformer path**

This path models temporal dependencies within and across match structure. For table tennis and badminton, the hierarchy is:

```text
Shot -> Rally -> Set
```

For tennis, the hierarchy is:

```text
Shot -> Rally -> Game -> Set
```

**Feature-token iTransformer path**

This path models interactions among semantic shot fields. Each feature field is one token. Time is compressed inside each feature token before feature-wise self-attention.

```text
feature history
  -> categorical embedding or numerical projection
  -> padding-aware right-aligned temporal projection
  -> one d_model vector per feature
  -> stack as (batch, num_features, d_model)
  -> Transformer self-attention across feature tokens
  -> feature-token summary
```

Example: if a prefix contains 10 observed shots and 8 semantic fields, the iTransformer path produces 8 feature tokens. Attention is performed among those 8 fields.

**Token definition**

In this project, the iTransformer path is feature-token only. Tokens correspond to semantic fields such as `type`, `location`, `spin`, `roundscore_A`, and `roundscore_B`. Categorical embeddings and player embeddings are input encoders; the iTransformer attention axis remains the feature-field axis.

## Supported Datasets

Sport names are defined by `src/default_config.py`.

| Sport | Default data path | Hierarchy | Typical targets |
| :--- | :--- | :--- | :--- |
| `table_tennis` | `data/table_tennis/processed_data` | L1-L2-L3 | type, backhand, location, strength, spin |
| `table_tennis_all` | `data/table_tennis/processed_data_all` | L1-L2-L3 | type, backhand, location, strength, spin |
| `badminton` | `data/badminton/processed_data_badminton` | L1-L2-L3 | type, backhand, landing area |
| `badminton_all` | `data/badminton/processed_data_badminton_all` | L1-L2-L3 | type, backhand, landing area |
| `tennis` | `data/tennis/Tennis_processed_data` | L1-L2-L3-L4 | type, backhand, location, spin |

Processed data directories contain:

```text
train_data.pkl
val_data.pkl
test_data.pkl
config.json
```

The training script reads only `train_data.pkl` and `val_data.pkl`. The test script reads `test_data.pkl` and the selected checkpoint.

## Models

The active model registry is in `src/default_config.py`; batch-run support is in `scripts/run_all_models.py`.

| Model type | Purpose |
| :--- | :--- |
| `task_attention_itransformer_feature_token` | Final MT-HTA; recommended thesis model |
| `task_attention` | Dual-path Task Attention model; currently also uses feature-token iTransformer |
| `task_attention_wo_itransformer` | Task Attention model without iTransformer path |
| `task_attention_final_wo_itransformer` | Final-module ablation without iTransformer path |
| `sequence_attention` | Sequence-fusion variant |
| `parallel` | CLS-token fusion variant |
| `task_project` | Task-projection fusion variant |
| `L1_L2` | Hierarchy ablation using L1 and L2 |
| `L1` | Hierarchy ablation using only shot-level L1 |
| `baseline_itransformer_feature_token` | Standalone feature-token iTransformer baseline |
| `baseline_transformer_flat` | Flat Transformer baseline |
| `baseline_h_lstm` | Hierarchical LSTM baseline |
| `baseline_lstm_flat` | Flat LSTM baseline |
| `baseline_lstm_context` | Rally-context LSTM baseline |
| `baseline_lstm` | Current-sequence LSTM baseline |
| `baseline_shuttlenet_full` | Adapted ShuttleNet-style Seq2Seq baseline |

## Project Layout

```text
configs/                         sport-specific features, targets, paths, defaults
data/                            raw and processed data
preprocessing/                   table tennis, badminton, tennis preprocessing
scripts/train_location_loss.py   single-run training entrypoint
scripts/test_location_loss.py    checkpoint evaluation entrypoint
scripts/run_all_models.py        batch train/test runner
scripts/analyze_rollout_prediction.py
                                  closed-loop rollout analysis
scripts/analysis/analysis_eda.py dataset analysis and figures
src/default_config.py            sport config loader and model registry
src/dataloader.py                3-level and 4-level hierarchical datasets
src/model_components.py          shared encoders and fusion components
src/models/model_fuse.py         Final MT-HTA and framework variants
src/models/                      baseline models
src/losses.py                    multi-task losses and location distance loss
utils/evaluator.py               metrics, reports, and plots
Thesis_data/content/             thesis text
Thesis_data/figures/             thesis figures
outputs/results/                 training, testing, and analysis outputs
```

## Quick Start: How to Run

所有指令請在專案根目錄執行：

```powershell
cd C:\Users\a3654\Desktop\antigravity
```

1. 啟用 Python / torch 環境。

```powershell
conda activate gpu_pyt
```

2. 確認 processed data 已存在。以下以原始桌球資料為例：

```powershell
Get-ChildItem data\table_tennis\processed_data
```

目錄中應包含：

```text
train_data.pkl
val_data.pkl
test_data.pkl
config.json
```

3. 訓練 Final MT-HTA。

```powershell
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating
```

訓練完成後，run 會輸出到：

```text
outputs/results/table_tennis/run_table_tennis_task_attention_itransformer_feature_token_<timestamp>/
```

4. 測試剛訓練好的 run。

```powershell
python scripts/test_location_loss.py --sport table_tennis --run_dir outputs/results/table_tennis/<run_dir>
```

其中 `<run_dir>` 請替換成上一個步驟產生的資料夾名稱。

5. 一次訓練並測試多個模型。

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token baseline_transformer_flat baseline_itransformer_feature_token --d_model 256
```

6. 執行 closed-loop rollout 分析。

```powershell
python scripts/analyze_rollout_prediction.py --run_dir outputs/results/table_tennis/<run_dir> --sport table_tennis
```

## Environment

On the current Windows setup:

```powershell
conda activate gpu_pyt
```

or call the environment Python directly:

```powershell
C:\Users\a3654\anaconda3\envs\gpu_pyt\python.exe --version
```

## Data Preparation

If processed data already exists, training can start directly. To rebuild from raw CSV, run the corresponding preprocessing script:

```powershell
python preprocessing/TableTennis_preprocess.py
python preprocessing/Badminton_preprocess.py
python preprocessing/Tennis_preprocess.py
```

Reproducibility note:

- preprocessing `random_seed` controls the match-level `train/val/test` split.
- training `--seed` does not recreate the split; it controls initialization, DataLoader shuffle, dropout, and RLW sampling.
- current processed datasets are match-level splits, so one match does not appear in multiple splits.

## Train Final MT-HTA

Single run on the original table tennis dataset:

```powershell
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating
```

Common arguments:

| Argument | Meaning |
| :--- | :--- |
| `--sport` | dataset key, e.g. `table_tennis`, `badminton_all`, `tennis` |
| `--model_type` | model name from the registry |
| `--seed` | training random seed |
| `--d_model` | hidden dimension |
| `--batch_size` | batch size |
| `--epochs` | number of training epochs |
| `--use_rlw` | enable Random Loss Weighting |
| `--use_gated_fusion` | enable path-level gated fusion |
| `--use_shot_aware_pe` | add serve/self semantic positional embeddings |
| `--use_top_down_attention` | enable high-level to low-level refinement |
| `--use_turn_based_gating` | route player embeddings by next-shot hitter/receiver role |

Runs are saved under the sport-specific `results_dir`, for example:

```text
outputs/results/table_tennis/run_table_tennis_task_attention_itransformer_feature_token_<timestamp>/
```

## Batch Training and Testing

Train and test selected models:

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token baseline_transformer_flat baseline_itransformer_feature_token --d_model 256
```

Train only:

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token --skip_test
```

Test latest matching runs only:

```powershell
python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token --skip_train
```

## Evaluate a Checkpoint

```powershell
python scripts/test_location_loss.py --sport table_tennis --run_dir outputs/results/table_tennis/<run_dir>
```

The test script reconstructs the model from `<run_dir>/config.json` and loads:

```text
<run_dir>/train/weights/best_model.pth
```

If `best_model.pth` is missing, it falls back to `last_model.pth`.

## Analysis

Generate EDA figures:

```powershell
python scripts/analysis/analysis_eda.py --source raw_csv --csv_path data/table_tennis/TableTennis_dataset.csv --out_dir outputs/eda --figure_formats pdf
```

Run closed-loop rollout analysis:

```powershell
python scripts/analyze_rollout_prediction.py --run_dir outputs/results/table_tennis/<run_dir> --sport table_tennis
```

## Quick Sanity Checks

Check that the final model and main entrypoints are documented:

```powershell
rg -n "task_attention_itransformer_feature_token|Feature-token iTransformer|train_location_loss.py|test_location_loss.py|run_all_models.py|analyze_rollout_prediction.py" README.md
```

Check that the main CLIs are available:

```powershell
C:\Users\a3654\anaconda3\envs\gpu_pyt\python.exe scripts/run_all_models.py --help
C:\Users\a3654\anaconda3\envs\gpu_pyt\python.exe scripts/train_location_loss.py --help
C:\Users\a3654\anaconda3\envs\gpu_pyt\python.exe scripts/test_location_loss.py --help
```
