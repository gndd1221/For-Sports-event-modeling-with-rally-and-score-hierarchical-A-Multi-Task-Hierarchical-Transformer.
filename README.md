# Final MT-HTA

Final MT-HTA (Multi-Task Hierarchical Task-Attention Transformer) 是用於持拍運動下一拍事件預測的多任務架構。模型輸入目前預測點以前的擊球事件與比賽階層脈絡，同時輸出球種、正反手、落點及運動特有屬性。此 repository 支援 table tennis、badminton 與 tennis。

[查看 Final MT-HTA 架構圖（PDF）](<Final MT-HTA Architecture.pdf>)

![Final MT-HTA closed-loop rollout](docs/figures/rollout_type_accuracy_surface_formal_3seed_mean.svg)

## 方法概觀

資料依比賽規則組成多層歷史：

- **L1 Shot-level**：目前 rally 中、目標拍以前的 observed prefix。
- **L2 Rally-level**：三層資料為目前 set 內已完成的 rallies；tennis 為目前 game 內已完成的 rallies。
- **L3 Set-level（三層資料）**：table tennis 與 badminton 中，目前 set 以前已完成的 sets。
- **L3 Game-level（tennis）**：tennis 目前 set 內已完成的 games。
- **L4 Set-level（tennis）**：tennis 目前 set 以前已完成的 sets。

因此 table tennis 與 badminton 使用 Shot -> Rally -> Set，tennis 使用 Shot -> Rally -> Game -> Set。

Final MT-HTA 使用兩條平行編碼路徑：

1. **Hierarchical Transformer path**：分層編碼 shot、rally、set；tennis 另編碼 game。
2. **Feature-token iTransformer path**：每個語意欄位是一個 token。各欄位先沿有效時間步做 padding-aware temporal projection，再於 feature tokens 之間做 self-attention，以建模 type、location、spin、score 等欄位交互。

兩條路徑在各階層以 sigmoid gate 融合。Final 設定再加入 shot-aware positional encoding、top-down attention、turn-based player-role routing 與最近一拍 gated skip connection。Task Attention Fusion 為每個預測任務學習獨立 query，從階層摘要與 hitter/opponent 表示擷取任務所需資訊。

## 論文結果

以下為 original table tennis test set，使用相同 match-level split、`d_model=256` 與 seeds `42/123/2024`。數值為 mean +/- sample standard deviation；Location Distance 越低越好。

| Model | Macro F1 | Type F1 | Backhand F1 | Location F1 | Strength F1 | Spin F1 | Loc. Dist. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current-rally BiLSTM | 0.5154 +/- 0.0034 | 0.4578 +/- 0.0054 | 0.7417 +/- 0.0028 | 0.2550 +/- 0.0123 | 0.5101 +/- 0.0018 | 0.6124 +/- 0.0021 | 0.8919 +/- 0.0080 |
| Rally-context LSTM | 0.5177 +/- 0.0030 | 0.4644 +/- 0.0140 | 0.7439 +/- 0.0012 | 0.2666 +/- 0.0057 | 0.5036 +/- 0.0040 | 0.6101 +/- 0.0109 | 0.9041 +/- 0.0144 |
| Hierarchical LSTM | 0.5206 +/- 0.0009 | 0.4715 +/- 0.0045 | 0.7443 +/- 0.0023 | 0.2627 +/- 0.0098 | 0.5063 +/- 0.0045 | 0.6180 +/- 0.0081 | 0.9001 +/- 0.0145 |
| ShuttleNet (adapted) | 0.3958 +/- 0.0260 | 0.2672 +/- 0.0439 | 0.5690 +/- 0.0411 | 0.2046 +/- 0.0062 | 0.4376 +/- 0.0015 | 0.5009 +/- 0.0423 | 0.9607 +/- 0.0061 |
| Flat Transformer | 0.5191 +/- 0.0059 | 0.4598 +/- 0.0090 | 0.7355 +/- 0.0024 | 0.2667 +/- 0.0122 | 0.5148 +/- 0.0041 | 0.6185 +/- 0.0060 | **0.8791 +/- 0.0012** |
| PatchTST (adapted) | 0.4635 +/- 0.0091 | 0.3648 +/- 0.0121 | 0.6598 +/- 0.0265 | 0.2373 +/- 0.0148 | 0.4855 +/- 0.0045 | 0.5700 +/- 0.0056 | 0.9157 +/- 0.0174 |
| FT-iTransformer | 0.5098 +/- 0.0037 | 0.4557 +/- 0.0057 | 0.7356 +/- 0.0034 | 0.2457 +/- 0.0088 | 0.5068 +/- 0.0079 | 0.6054 +/- 0.0073 | 0.8995 +/- 0.0077 |
| **Final MT-HTA** | **0.5315 +/- 0.0047** | **0.4891 +/- 0.0069** | 0.7429 +/- 0.0061 | **0.2686 +/- 0.0060** | **0.5310 +/- 0.0049** | **0.6259 +/- 0.0041** | 0.8874 +/- 0.0101 |

ShuttleNet-adapted 使用目前 rally 的 L1 sequence，保留 encoder-decoder、player branches 與 position-aware gated fusion，並將輸入/輸出介面調整為單步多任務分類。PatchTST-adapted 將當下可用的階層歷史展平，對各語意 feature channel 建立 temporal patches。這些結果只代表本 repository 的適配版本，不等同於原論文任務上的效能。

跨資料集的 majority-class baseline 與 Final MT-HTA：

| Dataset | Majority Macro F1 | Final MT-HTA Macro F1 |
| :--- | ---: | ---: |
| Original Table Tennis | 0.2532 | **0.5315 +/- 0.0047** |
| Expanded Table Tennis | 0.2379 | **0.5525 +/- 0.0007** |
| Badminton ShuttleSet | 0.2175 | **0.5309 +/- 0.0028** |
| Tennis MCP | 0.5349 | **0.7095 +/- 0.0064** |

## 環境

環境為 Python 3.9、PyTorch 2.3 與 CUDA 12.1。以下命令可直接逐行貼入 Windows `cmd.exe`：

```bat
git clone https://github.com/gndd1221/Sequence-Modeling-in-Table-Tennis-Event-Data.git
cd Sequence-Modeling-in-Table-Tennis-Event-Data
conda env create -f environment.yml
conda activate mt-hta
```

若已有可用的 CUDA/PyTorch 環境：

```bat
python -m pip install -r requirements.txt
```

確認環境：

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 資料準備

CSV 與用途列於 [data/README.md](data/README.md)。首次須先建立 processed data。

Original table tennis（主要論文實驗）：

```bat
python preprocessing/TableTennis_preprocess.py
```

Expanded table tennis：

```bat
python preprocessing/TableTennis_preprocess.py --variant all
```

Combined badminton（正式跨運動實驗的預設）：

```bat
python preprocessing/Badminton_preprocess.py
```

Original badminton 補充版本：

```bat
python preprocessing/Badminton_preprocess.py --variant original
```

Tennis Top100（正式跨運動實驗的預設）：

```bat
python preprocessing/Tennis_preprocess.py
```

所有 preprocessing 入口皆支援：

```text
--variant <variant> --input <csv> --output-dir <dir>
--seed 42 --val-ratio 0.1 --test-ratio 0.2
```

分割單位是 **match**，同一 match 不會跨 train/validation/test。正式資料切分使用 preprocessing seed `42`；模型訓練 seeds 不會重新切分資料。

## 訓練 Final MT-HTA

單一 seed：

```bat
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating --seed 42
```

正式三種子訓練與測試可在互動式 `cmd.exe` 中直接執行：

```bat
for %S in (42 123 2024) do python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating --seed %S
```

若將同一命令寫入 `.bat` 檔，須把 `%S` 改成 `%%S`。五個任務的固定權重 `0.2` 已定義於 `configs/table_tennis.yaml`，不需要另外傳入 JSON。

其餘論文固定訓練設定由 `configs/table_tennis.yaml` 與訓練程式預設值共同提供：40 epochs、batch size 128、AdamW learning rate `1e-4`、weight decay `0.01`、10% warm-up、gradient clipping `1.0`、label smoothing `0.1` 與 Location Expected Distance weight `1.0`。每個 run 以 validation loss 最低的 `best_model.pth` 進行正式測試。`run_all_models.py` 會依序完成訓練及測試。

## 測試與 Baselines

測試既有 run：

將 `RUN_DIRECTORY` 換成實際的 run 資料夾名稱，再執行：

```bat
python scripts/test_location_loss.py --sport table_tennis --run_dir outputs\results\table_tennis\RUN_DIRECTORY
```

執行主要 baseline（範例為 seed 42）：

```bat
python scripts/run_all_models.py --sport table_tennis --models baseline_lstm baseline_lstm_context baseline_h_lstm baseline_shuttlenet_full baseline_transformer_flat baseline_itransformer_feature_token --d_model 256 --seed 42
```

PatchTST-adapted 使用同一個 runner。三種子可由 `cmd.exe` 執行：

```bat
for %S in (42 123 2024) do python scripts/run_all_models.py --sport table_tennis --models baseline_patchtst --d_model 256 --seed %S
```

目前 registry 中可用的模型：

| Model ID | Purpose |
| :--- | :--- |
| `task_attention_itransformer_feature_token` | MT-HTA Core 與 Final MT-HTA 共用的正式 model ID；上方 final flags 決定是否啟用完整進階模組 |
| `task_attention` | 與 MT-HTA Core 相同的 dual-path Task Attention 設定別名 |
| `cls_token_itransformer_feature_token` | CLS-token fusion ablation |
| `task_project_itransformer_feature_token` | Task-projection fusion ablation |
| `task_attention_L1`, `task_attention_L1_L2` | Hierarchy-depth ablations |
| `task_attention_wo_itransformer`, `task_attention_final_wo_itransformer` | Hierarchical Transformer-only ablations |
| `sequence_attention` | Sequence-level task-attention variant |
| `baseline_lstm`, `baseline_lstm_context`, `baseline_lstm_flat`, `baseline_h_lstm` | Recurrent baselines |
| `baseline_transformer_flat` | Flat temporal Transformer |
| `baseline_itransformer_feature_token` | Standalone FT-iTransformer |
| `baseline_patchtst` | PatchTST-adapted |
| `baseline_shuttlenet_full` | ShuttleNet-adapted |

## Reproducibility 與評估注意事項

- Preprocessing seed 控制 match IDs 的 split；訓練 seed `42/123/2024` 控制模型初始化、DataLoader shuffle、dropout 與其他 PyTorch 隨機操作。
- Test DataLoader 使用 `shuffle=False`，同一 checkpoint 與資料在 deterministic evaluation 下不應因訓練 seed 再次改變。
- `outputs/results/<sport>/run_<sport>_<model>_<timestamp>/` 保存 run config、best/last checkpoint、TensorBoard logs 與 test metrics；這些大型產物不納入 Git。
- Location Expected Distance Loss 對 target 位於 3x3 grid 的樣本，計算 grid 內預測機率相對真實格子的期望歐氏距離，再與 cross entropy 相加。


## Repository Layout

```text
configs/                      sport-specific features, targets and defaults
data/                         publishable CSV inputs; generated data are ignored
docs/figures/                 README figures
preprocessing/                match-level data preparation
scripts/                      training, testing, runners and analyses
src/model_components.py       shared MT-HTA components
src/models/model_fuse.py      Final MT-HTA and framework ablations
src/models/                   adapted and general baselines
utils/                        shared base model and evaluator
Final MT-HTA Architecture.pdf architecture figure
```
