# Final MT-HTA

Final MT-HTA (Multi-Task Hierarchical Task-Attention Transformer) is a multi-task architecture for next-shot event prediction in racket sports. Given the observed shot prefix and the available match context before a target shot, the model jointly predicts shot type, backhand usage, location, and sport-specific attributes. The repository supports table tennis, badminton, and tennis.

[View the Final MT-HTA architecture (PDF)](<Final MT-HTA Architecture.pdf>)

![Final MT-HTA closed-loop rollout](docs/figures/rollout_type_accuracy_surface_formal_3seed_mean.svg)

## Architecture

The input history follows the competition hierarchy:

- **L1 Shot level:** the observed prefix before the target shot in the current rally.
- **L2 Rally level:** completed rallies in the current set for three-level datasets, or in the current game for tennis.
- **L3 Set level (three-level datasets):** completed sets before the current set in table tennis and badminton.
- **L3 Game level (tennis):** completed games in the current tennis set.
- **L4 Set level (tennis):** completed sets before the current tennis set.

Table tennis and badminton therefore use Shot -> Rally -> Set, while tennis uses Shot -> Rally -> Game -> Set.

Final MT-HTA contains two parallel encoding paths:

1. **Hierarchical Transformer path:** encodes shot, rally, and set context; tennis additionally includes the game level.
2. **Feature-token iTransformer path:** treats each semantic field as one token. Each field is first projected along its valid, padding-aware temporal axis, after which self-attention models interactions among feature tokens such as type, location, spin, and score.

The two paths are combined at each hierarchy level through sigmoid-gated fusion. The final configuration also uses shot-aware positional encoding, top-down attention, turn-based player-role routing, and a one-shot gated skip connection. Task Attention Fusion learns an independent query for each prediction task and retrieves task-specific information from hierarchy summaries and hitter/opponent representations.

## Results

The following results use the original table-tennis test set, a fixed match-level split, `d_model=256`, and seeds `42`, `123`, and `2024`. Values are mean +/- sample standard deviation. Lower Location Distance is better.

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

ShuttleNet-adapted uses only the current-rally L1 sequence while retaining its encoder-decoder structure, player branches, and position-aware gated fusion. PatchTST-adapted flattens the available hierarchy and constructs temporal patches for each semantic feature channel. These results apply to the adapted implementations in this repository and are not results on the original papers' tasks.

Cross-dataset majority-class and Final MT-HTA results:

| Dataset | Majority Macro F1 | Final MT-HTA Macro F1 |
| :--- | ---: | ---: |
| Original Table Tennis | 0.2532 | **0.5315 +/- 0.0047** |
| Expanded Table Tennis | 0.2379 | **0.5525 +/- 0.0007** |
| Badminton ShuttleSet | 0.2175 | **0.5309 +/- 0.0028** |
| Tennis MCP | 0.5349 | **0.7095 +/- 0.0064** |

## Environment

The verified environment uses Python 3.9, PyTorch 2.3, and CUDA 12.1. Run all commands from the repository root. The following commands can be pasted directly into Windows `cmd.exe`:

```bat
conda env create -f environment.yml
conda activate mt-hta
```

To install the dependencies into an existing Python environment instead:

```bat
python -m pip install -r requirements.txt
```

Verify the PyTorch installation:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data Preparation

The available CSV files and their intended uses are listed in [data/README.md](data/README.md). Processed files must be generated before training.

Original table tennis, used for the main experiments:

```bat
python preprocessing/TableTennis_preprocess.py
```

Expanded table tennis:

```bat
python preprocessing/TableTennis_preprocess.py --variant all
```

Combined badminton, used for the main cross-sport experiments:

```bat
python preprocessing/Badminton_preprocess.py
```

Original badminton supplementary variant:

```bat
python preprocessing/Badminton_preprocess.py --variant original
```

Tennis Top100, used for the main cross-sport experiments:

```bat
python preprocessing/Tennis_preprocess.py
```

All preprocessing entry points support dataset-specific `--variant` choices and the following common options:

```text
--input INPUT_CSV --output-dir OUTPUT_DIRECTORY
--seed 42 --val-ratio 0.1 --test-ratio 0.2
```

Splits are created at the match level, so a match cannot occur in more than one of the training, validation, and test sets. The reported experiments use preprocessing seed `42`. Model-training seeds do not recreate the data split.

## Training Final MT-HTA

Train one seed:

```bat
python scripts/train_location_loss.py --sport table_tennis --model_type task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating --seed 42
```

Train and evaluate the three reported seeds from an interactive `cmd.exe` session:

```bat
for %S in (42 123 2024) do python scripts/run_all_models.py --sport table_tennis --models task_attention_itransformer_feature_token --d_model 256 --skip_window_size 1 --use_shot_aware_pe --use_gated_fusion --use_top_down_attention --use_turn_based_gating --seed %S
```

When placing this command in a `.bat` file, replace `%S` with `%%S`. Equal task weights of `0.2` are already defined in `configs/table_tennis.yaml`; no command-line JSON is required.

The remaining reported training settings come from `configs/table_tennis.yaml` and the training defaults: 40 epochs, batch size 128, AdamW with a learning rate of `1e-4`, weight decay `0.01`, 10% warm-up, gradient clipping at `1.0`, label smoothing `0.1`, and Location Expected Distance weight `1.0`. Formal evaluation uses `best_model.pth`, selected by the lowest validation loss. `run_all_models.py` performs training followed by testing.

## Testing and Baselines

To test an existing run, replace `RUN_DIRECTORY` with the actual run directory name:

```bat
python scripts/test_location_loss.py --sport table_tennis --run_dir outputs\results\table_tennis\RUN_DIRECTORY
```

Run the main baselines for seed 42:

```bat
python scripts/run_all_models.py --sport table_tennis --models baseline_lstm baseline_lstm_context baseline_h_lstm baseline_shuttlenet_full baseline_transformer_flat baseline_itransformer_feature_token --d_model 256 --seed 42
```

Run PatchTST-adapted for the three reported seeds:

```bat
for %S in (42 123 2024) do python scripts/run_all_models.py --sport table_tennis --models baseline_patchtst --d_model 256 --seed %S
```

Available model IDs:

| Model ID | Purpose |
| :--- | :--- |
| `task_attention_itransformer_feature_token` | Official model ID shared by MT-HTA Core and Final MT-HTA; the final-module flags shown above enable the complete configuration |
| `task_attention` | Alias for the dual-path MT-HTA Core Task Attention configuration |
| `cls_token_itransformer_feature_token` | CLS-token fusion ablation |
| `task_project_itransformer_feature_token` | Task-projection fusion ablation |
| `task_attention_L1`, `task_attention_L1_L2` | Hierarchy-depth ablations |
| `task_attention_wo_itransformer`, `task_attention_final_wo_itransformer` | Hierarchical Transformer-only ablations |
| `sequence_attention` | Sequence-level task-attention variant |
| `baseline_lstm`, `baseline_lstm_context`, `baseline_lstm_flat`, `baseline_h_lstm` | Recurrent baselines |
| `baseline_transformer_flat` | Flat temporal Transformer |
| `baseline_itransformer_feature_token` | Standalone FT-iTransformer |
| `baseline_patchtst` | PatchTST-adapted baseline |
| `baseline_shuttlenet_full` | ShuttleNet-adapted baseline |

## Reproducibility and Evaluation Notes

- The preprocessing seed controls the match-level split. Training seeds `42`, `123`, and `2024` control model initialization, DataLoader shuffling, dropout, and other PyTorch random operations.
- Test DataLoaders use `shuffle=False`. Under deterministic evaluation, repeatedly evaluating the same checkpoint and data should not depend on the training seed.
- Each run is stored under `outputs/results/SPORT/run_SPORT_MODEL_TIMESTAMP/` with its configuration, best/last checkpoints, TensorBoard logs, and test metrics. These generated artifacts are not tracked by Git.
- For targets inside the 3x3 grid, Location Expected Distance Loss computes the expected Euclidean distance between the normalized in-grid prediction probabilities and the true grid cell, then combines it with cross-entropy loss.
- Location Average Distance only includes samples for which both the target and the argmax prediction are valid grid classes. A grid target predicted as a non-grid class is excluded from this average, so the metric should be interpreted together with Location F1 and accuracy.

## Repository Layout

```text
configs/                      sport-specific features, targets, and defaults
data/                         publishable CSV inputs; generated files are ignored
docs/                         supplementary usage documentation and README figures
preprocessing/                match-level data preparation
scripts/                      training, testing, runners, and analyses
src/model_components.py       shared MT-HTA components
src/models/model_fuse.py      Final MT-HTA and framework ablations
src/models/                   adapted and general baselines
utils/                        shared base model and evaluator
Final MT-HTA Architecture.pdf architecture figure
```
