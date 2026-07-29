# Data

This folder contains the CSV files to reproduce the paper's experiments. `preprocessing/` splits the train/validation/test sets at the match level; the generated `.pkl` and `.json` files are ignored by git.

| Sport / variant | CSV | Default processed output | Thesis usage |
| :--- | :--- | :--- | :--- |
| Table tennis (original) | `table_tennis/TableTennis_dataset.csv` | `table_tennis/processed_data/` | main experiment, baseline, ablation, rollout, case study |
| Table tennis (expanded) | `table_tennis/TableTennis_dataset_all.csv` | `table_tennis/processed_data_all/` | Expanded table-tennis experiments |
| Badminton (original) | `badminton/badminton_dataset.csv` | `badminton/processed_data_badminton/` | affiliated results |
| Badminton (combined/all) | `badminton/badminton_dataset_all.csv` | `badminton/processed_data_badminton_all/` | cross-sport experiments |
| Tennis Top100 | `tennis/Tennis_Converted_Top100.csv` | `tennis/Tennis_processed_data/` | cross-sport experiments |
| Tennis Top200 | `tennis/Tennis_Converted_Top200.csv` | `tennis/Tennis_processed_data_top200/` | affilated results |

## Provenance

- Table tennis: professional shot-by-shot annotations exported from the 3S technical and tactical statistics system.
- Badminton: processed and combined ShuttleSet and ShuttleSet22 stroke annotations.
- Tennis: converted point-by-point records from the Match Charting Project (MCP).
