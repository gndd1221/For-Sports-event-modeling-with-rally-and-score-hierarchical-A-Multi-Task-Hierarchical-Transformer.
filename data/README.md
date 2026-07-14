# Data

本目錄保留可用於重建論文實驗資料的 CSV。`preprocessing/` 會在 match 層級切分 train/validation/test，產生的 `.pkl` 與 `.json` 均由 Git 忽略。

| Sport / variant | CSV | Default processed output | Thesis usage |
| :--- | :--- | :--- | :--- |
| Table tennis (original) | `table_tennis/TableTennis_dataset.csv` | `table_tennis/processed_data/` | 主要實驗、baseline、ablation、rollout 與 case study |
| Table tennis (expanded) | `table_tennis/TableTennis_dataset_all.csv` | `table_tennis/processed_data_all/` | Expanded table-tennis 實驗 |
| Badminton (original) | `badminton/badminton_dataset.csv` | `badminton/processed_data_badminton/` | 補充版本 |
| Badminton (combined/all) | `badminton/badminton_dataset_all.csv` | `badminton/processed_data_badminton_all/` | 正式跨運動實驗 |
| Tennis Top100 | `tennis/Tennis_Converted_Top100.csv` | `tennis/Tennis_processed_data/` | 正式跨運動實驗 |
| Tennis Top200 | `tennis/Tennis_Converted_Top200.csv` | `tennis/Tennis_processed_data_top200/` | 補充版本，未列入主要結果表 |

## Provenance

- Table tennis: professional shot-by-shot annotations exported from the 3S technical and tactical statistics system.
- Badminton: processed and combined ShuttleSet and ShuttleSet22 stroke annotations.
- Tennis: converted point-by-point records from the Match Charting Project (MCP).

資料不因本 repository 採用 MIT License 而重新授權。公開或再散布前，使用者應自行確認各來源的授權、引用與個資要求。
