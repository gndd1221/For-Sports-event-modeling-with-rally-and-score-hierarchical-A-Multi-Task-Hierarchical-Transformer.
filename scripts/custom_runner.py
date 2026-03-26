"""
本腳本讓您可以自訂一系列實驗，程式會自動：
1. 執行訓練 (Train)
2. 訓練完後等待 3 秒 (釋放顯存)
3. 自動找到剛剛產生的結果資料夾
4. 執行測試 (Test) 並產生報表

=====================================================================================
完整參數指南 (Full Parameter Guide) - 括號內為預設值
=====================================================================================

【基本與路徑設定】
- 'model_type'         : 模型變體 (預設: 'task_attention')
    - 可選: 'sequence_attention', 'task_attention', 'parallel', 'task_project', 'L1', 'L1_L2'
- 'sport'              : 運動類型 (預設: 'table_tennis',有'badminton','tennis')
- 'data_dir'            : 資料路徑 (預設: 'data/table_tennis/processed_data')
- 'base_config_file'    : 設定檔路徑 (預設: 來自 YAML 設定)
- 'results_base_dir'    : 結果儲存根目錄 (預設: 'outputs/results/table_tennis')
- 'seed'               : 隨機種子 (預設: 42)

【訓練超參數】
- 'epochs'             : 訓練輪數 (桌球預設: 40)
- 'batch_size'         : 批次大小 (桌球預設: 128)
- 'learning_rate'      : 學習率 (桌球預設: 0.0001)
- 'weight_decay'       : 權重衰減 (預設: 0.01)
- 'warmup_ratio'       : 學習率預熱比例 (預設: 0.1)
- 'clip'               : 梯度裁剪 (預設: 1.0)
- 'label_smoothing'    : 標籤平滑 (桌球預設: 0.1)
- 'num_workers'        : 資料載入執行緒 (Windows 預設 0, Linux 預設 4)

【損失函數設定】
- 'lambda_dist'        : 位置距離損失權重 (桌球預設: 1.0)
- 'grid_offset'        : 位置坐標偏移 (桌球預設: 2)
- 'use_rlw'            : 是否啟用隨機損失權重 (預設: False)
- 'loss_weights'       : 各任務手動權重 (JSON格式，例如: '{"type":0.2, "location":0.2, ...}')

【Transformer 架構設定】
- 'd_model'            : 隱藏層維度 (預設: 128)
- 'nhead'              : 注意力頭數 (預設: 4)
- 'num_encoder_layers' : 編碼器層數 (預設: 2)
- 'dim_feedforward'    : 前饋層維度 (預設: 256)
- 'dropout'            : 防止過擬和 (預設: 0.3)
- 'player_embedding_dim': 選手嵌入維度 (預設: 32)
- 'pooling_type'       : 序列特徵聚合 ('last', 'mean', 'attention') (預設: 'last')
- 'head_depth'         : 預測頭 MLP 深度 (預設: 1)

【進階功能 (消融實驗)】
- 'skip_window_size'   : 跳轉連接窗口大小 (預設: 0 表示關閉)
- 'use_gated_fusion'   : 是否啟用門控融合 (預設: False)
- 'use_shot_aware_pe'  : 是否啟用拍號感知位置編碼 (預設: False)
- 'use_skip_connection': 是否啟用跳轉連接 (等同於 skip_window_size=1) (預設: False)

=====================================================================================
"""

import subprocess
import sys
import os
import glob
import time
import json

# 嘗試讀取 YAML 以獲取精確路徑
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =====================================================================================
# 1. 實驗排程設定區域
# =====================================================================================
EXPERIMENTS = [
    {
        "model_type": "sequence_attention",
        "skip_window_size": 3,
        "epochs": 50,
        "use_gated_fusion": True,
        "use_shot_aware_pe": True
    },
    {
        "model_type": "task_attention",
        "skip_window_size": 1,
        "epochs": 50,
    },
]

# 預設通用設定
DEFAULT_SPORT = "table_tennis"

# =====================================================================================
# 2. 自動執行邏輯
# =====================================================================================

def get_sport_results_dir(sport):
    """取得該運動對應的結果路徑，優先讀取 YAML 設定"""
    if HAS_YAML:
        cfg_path = os.path.join("configs", f"{sport}.yaml")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    return data.get('results_dir', f"outputs/results/{sport}")
            except:
                pass
    return f"outputs/results/{sport}"

def get_latest_run_dir(model_type, sport):
    """搜尋指定模型最新的結果資料夾"""
    res_dir = get_sport_results_dir(sport)
    # 根據命名規則搜尋
    pattern = os.path.join(res_dir, f"run_{sport}_{model_type}_*")
    folders = glob.glob(pattern)
    
    if not folders:
        # 嘗試無 sport 前綴的格式
        pattern_alt = os.path.join(res_dir, f"run_{model_type}_*")
        folders = glob.glob(pattern_alt)
        
    if not folders:
        return None
        
    return sorted(folders)[-1]

def run_experiment_pipeline():
    python_exe = sys.executable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    total_experiments = len(EXPERIMENTS)
    print(f"\n{'#'*80}")
    print(f"#  開始自動化實驗流程 - 共 {total_experiments} 個項目")
    print(f"{'#'*80}")

    for i, cfg in enumerate(EXPERIMENTS, 1):
        model_name = cfg.get("model_type", "task_attention")
        sport = cfg.get("sport", DEFAULT_SPORT)
        
        print(f"\n▶️ [實驗 {i}/{total_experiments}] 正在處理模型: {model_name} ({sport})")
        
        # --- (A) 執行訓練 ---
        train_cmd = [python_exe, "scripts/train_location_loss.py"]
        
        for key, value in cfg.items():
            if isinstance(value, bool):
                if value: train_cmd.append(f"--{key}")
            else:
                train_cmd.extend([f"--{key}", str(value)])
        
        if "sport" not in cfg:
            train_cmd.extend(["--sport", sport])
        
        print(f"   [Step 1] 正在訓練... \n   指令: {' '.join(train_cmd)}")
        train_result = subprocess.run(train_cmd, cwd=project_root)
        
        if train_result.returncode != 0:
            print(f"   ❌ 訓練失敗，跳過此項目的測試。")
            continue
            
        # --- (B) 等待並尋找資料夾 ---
        print(f"   [Step 2] 訓練成功，等待顯存釋放...")
        time.sleep(3)
        
        run_dir = get_latest_run_dir(model_name, sport)
        
        if not run_dir:
            print(f"   ⚠️ 錯誤: 找不到剛產生的結果目錄。")
            continue
            
        # --- (C) 執行測試 ---
        print(f"   [Step 3] 發現目錄: {run_dir}，開始測試評估...")
        test_cmd = [
            python_exe, "scripts/test_location_loss.py",
            "--run_dir", run_dir,
            "--sport", sport
        ]
        
        if "lambda_dist" in cfg:
            test_cmd.extend(["--lambda_dist", str(cfg["lambda_dist"])])
            
        test_result = subprocess.run(test_cmd, cwd=project_root)
        
        if test_result.returncode == 0:
            print(f"   ✅ 實驗 {i} 全部流程完成！")
        else:
            print(f"   ❌ 測試階段失敗。")

    print(f"\n{'#'*80}")
    print(f"#  所有實驗總表執行完畢！")
    print(f"{'#'*80}")

if __name__ == "__main__":
    run_experiment_pipeline()
