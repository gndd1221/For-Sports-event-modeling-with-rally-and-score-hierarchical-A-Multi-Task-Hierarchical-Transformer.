"""
run_all_models.py — 一鍵訓練與測試所有模型變體

用法：
    python run_all_models.py                         # 訓練+測試所有 5 種模型
    python run_all_models.py --models task_attention L1  # 只跑指定模型
    python run_all_models.py --skip_train             # 跳過訓練，只測試 (需指定 --checkpoints_base_dir)
    python run_all_models.py --skip_test              # 只訓練，不測試
    python run_all_models.py --epochs 10              # 傳遞額外參數給訓練腳本
"""

import subprocess
import sys
import os
import time
import glob
import argparse
import csv

# 修正 Windows CMD 預設 cp950 無法印出 Emoji 的問題
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 確保專案根目錄在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.default_config import AVAILABLE_SPORTS


# 所有可用的模型變體
ALL_MODELS = ['sequence_attention', 'parallel', 'task_project', 'task_attention', 'L1_L2', 'L1']


def find_latest_run_dir(results_base_dir, model_type, sport=None):
    """找到指定 model_type 最新的訓練結果資料夾"""
    if sport:
        pattern = os.path.join(results_base_dir, f'run_{sport}_{model_type}_*')
        prefix = f'run_{sport}_{model_type}_'
    else:
        pattern = os.path.join(results_base_dir, f'run_{model_type}_*')
        prefix = f'run_{model_type}_'
        
    potential_dirs = glob.glob(pattern)
    matching_dirs = []
    
    # 嚴格驗證結尾是否正好是 YYYYMMDD_HHMMSS，避免 L1 匹配到 L1_L2 的資料夾
    for d in potential_dirs:
        folder_name = os.path.basename(d)
        if folder_name.startswith(prefix):
            suffix = folder_name[len(prefix):]
            if len(suffix) == 15 and suffix[:8].isdigit() and suffix[8] == '_' and suffix[9:].isdigit():
                matching_dirs.append(d)
                
    matching_dirs = sorted(matching_dirs)
    if matching_dirs:
        return matching_dirs[-1]
        
    # fallback: 不含 sport 前綴的舊格式
    if sport:
        pattern_legacy = os.path.join(results_base_dir, f'train_{sport}_{model_type}_*')
        prefix_legacy = f'train_{sport}_{model_type}_'
        potential_legacy = glob.glob(pattern_legacy)
        matching_legacy = []
        for d in potential_legacy:
            folder_name = os.path.basename(d)
            if folder_name.startswith(prefix_legacy):
                suffix = folder_name[len(prefix_legacy):]
                if len(suffix) == 15 and suffix[:8].isdigit() and suffix[8] == '_' and suffix[9:].isdigit():
                    matching_legacy.append(d)
        
        matching_legacy = sorted(matching_legacy)
        if matching_legacy:
            return matching_legacy[-1]
            
    return None


def run_command(cmd, description):
    """執行命令並即時顯示輸出"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if result.returncode != 0:
        print(f"\n[FAIL] 失敗: {description} (exit code: {result.returncode})")
        return False
    
    print(f"\n[SUCCESS] 完成: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="一鍵訓練與測試所有模型變體",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python run_all_models.py                                  # 所有模型，預設參數
  python run_all_models.py --models task_attention L1       # 只跑 2 種
  python run_all_models.py --epochs 10 --batch_size 64      # 自訂訓練參數
  python run_all_models.py --skip_train                     # 只測試已訓練的模型
        """
    )
    
    # 控制參數
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help=f"要執行的模型列表 (預設: 全部)")
    parser.add_argument('--skip_train', action='store_true',
                        help="跳過訓練，只執行測試")
    parser.add_argument('--skip_test', action='store_true',
                        help="跳過測試，只執行訓練")
    parser.add_argument('--results_base_dir', type=str, default=None,
                        help="結果根目錄 (不指定時從 sport config 取得)")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="資料目錄 (不指定時從 sport config 取得)")
    parser.add_argument('--sport', type=str, default='table_tennis',
                        choices=AVAILABLE_SPORTS,
                        help=f"運動類型 (預設: table_tennis, 可選: {AVAILABLE_SPORTS})")
    
    # 訓練超參數 (傳遞給 train_location_loss.py，None 表示使用 YAML 預設值)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--lambda_dist', type=float, default=None)
    parser.add_argument('--label_smoothing', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--warmup_ratio', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_encoder_layers', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--player_embedding_dim', type=int, default=None)
    parser.add_argument('--pooling_type', type=str, default=None,
                        choices=['last', 'mean', 'attention'])
    parser.add_argument('--head_depth', type=int, default=None)
    parser.add_argument('--loss_weights', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (預設 42)')
    parser.add_argument('--use_rlw', action='store_true',
                        help='啟用 RLW (Random Loss Weighting)')
    parser.add_argument('--use_skip_connection', action='store_true',
                        help='啟用 Gated Skip Connection (等同於 --skip_window_size 1)')
    parser.add_argument('--skip_window_size', type=int, default=None,
                        help='Skip Connection 聚合的最後 N 拍 (預設 0=關閉, 1=單拍, >1=多拍局部池化)')
    parser.add_argument('--use_gated_fusion', action='store_true',
                        help='啟用門控 PACT-iTransformer 融合 (取代原本的 Concatenate+Linear)')
    parser.add_argument('--use_shot_aware_pe', action='store_true',
                        help='啟用 Shot-Aware Positional Encoding (注入發球/我方拍語義)')
    parser.add_argument('--use_top_down_attention', action='store_true',
                        help='啟用 Top-Down Cross-Attention (由高層向低層序列提煉摘要)')
    parser.add_argument('--use_turn_based_gating', action='store_true',
                        help='啟用 Turn-Based Style Gating (根據出手權動態混合球員風格)')
    parser.add_argument('--use_temporal_scale_gating', action='store_true',
                        help='啟用 Temporal-Scale Adaptive Gating (根據拍數動態調整階層信任權重)')
    
    args = parser.parse_args()
    
    python_cmd = sys.executable
    models_to_run = args.models
    results = {}

    # 從 sport config 取得預設路徑
    from src.default_config import load_sport_config
    sport_config = load_sport_config(args.sport)
    sport_train_args = sport_config.get('training_args', {})
    results_base_dir = args.results_base_dir or sport_config.get('results_dir', f'results/{args.sport}')
    
    # 顯示實際使用的值 (CLI 覆寫 > YAML 預設)
    display_epochs = args.epochs if args.epochs is not None else sport_train_args.get('epochs', '(YAML)')
    display_batch = args.batch_size if args.batch_size is not None else sport_train_args.get('batch_size', '(YAML)')
    display_lr = args.learning_rate if args.learning_rate is not None else sport_train_args.get('learning_rate', '(YAML)')
    
    print(f"\n{'#'*70}")
    print(f"#  PACT 模型批次執行器")
    print(f"#  要執行的模型: {', '.join(models_to_run)}")
    print(f"#  訓練: {'跳過' if args.skip_train else '是'}")
    print(f"#  測試: {'跳過' if args.skip_test else '是'}")
    print(f"#  Epochs: {display_epochs}  |  Batch: {display_batch}  |  LR: {display_lr}")
    print(f"{'#'*70}")
    
    total_start = time.time()
    
    for i, model_type in enumerate(models_to_run, 1):
        model_start = time.time()
        print(f"\n\n{'*'*70}")
        print(f"*  [{i}/{len(models_to_run)}] 模型: {model_type}")
        print(f"{'*'*70}")
        
        train_success = True
        test_success = True
        run_dir = None
        
        # ===== 訓練 =====
        if not args.skip_train:
            train_cmd = [
                python_cmd, 'scripts/train_location_loss.py',
                '--model_type', model_type,
                '--sport', args.sport,
            ]
            # 只在 CLI 明確指定時才傳遞參數，否則讓 train_location_loss.py 從 YAML 讀取
            optional_args = {
                '--epochs': args.epochs,
                '--batch_size': args.batch_size,
                '--learning_rate': args.learning_rate,
                '--lambda_dist': args.lambda_dist,
                '--label_smoothing': args.label_smoothing,
                '--weight_decay': args.weight_decay,
                '--warmup_ratio': args.warmup_ratio,
                '--num_workers': args.num_workers,
                '--d_model': args.d_model,
                '--nhead': args.nhead,
                '--num_encoder_layers': args.num_encoder_layers,
                '--dim_feedforward': args.dim_feedforward,
                '--dropout': args.dropout,
                '--player_embedding_dim': args.player_embedding_dim,
                '--pooling_type': args.pooling_type,
                '--head_depth': args.head_depth,
                '--results_base_dir': args.results_base_dir,
                '--data_dir': args.data_dir,
                '--loss_weights': args.loss_weights,
                '--seed': args.seed,
                '--skip_window_size': args.skip_window_size,
            }
            for flag, value in optional_args.items():
                if value is not None:
                    train_cmd.extend([flag, str(value)])
            
            # Boolean flags (store_true 類型)
            if args.use_rlw:
                train_cmd.append('--use_rlw')
            if args.use_skip_connection:
                train_cmd.append('--use_skip_connection')
            if args.use_gated_fusion:
                train_cmd.append('--use_gated_fusion')
            if args.use_shot_aware_pe:
                train_cmd.append('--use_shot_aware_pe')
            if args.use_top_down_attention:
                train_cmd.append('--use_top_down_attention')
            if args.use_turn_based_gating:
                train_cmd.append('--use_turn_based_gating')
            if args.use_temporal_scale_gating:
                train_cmd.append('--use_temporal_scale_gating')
            
            train_success = run_command(train_cmd, f"訓練 {model_type} ({args.sport})")
        
        # ===== 測試 =====
        if not args.skip_test and train_success:
            # Windows 環境下，PyTorch subprocess 結束後顯存釋放可能會有幾百毫秒的延遲
            # 加上短暫休眠，避免緊接著的 test_location_loss.py 發生 CUDA OOM
            if not args.skip_train:
                time.sleep(3)
                
            # 找到最新的 run_dir
            res_dir = results_base_dir
            run_dir = find_latest_run_dir(res_dir, model_type, args.sport)
            
            if run_dir is None:
                print(f"\n[WARN] 找不到 {model_type} 的訓練結果，跳過測試")
                test_success = False
            else:
                test_cmd = [
                    python_cmd, 'scripts/test_location_loss.py',
                    '--run_dir', run_dir,
                    '--sport', args.sport,
                ]
                if args.data_dir:
                    test_cmd.extend(['--data_dir', args.data_dir])
                test_success = run_command(test_cmd, f"測試 {model_type} ({os.path.basename(run_dir)})")
        
        model_elapsed = time.time() - model_start
        results[model_type] = {
            'train': '[PASS]' if train_success else '[FAIL]',
            'test': '[PASS]' if test_success else ('[SKIP]' if args.skip_test else '[FAIL]'),
            'run_dir': run_dir or '—',
            'time': f'{model_elapsed/60:.1f} min'
        }
    
    # ===== 結果彙整 =====
    total_elapsed = time.time() - total_start
    
    print(f"\n\n{'='*70}")
    print(f"  執行結果彙整  (總耗時: {total_elapsed/60:.1f} 分鐘)")
    print(f"{'='*70}")
    print(f"{'模型':<20} {'訓練':<6} {'測試':<6} {'耗時':<12} {'結果目錄'}")
    print(f"{'-'*70}")
    for model_type, result in results.items():
        dir_name = os.path.basename(result['run_dir']) if result['run_dir'] != '—' else '—'
        print(f"{model_type:<20} {result['train']:<6} {result['test']:<6} {result['time']:<12} {dir_name}")
    print(f"{'='*70}")

    # ===== CSV 彙總 =====
    if not args.skip_test:
        all_csv_rows = []
        # 基本欄位順序 (確保核心指標在前)
        base_fieldnames = ['model_type', 'sport', 'task', 'accuracy', 'precision', 'recall',
                      'f1', 'roc_auc_weighted', 'roc_auc_micro', 'loss']
        extra_fieldnames = set()
        
        for model_type, result in results.items():
            run_dir = result.get('run_dir', '—')
            if run_dir == '—':
                continue
            csv_file = os.path.join(run_dir, 'test', 'evaluation_results.csv')
            if os.path.exists(csv_file):
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_csv_rows.append(row)
                        # 收集所有出現過的額外欄位 (如 avg_distance)
                        extra_fieldnames.update(k for k in row.keys() if k not in base_fieldnames)
        
        if all_csv_rows:
            fieldnames = base_fieldnames + sorted(extra_fieldnames)
            summary_path = os.path.join(results_base_dir, 'results_summary.csv')
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_csv_rows)
            
            print(f"\n📊 結果彙總 CSV 已儲存: {summary_path}")
            print(f"\n{'模型':<18} {'任務':<12} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
            print(f"{'-'*60}")
            for row in all_csv_rows:
                print(f"{row['model_type']:<18} {row['task']:<12} {row['accuracy']:>7} {row['precision']:>7} {row['recall']:>7} {row['f1']:>7}")


if __name__ == '__main__':
    main()
