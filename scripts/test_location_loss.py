import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
import os
import sys
import csv
from tqdm import tqdm
import numpy as np

# 確保專案根目錄在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.dataloader import get_dataloader

from src.default_config import (
    get_model_class, load_sport_config, AVAILABLE_SPORTS,
    DEFAULT_LABEL_SMOOTHING, DEFAULT_GRID_OFFSET
)
from src.losses import build_criterion_dict, GridDistanceCalculator

# 導入機器學習評估工具 (經由 Evaluator)
from utils.evaluator import ModelEvaluator

# =====================================================================================
# 評估函數 (整合 Distance 與 ModelEvaluator)
# =====================================================================================
def evaluate_model(model, test_loader, config, device, run_dir, output_dir, writer, lambda_dist_override=None):
    model.eval()
    
    training_args = config.get('training_args', {})
    
    # 決定 Loss 參數
    lambda_dist = lambda_dist_override if lambda_dist_override is not None else training_args.get('lambda_dist', 1.0)
    
    # 防止 override 丟失，把覆寫的 Lambda 寫回 config 給 Loss Registry 吃
    config.setdefault('training_args', {})['lambda_dist'] = lambda_dist
    
    # 動態產生 Registry
    criterions = build_criterion_dict(config, device=device)
    grid_calculator = GridDistanceCalculator(
        grid_width=3, 
        grid_offset=training_args.get('grid_offset', DEFAULT_GRID_OFFSET),
        device=device
    ) if ('location' in config.get('targets', []) and training_args.get('use_distance_loss', True)) else None

    print(f"Testing with Lambda Distance: {lambda_dist}")
    print(f"Results will be saved to: {output_dir}")

    total_loss = 0
    # 儲存預測結果供後續計算 ROC/CM
    all_preds = {task: [] for task in config['targets']}
    all_targets = {task: [] for task in config['targets']}
    all_probs = {task: [] for task in config['targets']}
    
    # 記錄每個任務的 loss
    task_losses_accum = {task: 0 for task in config['targets']}
    total_ce_loss = 0
    total_dist_loss = 0
    
    batch_count = 0
    
    # 用於計算 Location 平均距離誤差
    total_grid_distance = 0.0
    total_grid_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            logits = model(batch)
            targets = batch['targets']
            
            batch_loss = 0
            for task, pred in logits.items():
                if task not in targets or task not in criterions: continue
                
                target = targets[task].to(device)
                
                # --- Loss 計算 (透過 Registry) ---
                loss_output = criterions[task](pred, target)
                if isinstance(loss_output, tuple):
                    loss, ce, dist = loss_output
                    total_ce_loss += ce.item()
                    total_dist_loss += dist.item() if isinstance(dist, torch.Tensor) else dist
                else:
                    loss = loss_output
                
                # --- Distance 計算 (Metrics) ---
                pred_idx = torch.argmax(pred, dim=1)
                if task == 'location' and grid_calculator:
                    t_dist, t_cnt = grid_calculator.calculate_distance(pred_idx, target)
                    total_grid_distance += t_dist
                    total_grid_count += t_cnt
                
                batch_loss += loss.item()
                task_losses_accum[task] += loss.item()
                
                # 收集結果
                probs = F.softmax(pred, dim=1)
                _, predicted = torch.max(pred, 1)
                
                # 移除 padding (0) 以便進行準確評估
                mask = target != 0
                all_preds[task].extend(predicted[mask].cpu().numpy())
                all_targets[task].extend(target[mask].cpu().numpy())
                all_probs[task].extend(probs[mask].cpu().numpy())
            
            total_loss += batch_loss
            batch_count += 1

    avg_loss = total_loss / max(batch_count, 1)
    avg_grid_distance = total_grid_distance / total_grid_count if total_grid_count > 0 else 0.0
    avg_task_losses = {task: t_loss / max(batch_count, 1) for task, t_loss in task_losses_accum.items()}
    avg_ce_loss = total_ce_loss / max(batch_count, 1)
    avg_dist_loss = total_dist_loss / max(batch_count, 1)
    
    # 計算 Total Loss (所有任務 loss 加總)
    total_task_loss_sum = sum(avg_task_losses.values())
    
    writer.add_scalar('Loss/Total_Loss', avg_loss, 0)
    writer.add_scalar('Loss/Total_Task_Loss_Sum', total_task_loss_sum, 0)
    writer.add_scalar('Loss/CE_Loss', avg_ce_loss, 0)
    writer.add_scalar('Loss/Dist_Loss', avg_dist_loss, 0)
    for task, t_loss in avg_task_losses.items():
        writer.add_scalar(f'Loss/{task}', t_loss, 0)
    writer.add_scalar('Metric/Location_Distance', avg_grid_distance, 0)
    
    # 準備 Custom Metrics (例如 Location Distance) 給 Evaluator 寫進報表
    custom_metrics = {}
    if total_grid_count > 0:
        custom_metrics['location'] = {'avg_distance': avg_grid_distance}
        
    model_type = config.get('training_args', {}).get('model_type', 'unknown')

    # 交給 Evaluator 完成產出
    evaluator = ModelEvaluator(output_dir, writer)
    evaluator.evaluate(
        config=config,
        all_targets=all_targets,
        all_preds=all_preds,
        all_probs=all_probs,
        avg_task_losses=avg_task_losses,
        custom_metrics=custom_metrics,
        model_type=model_type,
        step=0
    )

    print(f"評估完成。結果與圖片已儲存至資料夾: {output_dir}")
    if total_grid_count > 0:
        print(f"Location Avg Distance: {avg_grid_distance:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="模型資料夾路徑")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="資料夾路徑 (不指定時從 sport config 取得)")
    parser.add_argument('--sport', type=str, default='table_tennis',
                        choices=AVAILABLE_SPORTS,
                        help=f"運動類型 (預設 table_tennis)")
    
    # 可覆寫超參數
    parser.add_argument('--lambda_dist', type=float, default=None, help="覆寫 Lambda Distance")
    parser.add_argument('--model_type', type=str, default=None,
                        help="覆寫模型類型 (若不填則從 config.json 讀取)")
    
    args = parser.parse_args()

    run_id = os.path.basename(args.run_dir)
    log_path = os.path.join(args.run_dir, 'test', 'logs')
    output_dir = os.path.join(args.run_dir, 'test')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    print(f"注意: 所有評估結果 (TXT, PNG, TensorBoard Logs) 將統一儲存於: {output_dir}")

    writer = SummaryWriter(log_dir=log_path)

    config_path = os.path.join(args.run_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"錯誤: 找不到 config.json 在 {args.run_dir}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 決定模型類型：命令列參數 > config 中的設定 > 預設 task_attention
    model_type = args.model_type or config.get('training_args', {}).get('model_type', 'task_attention')
    
    PACTModel, config_overrides = get_model_class(model_type)
    
    # Preserve the model structure from saved config (matches trained checkpoint)
    # Priority: model_args.hierarchy_levels (new format) > root hierarchy_levels (old format)
    saved_model_args = config.get('model_args', {})
    saved_hierarchy = saved_model_args.get('hierarchy_levels') or config.get('hierarchy_levels')
    saved_fusion_type = saved_model_args.get('fusion_type')
    
    config.setdefault('model_args', {}).update(config_overrides)
    
    # Restore saved model structure to ensure it matches the checkpoint
    if saved_hierarchy:
        config['model_args']['hierarchy_levels'] = saved_hierarchy
    if saved_fusion_type:
        config['model_args']['fusion_type'] = saved_fusion_type
    model = PACTModel(config).to(device)
    model_path = os.path.join(args.run_dir, 'train', 'weights', 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"找不到 best_model.pth，嘗試載入 last_model.pth")
        model_path = os.path.join(args.run_dir, 'train', 'weights', 'last_model.pth')
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"已載入模型: {model_path} (類型: {model_type})")

    # 決定 data_dir: CLI > config > sport config > 預設
    sport_cfg = load_sport_config(args.sport)
    data_dir = args.data_dir or config.get('data_args', {}).get('data_dir') or sport_cfg.get('data_dir', 'processed_data')

    test_dataset, collate_fn = get_dataloader(
        args.sport,
        data_path=os.path.join(data_dir, 'test_data.pkl'),
        config_path=os.path.join(data_dir, 'config.json')
    )
    test_loader = DataLoader(test_dataset, batch_size=config['training_args']['batch_size'], 
                             shuffle=False, collate_fn=collate_fn)
    
    evaluate_model(model, test_loader, config, device, args.run_dir, output_dir, writer, args.lambda_dist)
    
    writer.close()

if __name__ == '__main__':
    main()