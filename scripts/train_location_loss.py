import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
import os
import sys
import random
import numpy as np
from tqdm import tqdm
import time

# 確保專案根目錄在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 導入自定義模組
from src.default_config import (
    get_model_class, build_feature_config, build_full_config, load_sport_config,
    AVAILABLE_SPORTS,
    DEFAULT_LOSS_WEIGHTS, DEFAULT_LABEL_SMOOTHING,
    DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_RATIO,
    DEFAULT_GRID_OFFSET, DEFAULT_NUM_WORKERS
)
from src.losses import build_criterion_dict, GridDistanceCalculator
from src.dataloader import get_dataloader

# 導入 AdamW 和 Cosine Schedule
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# =====================================================================================
# Random Seed 控制
# =====================================================================================
def set_seed(seed):
    """統一設定所有隨機源的 seed，確保訓練可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 環境
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# =====================================================================================
# Trainer Class
# =====================================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_dir = run_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        training_args = config['training_args']
        
        # 使用 Task Registry 動態生成所有任務的 Loss Function
        self.criterions = build_criterion_dict(config, device=self.device)
        
        # 指標輔助計算器 (目前僅供 location 使用，且需要啟用 distance loss)
        use_distance_loss = training_args.get('use_distance_loss', True)
        self.grid_calculator = GridDistanceCalculator(
            grid_width=3, 
            grid_offset=training_args.get('grid_offset', DEFAULT_GRID_OFFSET),
            device=self.device
        ) if ('location' in config.get('targets', []) and use_distance_loss) else None
        
        # 任務 Loss 權重
        self.loss_weights = training_args.get('loss_weights', DEFAULT_LOSS_WEIGHTS)
        self.use_rlw = training_args.get('use_rlw', False)
        self.task_list = config.get('targets', list(self.loss_weights.keys()))
        self.num_tasks = len(self.task_list)
        
        if self.use_rlw:
            print(f"✅ RLW (Random Loss Weighting) 已啟用，共 {self.num_tasks} 個任務")
            print(f"   每個 training step 會從 N(0,1) + Softmax 隨機生成權重")
        else:
            print(f"使用固定 Loss 權重: {self.loss_weights}")
        
        # 優化器設定
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=training_args['learning_rate'], 
            weight_decay=training_args.get('weight_decay', DEFAULT_WEIGHT_DECAY)
        )
        
        # 學習率排程
        num_training_steps = len(self.train_loader) * training_args['epochs']
        warmup_ratio = training_args.get('warmup_ratio', DEFAULT_WARMUP_RATIO)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, 'train', 'logs'))
        self.best_val_loss = float('inf')

    def _calculate_loss_and_acc_and_dist(self, logits, targets):
        """計算單一批次的損失、準確率以及附加指標"""
        total_loss = 0
        accuracies = {}
        distances = {} # 儲存距離誤差
        task_losses = {}  # 儲存每個任務的 loss
        ce_loss_val = 0
        dist_loss_val = 0

        for task, pred in logits.items():
            if task not in targets or task not in self.criterions: continue
            
            target = targets[task].to(self.device)
            
            # --- Loss 計算 (透過 Registry) ---
            loss_output = self.criterions[task](pred, target)
            
            if isinstance(loss_output, tuple):
                loss, ce, dist = loss_output
                ce_loss_val = ce.item()
                dist_loss_val = dist.item() if isinstance(dist, torch.Tensor) else dist
            else:
                loss = loss_output

            weighted_loss = loss * self._current_weights.get(task, 1.0)
            total_loss += weighted_loss
            task_losses[task] = loss.item()  # 記錄未加權的 loss
            
            # --- Accuracy 計算 ---
            _, predicted_labels = torch.max(pred, 1)
            mask = target != 0
            if mask.sum() > 0:
                correct = (predicted_labels[mask] == target[mask]).sum().item()
                accuracies[task] = correct / mask.sum().item()
            else:
                accuracies[task] = 0.0
            
            # --- 額外 Metrics 計算 (Distance) ---
            if task == 'location' and self.grid_calculator:
                tot_dist, dist_count = self.grid_calculator.calculate_distance(predicted_labels, target)
                if dist_count > 0:
                    distances[task] = tot_dist / dist_count
                else:
                    distances[task] = 0.0
            
        return total_loss, accuracies, distances, ce_loss_val, dist_loss_val, task_losses

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        total_ce_loss = 0
        total_dist_loss = 0
        epoch_accuracies = {task: 0 for task in self.config['targets']}
        epoch_distances = {'location': 0.0}
        epoch_task_losses = {task: 0 for task in self.config['targets']}
        batch_count = 0
        dist_batch_count = 0
        
        pbar = tqdm(self.val_loader, desc=f"  Validate Epoch {epoch}/{self.config['training_args']['epochs']}")

        with torch.no_grad():
            # 驗證時固定使用原始 loss_weights
            self._current_weights = self.loss_weights
            for batch in pbar:
                logits = self.model(batch)
                loss, accuracies, distances, ce_loss, dist_loss, task_losses = self._calculate_loss_and_acc_and_dist(logits, batch['targets'])
                
                total_val_loss += loss.item()
                total_ce_loss += ce_loss
                total_dist_loss += dist_loss
                for task, acc in accuracies.items():
                    epoch_accuracies[task] += acc
                for task, t_loss in task_losses.items():
                    epoch_task_losses[task] += t_loss
                
                if 'location' in distances and distances['location'] > 0:
                    epoch_distances['location'] += distances['location']
                    dist_batch_count += 1

                batch_count += 1
                pbar.set_postfix(val_loss=loss.item())

        avg_loss = total_val_loss / max(batch_count, 1)
        avg_ce = total_ce_loss / max(batch_count, 1)
        avg_dist = total_dist_loss / max(batch_count, 1)
        avg_accuracies = {task: acc / max(batch_count, 1) for task, acc in epoch_accuracies.items()}
        avg_task_losses = {task: t_loss / max(batch_count, 1) for task, t_loss in epoch_task_losses.items()}
        avg_distance = epoch_distances['location'] / max(dist_batch_count, 1)
        
        # 計算所有任務 loss 的總和 (Total Loss)
        total_task_loss_sum = sum(avg_task_losses.values())
        
        # 輸出包含 Total Loss 和各任務 Loss
        task_loss_str = " | ".join([f"{task}: {avg_task_losses.get(task, 0):.4f}" for task in self.config['targets']])
        print(f"Epoch {epoch} Valid | Weighted Loss: {avg_loss:.4f} | Total Loss: {total_task_loss_sum:.4f} | {task_loss_str}")
        print(f"              | CE: {avg_ce:.4f}, Dist: {avg_dist:.4f} | Loc Acc: {avg_accuracies.get('location', 0):.4f} | Loc Dist: {avg_distance:.4f}")
        
        return avg_loss, avg_accuracies, avg_distance, avg_ce, avg_dist, avg_task_losses

    def train(self):
        print(f"開始訓練，使用設備: {self.device}")
        print(f"Lambda Distance: {self.config['training_args']['lambda_dist']}")
        print(f"所有結果將儲存於: {self.run_dir}")
        
        epochs = self.config['training_args']['epochs']
        clip_value = self.config['training_args']['clip']

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_epoch_loss = 0
            total_ce_loss = 0
            total_dist_loss = 0
            epoch_accuracies = {task: 0 for task in self.config['targets']}
            epoch_distances = {'location': 0.0}
            epoch_task_losses = {task: 0 for task in self.config['targets']}
            batch_count = 0
            dist_batch_count = 0
            
            pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{epochs}")

            for batch in pbar:
                self.optimizer.zero_grad()
                
                # --- RLW: 每個 step 生成隨機權重 ---
                if self.use_rlw:
                    random_logits = torch.randn(self.num_tasks)
                    random_probs = torch.softmax(random_logits, dim=0)
                    self._current_weights = {task: random_probs[i].item() for i, task in enumerate(self.task_list)}
                else:
                    self._current_weights = self.loss_weights
                
                logits = self.model(batch)
                loss, accuracies, distances, ce_loss, dist_loss, task_losses = self._calculate_loss_and_acc_and_dist(logits, batch['targets'])
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_epoch_loss += loss.item()
                total_ce_loss += ce_loss
                total_dist_loss += dist_loss
                for task, acc in accuracies.items():
                    epoch_accuracies[task] += acc
                for task, t_loss in task_losses.items():
                    epoch_task_losses[task] += t_loss
                
                if 'location' in distances and distances['location'] > 0:
                    epoch_distances['location'] += distances['location']
                    dist_batch_count += 1

                batch_count += 1
                
                # 更新進度條顯示
                total_task_loss = sum(task_losses.values())
                pbar.set_postfix(loss=loss.item(),
                                 total=total_task_loss,
                                 ce=ce_loss,
                                 dist=dist_loss,
                                 loc_acc=accuracies.get('location', 0),
                                 loc_dist=distances.get('location', 0))

            # --- 訓練日誌 ---
            avg_train_loss = total_epoch_loss / max(batch_count, 1)
            avg_train_ce = total_ce_loss / max(batch_count, 1)
            avg_train_dist = total_dist_loss / max(batch_count, 1)
            avg_train_accuracies = {task: acc / max(batch_count, 1) for task, acc in epoch_accuracies.items()}
            avg_train_task_losses = {task: t_loss / max(batch_count, 1) for task, t_loss in epoch_task_losses.items()}
            avg_train_distance = epoch_distances['location'] / max(dist_batch_count, 1)
            
            # 計算所有任務 loss 的總和
            total_task_loss_sum = sum(avg_train_task_losses.values())
            
            task_loss_str = " | ".join([f"{task}: {avg_train_task_losses.get(task, 0):.4f}" for task in self.config['targets']])
            print(f"Epoch {epoch} Train | Weighted Loss: {avg_train_loss:.4f} | Total Loss: {total_task_loss_sum:.4f} | {task_loss_str}")
            print(f"              | CE: {avg_train_ce:.4f}, Dist: {avg_train_dist:.4f} | Loc Acc: {avg_train_accuracies.get('location', 0):.4f} | Loc Dist: {avg_train_distance:.4f}")
            
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/train_ce', avg_train_ce, epoch)
            self.writer.add_scalar('Loss/train_dist', avg_train_dist, epoch)
            self.writer.add_scalar('Loss/train_total', total_task_loss_sum, epoch)
            for task, t_loss in avg_train_task_losses.items():
                self.writer.add_scalar(f'Loss/{task}_train', t_loss, epoch)
            self.writer.add_scalar('Metric/Location_Distance_train', avg_train_distance, epoch)
            for task, acc in avg_train_accuracies.items():
                self.writer.add_scalar(f'Accuracy/{task}_train', acc, epoch)
            
            # --- 驗證步驟 ---
            val_loss, val_accuracies, val_dist, val_ce, val_dist_loss, val_task_losses = self._validate_epoch(epoch)
            
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Loss/val_ce', val_ce, epoch)
            self.writer.add_scalar('Loss/val_dist', val_dist_loss, epoch)
            self.writer.add_scalar('Loss/val_total', sum(val_task_losses.values()), epoch)
            for task, t_loss in val_task_losses.items():
                self.writer.add_scalar(f'Loss/{task}_val', t_loss, epoch)
            self.writer.add_scalar('Metric/Location_Distance_val', val_dist, epoch)
            for task, acc in val_accuracies.items():
                self.writer.add_scalar(f'Accuracy/{task}_val', acc, epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_dir = os.path.join(self.run_dir, 'train', 'weights')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"找到更佳驗證損失 ({self.best_val_loss:.4f})，模型已儲存至 {save_path}")

        save_dir = os.path.join(self.run_dir, 'train', 'weights')
        os.makedirs(save_dir, exist_ok=True)
        last_model_path = os.path.join(save_dir, 'last_model.pth')
        torch.save(self.model.state_dict(), last_model_path)
        print(f"訓練完成。最終模型已儲存至 {last_model_path}")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description="PACT Model Training Script")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="資料夾路徑 (不指定時從 sport config 取得)")
    parser.add_argument('--base_config_file', type=str, default=None,
                        help="設定檔路徑 (不指定時從 sport config 取得)")
    parser.add_argument('--results_base_dir', type=str, default=None,
                        help="結果儲存路徑 (不指定時從 sport config 取得)")
    parser.add_argument('--sport', type=str, default='table_tennis',
                        choices=AVAILABLE_SPORTS,
                        help=f"運動類型 (預設 table_tennis, 可選: {AVAILABLE_SPORTS})")
    
    # 訓練超參數 (改為 None 以便偵測使用者是否有覆寫)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--clip', type=float, default=None)
    
    # Distance-Weighted Loss 超參數
    parser.add_argument('--lambda_dist', type=float, default=None, help="Distance Loss 的權重")
    parser.add_argument('--grid_offset', type=int, default=None, help="Grid 起始索引 (預設 2)")

    # 模型架構參數 (None 表示使用 YAML > fallback 預設值)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--player_embedding_dim', type=int, default=None)
    parser.add_argument('--pooling_type', type=str, default=None,
                        choices=['last', 'mean', 'attention'],
                        help='序列聚合策略: last(預設), mean, attention')
    parser.add_argument('--head_depth', type=int, default=None,
                        help='預測頭 MLP 深度 (1=單層Linear, 2+=MLP)')
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_encoder_layers', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    
    # 新增: 可配置的訓練參數
    parser.add_argument('--model_type', type=str, default='task_attention',
                        help="模型類型 (可選: task_project, task_attention, location, ...)")
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help="Label Smoothing 值")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help="AdamW Weight Decay")
    parser.add_argument('--warmup_ratio', type=float, default=None,
                        help="學習率 Warmup 比例")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="DataLoader 的 worker 數量")
    parser.add_argument('--loss_weights', type=str, default=None,
                        help='任務 Loss 權重 (JSON 格式, 例: \'{"type":0.3,"location":0.3,...}\')')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (預設 42)')
    parser.add_argument('--use_rlw', action='store_true',
                        help='啟用 RLW (Random Loss Weighting)，每個 step 隨機生成任務權重')
    parser.add_argument('--use_skip_connection', action='store_true',
                        help='啟用 Gated Skip Connection 讓預測頭直接看到最後一拍的原始特徵')
    
    args = parser.parse_args()
    
    # 從 sport config 取得預設路徑 (CLI 參數優先)
    sport_config = load_sport_config(args.sport)
    data_dir = args.data_dir or sport_config.get('data_dir', 'processed_data')
    base_config_file = args.base_config_file or sport_config.get('config_file', f'{data_dir}/config.json')
    results_base_dir = args.results_base_dir or sport_config.get('results_dir', 'results')

    run_id = f'run_{args.sport}_{args.model_type}_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir = os.path.join(results_base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    with open(base_config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 取出 YAML 中的 training_args (注意：要從 sport_config 讀，不是 config.json)
    yaml_train_args = sport_config.get('training_args', {})

    # 定義優先級: CLI 傳入 > YAML 設定檔 > fallback 預設參數
    def _resolve(cli_val, yaml_key, fallback):
        """CLI > YAML > fallback 優先級解析"""
        if cli_val is not None:
            return cli_val
        return yaml_train_args.get(yaml_key, fallback)

    epochs = _resolve(args.epochs, 'epochs', 40)
    batch_size = _resolve(args.batch_size, 'batch_size', 128)
    learning_rate = _resolve(args.learning_rate, 'learning_rate', 1e-4)
    lambda_dist = _resolve(args.lambda_dist, 'lambda_dist', 1.0)
    grid_offset = _resolve(args.grid_offset, 'grid_offset', DEFAULT_GRID_OFFSET)
    label_smoothing = _resolve(args.label_smoothing, 'label_smoothing', DEFAULT_LABEL_SMOOTHING)
    weight_decay = _resolve(args.weight_decay, 'weight_decay', DEFAULT_WEIGHT_DECAY)
    warmup_ratio = _resolve(args.warmup_ratio, 'warmup_ratio', DEFAULT_WARMUP_RATIO)
    num_workers = _resolve(args.num_workers, 'num_workers', DEFAULT_NUM_WORKERS)
    clip = _resolve(args.clip, 'clip', 1.0)
    seed = _resolve(args.seed, 'seed', 42)
    use_distance_loss = yaml_train_args.get('use_distance_loss', True)
    use_rlw = args.use_rlw or yaml_train_args.get('use_rlw', False)

    # 設定 Random Seed (在所有隨機操作之前)
    set_seed(seed)

    # 模型架構參數也遵循 CLI > YAML > fallback
    d_model = _resolve(args.d_model, 'd_model', 128)
    player_embedding_dim = _resolve(args.player_embedding_dim, 'player_embedding_dim', 32)
    nhead = _resolve(args.nhead, 'nhead', 4)
    num_encoder_layers = _resolve(args.num_encoder_layers, 'num_encoder_layers', 2)
    dim_feedforward = _resolve(args.dim_feedforward, 'dim_feedforward', 256)
    dropout = _resolve(args.dropout, 'dropout', 0.3)
    pooling_type = _resolve(args.pooling_type, 'pooling_type', 'last')
    head_depth = _resolve(args.head_depth, 'head_depth', 1)

    # 使用運動別配置建構完整特徵配置
    config = build_full_config(args.sport, config)

    # 解析 loss_weights (CLI > sport config > 預設)
    if args.loss_weights:
        loss_weights = json.loads(args.loss_weights)
    else:
        loss_weights = config.get('loss_weights', DEFAULT_LOSS_WEIGHTS)

    config['data_args'] = {
        'data_dir': data_dir,
        'base_config_file': base_config_file
    }
    config['model_args'] = {
        'd_model': d_model, 'player_embedding_dim': player_embedding_dim,
        'nhead': nhead, 'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward, 'dropout': dropout,
        'pooling_type': pooling_type, 'head_depth': head_depth,
        'use_skip_connection': args.use_skip_connection,
    }
    config['training_args'] = {
        'epochs': epochs, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'clip': clip,
        'lambda_dist': lambda_dist,
        'label_smoothing': label_smoothing,
        'weight_decay': weight_decay,
        'warmup_ratio': warmup_ratio,
        'grid_offset': grid_offset,
        'num_workers': num_workers,
        'loss_weights': loss_weights,
        'model_type': args.model_type,
        'seed': seed,
        'use_distance_loss': use_distance_loss,
        'use_rlw': use_rlw,
    }
    
    num_workers = config['training_args']['num_workers']
    
    train_dataset, collate_fn = get_dataloader(
        args.sport,
        data_path=os.path.join(config['data_args']['data_dir'], 'train_data.pkl'),
        config_path=config['data_args']['base_config_file']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training_args']['batch_size'], shuffle=True, 
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    
    val_dataset, _ = get_dataloader(
        args.sport,
        data_path=os.path.join(config['data_args']['data_dir'], 'val_data.pkl'),
        config_path=config['data_args']['base_config_file']
    )
    val_loader = DataLoader(val_dataset, batch_size=config['training_args']['batch_size'], shuffle=False, 
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    config['targets'] = list(train_dataset[0]['targets'].keys())

    PACTModel, config_overrides = get_model_class(args.model_type)
    # Apply model type overrides (e.g., L1, L1_L2) to config
    # Overrides should take precedence over sport defaults
    if 'hierarchy_levels' in config and 'hierarchy_levels' not in config_overrides:
        # If model type doesn't specify hierarchy (e.g. task_attention defaults), use sport config
        config['model_args']['hierarchy_levels'] = config['hierarchy_levels']
    
    config['model_args'].update(config_overrides)

    print(f"\n[{args.sport.upper()} - {args.model_type}]")
    print(f"訓練參數: Epochs={config['training_args']['epochs']}, BatchSize={config['training_args']['batch_size']}, LR={config['training_args']['learning_rate']}")
    print(f"結果儲存於: {run_dir}")
    print(f"{'='*50}\n")
    
    # Save config AFTER overrides are applied, so test script can reconstruct exact model
    final_config_path = os.path.join(run_dir, 'config.json')
    with open(final_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    model = PACTModel(config)
    trainer = Trainer(model, train_loader, val_loader, config, run_dir)
    trainer.train()

if __name__ == '__main__':
    main()