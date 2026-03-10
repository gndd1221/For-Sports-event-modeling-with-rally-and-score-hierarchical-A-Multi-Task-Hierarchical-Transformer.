import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# Expected Distance Loss (距離的期望值)
# 核心概念:
# - Loss A (CE): 標準 CrossEntropyLoss，保持 One-hot Target 的精確度
# - Loss B (Expected Distance): 計算預測分佈中每一點到目標的加權距離總和
# - Total Loss = CE + lambda_dist * Expected Distance Loss
# 
# 數學公式: Loss = Σ P_i * || Coord_i - TargetCoord ||
# 直觀意義: 類似「搬運功」(Earth Mover's Distance 近似)
#           任何分配給錯誤位置的機率，都會直接貢獻 Loss，無法被抵銷
# =====================================================================================
class DistanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, grid_width=3, lambda_dist=0.5, device='cuda', grid_offset=2, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dist = lambda_dist
        self.device = device
        self.grid_offset = grid_offset
        self.grid_size = grid_width * grid_width  # 9
        self.grid_width = grid_width
        
        # 建立九宮格座標映射 (Row-Major)
        # Zone 2-10 對應 3x3 Grid:
        # [2, 3, 4]    [(0,0), (0,1), (0,2)]
        # [5, 6, 7] -> [(1,0), (1,1), (1,2)]
        # [8, 9, 10]   [(2,0), (2,1), (2,2)]
        self.coords = torch.tensor([
            [r, c] for r in range(grid_width) for c in range(grid_width)
        ], dtype=torch.float32, device=device)
        
        # 預先計算所有格子間的距離矩陣 (9x9)
        # distance_matrix[i, j] = || Coord_i - Coord_j ||
        self.distance_matrix = self._compute_distance_matrix()
        
        # 標準 Cross Entropy Loss (忽略 padding index 0)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    
    def _compute_distance_matrix(self):
        """預先計算 9x9 距離矩陣"""
        # coords: (9, 2)
        # 計算所有 pair 的距離
        diff = self.coords.unsqueeze(0) - self.coords.unsqueeze(1)  # (9, 9, 2)
        distance_matrix = torch.norm(diff, dim=2)  # (9, 9)
        return distance_matrix
    
    def forward(self, logits, targets):
        """
        logits: (Batch, Num_Classes)
        targets: (Batch) - Hard Label 索引
        """
        # ============================================
        # Loss A: 標準 Cross Entropy (One-hot Target)
        # ============================================
        ce = self.ce_loss(logits, targets)
        
        # ============================================
        # Loss B: Expected Distance Loss (距離期望值)
        # 公式: Σ P_i * distance(Coord_i, TargetCoord)
        # ============================================
        grid_start = self.grid_offset
        grid_end = self.grid_offset + self.grid_size
        
        # 找出哪些樣本在 Grid 內 (Zone 2-10)
        is_grid = (targets >= grid_start) & (targets < grid_end)
        
        if is_grid.any():
            # 取出 Grid 樣本的 logits (只取 Zone 2-10 對應的 9 個 class)
            grid_logits = logits[is_grid, grid_start:grid_end]  # (N_grid, 9)
            grid_targets = targets[is_grid] - grid_start  # (N_grid,) 轉為 0-8
            
            # 計算預測機率分佈
            probs = F.softmax(grid_logits, dim=1)  # (N_grid, 9)
            
            # 取出每個樣本對應的目標距離向量
            # distance_matrix[target_idx] 給出從 target 到所有 9 個格子的距離
            # costs: (N_grid, 9)
            costs = self.distance_matrix[grid_targets]  # (N_grid, 9)
            
            # 計算期望距離: Σ P_i * distance(Coord_i, TargetCoord)
            # probs * costs: (N_grid, 9) 對應每個格子的加權距離
            # sum over dim=1: 每個樣本的期望距離
            expected_dist = torch.sum(probs * costs, dim=1)  # (N_grid,)
            dist_loss = expected_dist.mean()
        else:
            dist_loss = torch.tensor(0.0, device=logits.device)
        
        # ============================================
        # 組合 Loss
        # ============================================
        total_loss = ce + self.lambda_dist * dist_loss
        
        return total_loss, ce, dist_loss

# =====================================================================================
# Loss Factory / Task Registry 
# =====================================================================================
def build_criterion_dict(config, device='cuda'):
    """
    根據 config 設定動態生成所有任務的 Loss Function。
    未特別指定的任務，預設使用 CrossEntropyLoss。
    """
    criterions = {}
    targets = config.get('targets', [])
    training_args = config.get('training_args', {})
    
    label_smoothing = training_args.get('label_smoothing', 0.1)
    lambda_dist = training_args.get('lambda_dist', 1.0)
    grid_offset = training_args.get('grid_offset', 2)

    use_distance_loss = training_args.get('use_distance_loss', True)

    for task in targets:
        if task == 'location' and use_distance_loss:
            criterions[task] = DistanceWeightedCrossEntropyLoss(
                num_classes=config.get(f'num_{task}', 11),
                lambda_dist=lambda_dist,
                device=device,
                grid_offset=grid_offset,
                label_smoothing=label_smoothing
            )
        else:
            criterions[task] = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing, 
                ignore_index=0
            )
            
    return criterions

# =====================================================================================
# Extensible Metric Calculators
# =====================================================================================
class GridDistanceCalculator:
    """獨立的距離計算器，專門服務於 Test/Eval 階段的 Location Distance 指標計算"""
    def __init__(self, grid_width=3, grid_offset=2, device='cpu'):
        self.grid_size = grid_width * grid_width
        self.grid_offset = grid_offset
        self.coords = torch.tensor([
            [r, c] for r in range(grid_width) for c in range(grid_width)
        ], dtype=torch.float32, device=device)

    def calculate_distance(self, predictions: torch.Tensor, targets: torch.Tensor):
        grid_end = self.grid_offset + self.grid_size
        mask_grid = (targets >= self.grid_offset) & (targets < grid_end) & \
                    (predictions >= self.grid_offset) & (predictions < grid_end)
        
        if mask_grid.any():
            t_idx = targets[mask_grid] - self.grid_offset
            p_idx = predictions[mask_grid] - self.grid_offset
            t_coord = self.coords[t_idx]
            p_coord = self.coords[p_idx]
            dists = torch.norm(t_coord - p_coord, dim=1)
            return dists.sum().item(), dists.size(0)
        return 0.0, 0
