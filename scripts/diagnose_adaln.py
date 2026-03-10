"""
AdaLN 診斷工具
用於檢查訓練後的模型：
1. AdaLN 的 scale/shift 參數是否已從零初始化中學到了有意義的值
2. 不同球員的 Embedding 是否已經分化
3. 同一個球員 vs 不同球員經過 AdaLN 後的特徵差異

用法: python diagnose_adaln.py
"""
import torch
import json
import os
import sys
from collections import defaultdict

# 確保專案根目錄在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- 設定 ---
SPORT = 'table_tennis'
MODEL_TYPE = 'parallel'
# 自動找到最新的 AdaLN 訓練結果
RESULTS_DIR = f'outputs/results/{SPORT}'


def find_latest_checkpoint(results_dir, model_type):
    """找到最新的 checkpoint"""
    candidates = []
    for d in os.listdir(results_dir):
        if d.startswith(f'run_{SPORT}_{model_type}_'):
            path = os.path.join(results_dir, d, 'train', 'weights', 'best_model.pth')
            if os.path.exists(path):
                candidates.append((d, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def diagnose():
    print("=" * 70)
    print("AdaLN 診斷報告")
    print("=" * 70)

    # 1. 載入 checkpoint
    result = find_latest_checkpoint(RESULTS_DIR, MODEL_TYPE)
    if result is None:
        print(f"❌ 找不到 {MODEL_TYPE} 的 checkpoint！")
        return
    
    run_name, ckpt_path = result
    print(f"\n📂 使用 Checkpoint: {run_name}")
    print(f"   路徑: {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # 2. 檢查 AdaLN 參數
    print("\n" + "=" * 70)
    print("【檢查一】AdaLN Modulation 參數是否已脫離零初始化")
    print("=" * 70)
    
    adaln_params = {}
    for name, param in state_dict.items():
        if 'adaLN_modulation' in name:
            adaln_params[name] = param
    
    if not adaln_params:
        print("❌ 找不到任何 adaLN_modulation 參數！")
        print("   這表示模型可能沒有正確使用 AdaLN，或是載入了舊版模型。")
        print("\n   所有參數名稱列表（前 20 個）：")
        for i, name in enumerate(state_dict.keys()):
            if i >= 20:
                print("   ...")
                break
            print(f"     {name}")
        return
    
    print(f"\n✅ 找到 {len(adaln_params)} 個 AdaLN 參數")
    
    all_zero_count = 0
    non_zero_count = 0
    
    for name, param in adaln_params.items():
        abs_mean = param.abs().mean().item()
        abs_max = param.abs().max().item()
        std = param.std().item()
        is_zero = abs_max < 1e-7
        
        status = "⚠️ 全零（未學習！）" if is_zero else "✅ 已學習"
        if is_zero:
            all_zero_count += 1
        else:
            non_zero_count += 1
        
        # 只顯示 weight 層的統計 (bias 通常較小)
        if 'weight' in name:
            print(f"\n  {status} {name}")
            print(f"    |mean|={abs_mean:.6f}  |max|={abs_max:.6f}  std={std:.6f}")
    
    print(f"\n📊 總結: {non_zero_count} 個參數已學習, {all_zero_count} 個仍為零")
    
    if all_zero_count > 0 and non_zero_count == 0:
        print("❌ AdaLN 完全沒有被訓練！梯度可能沒有正確回傳。")
    elif all_zero_count > 0:
        print("⚠️  部分 AdaLN 層未被訓練，可能是某些 encoder 沒有使用到。")
    else:
        print("✅ 所有 AdaLN 層都已經學到了非零參數！")
    
    # 3. 檢查球員 Embedding 的分化程度
    print("\n" + "=" * 70)
    print("【檢查二】球員 Embedding 差異化程度")
    print("=" * 70)
    
    player_emb_key = None
    for name in state_dict:
        if 'player_encoder.embedding.weight' in name:
            player_emb_key = name
            break
    
    if player_emb_key is None:
        print("⚠️ 找不到 player_encoder.embedding.weight")
        return
    
    player_emb = state_dict[player_emb_key]  # (num_players, dim)
    num_players, dim = player_emb.shape
    print(f"\n球員數量: {num_players}, Embedding 維度: {dim}")
    
    # 計算球員之間的餘弦相似度
    norms = player_emb.norm(dim=1, keepdim=True)
    # 排除 padding (index 0) 和 norm 為 0 的球員
    valid_mask = (norms.squeeze() > 1e-6)
    valid_emb = player_emb[valid_mask]
    
    if valid_emb.shape[0] < 2:
        print("⚠️ 有效球員數量不足，無法計算相似度")
        return
    
    valid_norms = valid_emb.norm(dim=1, keepdim=True)
    normalized = valid_emb / valid_norms
    cosine_sim = torch.mm(normalized, normalized.t())
    
    # 取上三角（排除自己跟自己的相似度）
    mask = torch.triu(torch.ones_like(cosine_sim), diagonal=1).bool()
    pairwise_sims = cosine_sim[mask]
    
    print(f"\n球員間餘弦相似度統計（有效球員: {valid_emb.shape[0]}）:")
    print(f"  平均: {pairwise_sims.mean():.4f}")
    print(f"  標準差: {pairwise_sims.std():.4f}")
    print(f"  最大: {pairwise_sims.max():.4f}")
    print(f"  最小: {pairwise_sims.min():.4f}")
    
    if pairwise_sims.std() < 0.05:
        print("⚠️  球員 Embedding 差異很小！所有球員看起來幾乎一樣。")
        print("   → AdaLN 無法有效區分不同球員。")
    elif pairwise_sims.std() < 0.15:
        print("🔶 球員 Embedding 有一定程度的分化，但差異不大。")
    else:
        print("✅ 球員 Embedding 已經高度分化！不同球員有明顯差異。")
    
    # 4. 模擬 AdaLN 的實際效果
    print("\n" + "=" * 70)
    print("【檢查三】AdaLN 對不同球員產生的 scale/shift 差異")
    print("=" * 70)
    
    # 找第一個 adaLN_modulation 的 weight 和 bias
    weight_key = None
    bias_key = None
    for name in adaln_params:
        if 'shot_encoder.layers.0.adaLN_modulation.1.weight' in name:
            weight_key = name
        if 'shot_encoder.layers.0.adaLN_modulation.1.bias' in name:
            bias_key = name
    
    if weight_key and bias_key and valid_emb.shape[0] >= 2:
        W = adaln_params[weight_key]  # (4*d_model, player_dim)
        b = adaln_params[bias_key]    # (4*d_model,)
        
        # 模擬 SiLU + Linear
        player_a = valid_emb[0]  # 第一個有效球員
        player_b = valid_emb[1]  # 第二個有效球員
        
        import torch.nn.functional as Fn
        out_a = Fn.linear(Fn.silu(player_a), W, b)
        out_b = Fn.linear(Fn.silu(player_b), W, b)
        
        d_model = out_a.shape[0] // 4
        shift_a, scale_a, _, _ = out_a.chunk(4)
        shift_b, scale_b, _, _ = out_b.chunk(4)
        
        scale_diff = (scale_a - scale_b).abs().mean().item()
        shift_diff = (shift_a - shift_b).abs().mean().item()
        
        print(f"\n球員 A vs 球員 B (shot_encoder.layers.0):")
        print(f"  Scale 差異 (|γ_A - γ_B|): {scale_diff:.6f}")
        print(f"  Shift 差異 (|β_A - β_B|): {shift_diff:.6f}")
        print(f"  Scale A 範圍: [{scale_a.min():.4f}, {scale_a.max():.4f}]")
        print(f"  Scale B 範圍: [{scale_b.min():.4f}, {scale_b.max():.4f}]")
        
        if scale_diff < 0.001 and shift_diff < 0.001:
            print("\n⚠️  兩個球員產生的 AdaLN 參數幾乎相同！")
            print("   → AdaLN 目前沒有在區分球員。")
        else:
            print(f"\n✅ AdaLN 正在為不同球員產生不同的特徵調變！")
    
    print("\n" + "=" * 70)
    print("診斷完成")
    print("=" * 70)


if __name__ == '__main__':
    diagnose()
