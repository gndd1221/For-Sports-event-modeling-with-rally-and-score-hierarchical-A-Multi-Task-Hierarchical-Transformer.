"""
Task Attention Multi-Task Decoupling Analysis
==============================================
Computes the GLOBAL average Task Cross-Attention weights for each prediction
task (type, backhand, location, strength, spin) across the entire test set,
then visualises the comparison as a grouped bar chart + heatmap.

This answers: "When the model predicts different targets, does it look at
different information sources (L1 vs L2 vs L3 vs Player vs Opponent)?"
"""

import os, sys, json, torch, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.dataloader import get_dataloader
from src.default_config import get_model_class, load_sport_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--sport', type=str, default='table_tennis')
    args = parser.parse_args()

    # ─── Load config & model ─────────────────────────────────────────────────
    config = json.load(open(os.path.join(args.run_dir, 'config.json'), encoding='utf-8'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = config.get('training_args', {}).get('model_type', 'task_attention')
    PACTModel, overrides = get_model_class(model_type)
    saved_args = config.get('model_args', {})
    saved_hierarchy = saved_args.get('hierarchy_levels') or config.get('hierarchy_levels')
    saved_fusion = saved_args.get('fusion_type')
    config.setdefault('model_args', {}).update(overrides)
    if saved_hierarchy:
        config['model_args']['hierarchy_levels'] = saved_hierarchy
    if saved_fusion:
        config['model_args']['fusion_type'] = saved_fusion

    model = PACTModel(config).to(device)
    model_path = os.path.join(args.run_dir, 'train', 'weights', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # ─── Load test data ──────────────────────────────────────────────────────
    sport_cfg = load_sport_config(args.sport)
    data_dir = (config.get('data_args', {}).get('data_dir')
                or sport_cfg.get('data_dir', 'processed_data'))
    test_dataset, collate_fn = get_dataloader(
        args.sport,
        data_path=os.path.join(data_dir, 'test_data.pkl'),
        config_path=os.path.join(data_dir, 'config.json'),
    )
    batch_size = config['training_args']['batch_size']
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    targets = config.get('targets', [])
    level_names = config['model_args'].get('hierarchy_levels', ['L1', 'L2', 'L3'])
    use_seq_fusion = config.get('model_args', {}).get('use_sequence_fusion', False)

    # KV token labels (for task_attention mode: summary tokens only)
    if not use_seq_fusion:
        kv_labels = [f'{lv}' for lv in level_names] + ['Player', 'Opponent']
    else:
        kv_labels = None  # will be determined dynamically

    # ─── Collect task attention across entire test set ────────────────────────
    # Accumulate per-task, per-KV-position sums
    task_attn_accum = {t: None for t in targets}
    task_attn_count = {t: 0 for t in targets}

    print("Collecting Task Attention across entire Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
                elif isinstance(val, dict):
                    batch[key] = {k: v.to(device) for k, v in val.items()}

            output = model(batch)
            logits, debug_info = output if isinstance(output, tuple) else (output, {})

            if debug_info.get('task_attn_weights') is None:
                continue

            for task_name, attn_tensor in debug_info['task_attn_weights'].items():
                if task_name not in targets:
                    continue
                # attn_tensor: (B, nhead, 1, kv_len)
                attn_np = attn_tensor.cpu().numpy()
                B = attn_np.shape[0]
                # Average across heads, squeeze query dim -> (B, kv_len)
                attn_avg = np.mean(attn_np, axis=1).squeeze(1)  # (B, kv_len)

                if task_attn_accum[task_name] is None:
                    task_attn_accum[task_name] = np.sum(attn_avg, axis=0)
                else:
                    task_attn_accum[task_name] += np.sum(attn_avg, axis=0)
                task_attn_count[task_name] += B

    # ─── Compute global averages ─────────────────────────────────────────────
    task_global_avg = {}
    for t in targets:
        if task_attn_accum[t] is not None and task_attn_count[t] > 0:
            task_global_avg[t] = task_attn_accum[t] / task_attn_count[t]

    if not task_global_avg:
        print("No task attention data collected. Exiting.")
        return

    # Determine KV labels if not set
    kv_len = len(next(iter(task_global_avg.values())))
    if kv_labels is None or len(kv_labels) != kv_len:
        kv_labels = [f'{lv}' for lv in level_names] + ['Player', 'Opponent']
        kv_labels = kv_labels[:kv_len]

    output_dir = os.path.join(args.run_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # ─── Print results ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Global Average Task Attention Weights (across entire Test Set)")
    print("=" * 70)
    header = f"{'Task':>12} | " + " | ".join(f"{kv:>10}" for kv in kv_labels)
    print(header)
    print("-" * len(header))
    for t in targets:
        if t in task_global_avg:
            vals = " | ".join(f"{v:>10.4f}" for v in task_global_avg[t])
            print(f"{t:>12} | {vals}")

    # ─── Plot 1: Grouped Bar Chart ──────────────────────────────────────────
    task_names = [t for t in targets if t in task_global_avg]
    data_matrix = np.array([task_global_avg[t] for t in task_names])  # (num_tasks, kv_len)

    x = np.arange(len(kv_labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(task_names)))

    for i, task in enumerate(task_names):
        offset = (i - len(task_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data_matrix[i], width, label=task.capitalize(),
                      color=colors[i], edgecolor='gray', linewidth=0.5)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Information Source (KV Tokens)', fontsize=12)
    ax.set_ylabel('Average Attention Weight', fontsize=12)
    ax.set_title('Multi-Task Attention Decoupling:\nHow different prediction tasks attend to different information sources',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(kv_labels, fontsize=11)
    ax.legend(title='Prediction Task', fontsize=10, title_fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_attention_decoupling_bar.png'), dpi=300)
    plt.close()
    print(f"\nSaved task_attention_decoupling_bar.png")

    # ─── Plot 2: Heatmap ────────────────────────────────────────────────────
    df = pd.DataFrame(data_matrix, index=[t.capitalize() for t in task_names],
                      columns=kv_labels)
    plt.figure(figsize=(10, 5))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5,
                cbar_kws={'label': 'Attention Weight'})
    plt.title('Task × Information Source Attention Heatmap\n'
              '(Global average across entire Test Set)', fontsize=13)
    plt.xlabel('Information Source (KV Token)')
    plt.ylabel('Prediction Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_attention_decoupling_heatmap.png'), dpi=300)
    plt.close()
    print(f"Saved task_attention_decoupling_heatmap.png")


if __name__ == '__main__':
    main()
