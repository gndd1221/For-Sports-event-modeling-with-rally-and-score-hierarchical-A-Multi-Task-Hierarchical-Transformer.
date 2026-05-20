import os
import sys
import json
import torch
import argparse
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

SHOT_TYPE_MAP = {
    0: "Zero", 1: "Drive", 2: "Counter", 3: "Smash", 4: "Twist",
    5: "FastDrive", 6: "FastPush(A)", 7: "Flip", 8: "LongPush(P)", 
    9: "FastPush(P)", 10: "LongPush", 11: "Drop", 12: "Chop", 
    13: "Block", 14: "Lob", 15: "Serve(Trad)", 16: "Serve(Hook)", 
    17: "Serve(Rev)", 18: "Serve(Squat)"
}


def main():
    parser = argparse.ArgumentParser(description="Attention and Gating Visualization Analysis")
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--sport', type=str, default='table_tennis')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    # ─── 1. Load config and model ───────────────────────────────────────────────
    config_path = os.path.join(args.run_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: Cannot find config.json in {args.run_dir}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config.get('training_args', {}).get('model_type', 'task_attention')

    PACTModel, config_overrides = get_model_class(model_type)
    saved_model_args = config.get('model_args', {})
    saved_hierarchy = saved_model_args.get('hierarchy_levels') or config.get('hierarchy_levels')
    saved_fusion_type = saved_model_args.get('fusion_type')
    config.setdefault('model_args', {}).update(config_overrides)
    if saved_hierarchy:
        config['model_args']['hierarchy_levels'] = saved_hierarchy
    if saved_fusion_type:
        config['model_args']['fusion_type'] = saved_fusion_type

    model = PACTModel(config).to(device)
    model_path = os.path.join(args.run_dir, 'train', 'weights', 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.run_dir, 'train', 'weights', 'last_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: No model weights found in {args.run_dir}/train/weights/")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # ─── Feature index lookup ────────────────────────────────────────────────────
    # shot_seq_current shape: (B, T, F), F columns follow features_to_extract order
    features = config.get('features_to_extract', [])
    type_feat_idx = features.index('type') if 'type' in features else None

    # ─── 2. Load Test Data ───────────────────────────────────────────────────────
    sport_cfg = load_sport_config(args.sport)
    data_dir = (
        args.data_dir
        or config.get('data_args', {}).get('data_dir')
        or sport_cfg.get('data_dir', 'processed_data')
    )
    test_dataset, collate_fn = get_dataloader(
        args.sport,
        data_path=os.path.join(data_dir, 'test_data.pkl'),
        config_path=os.path.join(data_dir, 'config.json'),
    )
    batch_size = config['training_args']['batch_size']
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    output_dir = args.output_dir or os.path.join(args.run_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # ─── 3. Collect Data ─────────────────────────────────────────────────────────
    # Each batch record: (td_attn, task_attn, shot_seq_current_cpu, lengths_in_this_batch)
    #   td_attn:        np.ndarray (B, nhead, 1, T) or None
    #   task_attn:      dict {task: np.ndarray (B, nhead, 1, T)} or None
    #   shot_seq:       np.ndarray (B, T, F)   ← for X-axis labels
    #   lengths:        list[int]               ← rally_lengths per sample

    all_lengths = []          # flat list, length = total samples
    batch_records = []        # list of dicts per batch
    global_tsag = []
    td_gate_val = None

    print("Collecting debug info from Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move tensors to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
                elif isinstance(val, dict):
                    batch[key] = {k: v.to(device) for k, v in val.items()}

            output = model(batch)
            logits, debug_info = output if isinstance(output, tuple) else (output, {})

            lengths_t = debug_info.get('rally_lengths')
            if lengths_t is None:
                # fallback: use shot_seq_current_lengths
                lengths_t = batch.get('shot_seq_current_lengths')
            lengths = lengths_t.cpu().numpy().tolist() if lengths_t is not None else []
            all_lengths.extend(lengths)

            # TSAG
            if debug_info.get('tsag_weights') is not None:
                global_tsag.append(debug_info['tsag_weights'].cpu().numpy())

            # Gate
            if debug_info.get('td_gate_value') is not None:
                td_gate_val = debug_info['td_gate_value']

            # Attention tensors
            td_attn_np = (
                debug_info['td_attn_weights'].cpu().numpy()
                if debug_info.get('td_attn_weights') is not None
                else None
            )
            task_attn_np = None
            if debug_info.get('task_attn_weights') is not None:
                task_attn_np = {
                    t: a.cpu().numpy() for t, a in debug_info['task_attn_weights'].items()
                }

            # shot_seq_current is (B, T, F) on device; copy to CPU for labels
            shot_seq_cpu = batch['shot_seq_current'].cpu().numpy()

            batch_records.append({
                'lengths': lengths,
                'shot_seq': shot_seq_cpu,   # (B, T, F)
                'td_attn': td_attn_np,
                'task_attn': task_attn_np,
            })

    # ─── 4. Derived config ───────────────────────────────────────────────────────
    targets = config.get('targets', [])
    level_names = config['model_args'].get('hierarchy_levels', ['L1', 'L2', 'L3'])

    sns.set_theme(style="whitegrid")

    # ─── Plot 1: TSAG vs Rally Length ───────────────────────────────────────────
    if global_tsag:
        all_tsag = np.concatenate(global_tsag, axis=0)
        all_len_arr = np.array(all_lengths)
        avg_tsag = np.mean(all_tsag, axis=1)
        df = pd.DataFrame(avg_tsag, columns=level_names[:avg_tsag.shape[1]])
        df['rally_length'] = all_len_arr
        bins = [0, 3, 6, 9, 200]
        labels = ['1-3', '4-6', '7-9', '>=10']
        df['length_group'] = pd.cut(df['rally_length'], bins=bins, labels=labels)
        group_means = df.groupby('length_group')[level_names[:avg_tsag.shape[1]]].mean()

        plt.figure(figsize=(10, 6))
        group_means.plot(kind='bar', stacked=False, colormap='viridis')
        plt.title('TSAG Level Importance vs Rally Length')
        plt.xlabel('Rally Length (shots)')
        plt.ylabel('Average TSAG Weight')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tsag_vs_length.png'), dpi=300)
        plt.close()
        print(f"Saved TSAG plot.")

    # ─── Plot 2: Top-Down Gate ───────────────────────────────────────────────────
    if td_gate_val is not None:
        print(f"Top-Down Gate Value: {td_gate_val:.4f}")
        with open(os.path.join(output_dir, 'td_gate_val.txt'), 'w') as f:
            f.write(f"Top-Down Gate Value (after training): {td_gate_val:.4f}\n")

    # ─── Helper: build X-axis labels from actual shot types in a sample ─────────
    def make_x_labels(shot_seq, length):
        """
        shot_seq: (T, F) numpy array, one sample
        length:   int, actual rally length (no padding)
        returns:  list of strings, len == length
        """
        labels = []
        for t in range(length):
            if type_feat_idx is not None:
                shot_type_token = int(shot_seq[t, type_feat_idx])
                real_type = shot_type_token - 1
                name = SHOT_TYPE_MAP.get(real_type, f"Type-{real_type}")
                labels.append(name)
            else:
                labels.append(f"Shot-{t+1}")
        return labels

    # ─── Find best case-study (longest rally > 10 across all batches) ───────────
    best = {'len': 0, 'b_idx': 0, 's_idx': 0}
    for b_idx, rec in enumerate(batch_records):
        for s_idx, rl in enumerate(rec['lengths']):
            if rl > best['len']:
                best = {'len': rl, 'b_idx': b_idx, 's_idx': s_idx}

    b_idx_best = best['b_idx']
    s_idx_best = best['s_idx']
    rl_best = best['len']
    print(f"Case Study: batch {b_idx_best}, sample {s_idx_best}, rally length {rl_best}")

    shot_seq_best = batch_records[b_idx_best]['shot_seq'][s_idx_best]   # (T, F)
    x_labels_best = make_x_labels(shot_seq_best, rl_best)

    # ─── Plot 3: All-task Attention Heatmaps (case study) ───────────────────────
    # task_attention (use_sequence_fusion=False):
    #   KV = [L1_Summary(1)] + [L2_Summary(1)] + [L3_Summary(1)] + [Player(1)] + [Opponent(1)]
    #   Total KV = len(hierarchy_levels) + 2
    # sequence_attention (use_sequence_fusion=True):
    #   KV = [Shot_1..Shot_T (padded)] + [L2_Summary] + [L3_Summary] + [Player] + [Opponent]
    use_sequence_fusion = config.get('model_args', {}).get('use_sequence_fusion', False)
    task_attn_best = batch_records[b_idx_best]['task_attn']
    if task_attn_best is not None:
        tasks_available = [t for t in targets if t in task_attn_best]
        num_tasks = len(tasks_available)
        if num_tasks > 0:
            first_task = tasks_available[0]
            total_kv_len = task_attn_best[first_task][s_idx_best].shape[-1]

            if not use_sequence_fusion:
                # task_attention: KV = [L1_Summary, L2_Summary, L3_Summary, Player, Opponent]
                # Matches model_components.py else-branch: all hierarchy_features + player/opponent
                full_x_labels = [f'[{lv}_Summary]' for lv in level_names]
                full_x_labels += ['[Player]', '[Opponent]']
                while len(full_x_labels) < total_kv_len:
                    full_x_labels.append(f'[CTX-{len(full_x_labels)}]')
                full_x_labels = full_x_labels[:total_kv_len]
                x_axis_title = ('KV Tokens (task_attention): '
                                '[L1_Summary] + [L2_Summary] + [L3_Summary] + [Player] + [Opponent]')
                fig_width = max(10, total_kv_len * 1.4)
            else:
                # sequence_attention: KV = [Shot_1..Shot_T + PAD] + [L2_Summary] + [L3_Summary] + [Player] + [Opponent]
                padded_shot_len = batch_records[b_idx_best]['shot_seq'].shape[1]
                num_context = total_kv_len - padded_shot_len
                full_x_labels = list(x_labels_best)
                for t in range(rl_best, padded_shot_len):
                    full_x_labels.append(f'[PAD-{t}]')
                for lv_name in level_names[1:]:
                    full_x_labels.append(f'[{lv_name}_Summary]')
                full_x_labels += ['[Player]', '[Opponent]']
                while len(full_x_labels) < total_kv_len:
                    full_x_labels.append(f'[CTX-{len(full_x_labels)}]')
                full_x_labels = full_x_labels[:total_kv_len]
                x_axis_title = ('KV Tokens (sequence_attention): '
                                '[Shot_1..Shot_T + PAD] + [L2_Summary] + [L3_Summary] + [Player] + [Opponent]')
                fig_width = max(14, total_kv_len * 0.6)

            fig, axes = plt.subplots(num_tasks, 1, figsize=(fig_width, 2.8 * num_tasks))
            if num_tasks == 1:
                axes = [axes]

            for i, task in enumerate(tasks_available):
                attn = task_attn_best[task][s_idx_best]       # (nhead, 1, total_kv_len)
                attn_avg = np.mean(attn, axis=0).squeeze()    # (total_kv_len,)

                sns.heatmap(
                    attn_avg.reshape(1, -1),
                    annot=True, fmt='.3f',
                    cmap='viridis', cbar=True,
                    ax=axes[i],
                    xticklabels=full_x_labels[:len(attn_avg)],
                )
                axes[i].set_title(
                    f'Task Cross-Attention: {task.capitalize()}  '
                    f'(case rally len={rl_best}, KV tokens={total_kv_len})'
                )
                axes[i].set_yticks([])
                if i == num_tasks - 1:
                    axes[i].set_xlabel(x_axis_title)
                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                else:
                    axes[i].set_xticklabels([])

            plt.suptitle('Task-Specific Cross-Attention — Full KV Token Pool', fontsize=13, y=1.01)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'task_attention_all.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Saved task_attention_all.png (use_sequence_fusion={use_sequence_fusion})')

    # ─── Plot 4: Top-Down Attention (case study) ─────────────────────────────────
    td_attn_best = batch_records[b_idx_best]['td_attn']
    if td_attn_best is not None:
        attn_map = np.mean(td_attn_best[s_idx_best], axis=0).squeeze()  # (T,)
        plot_len = min(rl_best, len(attn_map))

        plt.figure(figsize=(max(10, plot_len * 0.7), 3))
        sns.heatmap(
            attn_map[:plot_len].reshape(1, -1),
            annot=True, fmt='.2f',
            cmap='magma', cbar=True,
            xticklabels=x_labels_best[:plot_len],
        )
        plt.title(f'Top-Down Refinement Attention (rally len={rl_best})\n'
                  f'High-level context locates key shots in L1 sequence')
        plt.xlabel('Shot Sequence (Shot Type ID)')
        plt.yticks([])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_down_attention.png'), dpi=300)
        plt.close()
        print(f"Saved top_down_attention.png")

    # ─── Plot 5: Global Aggregation — avg attention per shot type ────────────────
    print("Calculating Global Aggregation...")
    if any(rec['td_attn'] is not None for rec in batch_records):
        type_attn_sum = {}
        type_counts = {}

        for rec in batch_records:
            td = rec['td_attn']
            if td is None:
                continue
            for s_idx, rl in enumerate(rec['lengths']):
                if rl < 1:
                    continue
                attn = np.mean(td[s_idx], axis=0).squeeze()  # (T,)
                # Normalize across this sample so short/long rallies are comparable
                seg = attn[:rl]
                total = seg.sum()
                if total > 0:
                    seg = seg / total

                shot_seq = rec['shot_seq'][s_idx]  # (T, F)
                for t in range(rl):
                    if type_feat_idx is not None:
                        stype_token = int(shot_seq[t, type_feat_idx])
                        stype = stype_token - 1
                    else:
                        stype = 0
                    type_attn_sum[stype] = type_attn_sum.get(stype, 0.0) + float(seg[t])
                    type_counts[stype] = type_counts.get(stype, 0) + 1

        # Filter: need at least 10 observations per type
        valid = {s: type_attn_sum[s] / type_counts[s]
                 for s in type_attn_sum if type_counts[s] >= 10}

        if valid:
            stypes = sorted(valid.keys())
            vals = [valid[s] for s in stypes]
            bar_labels = [f"{SHOT_TYPE_MAP.get(s, f'Type-{s}')}\n(n={type_counts[s]})" for s in stypes]

            plt.figure(figsize=(max(12, len(stypes) * 0.7), 5))
            bars = plt.bar(bar_labels, vals, color='coral', edgecolor='darkred', linewidth=0.5)
            # Highlight top-3
            top3_idx = np.argsort(vals)[-3:]
            for idx in top3_idx:
                bars[idx].set_color('crimson')
            plt.title('Global Aggregation: Average Top-Down Attention Weight per Shot Type\n'
                      '(Crimson = Top-3 most attended types across entire test set)')
            plt.ylabel('Normalized Attention Weight (avg)')
            plt.xlabel('Shot Type ID  (x-axis label includes observation count n)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'global_attention_by_type.png'), dpi=300)
            plt.close()
            print(f"Saved global_attention_by_type.png")

            # Print top types
            print("\n=== Top-5 most attended Shot Types (global) ===")
            for stype in sorted(valid, key=lambda x: -valid[x])[:5]:
                name = SHOT_TYPE_MAP.get(stype, f"Type-{stype}")
                print(f"  {name:>12}: avg_attn={valid[stype]:.4f}, n={type_counts[stype]}")


if __name__ == '__main__':
    main()
