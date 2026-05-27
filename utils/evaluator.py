import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

class ModelEvaluator:
    """
    負責所有的指標計算、圖表繪製以及報告生成。
    將預測結果傳入 evaluate()，它會自動為所有支援的任務輸出完整的評估報告。
    """
    def __init__(self, output_dir, writer=None):
        self.output_dir = output_dir
        self.writer = writer
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    def _plot_confusion_matrix(self, cm, class_names, task_name):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Confusion Matrix for {task_name.capitalize()}', fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def _plot_metrics_barchart(self, metrics_dict, metric_name):
        tasks = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.bar(tasks, values, color=sns.color_palette("viridis", len(tasks)))
        
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison Across All Tasks', fontsize=16)
        ax.set_ylim(0, 1.05)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

        fig.tight_layout()
        plt.close(fig)
        return fig

    def _plot_save_roc(self, targets_one_hot, probs, num_classes, task_name):
        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(num_classes):
            if np.sum(targets_one_hot[:, i]) == 0:
                continue
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(targets_one_hot.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-ROC (area = {roc_auc["micro"]:0.4f})',
                 color='deeppink', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(num_classes), colors):
            if i in roc_auc:
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (area = {roc_auc[i]:0.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {task_name.capitalize()}')
        plt.legend(loc="lower right")
        
        save_path = os.path.join(self.output_dir, 'plots', f'ROC_{task_name}.png')
        plt.savefig(save_path)
        plt.close(fig)

    def evaluate(self, config, all_targets, all_preds, all_probs, avg_task_losses, custom_metrics=None, model_type="unknown", step=0):
        """
        config: 用來擷取 target, num_classes 資訊
        all_targets / all_preds / all_probs: 預測與標籤紀錄 {task_name: list}
        avg_task_losses: dict 標示各任務的 Validation/Test Loss
        custom_metrics: 紀錄距離或其他自訂指標如 {"location": {"avg_distance": 1.5}}
        """
        sport = config.get('sport', 'unknown')
        targets_list = config.get('targets', [])
        custom_metrics = custom_metrics or {}
        
        avg_loss = sum(avg_task_losses.values())

        # ============================================
        # Text 報告
        # ============================================
        result_path = os.path.join(self.output_dir, 'evaluation_results.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            def log_msg(msg):
                print(msg)
                f.write(msg + "\n")

            log_msg("="*60)
            log_msg(f"{'模型評估報告':^60}")
            log_msg("="*60)
            log_msg(f"  - Total Task Loss Sum    : {avg_loss:.4f}")
            log_msg("\n--- 各任務 Loss ---")
            for task, t_loss in avg_task_losses.items():
                log_msg(f"  - {task.capitalize():<15}: {t_loss:.4f}")
            
            for task_name, items in custom_metrics.items():
                for m_key, m_val in items.items():
                   log_msg(f"  - {task_name.capitalize()} {m_key}: {m_val:.4f}")
            log_msg("")

            accuracies, precisions, recalls, f1_scores = {}, {}, {}, {}
            csv_rows = []

            for task in targets_list:
                if task not in all_targets or not all_targets[task]: 
                    continue

                y_true = np.array(all_targets[task])
                y_pred = np.array(all_preds[task])
                y_prob = np.array(all_probs[task])
                
                num_classes = config.get('model_args', {}).get(f'num_{task}', config.get(f'num_{task}'))

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                targets_one_hot = np.eye(num_classes)[y_true]
                
                try:
                    # 過濾掉沒有正樣本的類別 (例如被 padding mask 濾除的 0，或測試集中未出現的類別)
                    valid_classes = [i for i in range(num_classes) if np.sum(targets_one_hot[:, i]) > 0]
                    if len(valid_classes) > 1 and y_prob.shape[1] == num_classes:
                        targets_one_hot_valid = targets_one_hot[:, valid_classes]
                        y_prob_valid = y_prob[:, valid_classes]
                        roc_w = roc_auc_score(targets_one_hot_valid, y_prob_valid, multi_class='ovr', average='weighted')
                    else:
                        roc_w = 0.0
                except (ValueError, IndexError):
                    roc_w = 0.0
                
                try:
                    fpr_m, tpr_m, _ = roc_curve(targets_one_hot.ravel(), y_prob.ravel())
                    roc_m = auc(fpr_m, tpr_m)
                except (ValueError, IndexError):
                    roc_m = 0.0

                task_cap = task.capitalize()
                accuracies[task_cap] = acc
                precisions[task_cap] = prec
                recalls[task_cap] = rec
                f1_scores[task_cap] = f1

                log_msg(f"--- 任務: {task_cap} ---")
                log_msg(f"  - Accuracy          : {acc:.4f}")
                log_msg(f"  - Precision         : {prec:.4f}")
                log_msg(f"  - Recall            : {rec:.4f}")
                log_msg(f"  - F1-Score          : {f1:.4f}")
                log_msg(f"  - ROC AUC (Weighted): {roc_w:.4f}")
                log_msg(f"  - ROC AUC (Micro)   : {roc_m:.4f}")
                
                custom_vals = custom_metrics.get(task, {})
                for m_key, m_val in custom_vals.items():
                    log_msg(f"  - {m_key.replace('_', ' ').capitalize()}: {m_val:.4f}")
                log_msg("")
                
                # TensorBoard Metrics
                if self.writer:
                    self.writer.add_scalar(f'Accuracy/{task}_val', acc, step)
                    self.writer.add_scalar(f'Loss/{task}_val', avg_task_losses.get(task, 0), step)
                
                # 繪圖
                cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
                class_names = [str(i) for i in range(num_classes)]
                cm_figure = self._plot_confusion_matrix(cm, class_names, task)
                cm_save_path = os.path.join(self.output_dir, 'plots', f'cm_{task}.png')
                cm_figure.savefig(cm_save_path)
                
                if self.writer:
                    self.writer.add_figure(f'ConfusionMatrix/{task_cap}', cm_figure, step)
                
                self._plot_save_roc(targets_one_hot, y_prob, num_classes, task)

                # CSV Report Preparation
                row = {
                    'model_type': model_type,
                    'sport': sport,
                    'task': task,
                    'accuracy': f'{acc:.4f}',
                    'precision': f'{prec:.4f}',
                    'recall': f'{rec:.4f}',
                    'f1': f'{f1:.4f}',
                    'roc_auc_weighted': f'{roc_w:.4f}',
                    'roc_auc_micro': f'{roc_m:.4f}',
                    'loss': f'{avg_task_losses.get(task, 0):.4f}',
                }
                # Attach custom metric values like 'avg_distance' into csv
                if custom_vals:
                     for k, v in custom_vals.items():
                         row[k] = f'{v:.4f}'
                csv_rows.append(row)

            # TensorBoard Summary
            if self.writer:
                if accuracies:
                    self.writer.add_figure('Metrics_Comparison/Accuracy', self._plot_metrics_barchart(accuracies, "Accuracy"), step)
                    self.writer.add_figure('Metrics_Comparison/Precision', self._plot_metrics_barchart(precisions, "Precision"), step)
                    self.writer.add_figure('Metrics_Comparison/Recall', self._plot_metrics_barchart(recalls, "Recall"), step)
                    self.writer.add_figure('Metrics_Comparison/F1-Score', self._plot_metrics_barchart(f1_scores, "F1-Score"), step)
                    
                self.writer.add_scalar('Loss/Total_Task_Loss_Sum', avg_loss, step)
                for task_name, items in custom_metrics.items():
                     for m_key, m_val in items.items():
                          self.writer.add_scalar(f'Metric/{task_name}_{m_key}', m_val, step)
                          
        # ============================================
        # CSV 報告
        # ============================================
        if csv_rows:
            csv_path = os.path.join(self.output_dir, 'evaluation_results.csv')
            
            # Combine all custom keys globally for fieldnames
            all_custom_keys = set()
            for items in custom_metrics.values():
                all_custom_keys.update(items.keys())
                
            fieldnames = ['model_type', 'sport', 'task', 'accuracy', 'precision', 'recall',
                          'f1', 'roc_auc_weighted', 'roc_auc_micro', 'loss'] + list(sorted(all_custom_keys))
                          
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer_csv.writeheader()
                writer_csv.writerows(csv_rows)
            print(f"CSV 結果已儲存: {csv_path}")
