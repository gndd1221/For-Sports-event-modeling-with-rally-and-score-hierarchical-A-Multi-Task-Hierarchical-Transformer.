import os
import sys
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.default_config import get_model_class

SHOT_TYPE_MAP = {
    0: "Zero", 1: "Drive", 2: "Counter", 3: "Smash", 4: "Twist",
    5: "FastDrive", 6: "FastPush(A)", 7: "Flip", 8: "LongPush(P)", 
    9: "FastPush(P)", 10: "LongPush", 11: "Drop", 12: "Chop", 
    13: "Block", 14: "Lob", 15: "Serve(Trad)", 16: "Serve(Hook)", 
    17: "Serve(Rev)", 18: "Serve(Squat)"
}

def cosine_similarity_matrix(weights):
    # weights: (N, D)
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-8, norm) # avoid div by zero
    w_norm = weights / norm
    return np.dot(w_norm, w_norm.T)

def analyze_embeddings(run_dir):
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config at {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    model_type = config.get('model_type', 'sequence_attention')
    legacy = config.get('legacy_model', False)
    
    # Load Model
    ModelClass, overrides = get_model_class(model_type, legacy=legacy)
    if 'model_args' not in config:
        config['model_args'] = {}
    config['model_args'].update(overrides)
    
    model = ModelClass(config)
    ckpt_path = os.path.join(run_dir, 'train', 'weights', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        # Fallback for other potential locations
        fallback_path = os.path.join(run_dir, 'checkpoint_best.pth')
        if os.path.exists(fallback_path):
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path} or {fallback_path}")
        
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    print(f"Successfully loaded model weights from {ckpt_path}")
    
    out_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    # ─── 1. Analyze Shot Type Embeddings ──────────────────────────────────────
    print("Analyzing Shot Type Embeddings...")
    if hasattr(model, 'embedding_layers') and hasattr(model.embedding_layers, 'type_embedding'):
        type_emb_layer = model.embedding_layers.type_embedding
        type_weights = type_emb_layer.weight.detach().numpy() 
        
        # Index 0 is padding. Active indices are 1 to num_classes-1.
        num_classes = type_weights.shape[0]
        active_indices = list(range(1, num_classes))
        active_weights = type_weights[1:num_classes]
        
        # Raw index = model_index - 1 (because of +1 padding shift)
        labels = [SHOT_TYPE_MAP.get(i-1, f"Type-{i-1}") for i in active_indices] 
        
        # a) Cosine Similarity Heatmap
        sim_matrix = cosine_similarity_matrix(active_weights)
        plt.figure(figsize=(14, 12))  # 稍微放大畫布以容納數值
        sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 7})
        plt.title('Cosine Similarity between Shot Type Embeddings', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'embedding_type_similarity.png'), dpi=300)
        plt.close()
        print("Saved embedding_type_similarity.png")
        
        # b) PCA 2D Projection
        pca = PCA(n_components=2)
        type_pca = pca.fit_transform(active_weights)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(type_pca[:, 0], type_pca[:, 1], c='blue', s=50, alpha=0.6)
        for i, label in enumerate(labels):
            # Highlight serves differently
            if "Serve" in label:
                color = "red"
                fontweight = "bold"
            else:
                color = "black"
                fontweight = "normal"
            plt.annotate(label, (type_pca[i, 0], type_pca[i, 1]), 
                         fontsize=10, color=color, fontweight=fontweight,
                         xytext=(5, 5), textcoords='offset points')
            
        plt.title('PCA Projection of Shot Type Embeddings', fontsize=14)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'embedding_type_pca.png'), dpi=300)
        plt.close()
        print("Saved embedding_type_pca.png")
    else:
        print("Model does not have model.embedding_layers.type_embedding. Skipping.")

    # ─── 2. Analyze Player Embeddings ─────────────────────────────────────────
    print("Analyzing Player Embeddings...")
    if hasattr(model, 'player_encoder') and hasattr(model.player_encoder, 'embedding'):
        player_weights = model.player_encoder.embedding.weight.detach().numpy()
        active_player_weights = player_weights[1:] # Skip padding index 0
        num_players = active_player_weights.shape[0]
        
        if num_players >= 3:
            # KMeans Clustering
            n_clusters = min(3, num_players)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(active_player_weights)
            
            # UMAP
            n_neighbors = min(15, num_players - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            player_umap = reducer.fit_transform(active_player_weights)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(player_umap[:, 0], player_umap[:, 1], 
                                  c=clusters, cmap='viridis', s=40, alpha=0.8)
                                  
            handles, _ = scatter.legend_elements()
            plt.legend(handles, [f"Style Cluster {i+1}" for i in range(n_clusters)], title="K-Means Clusters")
            
            plt.title(f'UMAP Projection of Player Style Embeddings (N={num_players})', fontsize=14)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'embedding_player_umap.png'), dpi=300)
            plt.close()
            print("Saved embedding_player_umap.png")
        else:
            print("Not enough players to do UMAP/Clustering.")
    else:
        print("Model does not have model.player_encoder.embedding. Skipping.")

    print(f"\nAll embedding plots saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Directory containing config.json and checkpoint_best.pth')
    args = parser.parse_args()
    
    analyze_embeddings(args.run_dir)

if __name__ == '__main__':
    main()
