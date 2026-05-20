import torch, json, pickle, os, sys
sys.path.insert(0, '.')
from src.default_config import get_model_class
import numpy as np
from sklearn.cluster import KMeans

# 1. Get Clusters
run_dir = 'outputs/results/table_tennis/run_table_tennis_task_attention_20260421_205443'
c = json.load(open(os.path.join(run_dir, 'config.json')))
ModelClass, over = get_model_class(c.get('model_type', 'sequence_attention'), False)
c.setdefault('model_args', {}).update(over)
model = ModelClass(c)
ckpt = torch.load(os.path.join(run_dir, 'train/weights/best_model.pth'), map_location='cpu')
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=False)
w = model.player_encoder.embedding.weight.detach().numpy()[1:]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(w)

player_to_cluster = {player_id: cluster for player_id, cluster in enumerate(clusters, start=1)}

# 2. Load Data
with open('data/table_tennis/processed_data/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 3. Analyze Shot Frequencies
SHOT_MAP = {0: 'Zero', 1: 'Drive', 2: 'Counter', 3: 'Smash', 4: 'Twist', 5: 'FastDrive', 6: 'FastPush(A)', 7: 'Flip', 8: 'LongPush(P)', 9: 'FastPush(P)', 10: 'LongPush', 11: 'Drop', 12: 'Chop', 13: 'Block', 14: 'Lob', 15: 'Serve(Trad)', 16: 'Serve(Hook)', 17: 'Serve(Rev)', 18: 'Serve(Squat)'}

cluster_counts = {0: {}, 1: {}, 2: {}}
cluster_total = {0: 0, 1: 0, 2: 0}

for match in data:
    for s in match['sets']:
        for r in s['rallies']:
            for shot in r['shots']:
                pid = int(shot[2])
                stype = int(shot[4]) - 1
                if pid in player_to_cluster and stype >= 0:
                    cid = player_to_cluster[pid]
                    cluster_counts[cid][stype] = cluster_counts[cid].get(stype, 0) + 1
                    cluster_total[cid] += 1

print('Cluster Analysis:')
for cid in range(3):
    print(f'\nCluster {cid} Profile (Total shots: {cluster_total[cid]}):')
    sorted_shots = sorted(cluster_counts[cid].items(), key=lambda x: x[1], reverse=True)
    for stype, count in sorted_shots[:10]:
        pct = count / cluster_total[cid] * 100
        print(f'  {SHOT_MAP.get(stype, str(stype))}: {pct:.2f}%')
