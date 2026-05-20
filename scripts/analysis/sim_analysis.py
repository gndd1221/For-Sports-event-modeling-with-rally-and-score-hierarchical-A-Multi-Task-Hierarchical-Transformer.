import torch, json, os, sys, numpy as np
sys.path.insert(0, '.')
from src.default_config import get_model_class

run_dir = 'outputs/results/table_tennis/run_table_tennis_task_attention_20260421_205443'
c = json.load(open(os.path.join(run_dir, 'config.json')))
ModelClass, over = get_model_class(c.get('model_type', 'sequence_attention'), False)
c.setdefault('model_args', {}).update(over)
model = ModelClass(c)

ckpt = torch.load(os.path.join(run_dir, 'train/weights/best_model.pth'), map_location='cpu')
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=False)

w = model.embedding_layers.type_embedding.weight.detach().numpy()[1:]
norm = np.linalg.norm(w, axis=1, keepdims=True)
norm = np.where(norm==0, 1e-8, norm)
w_norm = w / norm
sim = np.dot(w_norm, w_norm.T)

M = {0:'Zero', 1:'Drive', 2:'Counter', 3:'Smash', 4:'Twist', 5:'FastDrive', 6:'FastPush(A)', 7:'Flip', 8:'LongPush(P)', 9:'FastPush(P)', 10:'LongPush', 11:'Drop', 12:'Chop', 13:'Block', 14:'Lob', 15:'Serve(Trad)', 16:'Serve(Hook)', 17:'Serve(Rev)', 18:'Serve(Squat)'}

print('--- HIGH SIMILARITY (> 0.25) ---')
for i in range(len(M)):
    for j in range(i+1, len(M)):
        if sim[i, j] > 0.25:
            print(f'{M[i]} vs {M[j]}: {sim[i,j]:.2f}')

print('\n--- HIGH DISSIMILARITY (< -0.3) ---')
for i in range(len(M)):
    for j in range(i+1, len(M)):
        if sim[i, j] < -0.3:
            print(f'{M[i]} vs {M[j]}: {sim[i,j]:.2f}')
