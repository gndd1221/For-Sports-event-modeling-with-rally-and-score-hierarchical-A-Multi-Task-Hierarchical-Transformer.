import torch, json, os, sys, numpy as np
sys.path.insert(0, '.')
from src.default_config import get_model_class
from sklearn.decomposition import PCA

run_dir = 'outputs/results/table_tennis/run_table_tennis_task_attention_20260421_205443'
c = json.load(open(os.path.join(run_dir, 'config.json')))
ModelClass, over = get_model_class(c.get('model_type', 'sequence_attention'), False)
c.setdefault('model_args', {}).update(over)
model = ModelClass(c)

ckpt = torch.load(os.path.join(run_dir, 'train/weights/best_model.pth'), map_location='cpu')
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state, strict=False)

w = model.embedding_layers.type_embedding.weight.detach().numpy()[1:]
pca = PCA(n_components=2)
coords = pca.fit_transform(w)

M = {0:'Zero', 1:'Drive', 2:'Counter', 3:'Smash', 4:'Twist', 5:'FastDrive', 6:'FastPush(A)', 7:'Flip', 8:'LongPush(P)', 9:'FastPush(P)', 10:'LongPush', 11:'Drop', 12:'Chop', 13:'Block', 14:'Lob', 15:'Serve(Trad)', 16:'Serve(Hook)', 17:'Serve(Rev)', 18:'Serve(Squat)'}

for i in range(len(M)):
    print(f'{M[i]:>12}: X={coords[i,0]:.3f}, Y={coords[i,1]:.3f}')
