"""
default_config.py - 集中管理所有預設配置值

集中管理 train/test 共用的模型與運動設定，支援桌球、羽球與網球。
"""
import importlib
import os

# =====================================================================================
# 模型註冊表
# MT-HTA 變體由 model_fuse.py 與 config overrides 區分；baseline 使用各自的模型檔。
# =====================================================================================
MODEL_REGISTRY = {
    'cls_token_itransformer_feature_token': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'cls_token',
            'encoder_path_mode': 'dual',
            'skip_window_size': 0,
            'use_gated_fusion': False,
            'use_shot_aware_pe': False,
            'use_top_down_attention': False,
            'use_turn_based_gating': False,
            'use_temporal_scale_gating': False,
            'use_task_decoder': False,
        }
    },
    'task_project_itransformer_feature_token': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'task_project',
            'encoder_path_mode': 'dual',
            'skip_window_size': 0,
            'use_gated_fusion': False,
            'use_shot_aware_pe': False,
            'use_top_down_attention': False,
            'use_turn_based_gating': False,
            'use_temporal_scale_gating': False,
            'use_task_decoder': False,
        }
    },
    'task_attention': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'dual',
        }
    },
    'task_attention_itransformer_feature_token': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'dual',
        }
    },
    'task_attention_L1': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'hierarchy_levels': ['L1'],
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'dual',
            'skip_window_size': 0,
            'use_gated_fusion': False,
            'use_shot_aware_pe': False,
            'use_top_down_attention': False,
            'use_turn_based_gating': False,
            'use_temporal_scale_gating': False,
            'use_task_decoder': False,
        }
    },
    'task_attention_L1_L2': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'hierarchy_levels': ['L1', 'L2'],
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'dual',
            'skip_window_size': 0,
            'use_gated_fusion': False,
            'use_shot_aware_pe': False,
            'use_top_down_attention': False,
            'use_turn_based_gating': False,
            'use_temporal_scale_gating': False,
            'use_task_decoder': False,
        }
    },
    'task_attention_wo_itransformer': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'pact_only',
            'skip_window_size': 0,
            'use_gated_fusion': False,
            'use_shot_aware_pe': False,
            'use_top_down_attention': False,
            'use_turn_based_gating': False,
            'use_temporal_scale_gating': False,
            'use_task_decoder': False,
        }
    },
    'task_attention_final_wo_itransformer': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'pact_only',
            'skip_window_size': 1,
            'use_gated_fusion': True,
            'use_shot_aware_pe': True,
            'use_top_down_attention': True,
            'use_turn_based_gating': True,
            'use_temporal_scale_gating': False,
            'use_task_decoder': False,
        }
    },
    'sequence_attention': {
        'module': 'src.models.model_fuse',
        'config_overrides': {
            'fusion_type': 'task_attention',
            'encoder_path_mode': 'dual',
            'use_sequence_fusion': True,
            'num_fusion_layers': 2,
        }
    },
    'baseline_lstm': {
        'module': 'src.models.baseline_lstm',
        'class_name': 'BaselineLSTM',
        'config_overrides': {} 
    },
    'baseline_shuttlenet_full': {
        'module': 'src.models.baseline_shuttlenet_full',
        'class_name': 'PACTModel',
        'config_overrides': {
            'pooling_type': 'last'
        }
    },
    'baseline_h_lstm': {
        'module': 'src.models.baseline_h_lstm',
        'class_name': 'BaselineHLSTM',
        'config_overrides': {}
    },
    'baseline_lstm_flat': {
        'module': 'src.models.baseline_lstm_flat',
        'class_name': 'BaselineLSTMFlat',
        'config_overrides': {}
    },
    'baseline_lstm_context': {
        'module': 'src.models.baseline_lstm_context',
        'class_name': 'BaselineLSTMContext',
        'config_overrides': {}
    },
    'baseline_transformer_flat': {
        'module': 'src.models.baseline_transformer_flat',
        'class_name': 'BaselineTransformerFlat',
        'config_overrides': {}
    },
    'baseline_patchtst': {
        'module': 'src.models.baseline_patchtst',
        'class_name': 'BaselinePatchTST',
        'config_overrides': {
            'patch_len': 16,
            'patch_stride': 8,
            'patch_feature_dim': 8,
            'patch_pooling': 'mean',
            'patch_max_tokens': 1024,
        }
    },
    'baseline_itransformer_feature_token': {
        'module': 'src.models.baseline_itransformer_feature_token',
        'class_name': 'BaselineITransformerFeatureToken',
        'config_overrides': {}
    }
}

# 可用的運動列表
AVAILABLE_SPORTS = ['table_tennis', 'badminton', 'tennis', 'table_tennis_all', 'badminton_all']


def get_model_class(model_type):
    """
    依名稱動態載入模型類別。
    
    Args:
        model_type: 模型類型名稱
    
    Returns:
        (model_class, config_overrides_dict)
        - config_overrides 應被合併到 config['model_args'] 中
    """
    # 新版統一模型
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"未知的模型類型: '{model_type}'。可用的類型: {available}")
    
    entry = MODEL_REGISTRY[model_type]
    module_name = entry['module']
         
    module = importlib.import_module(module_name)
        
    class_name = entry.get('class_name', 'PACTModel')
    ModelClass = getattr(module, class_name)
    
    return ModelClass, entry['config_overrides']


# =====================================================================================
# 運動別配置載入
# =====================================================================================
def load_sport_config(sport_name):
    """
    從 configs/<sport_name>.yaml 載入運動別配置。

    Args:
        sport_name: 運動名稱 (table_tennis, badminton, tennis)

    Returns:
        dict: 運動別配置 (features, targets, loss_weights, embedding_dims, ...)
    """
    import yaml
    
    # 從專案根目錄查找 configs/ (default_config.py 現在在 src/ 中，需上一層)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', f'{sport_name}.yaml')
    
    if not os.path.exists(config_path):
        available = [f.replace('.yaml', '') for f in os.listdir(os.path.join(base_dir, 'configs')) 
                     if f.endswith('.yaml')]
        raise FileNotFoundError(
            f"找不到運動配置: '{config_path}'。可用的運動: {available}"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        sport_config = yaml.safe_load(f)
    
    return sport_config


def build_full_config(sport_name, base_config):
    """
    將運動別配置與資料預處理產生的 base_config 合併為完整配置。

    特徵與任務定義皆由對應的運動設定檔提供。

    Args:
        sport_name: 運動名稱 (table_tennis, badminton, tennis)
        base_config: 從 processed_data/config.json 載入的 dict
            (包含 num_players, num_type, num_location 等來自預處理的數值)

    Returns:
        更新後的 config dict (原地修改並回傳)
    """
    sport_config = load_sport_config(sport_name)

    # 注入運動別特徵定義
    base_config['sport'] = sport_config['sport']
    base_config['features_to_extract'] = sport_config['features_to_extract']
    base_config['categorical_features'] = sport_config['categorical_features']
    base_config['numerical_features'] = sport_config['numerical_features']
    base_config['embedding_dims'] = sport_config['embedding_dims']
    base_config['targets'] = sport_config['targets']
    base_config['loss_weights'] = sport_config.get('loss_weights', {})

    # 從 base_config 建構 vocab_sizes (每個類別特徵的字典大小)
    base_config['vocab_sizes'] = {
        feat: base_config[f'num_{feat}']
        for feat in sport_config['categorical_features']
    }

    # 注入序列長度上限 (優先用 sport_config，fallback 到 base_config)
    for key in ['max_shot_seq_len', 'max_rally_seq_len', 'max_set_seq_len', 'max_game_seq_len']:
        if key in sport_config:
            base_config[key] = sport_config[key]

    # 注入階層配置 (4 層運動如網球會指定 hierarchy_levels)
    if 'hierarchy_levels' in sport_config:
        base_config['hierarchy_levels'] = sport_config['hierarchy_levels']

    return base_config


# =====================================================================================
# 預設訓練配置
# =====================================================================================
DEFAULT_LOSS_WEIGHTS = {
    'type': 0.2, 'location': 0.2, 'strength': 0.2, 'spin': 0.2, 'backhand': 0.2
}

DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_GRID_OFFSET = 2
DEFAULT_NUM_WORKERS = 0 if os.name == 'nt' else 4
