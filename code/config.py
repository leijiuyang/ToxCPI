




import os
from pathlib import Path
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = str(REPOSITORY_ROOT)
DATA_DIR = str(REPOSITORY_ROOT / 'data')



BINDING_DATA_PATH = os.path.join(DATA_DIR, 'binding', 'combined_balanced_n72045_with_seq.csv')
EFFECT_DATA_PATH = os.path.join(DATA_DIR, 'activation')
NO_EFFECT_DATA_PATH = os.path.join(DATA_DIR, 'inhibition')


MODEL_SAVE_DIR = str(REPOSITORY_ROOT / 'runtime' / 'checkpoints')
LOG_DIR = str(REPOSITORY_ROOT / 'runtime' / 'logs')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"


class ModelConfig:

    chemberta_hidden_dim = 768
    chemberta_freeze_layers = 6


    attentivefp_node_feat_size = 39
    attentivefp_edge_feat_size = 10
    attentivefp_num_layers = 3
    attentivefp_num_timesteps = 2
    attentivefp_graph_feat_size = 256


    toxprint_dim = 729
    benigni_dim = 33
    edc_dim = 50
    toxalert_hidden_dim = 256


    esm2_hidden_dim = 1280
    esm2_freeze_layers = 20


    hidden_dim = 256


    bilinear_hidden_dim = 256
    interaction_dim = 512


    predictor_hidden_dim = 256
    dropout = 0.3


    max_smiles_length = 512
    max_protein_length = 1024


class TrainConfig:

    batch_size = 128
    eval_batch_size = 128


    num_epochs = 30


    learning_rate = 1e-4
    weight_decay = 1e-5


    lr_scheduler = 'cosine'
    warmup_epochs = 5


    early_stopping_patience = 10


    binding_loss_weight = 1.0
    effect_loss_weight = 1.0
    type_loss_weight = 1.0


    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15


    cold_split = True


    mc_dropout_samples = 30


    seed = 42


    log_interval = 100
    save_interval = 1


class EvalConfig:

    metrics = ['auroc', 'auprc', 'accuracy', 'f1', 'mcc']


    binding_threshold = 0.5
    effect_threshold = 0.5


    save_attention_maps = True
    top_k_atoms = 10
    top_k_residues = 10


model_config = ModelConfig()
train_config = TrainConfig()
eval_config = EvalConfig()
