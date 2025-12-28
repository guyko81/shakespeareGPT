import lightgbm as lgb
import numpy as np
import torch
from dataset import ShakespeareDataset
from dataclasses import dataclass
from tokenizers import Tokenizer
import time
from pathlib import Path

# Setup paths (similar to train.py)
tokenizer_path = Path('./tokenizer/shakespeare.json')
tokenizer = Tokenizer.from_file(str(tokenizer_path))

@dataclass
class Config:
    use_characters = True # Set to False to use BPE tokens
    device = 'cpu'        # 'cpu' or 'gpu'
    block_size = 256 # context-length
    batch_size = 10000 # Increased chunk size
    
    vocab_size = 0 # Set dynamically
    n_embed = 4 # Embedding dimension
    
    train_size = 0.8 
    
    # NN params
    train_iters = 500 
    val_iters = 500 
    eval_interval = 10 # More frequent evaluation
    early_stopping_rounds = 5 # Stop if no improvement for 50 rounds

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
    def encode(self, s):
        return [self.stoi[c] for c in s]
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    def get_vocab_size(self):
        return self.vocab_size

def prepare_full_data(data, block_size, token_embeddings, pos_embeddings):
    from numpy.lib.stride_tricks import sliding_window_view
    data_np = data.numpy().astype(np.int32)
    print(f"Preparing sliding windows for {len(data_np)} tokens...")
    windows = sliding_window_view(data_np, window_shape=block_size + 1)
    
    # Split into inputs and targets
    # X_ids: (N, block_size)
    # y: (N,)
    X_ids = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, -1])
    
    # Map to embeddings
    # token_embeddings: (vocab_size, n_embed)
    # pos_embeddings: (block_size, n_embed)
    
    # 1. Get token embeddings for all inputs
    # Shape: (N, block_size, n_embed)
    X_emb = token_embeddings[X_ids] 
    
    # 2. Add positional embeddings
    # Shape broadcasts: (N, block_size, n_embed) + (1, block_size, n_embed)
    X_emb = X_emb + pos_embeddings[None, :, :]
    
    # 3. Flatten
    # Shape: (N, block_size * n_embed)
    N, T, C = X_emb.shape
    X_flat = X_emb.reshape(N, T * C)
    
    return X_flat, y

def main():
    print("Initializing datasets...")
    
    # Load raw data to build character tokenizer if needed
    with open('./data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    if Config.use_characters:
        tokenizer = CharTokenizer(text)
        print(f"Using character-level tokenization. Vocab size: {tokenizer.get_vocab_size()}")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Using BPE tokenization. Vocab size: {tokenizer.get_vocab_size()}")

    Config.vocab_size = tokenizer.get_vocab_size()
    
    # We still use ShakespeareDataset but prepare full data arrays
    train_ds = ShakespeareDataset(Config, is_test=False)
    val_ds = ShakespeareDataset(Config, is_test=True)
    
    if Config.use_characters:
        full_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(full_data))
        train_data_raw = full_data[:n]
        val_data_raw = full_data[n:]
    else:
        train_data_raw = train_ds.data
        val_data_raw = val_ds.data

    # Create random embeddings
    print("Creating random embeddings...")
    # Fixed random seed for reproducibility of embeddings
    rng = np.random.RandomState(42)
    token_embeddings = rng.randn(Config.vocab_size, Config.n_embed).astype(np.float32)
    pos_embeddings = rng.randn(Config.block_size, Config.n_embed).astype(np.float32)

    print("Preparing Training Data...")
    X_train, y_train = prepare_full_data(train_data_raw, Config.block_size, token_embeddings, pos_embeddings)
    print(f"X_train shape: {X_train.shape}")

    print("Preparing Validation Data...")
    X_val, y_val = prepare_full_data(val_data_raw, Config.block_size, token_embeddings, pos_embeddings)
    print(f"X_val shape: {X_val.shape}")

    # No categorical features anymore
    cat_features = None # list(range(Config.block_size))
    
    # Parameters
    params = {
        'objective': 'multiclass',
        'num_class': Config.vocab_size,
        'metric': 'multi_logloss',
        'device': Config.device,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'seed': 42,
        'num_threads': -1,    # Use all available cores
        'max_depth': 16,
        'num_leaves': 20000,
        'learning_rate': 0.1,
        'min_data_in_leaf': 1,
        'cat_l2': 1.0,
        'subsample': 1.0,
        'feature_fraction': 1.0
    }
    
    print(f"Starting full training for {Config.train_iters} rounds...")
    print(f"Vocab size: {Config.vocab_size}")
    
    print("Creating training Dataset (this may take a minute with categorical features)...")
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    
    print("Creating validation Dataset...")
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)
    
    start_time = time.time()
    print("Dataset ready. Starting LightGBM train...")
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=Config.train_iters,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=1),
            lgb.early_stopping(stopping_rounds=Config.early_stopping_rounds)
        ]
    )
    
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds")
    
    # Save model
    model_path = Path('shakespeare_lgbm_model.txt')
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
