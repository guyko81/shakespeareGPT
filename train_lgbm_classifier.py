import numpy as np
import torch
from dataset import ShakespeareDataset
from dataclasses import dataclass
from tokenizers import Tokenizer
import time
from pathlib import Path
from catboost import CatBoostClassifier

# Setup paths (similar to train.py)
tokenizer_path = Path('./tokenizer/shakespeare.json')

@dataclass
class Config:
    use_characters = True # Set to False to use BPE tokens
    block_size = 256 # context-length
    batch_size = 10000 # Required by ShakespeareDataset
    
    vocab_size = 0 # Set dynamically
    
    train_size = 0.8 
    
    # Model params
    iterations = 500
    learning_rate = 0.05
    depth = 6                    # Tree depth (max 16 for GPU)
    l2_leaf_reg = 3.0            # L2 regularization coefficient
    random_seed = 42
    
    # GPU params
    task_type = 'GPU'            # 'CPU' or 'GPU'
    devices = '0'                # GPU device id(s), e.g. '0' or '0:1' for multi-gpu
    gpu_ram_part = 0.95          # Fraction of GPU RAM to use
    
    # Categorical feature params
    one_hot_max_size = 2         # Max size for one-hot encoding (use CTR for larger)
    max_ctr_complexity = 4       # Max number of categorical features to combine
    
    # Tree structure
    grow_policy = 'SymmetricTree'  # 'SymmetricTree', 'Depthwise', 'Lossguide'
    min_data_in_leaf = 1         # Minimum samples in leaf
    
    # Boosting params
    bootstrap_type = 'Bayesian'  # 'Bayesian', 'Bernoulli', 'MVS', 'Poisson' (GPU multiclass: Bayesian/Bernoulli)
    bagging_temperature = 1.0    # For Bayesian bootstrap (higher = more randomness)
    sampling_frequency = 'PerTree'  # 'PerTree' or 'PerTreeLevel'
    
    # Early stopping
    early_stopping_rounds = 50   # Stop if no improvement for N rounds (None to disable)
    use_best_model = True        # Use best iteration based on eval metric
    
    # Logging
    verbose = 1                  # Print every N iterations (1 = every iter)

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

def prepare_full_data(data, block_size):
    from numpy.lib.stride_tricks import sliding_window_view
    data_np = data.numpy().astype(np.int32)
    print(f"Preparing sliding windows for {len(data_np)} tokens...")
    windows = sliding_window_view(data_np, window_shape=block_size + 1)
    
    X = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, -1])
    return X, y

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

    print("Preparing Training Data...")
    X_train, y_train = prepare_full_data(train_data_raw, Config.block_size)
    print(f"X_train shape: {X_train.shape}")

    print("Preparing Validation Data...")
    X_val, y_val = prepare_full_data(val_data_raw, Config.block_size)
    print(f"X_val shape: {X_val.shape}")

    print(f"Starting full training for {Config.iterations} rounds...")
    print(f"Vocab size: {Config.vocab_size}")
    
    # All context positions are categorical features (token IDs)
    cat_features = list(range(Config.block_size))
    
    print("Creating CatBoostClassifier (this may take a minute with categorical features)...")
    model = CatBoostClassifier(
        # Core params
        iterations=Config.iterations,
        learning_rate=Config.learning_rate,
        depth=Config.depth,
        l2_leaf_reg=Config.l2_leaf_reg,
        random_seed=Config.random_seed,
        
        # Loss and eval
        loss_function='MultiClass',
        eval_metric='MultiClass',
        
        # GPU params
        task_type=Config.task_type,
        devices=Config.devices,
        gpu_ram_part=Config.gpu_ram_part,
        
        # Categorical feature params
        cat_features=cat_features,
        one_hot_max_size=Config.one_hot_max_size,
        max_ctr_complexity=Config.max_ctr_complexity,
        
        # Tree structure
        grow_policy=Config.grow_policy,
        min_data_in_leaf=Config.min_data_in_leaf,
        
        # Boosting params
        bootstrap_type=Config.bootstrap_type,
        bagging_temperature=Config.bagging_temperature,
        sampling_frequency=Config.sampling_frequency,
        
        # Early stopping
        early_stopping_rounds=Config.early_stopping_rounds,
        use_best_model=Config.use_best_model,
        
        # Logging
        verbose=Config.verbose
    )
    
    start_time = time.time()
    print("Model ready. Starting CatBoost train...")
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=Config.early_stopping_rounds)
    
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds")
    
    # Save model using CatBoost's native format
    model_path = Path('shakespeare_catboost_classifier.cbm')
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
