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
    n_estimators = 500
    learning_rate = 0.05

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

    print(f"Starting full training for {Config.n_estimators} rounds...")
    print(f"Vocab size: {Config.vocab_size}")
    
    # All context positions are categorical features (token IDs)
    cat_features = list(range(Config.block_size))
    
    print("Creating CatBoostClassifier (this may take a minute with categorical features)...")
    model = CatBoostClassifier(
        iterations=Config.n_estimators,
        learning_rate=Config.learning_rate,
        random_seed=42,
        cat_features=cat_features,
        task_type='GPU',
        depth=6,
        verbose=1
    )
    
    start_time = time.time()
    print("Model ready. Starting CatBoost train...")
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds")
    
    # Save model using CatBoost's native format
    model_path = Path('shakespeare_catboost_classifier.cbm')
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
