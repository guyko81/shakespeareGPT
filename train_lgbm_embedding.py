import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from dataset import ShakespeareDataset
from dataclasses import dataclass
from tokenizers import Tokenizer
import time
from pathlib import Path
from gpt import ShakespeareGPT

# Setup paths (similar to train.py)
tokenizer_path = Path('./tokenizer/shakespeare.json')
tokenizer = Tokenizer.from_file(str(tokenizer_path))

@dataclass
class Config:
    use_characters = True # Set to False to use BPE tokens
    device = 'cuda' if torch.cuda.is_available() else 'cpu'        # 'cpu' or 'cuda'
    block_size = 256 # context-length
    batch_size = 10000 # Increased chunk size
    
    vocab_size = 0 # Set dynamically
    
    # Transformer params
    n_embed = 32 # Embedding dimension (small for feature extraction)
    n_heads = 2 # Number of attention heads
    n_layers = 2 # Number of transformer layers
    head_size = n_embed // n_heads # Computed automatically
    attn_dropout = 0.0 # No dropout for feature extraction
    block_dropout = 0.0 # No dropout for feature extraction
    
    train_size = 0.8 
    
    # LGBM params
    train_iters = 500 
    val_iters = 500 
    eval_interval = 10 # More frequent evaluation
    early_stopping_rounds = 5 # Stop if no improvement for 5 rounds

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

def prepare_full_data(data, block_size, transformer_model, device):
    """
    Extract features using the transformer model.
    
    Args:
        data: torch.Tensor of token IDs
        block_size: context length
        transformer_model: ShakespeareGPT model (without LM head, or we ignore its output)
        device: 'cpu' or 'cuda'
    
    Returns:
        X_flat: (N, block_size * n_embed) numpy array
        y: (N,) numpy array of target tokens
    """
    from numpy.lib.stride_tricks import sliding_window_view
    data_np = data.numpy().astype(np.int32)
    print(f"Preparing sliding windows for {len(data_np)} tokens...")
    windows = sliding_window_view(data_np, window_shape=block_size + 1)
    
    # Split into inputs and targets
    # X_ids: (N, block_size)
    # y: (N,)
    X_ids = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, -1])
    
    print(f"Extracting features with transformer (batch processing)...")
    
    # Convert to torch tensor
    X_ids_torch = torch.from_numpy(X_ids).long().to(device)
    
    # Extract features in batches to avoid OOM
    batch_size = 512  # Process 512 samples at a time
    all_features = []
    
    transformer_model.eval()
    with torch.no_grad():
        for i in range(0, len(X_ids_torch), batch_size):
            batch = X_ids_torch[i:i+batch_size]
            
            # Get token embeddings + positional embeddings
            B, T = batch.shape
            token_embs = transformer_model.token_embedding_table(batch)
            pos_embs = transformer_model.pos_embedding_table(torch.arange(T, device=device))
            x = token_embs + pos_embs
            
            # Pass through transformer blocks
            x = transformer_model.blocks(x)
            
            # x shape: (B, T, n_embed)
            # Flatten to (B, T * n_embed)
            x_flat = x.reshape(B, -1)
            
            all_features.append(x_flat.cpu().numpy())
            
            if i % 10000 == 0:
                print(f"Processed {i}/{len(X_ids_torch)} samples...", end='\r')
    
    X_flat = np.concatenate(all_features, axis=0).astype(np.float32)
    print(f"\nFeature extraction complete. Shape: {X_flat.shape}")
    
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

    # Initialize transformer for feature extraction
    print(f"Initializing transformer model on {Config.device}...")
    transformer = ShakespeareGPT(Config).to(Config.device)
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Keep transformer frozen (random initialization, no training)
    for param in transformer.parameters():
        param.requires_grad = False

    print("Preparing Training Data...")
    X_train, y_train = prepare_full_data(train_data_raw, Config.block_size, transformer, Config.device)
    print(f"X_train shape: {X_train.shape}")

    print("Preparing Validation Data...")
    X_val, y_val = prepare_full_data(val_data_raw, Config.block_size, transformer, Config.device)
    print(f"X_val shape: {X_val.shape}")

    # No categorical features anymore
    cat_features = None # list(range(Config.block_size))
    
    # Map torch device to LightGBM device
    lgbm_device = 'gpu' if Config.device == 'cuda' else 'cpu'
    
    # Parameters
    params = {
        'objective': 'multiclass',
        'num_class': Config.vocab_size,
        'metric': 'multi_logloss',
        'device': lgbm_device,
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
    model_path = Path('shakespeare_lgbm_model_embedding.txt')
    model.save_model(str(model_path))
    print(f"LightGBM model saved to {model_path}")
    
    # Save transformer model
    transformer_path = Path('shakespeare_transformer_feature_extractor.pth')
    torch.save(transformer.state_dict(), transformer_path)
    print(f"Transformer feature extractor saved to {transformer_path}")

if __name__ == "__main__":
    main()
