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
    n_embed = 384 # Embedding dimension (small for feature extraction)
    n_heads = 6 # Number of attention heads
    n_layers = 6 # Number of transformer layers
    head_size = n_embed // n_heads # Computed automatically
    attn_dropout = 0.0 # No dropout for feature extraction
    block_dropout = 0.0 # No dropout for feature extraction
    
    # Feature projection params
    output_dimension = 64 # Dimensionality reduction: flatten(block_size * n_embed) -> output_dimension
    
    # Pre-training params
    pretrain_epochs = 10 # Number of epochs to pre-train the feature extractor
    pretrain_batch_size = 64 # Batch size for pre-training
    pretrain_learning_rate = 1e-3 # Learning rate for pre-training
    
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

class FeatureExtractor(nn.Module):
    """Wrapper around ShakespeareGPT that adds dimensionality reduction."""
    def __init__(self, transformer_model, input_dim, output_dim):
        super().__init__()
        self.transformer = transformer_model
        self.projection = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, idx):
        """
        Args:
            idx: (B, T) token indices
        Returns:
            (B, output_dim) projected features
        """
        B, T = idx.shape
        device = idx.device
        
        # Get token embeddings + positional embeddings
        token_embs = self.transformer.token_embedding_table(idx)
        pos_embs = self.transformer.pos_embedding_table(torch.arange(T, device=device))
        x = token_embs + pos_embs
        
        # Pass through transformer blocks
        x = self.transformer.blocks(x)
        
        # x shape: (B, T, n_embed)
        # Flatten to (B, T * n_embed)
        x_flat = x.reshape(B, -1)
        
        # Project to output dimension
        features = self.projection(x_flat)
        
        return features

class FeatureExtractorWithLMHead(nn.Module):
    """Feature extractor with language modeling head for pre-training."""
    def __init__(self, feature_extractor, vocab_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.lm_head = nn.Linear(feature_extractor.output_dim, vocab_size)
        
    def forward(self, idx):
        """
        Args:
            idx: (B, T) token indices
        Returns:
            logits: (B, vocab_size) predictions for next token
        """
        features = self.feature_extractor(idx)
        logits = self.lm_head(features)
        return logits

def pretrain_feature_extractor(model, train_data, val_data, config, device):
    """
    Pre-train the feature extractor with language modeling objective.
    
    Args:
        model: FeatureExtractorWithLMHead
        train_data: torch.Tensor of training token IDs
        val_data: torch.Tensor of validation token IDs
        config: Config object
        device: 'cpu' or 'cuda'
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    print("\n" + "="*60)
    print("PRE-TRAINING FEATURE EXTRACTOR")
    print("="*60)
    
    # Prepare data
    train_np = train_data.numpy().astype(np.int32)
    val_np = val_data.numpy().astype(np.int32)
    
    train_windows = sliding_window_view(train_np, window_shape=config.block_size + 1)
    val_windows = sliding_window_view(val_np, window_shape=config.block_size + 1)
    
    X_train = torch.from_numpy(np.ascontiguousarray(train_windows[:, :-1])).long()
    y_train = torch.from_numpy(np.ascontiguousarray(train_windows[:, -1])).long()
    
    X_val = torch.from_numpy(np.ascontiguousarray(val_windows[:, :-1])).long()
    y_val = torch.from_numpy(np.ascontiguousarray(val_windows[:, -1])).long()
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config.pretrain_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        print(f"\nEpoch {epoch + 1}/{config.pretrain_epochs}")
        
        for i in range(0, len(X_train), config.pretrain_batch_size):
            batch_X = X_train_shuffled[i:i+config.pretrain_batch_size].to(device)
            batch_y = y_train_shuffled[i:i+config.pretrain_batch_size].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress update every 1000 batches
            if num_batches % 1000 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {num_batches}: Train Loss = {avg_loss:.4f}", end='\r')
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), config.pretrain_batch_size):
                batch_X = X_val[i:i+config.pretrain_batch_size].to(device)
                batch_y = y_val[i:i+config.pretrain_batch_size].to(device)
                
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                val_batches += 1
                
                # Calculate accuracy
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total
        
        print(f"  Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    
    print("="*60 + "\n")
    print("Pre-training complete! Feature extractor now has learned representations.")
    
    return model

def prepare_full_data(data, block_size, feature_extractor, device):
    """
    Extract features using the feature extractor model (transformer + projection).
    
    Args:
        data: torch.Tensor of token IDs
        block_size: context length
        feature_extractor: FeatureExtractor model (transformer + projection layer)
        device: 'cpu' or 'cuda'
    
    Returns:
        X_features: (N, output_dimension) numpy array
        y: (N,) numpy array of target tokens
    """
    from numpy.lib.stride_tricks import sliding_window_view
    data_np = data.numpy().astype(np.int32)
    print(f"Step 1/4: Preparing sliding windows for {len(data_np):,} tokens...")
    windows = sliding_window_view(data_np, window_shape=block_size + 1)
    
    # Split into inputs and targets
    # X_ids: (N, block_size)
    # y: (N,)
    X_ids = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, -1])
    
    print(f"Step 2/4: Created {len(X_ids):,} samples")
    print(f"Step 3/4: Extracting features with transformer + projection (batch processing)...")
    
    # Convert to torch tensor
    X_ids_torch = torch.from_numpy(X_ids).long()
    
    # Extract features in batches to avoid OOM
    batch_size = 4096
    total_batches = (len(X_ids_torch) + batch_size - 1) // batch_size
    
    # Preallocate output array to avoid memory issues during concatenation
    n_samples = len(X_ids_torch)
    n_features = feature_extractor.output_dim
    
    print(f"Step 3a: Preallocating output array ({n_samples:,} samples x {n_features:,} features)...")
    expected_size_mb = n_samples * n_features * 4 / (1024**2)  # float32 = 4 bytes
    print(f"  Expected memory usage: ~{expected_size_mb:.1f} MB")
    
    X_features = np.zeros((n_samples, n_features), dtype=np.float32)
    
    print(f"Step 3b: Extracting features and writing directly to preallocated array...")
    feature_extractor.eval()
    
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(X_ids_torch), batch_size)):
            batch = X_ids_torch[i:i+batch_size].to(device)
            
            # Get features from feature extractor (includes transformer + projection)
            features_batch = feature_extractor(batch).cpu().numpy()
            
            # Write directly to preallocated array
            end_idx = min(i + batch_size, n_samples)
            X_features[i:end_idx] = features_batch
            
            # Progress logging every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                processed = min((batch_idx + 1) * batch_size, len(X_ids_torch))
                print(f"  Batch {batch_idx + 1}/{total_batches}: Processed {processed:,}/{len(X_ids_torch):,} samples ({100 * processed / len(X_ids_torch):.1f}%)")
    
    print(f"Step 4/4: Feature extraction complete!")
    print(f"  Final shape: {X_features.shape}")
    print(f"  Memory usage: ~{X_features.nbytes / (1024**2):.1f} MB")
    
    return X_features, y

    """
    Extract features using the feature extractor model (transformer + projection).
    
    Args:
        data: torch.Tensor of token IDs
        block_size: context length
        feature_extractor: FeatureExtractor model (transformer + projection layer)
        device: 'cpu' or 'cuda'
    
    Returns:
        X_features: (N, output_dimension) numpy array
        y: (N,) numpy array of target tokens
    """
    from numpy.lib.stride_tricks import sliding_window_view
    data_np = data.numpy().astype(np.int32)
    print(f"Step 1/4: Preparing sliding windows for {len(data_np):,} tokens...")
    windows = sliding_window_view(data_np, window_shape=block_size + 1)
    
    # Split into inputs and targets
    # X_ids: (N, block_size)
    # y: (N,)
    X_ids = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, -1])
    
    print(f"Step 2/4: Created {len(X_ids):,} samples")
    print(f"Step 3/4: Extracting features with transformer + projection (batch processing)...")
    
    # Convert to torch tensor
    X_ids_torch = torch.from_numpy(X_ids).long()
    
    # Extract features in batches to avoid OOM
    batch_size = 4096
    total_batches = (len(X_ids_torch) + batch_size - 1) // batch_size
    
    # Preallocate output array to avoid memory issues during concatenation
    n_samples = len(X_ids_torch)
    n_features = feature_extractor.output_dim
    
    print(f"Step 3a: Preallocating output array ({n_samples:,} samples x {n_features:,} features)...")
    expected_size_mb = n_samples * n_features * 4 / (1024**2)  # float32 = 4 bytes
    print(f"  Expected memory usage: ~{expected_size_mb:.1f} MB")
    
    X_features = np.zeros((n_samples, n_features), dtype=np.float32)
    
    print(f"Step 3b: Extracting features and writing directly to preallocated array...")
    feature_extractor.eval()
    
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(X_ids_torch), batch_size)):
            batch = X_ids_torch[i:i+batch_size].to(device)
            
            # Get features from feature extractor (includes transformer + projection)
            features_batch = feature_extractor(batch).cpu().numpy()
            
            # Write directly to preallocated array
            end_idx = min(i + batch_size, n_samples)
            X_features[i:end_idx] = features_batch
            
            # Progress logging every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                processed = min((batch_idx + 1) * batch_size, len(X_ids_torch))
                print(f"  Batch {batch_idx + 1}/{total_batches}: Processed {processed:,}/{len(X_ids_torch):,} samples ({100 * processed / len(X_ids_torch):.1f}%)")
    
    print(f"Step 4/4: Feature extraction complete!")
    print(f"  Final shape: {X_features.shape}")
    print(f"  Memory usage: ~{X_features.nbytes / (1024**2):.1f} MB")
    
    return X_features, y

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

    # Initialize transformer and feature extractor
    print(f"Initializing transformer model on {Config.device} for feature extraction...")
    transformer = ShakespeareGPT(Config).to(Config.device)
    
    # Create feature extractor with projection layer
    input_dim = Config.block_size * Config.n_embed
    feature_extractor = FeatureExtractor(transformer, input_dim, Config.output_dimension).to(Config.device)
    
    total_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"Feature extractor parameters: {total_params:,}")
    print(f"  - Transformer: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"  - Projection layer: {input_dim} -> {Config.output_dimension}")
    
    # Pre-train the feature extractor if epochs > 0
    if Config.pretrain_epochs > 0:
        print(f"\nPre-training feature extractor for {Config.pretrain_epochs} epoch(s)...")
        model_with_head = FeatureExtractorWithLMHead(feature_extractor, Config.vocab_size).to(Config.device)
        model_with_head = pretrain_feature_extractor(model_with_head, train_data_raw, val_data_raw, Config, Config.device)
        # After pre-training, we only need the feature extractor (without LM head)
        feature_extractor = model_with_head.feature_extractor
    else:
        print("\nSkipping pre-training (pretrain_epochs = 0)")
    
    # Now freeze the feature extractor for feature extraction
    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    X_train, y_train = prepare_full_data(train_data_raw, Config.block_size, feature_extractor, Config.device)
    print(f"X_train shape: {X_train.shape}")

    print("\n" + "="*60)
    print("PREPARING VALIDATION DATA")
    print("="*60)
    X_val, y_val = prepare_full_data(val_data_raw, Config.block_size, feature_extractor, Config.device)
    print(f"X_val shape: {X_val.shape}")
    print("="*60 + "\n")

    # Free GPU memory after feature extraction
    print("Freeing GPU memory...")
    del feature_extractor
    del transformer
    if 'model_with_head' in locals():
        del model_with_head
    torch.cuda.empty_cache()
    print(f"GPU memory freed. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Check for NaN or Inf values
    print("\nChecking for invalid values in features...")
    train_has_nan = np.isnan(X_train).any()
    train_has_inf = np.isinf(X_train).any()
    val_has_nan = np.isnan(X_val).any()
    val_has_inf = np.isinf(X_val).any()
    
    if train_has_nan or train_has_inf or val_has_nan or val_has_inf:
        print(f"WARNING: Found invalid values - Train NaN: {train_has_nan}, Train Inf: {train_has_inf}, Val NaN: {val_has_nan}, Val Inf: {val_has_inf}")
        # Replace NaN and Inf with 0
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        print("Invalid values replaced with 0")
    else:
        print("No invalid values found - data is clean!")

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
        'max_depth': 10,
        'num_leaves': 1000,
        'learning_rate': 0.1,
        'min_data_in_leaf': 20,  # Increased from 1 to avoid splitting errors
        'lambda_l2': 1.0,
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
    
    # Save feature extractor (transformer + projection layer)
    feature_extractor_path = Path('shakespeare_feature_extractor.pth')
    torch.save(feature_extractor.state_dict(), feature_extractor_path)
    print(f"Feature extractor saved to {feature_extractor_path}")

if __name__ == "__main__":
    main()
