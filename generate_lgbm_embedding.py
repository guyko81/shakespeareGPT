import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from dataclasses import dataclass
from pathlib import Path
from gpt import ShakespeareGPT

# Reuse the same CharTokenizer logic
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

@dataclass
class Config:
    use_characters = True # Should match train_lgbm_embedding.py
    block_size = 256
    vocab_size = 0 # Set dynamically
    
    # Transformer params (must match training)
    n_embed = 32
    n_heads = 2
    n_layers = 2
    head_size = n_embed // n_heads
    attn_dropout = 0.0
    block_dropout = 0.0
    
    # Feature projection params (must match training)
    output_dimension = 256
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate():
    lgbm_model_path = Path('shakespeare_lgbm_model_embedding.txt')
    feature_extractor_path = Path('shakespeare_feature_extractor.pth')
    
    print(f"Loading models...")
    if not lgbm_model_path.exists():
        print(f"LightGBM model file not found at {lgbm_model_path}!")
        print("Run train_lgbm_embedding.py first.")
        return
    
    if not feature_extractor_path.exists():
        print(f"Feature extractor model file not found at {feature_extractor_path}!")
        print("Run train_lgbm_embedding.py first.")
        return

    # Load raw data to rebuild character tokenizer if needed
    with open('./data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    if Config.use_characters:
        tokenizer = CharTokenizer(text)
    else:
        tokenizer_path = Path('./tokenizer/shakespeare.json')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    Config.vocab_size = tokenizer.get_vocab_size()
    
    # Load feature extractor (transformer + projection)
    print(f"Loading feature extractor on {Config.device}...")
    transformer = ShakespeareGPT(Config).to(Config.device)
    input_dim = Config.block_size * Config.n_embed
    feature_extractor = FeatureExtractor(transformer, input_dim, Config.output_dimension).to(Config.device)
    feature_extractor.load_state_dict(torch.load(feature_extractor_path, map_location=Config.device))
    feature_extractor.eval()
    
    # Load LightGBM
    print("Loading LightGBM classifier...")
    lgbm_model = lgb.Booster(model_file=str(lgbm_model_path))
    
    # Start with a simple context (e.g., newline)
    context_char = '\n'
    context_ids = tokenizer.encode(context_char)
    
    # Pad to block_size
    full_context = [0] * (Config.block_size - len(context_ids)) + context_ids
    current_context = torch.tensor(full_context, dtype=torch.long).to(Config.device)
    
    generated_ids = []
    length = 500
    
    print(f"Generating {length} characters/tokens...")
    
    for i in range(length):
        # Extract features with feature extractor (transformer + projection)
        with torch.no_grad():
            features = feature_extractor(current_context.unsqueeze(0)).cpu().numpy()
        
        # Predict with LightGBM
        probs = lgbm_model.predict(features)[0]  # Shape: (vocab_size,)
        
        # Sample
        next_id = np.random.choice(len(probs), p=probs)
        generated_ids.append(next_id)
        
        # Update context (shift left and append new token)
        current_context = torch.cat([
            current_context[1:],
            torch.tensor([next_id], dtype=torch.long, device=Config.device)
        ])
        
        if i % 50 == 0:
            print(f"Generated {i}/{length}...", end='\r')

    print("\nDecoding...")
    generated_text = tokenizer.decode(generated_ids)
    
    print("="*50)
    print(generated_text)
    print("="*50)
    
    with open('generated_lgbm_embedding.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    
    print(f"\nGenerated text saved to generated_lgbm_embedding.txt")
        
if __name__ == "__main__":
    generate()

