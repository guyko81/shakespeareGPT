import numpy as np
from catboost import CatBoostClassifier
from tokenizers import Tokenizer
from dataclasses import dataclass
from pathlib import Path

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

@dataclass
class Config:
    use_characters = True # Should match train_lgbm_classifier.py
    block_size = 256
    vocab_size = 0 # Set dynamically
    top_k = 5  # Sample from top 5 tokens

def generate():
    model_path = Path('shakespeare_catboost_classifier.cbm')
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        print("Model file not found! Run train_lgbm_classifier.py first.")
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
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    
    # Start with a simple context (e.g., newline)
    context_char = '\n'
    context_ids = tokenizer.encode(context_char)
    
    # Pad to block_size (must be int for categorical features)
    full_context = [0] * (Config.block_size - len(context_ids)) + context_ids
    current_context = np.array(full_context, dtype=np.int32)
    
    generated_ids = []
    length = 500
    
    print(f"Generating {length} characters/tokens with top-{Config.top_k} sampling...")
    
    for i in range(length):
        input_feat = current_context.reshape(1, -1)
        probs = model.predict_proba(input_feat)[0]  # Shape (vocab_size,)
        
        # Top-k sampling
        top_k_indices = np.argsort(probs)[-Config.top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalize
        
        next_id = np.random.choice(top_k_indices, p=top_k_probs)
        generated_ids.append(int(next_id))
        
        # Update context (shift left and append new token)
        current_context[:-1] = current_context[1:]
        current_context[-1] = next_id
        
        if i % 50 == 0:
            print(f"Generated {i}/{length}...", end='\r')

    print("\nDecoding...")
    generated_text = tokenizer.decode(generated_ids)
    
    print("="*50)
    print(generated_text)
    print("="*50)
    
    with open('generated_lgbm_classifier.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
        
if __name__ == "__main__":
    generate()
