import lightgbm as lgb
import numpy as np
import torch
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
    use_characters = True # Should match train_lgbm.py
    block_size = 256
    vocab_size = 0 # Set dynamically

def generate():
    model_path = Path('shakespeare_lgbm_model.txt')
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        print("Model file not found! Run train_lgbm.py first.")
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
    model = lgb.Booster(model_file=str(model_path))
    
    # Start with a simple context (e.g., newline)
    # 61 is usually newline in Shakespeare GPT
    context_char = '\n'
    context_ids = tokenizer.encode(context_char)
    
    # Pad to block_size
    full_context = [0] * (Config.block_size - len(context_ids)) + context_ids
    current_context = np.array(full_context, dtype=np.int32)
    
    generated_ids = []
    length = 500
    
    print(f"Generating {length} characters/tokens...")
    
    for i in range(length):
        input_feat = current_context.reshape(1, -1)
        probs = model.predict(input_feat) # Shape (1, vocab_size)
        
        # Sample
        next_id = np.random.choice(len(probs[0]), p=probs[0])
        generated_ids.append(next_id)
        
        # Update context
        current_context = np.roll(current_context, -1)
        current_context[-1] = next_id
        
        if i % 50 == 0:
            print(f"Generated {i}/{length}...", end='\r')

    print("\nDecoding...")
    generated_text = tokenizer.decode(generated_ids)
    
    print("="*50)
    print(generated_text)
    print("="*50)
    
    with open('generated_lgbm.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
        
if __name__ == "__main__":
    generate()
