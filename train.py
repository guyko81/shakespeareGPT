import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from tokenizers import Tokenizer

from dataset import ShakespeareDataset

from gpt import ShakespeareGPT

from tqdm.auto import tqdm
from pathlib import Path


tokenizer = Tokenizer.from_file('./tokenizer/shakespeare.json')

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

@dataclass
class Config:
    batch_size = batch_size
    block_size = block_size
    max_iters = max_iters
    eval_interval = eval_interval
    learning_rate = learning_rate
    device = device
    eval_iters = eval_iters
    n_embd = n_embd
    n_head = n_head
    n_layer = n_layer
    dropout = dropout
    
    # needed for the code logic
    vocab_size = tokenizer.get_vocab_size()
    
    # compatibility with gpt.py and existing train.py logic
    n_embed = n_embd
    n_heads = n_head
    n_layers = n_layer
    lr = learning_rate
    train_iters = max_iters
    val_iters = eval_iters
    attn_dropout = dropout
    block_dropout = dropout
    head_size = n_embd // n_head
    



lm = ShakespeareGPT(Config)
lm = lm.to(device=Config.device)

train_ds = ShakespeareDataset(Config)
val_ds = ShakespeareDataset(Config,is_test=True)


optim = torch.optim.AdamW(lm.parameters(), lr=Config.lr)

def loss_fn(logits, targets):
    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits,targets)
    return loss




@torch.no_grad()
def valid_N_iters():
    val_step_losses = []
    for batch in tqdm(range(Config.val_iters)):
        inputs, targets = next(val_ds)
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        val_step_losses.append(loss.item())
        
        del inputs, targets, loss, logits
    
    val_loss = torch.tensor(val_step_losses).mean()
    print(f'val loss: {val_loss}')
    return val_loss


def train_N_iters():
    lm.train()
    train_step_losses = []
    val_losses = []
    for batch in tqdm(range(Config.train_iters)):
        optim.zero_grad()
        inputs, targets = next(train_ds)
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        loss.backward()
        optim.step()
        train_step_losses.append(loss.item())
        
        if batch%(Config.train_iters//10)==0 or batch==Config.train_iters-1:
            print(f"\n{'-'*50}\nbatch {batch} train step loss: {loss.item()}")
            print(f"train loss so far: {torch.tensor(train_step_losses).mean()}\n{'-'*50}\n")
            
        if batch%Config.eval_interval==0 or batch==Config.train_iters-1:
            lm.eval()
            val_loss = valid_N_iters()
            lm.train()
            val_losses.append(val_loss.item())
            
            del val_loss
            
        del inputs, targets, loss, logits
        
    return train_step_losses, val_losses


def save_lm():
    state_dict = lm.state_dict()
    save_path = Path('./').resolve() / 'shakespeareGPT'
    save_path.mkdir(exist_ok=True)
    model_path = save_path / f'shakespeareGPT.pth'
    torch.save(state_dict, model_path)


def train_lm():
    train_step_losses,val_losses = train_N_iters()
    save_lm()
    return train_step_losses,val_losses


tsl,vl=train_lm()
tsl_mean = torch.tensor(tsl).mean()
print('Train Loss:',tsl_mean.item())
print('Validation Loss:',vl[-1])