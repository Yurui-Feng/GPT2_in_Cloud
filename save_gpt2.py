import os
import torch
from model import GPT

out_dir = 'out_gpt2'
os.makedirs(out_dir, exist_ok=True)

init_from = 'gpt2' # 'gpt2', 'gpt2_xl', 'gpt2_medium', 'gpt2_large'

dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
model_args = dict(n_layer=None, n_head=None, n_embd=None, block_size=None,
                  bias=None, vocab_size=None, dropout=dropout) # start with model_args from command line

print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
# initialize from OpenAI GPT-2 weights
override_args = dict(dropout=dropout)
model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = getattr(model.config, k)

checkpoint = {'model_args': model_args,
              'model': model.state_dict()}
print(f"saving checkpoint to {out_dir}")
torch.save(checkpoint, os.path.join(out_dir, 'gpt2.pt'))