import argparse
import os
import torch
from model import GPT

if __name__=='__main__':
    # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()

    init_from = 'gpt2' # 'gpt2', 'gpt2_xl', 'gpt2_medium', 'gpt2_large'

    dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
    
    model_args = dict(n_layer=None, n_head=None, n_embd=None, block_size=None,
                  bias=None, vocab_size=None, dropout=dropout)
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
        
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)