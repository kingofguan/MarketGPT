"""
Sample from a trained model
"""
import os
# import pickle
import numpy as np
from contextlib import nullcontext
import torch
# import tiktoken
from model import GPTConfig, GPT
from data_processing.itch_encoding import Vocab, encode_msgs, decode_msg

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
dataset = '12302019.NASDAQ_ITCH50_AAPL_message_proc.npy' # dataset to use for initial prompt
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_context_msgs = 3 # number of messages from dataset to use as context
# num_samples = 10 # number of samples to draw
num_samples = 1 # number of samples to draw
# max_new_tokens = 500 # number of tokens generated in each sample
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('equities/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
# elif init_from.startswith('gpt2'):
#     # init from a given GPT-2 model
#     model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# # look for the meta pickle in case it is available in the dataset folder
# load_meta = False
# if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
#     meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
#     load_meta = os.path.exists(meta_path)
# if load_meta:
#     print(f"Loading meta from {meta_path}...")
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     # TODO want to make this more general to arbitrary encoder/decoder schemes
#     stoi, itos = meta['stoi'], meta['itos']
#     encode = lambda s: [stoi[c] for c in s]
#     decode = lambda l: ''.join([itos[i] for i in l])
# else:
#     # ok let's assume gpt-2 encodings by default
#     print("No meta.pkl found, assuming GPT-2 encodings...")
#     enc = tiktoken.get_encoding("gpt2")
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#     decode = lambda l: enc.decode(l)
    
# define dataset to use for initial prompt
data_dir = os.path.join('dataset/proc/ITCH/test/', dataset)

# grab and encode sample data to use as context
vocab = Vocab()
context_dataset = np.load(data_dir, mmap_mode='r')
X_raw = np.array(context_dataset[0:num_context_msgs])
X = encode_msgs(X_raw, vocab.ENCODING)
encoded_tok_len = X.shape[1]
print("X (encoded seq):", X)

# prepare context tensor
x = (torch.tensor(X.reshape(-1), dtype=torch.long, device=device)[None, ...])

# # encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = model.generate(x, max_new_tokens*encoded_tok_len, temperature=temperature, top_k=top_k)
            # print(decode(y[0].tolist()))
            print('---------------')
        print("new sequence", y[0].tolist())

# print the last message in the generated sequence
print("last generated msg:", y[0][-24:].tolist())
print("decoded msg:", decode_msg(np.array(y[0][-24:].tolist()), vocab.ENCODING))

# compare with real sequence
X_true = np.array(context_dataset[0:num_context_msgs+max_new_tokens])
print("true sequence:", X_true[-1])

# print decoding guide for reference
print([ "ticker", "NA_VAL",
        "event_type", "direction", "NA_VAL", "price", "fill_size", "remain_size",
        "delta_t_s", "delta_t_ns", "time_s", "time_ns",
        "NA_VAL", "price_ref", "fill_size_ref", "time_s_ref", "time_ns_ref", "NA_VAL"])