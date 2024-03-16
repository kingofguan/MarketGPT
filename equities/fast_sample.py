"""
Sample from a trained model
"""
import os
from tqdm import tqdm
import numpy as np
from contextlib import nullcontext
import time
import torch
from fast_model import Transformer, ModelArgs
from data_processing.itch_encoding import Vocab, encode_msgs, decode_msg

# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt_fast.pt'
# dataset = '12302019.NASDAQ_ITCH50_AAPL_message_proc.npy' # dataset to use for initial prompt
dataset = '03272019.NASDAQ_ITCH50_AAPL_message_proc.npy' # dataset to use for initial prompt
num_context_msgs = 100 # number of messages from dataset to use as context
# num_samples = 10 # number of samples to draw
num_samples = 1 # number of samples to draw
# max_new_tokens = 500 # number of tokens generated in each sample
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # False # use PyTorch 2.0 to compile the model to be faster
exec(open('equities/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
    
# define dataset to use for initial prompt
# data_dir = os.path.join('dataset/proc/ITCH/test/', dataset)
data_dir = os.path.join('dataset/proc/ITCH/full_view/', dataset)

# grab and encode sample data to use as context
vocab = Vocab()
# context_dataset = np.load(data_dir, mmap_mode='r')
# X_raw = np.array(context_dataset[0:num_context_msgs])
proc_messages = np.array(np.load(data_dir, mmap_mode='r')[0:(15700 + num_context_msgs)])
X_raw = proc_messages[-num_context_msgs:]
print("X_raw.shape:", X_raw.shape)
X = encode_msgs(X_raw, vocab.ENCODING)
print("X.shape:", X.shape)
gen_start_time = decode_msg(X[-1], vocab.ENCODING)[10] * 1000000000 + decode_msg(X[-1], vocab.ENCODING)[11]
print("current simulation time:", gen_start_time) # for computing simulation time elapsed in generation
encoded_tok_len = X.shape[1]

# prepare context tensor
x = (torch.tensor(X.reshape(-1), dtype=torch.long, device=device)[None, ...])
print("x.shape:", x.shape)

num_generation_steps = 500
start = time.perf_counter()

for t in tqdm(range(num_generation_steps)):
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens*encoded_tok_len, temperature=temperature, top_k=top_k)
                # print('---------------')
    # set new message as context for next iteration
    x_new = y[0][-24:]
    x_new = torch.unsqueeze(x_new, 0) # add batch dimension for concatenation purposes
    # append sampled index to the running sequence and continue
    x = torch.cat((x, x_new), dim=1)

    # if the sequence context is growing too long we must crop it at block_size
    if (x.size(1) + encoded_tok_len) > model.params.max_seq_len:
        x = x[:, encoded_tok_len:]
print(f"Time: {time.perf_counter() - start}")

# # print the last message in the generated sequence
# print("last generated msg:", y[0][-24:].tolist())
# print("decoded msg:", decode_msg(np.array(y[0][-24:].tolist()), vocab.ENCODING))

# # # compare with real sequence
# # X_true = np.array(context_dataset[0:num_context_msgs+max_new_tokens])
# # print("true sequence:", X_true[-1])

# [ "ticker", "order_id", "event_type", "direction", "price_abs", "price",
#  "fill_size", "remain_size", "delta_t_s", "delta_t_ns", "time_s", "time_ns",
#  "old_id", "price_ref", "fill_size_ref", "time_s_ref", "time_ns_ref", "old_price_abs"]