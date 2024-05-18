# MarketSimGPT Logbook
Goal: Get a transformer-based market simulator up and running with acceptable inference speeds.
Inspired by: https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf


## 100M Model Log

### Upcoming changes
Upcoming model revisions: Increase vocab size to nearest multiple of 64, and then re-train.

### 2024-05-15 11:14 ET
- SSM version #3 'ckpt_ssm_v3.pt' was trained on the ssm/mamba.py and ssm/train_ssm.py files. In this version, the model parameters were increased to 735M parameters (48 layers and 1536 embedding dimensions). All other training parameters were the same as version #2. The train and val loss decreased noticeably from version #2 but was still slightly higher than the best transformers version.
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimSSM/runs/zbxzystl?nw=nwuseraw843


### 2024-05-06 18:50 ET
- SSM version #2 'ckpt_ssm_v2.pt' was trained on the ssm/mamba.py and ssm/train_ssm.py files. Slight changes were made to the training parameters--most notably the number of training sets were increased to 4000 (same number as best Transformer model we trained) and the context length was decreased to 112 messages to make training times more manageable. The final train/val loss for the mamba-based was worse than the transformer-based model (although this is without doing any hyper-parameter sweeps for the ssm model). 
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimSSM/runs/2mqx0zdf?nw=nwuseraw843

### 2024-05-04 21:22 ET
First state-space model (ssm) is trained.

- SSM version #1 'ckpt_ssm.pt' was trained on the ssm/mamba.py and ssm/train_ssm.py files and was a proof-of-concept model based on the Mamba architecture by Gu and Dao (2023). The model has ~100M parameters and the same context length, training parameters, etc as the Transformer implementation.
    - Training Run (1000 steps): https://wandb.ai/aw843/MarketSimSSM/runs/u02sr5z7?nw=nwuseraw843

### 2024-04-01 18:22 ET
Over the last few days, the following changes have been made:

- Several hyperparameter sweeps were conducted using wandb and the 'fast_model.yaml' config file located in the config directory. The results for the hyperparameter sweeps can be found here:
    - Sweep 1 (50 iterations each): https://wandb.ai/aw843/MarketSimT_fast/sweeps/ierysdtl?nw=nwuseraw843
    - Sweep 2 (100 iterations each): https://wandb.ai/aw843/MarketSimT_fast/sweeps/u88bdp8c?nw=nwuseraw843
- Model version #5 'ckpt_fast_v4.pt' was trained using the equities/fast_model.py file and a version of the equities/fast_train.py file that utilized the best hyperparameters from the wandb sweep results. The model was initially trained for 2000 epochs, and then resumed for another 1000 epochs since the loss was continuing to drop. This allowed the model to reach a validation loss of 1.623. The results for the training cycles can be found here:
    - Run 1 (first 2000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/oqt5f9j0?nw=nwuseraw843
    - Run 2 (subsequent 1000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/bwkvixya?nw=nwuseraw843
- Model version #6 'ckpt_fast_v5.pt' was trained also using the equities/fast_model.py file and a version of the equities/fast_train.py file that utilized the best hyperparameters from the wandb sweep results. However, this model was trained for a continous 4000 epochs rather than a mix of 2000 and 1000 steps like the previous model version. Allowing the learning rate to degrade more slowly and continously across a larger number of timesteps allowed the model to more effectively and reach a more optimal validation loss of 1.364. The results of the training session can be found here:
    - Run 1 (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/ouoycbzf?nw=nwuseraw843

### Prior to 2024-03-28 8:00 ET
Logbook started on this date---a couple months after the project was started.

Model/Checkpoint Summary:
- Model version #1 'ckpt.pt' was trained on the initial equities/model.py and equities/train.py files and was a proof-of-concept model based on a vanilla transformer architecture.
- Model version #2 'ckpt_fast.pt' was trained on the initial equities/fast_model.py and equities/fast_train.py files and introduced many new model components to boost fidelity and latency performance. New components include RoPE embeddings, RMS Normalization, Grouped-Query Attention (GQA), Swish activation function, and KV caching. These changes resulted in a significant speedup at inference time (~7x) and a reduction in generation errors (hallucinating non-existant reference orders). However, performance degradation was observed at sequence lengths that were longer than the maximum context length.
- Model version #3 'ckpt_fast_v2.pt' was trained on a modifed version of the initial equities/fast_model.py and equities/fast_train.py files. This modification included the addition of an attention sink token at both training and inference time (https://arxiv.org/pdf/2309.17453.pdf). The addition of the sink token resulted in no noticeable model quality degradation past the maximum context length. At this time, I experimented with different sequence lengths and found that a sequence length of 112 messages (2688 tokens) was optimal and decreasing the context length to this amount resulted in the generation of an additional +3 tokens/sec at inference time.
- Model version #4 'ckpt_fast_v3.pt' was trained on a modifed version of the initial equities/fast_model.py and equities/fast_train.py files. This modification included the reducing the number of local kv heads from 12 to 6. This was done in hopes that it would reduce model latency by leveraging GQA, however, no latency improvement was observed. This checkpoint was subsequently deleted from the checkpoint directory.
