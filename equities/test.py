import os
import numpy as np

# poor man's data loader
dataset = '03272019.NASDAQ_ITCH50_AAPL_book_20_proc.npy'
data_dir = os.path.join('dataset/proc/ITCH/', dataset)

train_data = np.load(data_dir, mmap_mode='r')

print("train_data.shape:", train_data.shape)

