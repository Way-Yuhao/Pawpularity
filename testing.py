import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
print(torch.cuda.device_count())
