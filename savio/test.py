import torch
import os
print("GPU recognized:", torch.cuda.is_available())
print("Task ID:", os.environ.get('SLURM_TASK_ID'))