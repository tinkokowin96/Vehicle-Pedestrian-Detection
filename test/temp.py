import torch

file_dir = "D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/BEST_checkpoint.pth"
checkpoint = torch.load(file_dir)
best_loss = checkpoint['best_loss']

