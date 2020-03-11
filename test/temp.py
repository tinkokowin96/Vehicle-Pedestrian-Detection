import torch

file_dir = "D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/BEST_checkpoint.pth"
checkpoint = torch.load(file_dir)
best_loss = checkpoint['best_loss']

import torch
t = [torch.randn(5,4),torch.randn(3,4)]
r = []
for i in range(len(t)):
    r.extend([i] * t[i].size(0))
r = torch.LongTensor(r)



kitti_label = {'dontcare', 'car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc'}
for i, v in enumerate(kitti_label):
    print (v)