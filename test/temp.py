import torch
x = torch.zeros(10)
y = 1/x  # tensor with all infinities
c = y == float('inf')
y[c] = 0


