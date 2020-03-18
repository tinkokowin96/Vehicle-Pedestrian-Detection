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



kitti_label = {'car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc'}
label_map = {v: i+1 for i, v in enumerate(kitti_label)}
label_map['dontcare'] = 0
print(kitti_label)


import torch

a = []
t = torch.randn(5)
c = t<0.5
u = c.type(dtype=torch.long)
f_c = 1-u
r = f_c.type(dtype=torch.bool)
a.append(t[r])


t = torch.tensor([3, 2, 1, 4, 5, 6])
a = torch.tensor([1, 3, 1, 2, 2, 1])
r = (t>0) & (t<3) & ((a==1) | (a==2))

def bool_check(t, a):
    r = (t>0) & (t<3) & ((a==1) | (a==2))
    r = r.type(dtype=torch.uint8)
    n = t.size(0)
    for i in range(n):
        if r[i] == 1:
            print("I love U")
            break

for i in range(5):
    print("I love you")
    if i == 4:
        print("I hate you")

a = []
b = []
t = torch.randn(5,4)
for i in range(5):
    a.extend([t[i]])
    if i == 4:
        b = a

import torch

t = torch.randn(10)
c = t>0.5
c = c.type(dtype=torch.long)
r = torch.LongTensor((1-c).sum().item() * [3])



def find_intersection(set_1, set_2):

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    print("lower bond\n",lower_bounds)
    print("upper bond\n",upper_bounds)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

t = torch.tensor([[0.1763, 0.6638, 0.0815, 0.5439],
                [0.1789, 0.6456, 0.0737, 0.5372],
                [0.1838, 0.6329, 0.0651, 0.4880],
                [0.1843, 0.6129, 0.0663, 0.4568],
                [0.1744, 0.6950, 0.0899, 0.5324],
                [0.7718, 0.6467, 0.0749, 0.4170],
                [0.7678, 0.6347, 0.0680, 0.4045],
                [0.8037, 0.6404, 0.0642, 0.4077],
                [0.8029, 0.6525, 0.0719, 0.4246]])
overlap = find_intersection(t,t)

l_b = [torch.randn(3,4),torch.randn(2,4),torch.randn(1,4)]
l_l = [torch.randn(4),torch.randn(2),torch.randn(3)]

c_b = torch.cat(l_b,dim=0)
c_l = torch.cat(l_l,dim=0)

import torch
t = torch.tensor([2,1,3,5,6,3,7])

for i in range(t.size(0)):
    if (t[i]>0) & (t[i]<5) & ((t[i] == 3) | (t[i] == 1)):
        print("I love you")
    else:
        print("I hate you")

import torch
t = torch.tensor([0,1,3,0,6,3,0])
r = torch.nonzero(t == 3)

t = torch.randn(5,4)
n = t.size(0)
for i in range(n):
    print(i)
    if i == n-1:
        print("I love you")

level = {}
t = torch.tensor([1,3,4,2])
for i in range(t.size(0)):
    if t[i] == 1:
        level['1'] = 'I love U'
    if t[i] == 3:
        level['2'] = 'I hate U'



def calculate(n):
    if n%2 ==0:
        num = n * 3
        return num
    else:
        return "NUM is ODD"

