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


t = torch.tensor([1,3,4,2])
i = torch.tensor([3,2,0,1])
r = t[i]



def calculate(n):
    if n%2 ==0:
        num = n * 3
        return num
    else:
        return "NUM is ODD"

t = torch.randn(3,4)
r = torch.LongTensor(range(4)).unsqueeze(0).expand_as(t)

l = [1,1,2,2,1,1,3]
for i in range(len(l)):
    if l[i] == 1:
        print(1)
        continue
    if l[i] == 2:
        print(2)
        continue
    else:
        print(3)
        continue


def encxcy_to_cxcy(pre_box, pri_box):
    # decode the boxes
    return torch.cat([pre_box[:, :2] * pri_box[:, 2:] / 10 + pri_box[:, :2],  # cxcy
                      torch.exp(pre_box[:, 2:] / 5) * pri_box[:, 2:]], 1)  # wh


def find_intersection(set_1, set_2):

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
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

def cxcy_to_xy(box):
    # change from center coordinate to boundary coordinate
    return torch.cat([box[:, :2] - (box[:, 2:] / 2),  # xmin,ymin
                      box[:, :2] + (box[:, 2:] / 2)], 1)  # xmax,ymax


p = torch.tensor([[ 0.4375, -1.4266, -5.8974, -2.8486],
        [ 1.4008,  1.9228, -1.1341,  0.1821],
        [-0.6505,  2.7193,  0.5987,  1.9150],
        [-0.4600,  1.9228, -1.1341,  0.1821],
        [ 1.4008,  0.0620, -1.1341,  0.1821],
        [ 2.8017,  0.0620,  2.3316,  0.1821],
        [-0.6505,  0.0877,  0.5987,  1.9150],
        [-0.4600,  0.0620, -1.1341,  0.1821],
        [-0.9199,  0.0620,  2.3316,  0.1821],
        [-2.3208,  0.0620, -1.1341,  0.1821],
        [ 1.4008, -1.7988, -1.1341,  0.1821],
        [-0.6505, -2.5439,  0.5987,  1.9150],
        [-0.4600, -1.7988, -1.1341,  0.1821],
        [ 1.2236,  0.8035, -1.1592,  1.3374],
        [ 0.8936,  0.5868, -2.7307, -0.2341],
        [ 1.7305,  0.5682,  0.5737, -0.3955],
        [ 2.1204,  0.4637,  1.5898, -1.4116],
        [-1.4080,  0.8035, -1.1592,  1.3374],
        [-1.0282,  0.5868, -2.7307, -0.2341],
        [-1.9912,  0.5682,  0.5737, -0.3955]])

cxcy = torch.tensor([[0.0132, 0.0132, 0.1000, 0.1000],
        [0.0132, 0.0132, 0.1414, 0.1414],
        [0.0132, 0.0132, 0.1414, 0.0707],
        [0.0132, 0.0132, 0.0707, 0.1414],
        [0.0395, 0.0132, 0.1000, 0.1000],
        [0.0395, 0.0132, 0.1414, 0.1414],
        [0.0395, 0.0132, 0.1414, 0.0707],
        [0.0395, 0.0132, 0.0707, 0.1414],
        [0.0658, 0.0132, 0.1000, 0.1000],
        [0.0658, 0.0132, 0.1414, 0.1414],
        [0.0658, 0.0132, 0.1414, 0.0707],
        [0.0658, 0.0132, 0.0707, 0.1414],
        [0.0921, 0.0132, 0.1000, 0.1000],
        [0.0921, 0.0132, 0.1414, 0.1414],
        [0.0921, 0.0132, 0.1414, 0.0707],
        [0.0921, 0.0132, 0.0707, 0.1414],
        [0.1184, 0.0132, 0.1000, 0.1000],
        [0.1184, 0.0132, 0.1414, 0.1414],
        [0.1184, 0.0132, 0.1414, 0.0707],
        [0.1184, 0.0132, 0.0707, 0.1414]])

p = encxcy_to_cxcy(p, cxcy)
r = cxcy_to_xy(p)


import torch.nn.functional as F
import torch

t = torch.tensor([0.1, 0.001, 0.21, 0.0056, 0.0032, 0.12, 0.032])
r = F.softmax(t, 0)


t1 = torch.tensor([[2.6179e-01, 4.5332e-01, 3.1088e-01, 7.2495e-01]])
t = torch.tensor([[0.2577, 0.4332, 0.2981, 0.6818]])
r = find_jaccard_overlap(t1, t)