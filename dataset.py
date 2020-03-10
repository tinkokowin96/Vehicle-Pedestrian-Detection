from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch
from utils.augmentation import transform

class KITTI_Dataset (Dataset):
    def __init__ (self, data_folder, split):
        self.split = split.upper()       
        self.data_folder = data_folder
        
        with open(os.path.join(data_folder, self.split + '_Objects.json')) as f:
            self.object = json.load(f)
            
        with open(os.path.join(data_folder, self.split + '_Image.json')) as j:
            self.image = json.load(j)
            
    def __getitem__(self, i):
        image = Image.open(self.image[i],'r')
        image = image.convert('RGB')
        
        objects = self.object[i]
        label = torch.LongTensor(objects['label'])
        truncate = torch.FloatTensor(objects['truncate'])
        occlusion = torch.FloatTensor(objects['occlusion'])
        bndbox = torch.FloatTensor(objects['bndbox'])

        image, label, bndbox, truncate, occlusion = transform(image, label, bndbox, truncate, occlusion, self.split)

        return image, label, bndbox, truncate, occlusion
    
    def __len__(self):
        return len(self.image)
    
    def collate_fn(self, batch):
        image = list()
        label = list()
        box = list()
        truncate = list()
        occlusion = list()
        
        for b in batch:
            image.append(b[0])
            label.append(b[1])
            box.append(b[2])
            truncate.append(b[3])
            occlusion.append(b[4])
        #concentenate the sequence of tensor in one tensor.At former,the structure is like [tensor,tensor].After this,it will be 
        #one tensor of image
        image = torch.stack(image, dim=0)
        return image, label, box, truncate, occlusion