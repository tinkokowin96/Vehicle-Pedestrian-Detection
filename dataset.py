from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch
from augmentation import transform

class KITTI_Dataset (Dataset):
    def __init__ (self,data_folder,split):
        self.split = split.upper()       
        self.data_folder = data_folder
        
        with open(os.path.join(data_folder,split + '_Object.json')) as f:
            self.object = json.load(f)
            
        with open(os.path.join(data_folder,split + '_Image.json')) as j:
            self.image  = json.load(j)
            
    def __get__(self,i):
        image   = Image.open(self.image[i],'r')
        image   = image.convert('RGB')
        
        objects   = self.object(i)
        label     = torch.LongTensor(objects['label'])
        truncate  = torch.FloatTensor(objects['truncate'])
        occlusion = torch.FloatTensor(objects['occlusion'])
        bndbox    = torch.FloatTensor(objects['bndbox'])
        
        image,label,truncate,occlusion,bndbox = transform(image,label,truncate,occlusion,bndbox)
        
        return image,bndbox,label,truncate,occlusion
    
    def __len__(self):
        return len(self.image)
    
    def collate_fn (self,batch):
        image     = list()
        label     = list()
        truncate  = list()
        occlusion = list()
        box       = list()
        
        for b in batch:
            image.append(b[0])
            label.append(b[1])
            truncate.append(b[2])
            occlusion.append(b[3])
            box.append(b[4])
        
        #concentenate the sequence of tensor in one tensor.At former,the structure is like [tensor,tensor].After this,it will be 
        #one tensor of image
        image = torch.stack(image,dim = 0) 
        
        return image,box,label,truncate,occlusion