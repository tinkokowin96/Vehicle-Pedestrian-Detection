import random
import torch
from utility import iou
import torchvision.transforms.functional as F


def expend (img,boxes,filler):
    ori_h     = img.size(1)
    ori_w     = img.size(2)
    max_scale = 4
    scale     = random.uniform(1,max_scale)
    new_h     = int(scale*ori_h)
    new_w     = int(scale*ori_w)
#why filler dim to (3,1,1) is coz each param is for each color channel
    new_img   = torch.ones((3,new_h,new_w),dtype=torch.float)*filler.unsqueeze(1).unsqueeze(1)
    
    left      = random.randint(0,new_w - ori_w)
    right     = left + ori_w
    top       = random.randint(0,new_h - ori_h)
    botton    = top + ori_h
    
    new_img[:,top:botton,left:right] = img
    
    new_boxes = boxes + torch.FloatTensor([left,top,left,top]).unsqueeze(0)
    
    return new_img,new_boxes
   
def crop (img,label,truncate,occlusion,boxes):
    ori_h     = img.size(1)
    ori_w     = img.size(2)
    while True:
        min_overlap = random.choice([None,0.1,0.3,0.5,0.7,0.9])
        
        if min_overlap is None:
            continue
        
        max_trial = 50 
        
        for _  in range(max_trial):
            min_scale    = 0.3
            scale        = random.uniform(min_scale,1)
            new_h        = int(scale * ori_h)
            new_w        = int(scale * ori_w)
            
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2 :
                continue           
            
            left         = random.randint(0,ori_w - new_w)
            right        = ori_w - left
            top          = random.randint(0,ori_h - new_h)
            botton       = ori_h - top
                  
            crop         = torch.FloatTensor([left,top,right,botton])
            overlap      = iou(crop.unsqueeze(),boxes).squeeze(0)
            
            if overlap.max().item() < min_overlap :
                continue
            
            new_img      = img[:, top:botton , left:right]
            
#only retain the boxes that fall at least half in crop            
            box_half     = ( boxes[: , :2] + boxes[: , 2:] ) / 2
            half_in_crop = ( box_half[: , 0] > left ) * ( box_half[: , 0] < right ) * ( box_half[: , 1] > top ) * ( box_half[: , 1] < botton )  
            
            if not half_in_crop.any():
                continue
            
            bndbox       = boxes[half_in_crop , :]
            labels       = label[half_in_crop]
            truncates    = truncate[half_in_crop]
            occlusion    = occlusion[half_in_crop]
            
            bndbox[ : , :2]    = torch.max( bndbox[: , :2] , crop[ :2 ])
            bndbox[ : , :2]   -= crop[ :2 ]
            bndbox[ : , 2:]    = torch.max( bndbox[: , 2:] , crop[ 2: ])
            bndbox[ : , 2:]   -= crop[ 2: ]
            
    return new_img,labels,truncates,occlusion,bndbox

def h_flip(img,boxes):
    new_img  = F.h_flip(img)
    
    new_box  = boxes
    xmin     = img.width - new_box[: , 2]
    xmax     = img.width - new_box[: , 0]
    new_box  = new_box[: , xmin , 1 , xmax , 3]
    return new_img , new_box
    

def photometric_distortion(img):
    new_img = img
    distortions = [F.adjust_hue,
                   F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation()]
    random.shuffle(distortions)
    
    for dis in distortions:
        if random.random() < 0.5:
            if dis.__name__ == 'adjust_hue':
 # divide by 255 because PyTorch needs a normalized value
                adjust = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust = random.uniform( 0.5, 1.5 )
            new_img = dis(img , adjust)
    return new_img

        
def resize(img,boxes,dim=(300,300)):
    new_img   = F.resize(img, dim)
    old_dim   = torch.FloatTensor([img.width,img.height,img.width,img.height]) 
    box_per   = boxes / old_dim
    
    new_dim   = torch.FloatTensor(dim[0],dim[1],dim[0],dim[1])
    new_box   = box_per * new_dim
    
    return new_img,new_box

def transform(img,label,truncate,occlusion,box,split):
# Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
# see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    new_img,new_label,new_truncate,new_occlusion,new_box = img,label,truncate,occlusion,box
    
    img  = F.to_tensor(img)
    
    if split == 'TRAIN':

        new_img = photometric_distortion(img)
        
        if random.random() < 0.5:
            new_img,new_box = expend(img, box, mean)
            
        new_img,new_label,new_truncate,new_occlusion,new_box = crop(img,label,truncate,occlusion,box)
                   
        
        new_img = F.to_pil_image(new_img)
        if random.random() < 0.5:
            new_img,new_box = h_flip(img, box)
            
    new_img,new_box = expend(img, box, mean)     
    new_img         = F.to_tensor()
    #Normalize
    new_img         = F.normalize(new_img, mean=mean, std=std)
    
    return new_img,new_label,new_truncate,new_occlusion,new_box
    