from torch import nn
import torchvision.transforms.functional as F
from utils.box_utils import create_prior_boxes,iou,cxcy_to_encxcy,xy_to_cxcy
import torch

class MultiboxLoss(nn.Module):
    def __init__(self,no_class,threshold=0.5,alpha=1):
        super().__init__()

        self.no_class     = no_class
        self.threshold    = threshold
        self.alpha        = alpha
        self.smoothl1loss = F.smooth_l1_loss()
        self.crossentropy = F.cross_entropy(reduce = False)
        
    def forward (pre_box,pre_score,boxes,labels):
        prior_boxes = create_prior_boxes()
        no_prior    = prior_boxes.size(0)
        true_box    = torch.zeros(pre_box.size(0),no_prior) #N,8732
        true_loc    = torch.zeros(pre_box.size(0),no_prior)
        
        for i in range(pre_box.size(0)):
            overlap = iou(boxes[i],prior_boxes) #obj,8732
            
            overlap_to_obj , obj_for_priors = torch.max(overlap , dim = 0)
            
            _ , prior_for_obj = torch.max(overlap , dim = 1)
            #we'll assign the prior wrt best match coz for eg prior 65 got best match to obj 1 but obj 2's best match is it.so we need
            #to reassign these
            obj_for_priors[prior_for_obj] = torch.LongTensor([range(boxes[i].size(0))])
            #we will set true only if has overlap more than threshold and so we need to assign the best match to qualify for sure
            overlap_to_obj[prior_boxes] = 1
            
            label_for_priors = labels[i]*[obj_for_priors]
            label_for_priors[overlap_to_obj < self.threshold] = 0
            
            true_box[i] = label_for_priors
            true_loc[i] = cxcy_to_encxcy(xy_to_cxcy(true_box),prior_boxes)
            
        positive_priors = true_box !=0 #N,8732
        
        #location loss
        loc_loss = self.smoothl1loss(pre_box[positive_priors],true_loc[positive_priors]) #scalar
        
        #confident loss
        no_positive   = positive_priors.sum(dim = 1)
        conf_loss_all = self.crossentropy(pre_score(-1,self.no_class),labels(-1)) #N*8732
        conf_loss_all = conf_loss_all.view(pre_box[0],no_prior)#N,8732
        
        conf_loss_posi    = conf_loss_all[positive_priors]
        no_conf_loss_posi = conf_loss_posi.size()
        
        #hard negative mining
        conf_loss_neg     = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0 
        conf_loss_neg , _ = conf_loss_neg.sort(dim = 1,descending = True)
        no_hard_negative  = 3 * no_positive
        hard_negative     = conf_loss_neg[:,:no_hard_negative]

        confident_loss    = (conf_loss_posi.sum() + hard_negative.sum()) / no_positive.float()
        
        #multibox loss
        return confident_loss + loc_loss * self.alpha #scalar
        
        
            

            
            
        