from math import sqrt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_prior_boxes():
        obj_scales    = {'conv4_3' : 0.1,
                         'conv7'   : 0.2,
                         'conv8_2' : 0.375,
                         'conv9_2' : 0.55,
                         'conv10_2': 0.725,
                         'conv11_2': 0.9}

        aspect_ratios = {'conv4_3' : [1., 2., 0.5],
                         'conv7'   : [1., 2., 3., 0.5, .333],
                         'conv8_2' : [1., 2., 3., 0.5, .333],
                         'conv9_2' : [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
        
        fmap_dims     = {'conv4_3' : 38,
                         'conv7'   : 19,
                         'conv8_2' : 10,
                         'conv9_2' : 5,
                         'conv10_2': 3,
                         'conv11_2': 1}
            
        fmaps         = ['conv4_3','conv7','conv8_2','conv9_2','conv10_2','conv11_2']
        
        prior_boxes = []
        for i,j in enumerate (fmaps):
            for k in range (i):
                cx  = i / fmap_dims[j]
                cy  = k / fmap_dims[j]  
                
                for r in aspect_ratios[j]:
                    w   = obj_scales[j] * sqrt(aspect_ratios[j])
                    h   = obj_scales[j] / sqrt(aspect_ratios[j])
                    
                    prior_boxes.append([cx,cy,w,h])
                    
                    #for scale between current and next layer
                    if (r == 1.):
                        try:
                            additional_scale =  sqrt(obj_scales[j]*obj_scales[j+1])
                        #for the last layer there won't no next layer
                        except IndexError:
                            additional_scale = 1.
                            
                        prior_boxes.append(cx,cy,additional_scale,additional_scale)
                            
        #change to tensor form list
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp(0,1)
        
        return prior_boxes