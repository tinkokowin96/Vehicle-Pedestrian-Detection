from math import sqrt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_prior_boxes():
        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for i, fmap in enumerate(fmaps):
            for j in range(fmap_dims[fmap]):
                for k in range(fmap_dims[fmap]):
                    cx = (k + 0.5) / fmap_dims[fmap]
                    cy = (j + 0.5) / fmap_dims[fmap]

                    for r in aspect_ratios[fmap]:
                        w = obj_scales[fmap] * sqrt(r)
                        h = obj_scales[fmap] / sqrt(r)

                        prior_boxes.append([cx, cy, w, h])

                        #for scale between current and next layer
                        if r == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[i+1]])
                            #for the last layer there won't no next layer
                            except IndexError:
                                additional_scale = 1.

                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        #change to tensor form list
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)
        return prior_boxes

#intersection with bounding box presentation
def intersection(set1, set2):
    #print(set1.size(), set2.size())
    lower_bound = torch.max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0))  #n1,n2,2
    upper_bound = torch.min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0))
    
    intersect = torch.clamp(upper_bound-lower_bound, min=0)  #n1,n2,2
    return intersect[:, :, 0] * intersect[:, :, 1]  #n1,n2

#intersection over union with bounding box presentation           
def iou(set1, set2):
    area_set1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])  # (n1)
    area_set2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])  # (n2)

    intersect = intersection(set1, set2)
    sum_of_area = area_set1.unsqueeze(1) + area_set2.unsqueeze(0)
    union = sum_of_area - intersect
    
    return intersect / union
    
def xy_to_cxcy(box):
    #change to center coordinate form boundary coordinate
    return torch.cat([(box[:, 2:] + box[:, :2]) / 2,  #cxcy
                       box[:, 2:] - box[:, :2]], 1)  #wh
    
def cxcy_to_xy(box):
    #change from center coordinate to boundary coordinate
    return torch.cat([box[:, :2] - (box[:, 2:] / 2),  #xmin,ymin
                      box[:, :2] + (box[:, 2:] / 2)], 1)  #xmax,ymax
    
'''def cxcy_to_encxcy(pre_box,pri_box):
    #encode the boxes
    return torch.cat([(pre_box[:, :2] - pri_box[:, :2]) / (pri_box[:, 2:] / 10), #encxcy
                       torch.log(pre_box[:, 2:] / pri_box[:, 2:]) * 5], 1) #enwh '''

def cxcy_to_encxcy(pre_box, priors_cxcy):
    return torch.cat([(pre_box[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # en_c_x, en_c_y
                      torch.log(pre_box[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # en_w, en_h

def encxcy_to_cxcy(pre_box, pri_box):
    #decode the boxes
    return torch.cat([pre_box[:, :2] * pri_box[:, 2:] / 10 + pri_box[:, :2],  #cxcy
                      torch.exp(pre_box[:, 2:] / 5) * pri_box[:, 2:]], 1)  #wh