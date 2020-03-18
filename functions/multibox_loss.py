from torch import nn
from utils.box_utils import create_prior_boxes, iou, cxcy_to_encxcy, xy_to_cxcy, cxcy_to_xy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiboxLoss(nn.Module):
    def __init__(self, no_class, threshold=0.5, alpha=1.):
        super().__init__()

        self.no_class = no_class
        self.threshold = threshold
        self.alpha = alpha
        self.smoothl1loss = nn.SmoothL1Loss()
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.prior_cxcy = create_prior_boxes()
        self.prior_xy = cxcy_to_xy(self.prior_cxcy)

    def forward(self, pre_box, pre_score, boxes, labels):
        no_prior = self.prior_xy.size(0)
        true_loc = torch.zeros((pre_box.size(0), no_prior, 4), dtype=torch.float).to(device)  # N,8732,4
        true_class = torch.zeros((pre_box.size(0), no_prior), dtype=torch.long).to(device)

        for i in range(pre_box.size(0)):
            no_objects = boxes[i].size(0)
            overlap = iou(boxes[i], self.prior_xy)  # obj,8732
            overlap_to_obj, obj_for_priors = overlap.max(dim=0)

            _, prior_for_obj = overlap.max(dim=1)
            # we'll assign the prior wrt best match coz for eg prior 65 got best match to obj 1 but obj 2's best match is it.so we need
            # to reassign these
            obj_for_priors[prior_for_obj] = torch.LongTensor(range(no_objects)).to(device)
            # we will set true only if has overlap more than threshold and so we need to assign the best match to qualify for sure
            overlap_to_obj[prior_for_obj] = 1

            label_for_priors = labels[i][obj_for_priors]
            label_for_priors[overlap_to_obj < self.threshold] = 0

            true_class[i] = label_for_priors
            true_loc[i] = cxcy_to_encxcy(xy_to_cxcy(boxes[i][obj_for_priors]), self.prior_cxcy)

        positive_priors = true_class != 0  # N,8732

        # location loss
        loc_loss = self.smoothl1loss(pre_box[positive_priors], true_loc[positive_priors])  # scalar

        # confident loss
        no_positive = positive_priors.sum(dim=1)
        conf_loss_all = self.crossentropy(pre_score.view(-1, self.no_class), true_class.view(-1))  # N*8732
        conf_loss_all = conf_loss_all.view(pre_box.size(0), no_prior)  # N,8732
        conf_loss_posi = conf_loss_all[positive_priors]

        # hard negative mining
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        no_hard_negative = 3 * no_positive
        hard_neg_ser = torch.LongTensor(range(no_prior)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # N,8732
        hard_negative = hard_neg_ser < no_hard_negative.unsqueeze(1)
        conf_loss_hard_negative = conf_loss_neg[hard_negative]

        confident_loss = (conf_loss_posi.sum() + conf_loss_hard_negative.sum()) / no_positive.sum().float()
        # multibox loss
        return confident_loss + loc_loss * self.alpha  # scalar