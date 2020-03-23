import random
import torch
from utils.box_utils import iou
import torchvision.transforms.functional as FT


def expend(img, boxes, filler):
    ori_h = img.size(1)
    ori_w = img.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * ori_h)
    new_w = int(scale * ori_w)
    filler = torch.FloatTensor(filler)
    # why filler dim to (3,1,1) is coz each param is for each color channel
    new_img = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    left = random.randint(0, new_w - ori_w)
    right = left + ori_w
    top = random.randint(0, new_h - ori_h)
    botton = top + ori_h

    new_img[:, top:botton, left:right] = img

    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_img, new_boxes


def randn_crop(img, label, boxes, truncate, occlusion):
    ori_h = img.size(1)
    ori_w = img.size(2)
    while True:
        min_overlap = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])

        if min_overlap is None:
            return img, label, boxes, truncate, occlusion

        max_trial = 50

        for _ in range(max_trial):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * ori_h)
            new_w = int(scale_w * ori_w)

            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            left = random.randint(0, ori_w - new_w)
            right = new_w + left
            top = random.randint(0, ori_h - new_h)
            botton = new_h + top

            crop = torch.FloatTensor([left, top, right, botton])
            overlap = iou(crop.unsqueeze(0), boxes)  # 1,n_obj
            overlap = overlap.squeeze(0)  # n_obj

            if overlap.max().item() < min_overlap:
                continue

            # crop
            new_img = img[:, top:botton, left:right]

            # only retain the boxes that fall at least half in crop
            box_half = (boxes[:, :2] + boxes[:, 2:]) / 2
            half_in_crop = (box_half[:, 0] > left) * (box_half[:, 0] < right) * \
                           (box_half[:, 1] > top) * (box_half[:, 1] < botton)

            if not half_in_crop.any():
                continue

            bndbox = boxes[half_in_crop, :]
            labels = label[half_in_crop]
            truncates = truncate[half_in_crop]
            occlusion = occlusion[half_in_crop]

            bndbox[:, :2] = torch.max(bndbox[:, :2], crop[:2])
            bndbox[:, :2] -= crop[:2]
            bndbox[:, 2:] = torch.min(bndbox[:, 2:], crop[2:])
            bndbox[:, 2:] -= crop[:2]
            return new_img, labels, bndbox, truncates, occlusion


def h_flip(img, boxes):
    new_image = FT.hflip(img)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = img.width - boxes[:, 0]
    new_boxes[:, 2] = img.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def photometric_distortion(img):
    new_img = img
    distortions = [FT.adjust_hue,
                   FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation]
    random.shuffle(distortions)

    for dis in distortions:
        if random.random() < 0.5:
            if dis.__name__ == 'adjust_hue':
                # divide by 255 because PyTorch needs a normalized value
                adjust = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust = random.uniform(0.5, 1.5)
            new_img = dis(img, adjust)
    return new_img


def resize(img, boxes, dim=(300, 300), return_percent=True):
    new_img = FT.resize(img, dim)
    old_dim = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
    new_box = boxes / old_dim

    if not return_percent:
        new_dim = torch.FloatTensor([dim[1], dim[0], dim[1], dim[0]]).unsqueeze(0)
        new_box = new_box * new_dim

    return new_img, new_box


def transform(img, label, box, truncate, occlusion, split):
    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_img, new_label, new_box, new_truncate, new_occlusion = img, label, box, truncate, occlusion
    if split == 'TRAIN':

        new_img = photometric_distortion(new_img)
        new_img = FT.to_tensor(new_img)

        if random.random() < 0.5:
            new_img, new_box = expend(new_img, new_box, mean)
        new_img, new_label, new_box, new_truncate, new_occlusion = \
            randn_crop(new_img, new_label, new_box, new_truncate, new_occlusion)

        new_img = FT.to_pil_image(new_img)
        if random.random() < 0.5:
            new_img, new_box = h_flip(new_img, new_box)

    new_img, new_box = resize(new_img, new_box)
    new_img = FT.to_tensor(new_img)
    # Normalize
    new_img = FT.normalize(new_img, mean=mean, std=std)
    return new_img, new_label, new_box, new_truncate, new_occlusion
