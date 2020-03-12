import torch

kitti_label = {'car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc'}
label_map = {v: i+1 for i, v in enumerate(kitti_label)}
label_map['dontcare'] = 0
rev_label_map = {i: v for v, i in label_map.items()}
n_classes = len(label_map)

def mean_average_precision(det_boxes, det_labels, det_scores, true_boxes, true_labels, truncated, occlusion):
    true_objects = []
    assert det_boxes.size(0) == det_labels.size(0) == det_scores.size(0), \
        "Something is wrong here!! Number of Detected boxes and labels doesn't match"
    for i in range(true_boxes.size(0)):
        true_objects.extend([i] * true_labels[i].size(0))
    true_objects = torch.LongTensor(true_objects)  #concentenate into a singe tensor
    true_boxes = true_boxes.cat(dim=0)
    true_labels = true_labels.cat(dim=0)
    truncated = truncated.cat(dim=0)
    occlusion = occlusion.cat(dim=0)

    if 0.0 < truncated <= 0.15 and occlusion == 0:
        level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels = \
            easy_detect(det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels, truncated, occlusion)
        calculate_map(level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels)

    if 0.16 < truncated <= 0.30 and occlusion == 1:
        level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels = \
            moderate_detect(det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels, truncated, occlusion)
        calculate_map(level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels)

    if 0.31 < truncated <= 1.0 and occlusion == 2 or 3:
        level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels = \
            hard_detect(det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels, truncated, occlusion)
        calculate_map(level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels)

def calculate_map(level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels):
    global n_classes
    true_positive = torch.zeros(det_boxes.size(0), dtype=uint8)
    false_positive = torch.zeros(det_boxes.size(0), dtype=uint8)
    average_precision = torch.zeros(n_classes - 1)

    for c in range(1, n_classes):
        class_true_objects = true_objects[true_labels == c]
        class_true_boxes = true_boxes[true_labels == c]
        class_n_objects = class_true_objects.size(0)

        class_det_objects = det_objects[det_labels == c]
        class_det_boxes = det_boxes[det_labels == c]
        class_det_scores = det_scores[det_labels == c]
        n_class_det_objects = class_det_objects.size(0)

        true_class_boxes_det = torch.zeros(n_class_det_objects, dtype=torch.uint8)

        if n_class_det_objects == 0:
            continue

        class_det_scores, sort_id = class_det_scores.sort(dim=0, descending=True)
        class_det_objects = class_det_objects[sort_id]
        class_det_boxes = class_det_boxes[sort_id]

        for d in range(n_class_det_objects.size(0)):
            this_object = class_det_objects[d]
            this_det_box = class_det_boxes[d].unsqueeze(0)  #(1,4)

            object_boxes = class_true_boxes[class_true_objects == this_object]

            if object_boxes.size(0) == 0:
                false_positive[d] = 1
                continue

            overlap = iou(this_det_box, object_boxes)  # (1,n_objects)
            overlap, ind = overlap.squeeze(0).max(dim=0)  # ind is obj level ind we need to find the class level ind

            class_lvl_ind = torch.LongTensor(range(n_class_det_objects))[class_true_objects == this_object][ind]

            if overlap > 0.5:
                if true_class_boxes_det[class_lvl_ind] == 0:
                    true_positive[d] = 0
                    true_class_boxes_det[class_lvl_ind] = 1
                else:
                    false_positive[d] = 1
            else:
                false_positive[d] = 1

        cumul_true_positive = torch.cusmsum(true_positive, dim=0)
        cumul_false_positive = torch.cumsum(false_positive, dim=0)
        cumul_precision = cumul_true_positive / (cumul_true_positive + cumul_false_positive)
        cumul_recall = cumul_true_positive / class_n_objects

        recall_thresh = torch.arange(0., 1.1, .1, dtype=torch.float).tolist()
        precisions = torch.zeros(len(recall_thresh), dtype=torch.float)
        for i, thresh in enumerate(precision_thres):
            recall_abv_thresh = cumul_recall >= thresh
            if recall_abv_thresh.any():
                precisions[i] = cumul_precision[recall_abv_thres].max()
            else:
                precisions[i] = 0
        average_precision[c - 1] = precisions.mean()

    map = average_precision.mean()
    ap = {rev_label_map[i + 1]: v for i, v in enumerate(average_precision.tolist())}
    return level, ap, map

def easy_detect(det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels, truncated, occlusion):
    level = {'Difficulty Level': 'Easy'}
    moder_obj = 0.0 < truncated <= 0.15 and occlusion == 0
    det_boxes = det_boxes[moder_obj]
    det_labels = det_labels[moder_obj]
    det_scores = det_scores[moder_obj]

    det_objects = []
    for i in range(det_boxes.size(0)):
        det_objects.extends([i] * det_labels[i].size(0))
    det_objects = torch.LongTensor(det_objects)
    det_boxes = det_boxes.cat(dim=0)
    det_labels = det_labels.cat(dim=0)
    det_scores = det_scores.cat(dim=0)

    true_objects = true_objects[moder_obj]
    true_boxes = true_boxes[moder_obj]
    true_labels = true_labels[moder_obj]

    return level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels

def moderate_detect(det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels, truncated, occlusion):
    level = {'Difficulty Level': 'Moderate'}
    moder_obj = 0.16 < truncated <= 0.30 and occlusion == 1
    det_boxes = det_boxes[moder_obj]
    det_labels = det_labels[moder_obj]
    det_scores = det_scores[moder_obj]

    det_objects = []
    for i in range(det_boxes.size(0)):
        det_objects.extends([i] * det_labels[i].size(0))
    det_objects = torch.LongTensor(det_objects)
    det_boxes = det_boxes.cat(dim=0)
    det_labels = det_labels.cat(dim=0)
    det_scores = det_scores.cat(dim=0)

    true_objects = true_objects[moder_obj]
    true_boxes = true_boxes[moder_obj]
    true_labels = true_labels[moder_obj]

    return level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels

def hard_detect(det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels, truncated, occlusion):
    level = {'Difficulty Level': 'Hard'}
    moder_obj = 0.31 < truncated <= 1.0 and occlusion == 2 or 3
    det_boxes = det_boxes[moder_obj]
    det_labels = det_labels[moder_obj]
    det_scores = det_scores[moder_obj]

    det_objects = []
    for i in range(det_boxes.size(0)):
        det_objects.extends([i] * det_labels[i].size(0))
    det_objects = torch.LongTensor(det_objects)
    det_boxes = det_boxes.cat(dim=0)
    det_labels = det_labels.cat(dim=0)
    det_scores = det_scores.cat(dim=0)

    true_objects = true_objects[moder_obj]
    true_boxes = true_boxes[moder_obj]
    true_labels = true_labels[moder_obj]

    return level, det_objects, det_boxes, det_labels, det_scores, true_objects, true_boxes, true_labels
