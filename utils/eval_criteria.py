import torch
from utils.box_utils import iou

kitti_label = {'car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc'}
label_map = {v: i + 1 for i, v in enumerate(kitti_label)}
label_map['dontcare'] = 0
rev_label_map = {i: v for v, i in label_map.items()}
n_classes = len(label_map)
count = 0


def mean_average_precision(det_boxes, det_labels, det_scores, true_boxes, true_labels, truncated, occlusion):
    assert len(det_boxes) == len(det_labels) == len(det_scores), \
        "Something is wrong here!! Number of Detected boxes and labels doesn't match"

    true_objects = []
    for i in range(len(true_boxes)):
        true_objects.extend([i] * true_labels[i].size(0))
    true_objects = torch.LongTensor(true_objects)  # concatenate into a singe tensor
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    truncated = torch.cat(truncated, dim=0)
    occlusion = torch.cat(occlusion, dim=0)
    n_objects = true_boxes.size(0)

    det_objects = []
    for i in range(len(det_boxes)):
        det_objects.extend([i] * det_labels[i].size(0))
    det_objects = torch.LongTensor(det_objects)
    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)

    rank = level_check(truncated, occlusion)

    level = {}
    e_objects = []
    e_labels = []
    m_objects = []
    m_labels = []
    h_objects = []
    h_labels = []
    for i in range(n_objects):
        if rank[i] == 0:
            e_objects.append(true_objects[i])
            e_labels.append(true_labels[i])
            m_objects.append(true_objects[i])
            m_labels.append(true_labels[i])
            h_objects.append(true_objects[i])
            h_labels.append(true_labels[i])

        if rank[i] == 1:
            level['1'] = 'Easy'
            e_objects.append(true_objects[i])
            e_labels.append(true_labels[i])

        if rank[i] == 2:
            level['2'] = 'Moderate'
            m_objects.append(true_objects[i])
            m_labels.append(true_labels[i])

        else:
            level['3'] = 'Hard'
            h_objects.append(true_objects[i])
            h_labels.append(true_labels[i])

    e_objects = torch.tensor(e_objects)
    e_labels = torch.tensor(e_labels)
    m_objects = torch.tensor(m_objects)
    m_labels = torch.tensor(m_labels)
    h_objects = torch.tensor(h_objects)
    h_labels = torch.tensor(h_labels)

    calculate_map(level['1'], det_objects, det_boxes, det_labels, det_scores, e_objects, e_labels, true_objects,
                  true_labels, true_boxes)
    '''calculate_map(level['2'], det_objects, det_boxes, det_labels, det_scores, m_objects, m_labels, true_objects,
                  true_labels, true_boxes)
    calculate_map(level['3'], det_objects, det_boxes, det_labels, det_scores, h_objects, h_labels, true_objects,
                  true_labels, true_boxes)'''


def calculate_map(level, det_objects, det_boxes, det_labels, det_scores, r_objects, r_labels, true_objects, true_labels,
                  true_boxes):
    global n_classes, count
    true_positive = torch.zeros(det_boxes.size(0), dtype=torch.uint8)
    false_positive = torch.zeros(det_boxes.size(0), dtype=torch.uint8)
    average_precision = torch.zeros(n_classes - 1)
    for c in range(1, n_classes):
        class_r_objects = r_objects[r_labels == c]
        class_true_objects = true_objects[true_labels == c]
        class_true_boxes = true_boxes[true_labels == c]
        n_class_r_objects = class_r_objects.size(0)
        fl_n_class_r_objects = float(n_class_r_objects)
        n_class_true_objects = class_true_objects.size(0)

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

        for d in range(n_class_det_objects):
            this_object = class_det_objects[d]
            this_det_box = class_det_boxes[d].unsqueeze(0)  # (1,4)

            object_boxes = []
            object_boxes_ind = []
            match_t_objects = class_true_objects == this_object
            match_t_objects = match_t_objects.type(dtype=torch.long)
            match_t_obj_ind = torch.nonzero(match_t_objects).squeeze(1)
            for j in range(match_t_obj_ind.size(0)):
                for k in range(n_class_r_objects):
                    if class_true_objects[match_t_obj_ind][j] == class_r_objects[k]:
                        object_boxes.append(class_true_boxes[k].unsqueeze(0))
                        object_boxes_ind.append(k)

            if len(object_boxes) == 0:
                false_positive[d] = 1
                continue

            object_boxes = torch.cat(object_boxes, dim=0)
            object_boxes_ind = torch.tensor(object_boxes_ind)

            print(this_det_box, '\n', object_boxes)
            overlap = iou(this_det_box, object_boxes)  # (1,n_objects)
            overlap, ind = overlap.squeeze(0).max(dim=0)  # ind is obj level ind we need to find the class level ind

            class_lvl_ind = torch.LongTensor(range(n_class_true_objects))[object_boxes_ind][ind]

            if overlap > 0.5:
                if true_class_boxes_det[class_lvl_ind] == 0:
                    true_positive[d] = 1
                    true_class_boxes_det[class_lvl_ind] = 1
                else:
                    false_positive[d] = 1
            else:
                false_positive[d] = 1
            count += 1
            print(count)

        if true_positive.sum().item() == 0:
            continue

        cumul_true_positive = torch.cumsum(true_positive, dim=0)
        cumul_false_positive = torch.cumsum(false_positive, dim=0)
        cumul_precision = cumul_true_positive / (cumul_true_positive + cumul_false_positive + 1e-10)  # changing to
        # float if we don't change to float it will only produce 0 coz it think of it as int
        cumul_recall = cumul_true_positive / fl_n_class_r_objects

        count = 0

        recall_thresh = torch.arange(0., 1.1, .1, dtype=torch.float).tolist()
        precisions = torch.zeros(len(recall_thresh), dtype=torch.float)
        for i, thresh in enumerate(recall_thresh):
            recall_abv_thresh = cumul_recall >= thresh
            if recall_abv_thresh.any():
                precisions[i] = cumul_precision[recall_abv_thresh].max()
            else:
                precisions[i] = 0
        average_precision[c - 1] = precisions.mean()

    map = average_precision.mean()

    # print(map)
    ap = {rev_label_map[i + 1]: v for i, v in enumerate(average_precision.tolist())}
    print(ap)
    return map


def level_check(truncated, occlusion):
    r = []
    for i in range(truncated.size(0)):
        if (truncated[i] >= 0.0) & (truncated[i] <= 0.15) & ((occlusion[i] == 0) | (occlusion[i] == 1)):
            r.append(1)
            continue
        if (truncated[i] >= 0.16) & (truncated[i] <= 0.30) | ((occlusion[i] == 2) | (occlusion[i] == 3)):
            r.append(2)
            continue
        if (truncated[i] >= 0.31) & (truncated[i] <= 1.):
            r.append(3)
        else:
            r.append(0)
    return r
