from utils.box_utils import create_prior_boxes, encxcy_to_cxcy, cxcy_to_xy, iou
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(no_classes, min_score, max_overlap, top_k, predicted_location, predicted_scores):
    pri_box = create_prior_boxes()
    scores = F.softmax(predicted_scores, dim=2)
    batch_size = predicted_scores.size(0)

    all_boxes = list()
    all_labels = list()
    all_score = list()

    for i in range(batch_size):
        decoded_loc = encxcy_to_cxcy(predicted_location[i], pri_box)
        decoded_loc = cxcy_to_xy(decoded_loc)
        # print('\nPredicted Loc are ', decoded_loc[:30, :])
        # continue

        class_box = list()
        class_label = list()
        class_scores = list()

        for c in range(1, no_classes):
            class_score = scores[i][:, c]  # 8732
            score_abv_min = class_score > min_score  # 8732
            class_score = class_score[score_abv_min]  # no: qualified
            class_dec_loc = decoded_loc[score_abv_min]
            n_qualified = class_dec_loc.size(0)

            class_score, sort_id = class_score.sort(dim=0, descending=True)
            class_dec_loc = class_dec_loc[sort_id]
            overlap = iou(class_dec_loc, class_dec_loc)  # no: qualified,no:qualified (to check how much a box is
            # overlap to other boxes)

            suppress = torch.zeros(n_qualified, dtype=torch.long).to(device)
            for box in range(n_qualified):
                if suppress[box] == 1:
                    continue
                to_suppress = overlap[box] > max_overlap
                to_suppress = to_suppress.type(dtype=torch.long)
                suppress = torch.max(suppress, to_suppress)

                suppress[box] = 0  # we will set un-suppress the current box even if the overlap is 1
                # The way how NMS work is 1st find JO of all objects and suppress all the overlapping boxes that is
                # greater than threshold except the most possible obj location.The most possi obj loc is suppress[box]
                # as we sorted class_dec_loc wrt scores and so the former the more possible there is an object after
                # that we skip the loop if already suppressed.To clarify,let say obj 1,3,5 are overlapped than thres and
                # result 1,1,1 to these and then we keep the mostly likely so we got 0,1,1 and for 3,1 and 5,1 we skip
                # these,we only need to find 3,5.
            qual_obj = 1 - suppress
            qual_obj = qual_obj.type(dtype=torch.bool)
            class_box.append(class_dec_loc[qual_obj])
            class_label.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
            class_scores.append(class_score[qual_obj])

        no_object = len(class_box)
        # if no object is found ,we'll set it as background
        if no_object == 0:
            class_box.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            class_label.append(torch.LongTensor([0]).to(device))
            class_scores.append(torch.FloatTensor([0.]).to(device))

        # concatenate into a single tensor
        class_box = torch.cat(class_box, dim=0)
        class_label = torch.cat(class_label, dim=0)
        class_scores = torch.cat(class_scores, dim=0)

        if no_object > top_k:
            class_scores, sort_id = class_scores.sort(descending=True)
            class_scores = class_scores[:top_k]
            class_label = class_label[sort_id][:top_k]
            class_box = class_box[sort_id][:top_k]

        all_boxes.append(class_box)
        all_labels.append(class_label)
        all_score.append(class_scores)

    return all_boxes, all_labels, all_score
