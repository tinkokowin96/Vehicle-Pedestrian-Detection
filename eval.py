# tqdm means "progress" in Arabic and is an abbreviation for "I love you so much" in Spanish
from tqdm import tqdm
from pprint import PrettyPrinter
from functions.decoding import decode
from utils.eval_criteria import mean_average_precision
from dataset import KITTI_Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
n_worker = 6
data_folder = 'D:/Projects/Research/Vehicle & Pedestrian Detection/Demo/JSON'
checkpoint = 'D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/BEST_checkpoint.pth'
pp = PrettyPrinter()


def main():
    global batch_size, n_worker
    best_checkpoint = torch.load(checkpoint)
    model = best_checkpoint['model']

    test_data = KITTI_Dataset(data_folder, 'test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_data.collate_fn, num_workers=n_worker,
                                              pin_memory=True)
    true_boxes = []
    true_labels = []
    true_truncated = []
    true_occlusion = []
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    model.eval()
    with torch.no_grad():
        for i, (image, label, box, truncate, occlusion) in enumerate(tqdm(test_loader)):
            images = image.to(device)
            boxes = [b.to(device) for b in box]
            labels = [l.to(device) for l in label]
            truncated = [t.to(device) for t in truncate]
            occlusions = [o.to(device) for o in occlusion]

            pre_boxes, pre_scores = model(images)  # forward propagate
            dec_boxes, dec_labels, dec_scores = decode(9, 0.01, 0.45, 200, pre_boxes, pre_scores)

            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_truncated.extend(truncated)
            true_occlusion.extend(occlusions)
            pred_boxes.extend(dec_boxes)
            pred_labels.extend(dec_labels)
            pred_scores.extend(dec_scores)

        aps, map = mean_average_precision(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, true_truncated,
                                          true_occlusion)
    pp.pprint(aps)
    print('\nMean Average Precision (mAP): %.3f' % map)


if __name__ == '__main__':
    main()
