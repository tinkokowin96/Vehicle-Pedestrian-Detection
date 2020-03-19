from dataset import KITTI_Dataset
import torch
from functions.multibox_loss import MultiboxLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
n_worker = 6
data_folder = 'D:/Projects/Research/Vehicle & Pedestrian Detection/Demo/JSON'
checkpoint = 'D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/BEST_checkpoint.pth'


def validate():
    global batch_size, n_worker

    best_checkpoint = torch.load(checkpoint)
    model = best_checkpoint['model']

    val_data = KITTI_Dataset(data_folder, 'test')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                             collate_fn=val_data.collate_fn, num_workers=n_worker, pin_memory=True)
    criteria = MultiboxLoss(9)
    model.eval()  # not to perform backward propagation

    with torch.no_grad():
        for i, (images, labels, boxes, _, _) in enumerate(val_loader):
            image = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_socres = model(image)

            loss = criteria(predicted_locs, predicted_socres, boxes, labels)


if __name__ == '__main__':
    validate()
