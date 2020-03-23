import time
import torch.backends.cudnn as cudnn
from model import SSD
from dataset import KITTI_Dataset
import torch
from utils.utility import grad_clip, save_checkpoint, CalculateAvg
from functions.multibox_loss import MultiboxLoss

data_folder = "D:/Projects/Research/Vehicle & Pedestrian Detection/JSON"
n_classes = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_epoch = 0
no_epoch = 200
epo_since_imporv = 0
batch_size = 16
no_worker = 6
weight_decay = 1e-3
lr = 1e-5
momentum = 0.9
c_grad = None
checkpoint = "D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/BEST_checkpoint.pth"
# checkpoint = None
best_loss = 100.
print_freq = 100

cudnn.benchmark = True


def main():
    global n_classes, start_epoch, no_epoch, epo_since_imporv, batch_size, no_worker, weight_decay, lr, momentum, \
        checkpoint, best_loss

    biases = list()
    weight = list()

    if checkpoint is None:
        model = SSD(n_classes)

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    weight.append(param)

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': lr * 2}, {'params': weight}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epo_since_imporv = checkpoint['epoch_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        best_loss = checkpoint['best_loss']

    model.to(device)

    criteria = MultiboxLoss(n_classes).to(device)

    train_data = KITTI_Dataset(data_folder, 'train')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_data.collate_fn, num_workers=no_worker,
                                               pin_memory=True)

    val_data = KITTI_Dataset(data_folder, 'test')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_data.collate_fn, num_workers=no_worker,
                                             pin_memory=True)

    for epoch in range(start_epoch, no_epoch):
        train(train_loader, epoch, model, criteria, optimizer)
        val_loss = validate(val_loader, model, criteria)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if is_best:
            epo_since_imporv = 0
        else:
            epo_since_imporv += 1
            print("Number of epochs since imporvement :%d\n" % epo_since_imporv)

        save_checkpoint(epoch, epo_since_imporv, optimizer, model, val_loss, best_loss, is_best)


def train(train_loader, epoch, model, criteria, optimizer):
    model.train()

    load_time = CalculateAvg()  # forward and backward propagation time
    data_time = CalculateAvg()
    losses = CalculateAvg()

    start_time = time.time()

    for i, (images, labels, boxes, _, _) in enumerate(train_loader):
        data_time.update(time.time() - start_time)
        image = images.to(
            device)  # why we loop in othr is .to can't use in list object and images are not list coz of torch.stack
        # in collate_fn
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_socres = model(image)
        loss = criteria(predicted_locs, predicted_socres, boxes, labels)  # forward propagation
        optimizer.zero_grad()
        loss.backward()  # backward propagation

        if c_grad is not None:
            grad_clip(c_grad, optimizer)

        # update the model
        optimizer.step()

        load_time.update(time.time() - start_time)
        losses.update(loss.item(), image.size(0))

        start_time = time.time()

        if i % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}] \t Load Time:{load_time.param:.3f}({load_time.avg:.3f}) \t'
                  'Data Time:{data_time.param:.3f}({data_time.avg:.3f}) \t'
                  'Loss:{loss.param:.3f}({loss.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                 load_time=load_time,
                                                                 data_time=data_time,
                                                                 loss=losses))

    del predicted_locs, predicted_socres, image, boxes, labels  # delete history to release memory


def validate(val_loader, model, criteria):
    model.eval()  # not to perform backward propagation

    load_time = CalculateAvg()  # forward propagation time
    losses = CalculateAvg()

    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels, boxes, _, _) in enumerate(val_loader):
            image = images.to(
                device)  # why we loop in othr is .to can't use in list object and images are not list coz of
            # torch.stack in collate_fn
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_socres = model(image)
            loss = criteria(predicted_locs, predicted_socres, boxes, labels)

            load_time.update(time.time() - start_time)
            losses.update(loss.item(), image.size(0))

            start_time = time.time()

            if i % print_freq == 0:
                print('\n[{0}/{1}] \t Load Time:{load_time.param:.3f}({load_time.avg:.3f}) \t'
                      'Loss:{loss.param:.3f}({loss.avg:.3f})'.format(i, len(val_loader),
                                                                     load_time=load_time,
                                                                     loss=losses))

    print('\n * Loss: {loss.avg:.3f}'.format(loss=losses))
    return losses.avg


if __name__ == '__main__':
    main()
