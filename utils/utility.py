import os
import json
import xml.etree.cElementTree as ET
import torch

kitti_label = {'car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc'}
label_map = {v: i + 1 for i, v in enumerate(kitti_label)}
label_map['dontcare'] = 0

count = 0


def txt_to_xml(sourcedir, outputdir):
    file_list = []
    dir_list = []
    for root, dir, file in os.walk(sourcedir):
        for name in file:
            file_list.append(name)
            dir_list.append(os.path.join(root, name))

    num_file = len(file_list)
    for i in range(num_file):
        file_name = file_list[i].replace('.txt', '')

        xml_file = open(os.path.join(outputdir, file_name + '.xml'), 'w')

        xml_file.write('<?xml version="1.0" encoding="UTF-8"?>')
        xml_file.write('<annotation>')
        with open(os.path.join(dir_list[i]), 'r') as j:
            obj = j.read().splitlines()

        obj_label = list()
        truncate = list()
        occlusion = list()
        bndbox = list()
        index = 0
        for _ in obj:
            all_spec = [i for i in obj[index].split()]
            obj_label.append(all_spec[0])
            truncate.append(all_spec[1])
            occlusion.append(all_spec[2])
            bndbox.append(all_spec[4:8])

            index += 1
        for ind in range(index):
            xml_file.write('  ' + '<object>')
            xml_file.write('   ' + '<name>' + obj_label[ind] + '</name>')
            xml_file.write('   ' + '<truncate>' + truncate[ind] + '</truncate>')
            xml_file.write('   ' + '<occlusion>' + occlusion[ind] + '</occlusion>')
            xml_file.write('   ' + '<bndbox>' + '\n')
            xml_file.write('    ' + '<xmin>' + bndbox[ind][0] + '</xmin>')
            xml_file.write('    ' + '<ymin>' + bndbox[ind][1] + '</ymin>')
            xml_file.write('    ' + '<xmax>' + bndbox[ind][2] + '</xmax>')
            xml_file.write('    ' + '<ymax>' + bndbox[ind][3] + '</ymax>')
            xml_file.write('   ' + '</bndbox>')
            xml_file.write('  ' + '</object>')
        xml_file.write('</annotation>')
        xml_file.close()


def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    label = list()
    truncate = list()
    occlusion = list()
    bndbox = list()

    for object in root.iter('object'):
        obj_label = object.find('name').text.lower()

        if obj_label not in label_map:
            continue

        obj_truncate = float(object.find('truncate').text)
        obj_occlusion = float(object.find('occlusion').text)

        box = object.find('bndbox')
        xmin = float(box.find('xmin').text)
        xmin = int(xmin)
        ymin = float(box.find('ymin').text)
        ymin = int(ymin)
        xmax = float(box.find('xmax').text)
        xmax = int(xmax)
        ymax = float(box.find('ymax').text)
        ymax = int(ymax)

        label.append(label_map[obj_label])
        truncate.append(obj_truncate)
        occlusion.append(obj_occlusion)
        bndbox.append([xmin, ymin, xmax, ymax])
    return {'label': label, 'truncate': truncate, 'occlusion': occlusion, 'bndbox': bndbox}


# saving as json format
def save_as_json(split, outputdir):
    # train
    if split == 'train':
        train_xml = 'D:/Projects/Research/Vehicle & Pedestrian Detection/XML/train'
        img_source = 'D:/Projects/Research/Resources/kitti-object-detection/training/image_2'
        file_name = []
        train_objects = []
        img_dir = []
        for root, dir, file in os.walk(train_xml):
            for name in file:
                file_name.append(name)

        for ind in file_name:
            objects = parse_xml(os.path.join(train_xml, ind))
            train_objects.append(objects)
            img_ind = ind.replace('.xml', '.png')
            img_dir.append(os.path.join(img_source, img_ind))

        with open(os.path.join(outputdir, 'TRAIN_Objects.json'), 'w') as k:
            json.dump(train_objects, k)

        with open(os.path.join(outputdir, 'TRAIN_Image.json'), 'w') as k:
            json.dump(img_dir, k)

        with open(os.path.join(outputdir, 'Label_Map.json'), 'w') as j:
            json.dump(label_map, j)
    # test
    if split == 'test':
        test_xml = 'D:/Projects/Research/Vehicle & Pedestrian Detection/XML/test'
        img_source = 'D:/Projects/Research/Resources/kitti-object-detection/testing/image_2'
        file_name = []
        test_objects = []
        img_dir = []
        for root, dir, file in os.walk(test_xml):
            for name in file:
                file_name.append(name)

        for ind in file_name:
            objects = parse_xml(os.path.join(test_xml, ind))
            test_objects.append(objects)
            img_ind = ind.replace('.xml', '.png')
            img_dir.append(os.path.join(img_source, img_ind))

        with open(os.path.join(outputdir, 'TEST_Objects.json'), 'w') as k:
            json.dump(test_objects, k)

        with open(os.path.join(outputdir, 'TEST_Image.json'), 'w') as k:
            json.dump(img_dir, k)


def decimate(t, i):
    for d in range(t.dim()):
        if i[d] is not None:
            t = t.index_select(dim=d,
                               index=torch.arange(start=0, end=t.size(d), step=i[d]).long())
    return t


def save_checkpoint(epoch, epo_since_improv, optimizer, model, val_loss, best_loss, is_best):
    global count
    count += 1
    state = {'epoch': epoch,
             'epoch_since_improvement': epo_since_improv,
             'model': model,
             'loss': val_loss,
             'best_loss': best_loss,
             'optimizer': optimizer,
             'is_best': is_best
             }
    file_name = 'checkpoint'
    outputdir = 'D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/'
    if is_best:
        torch.save(state, outputdir + 'BEST_' + file_name + '.pth')
        count = 0
    else:
        if count != 5:
            torch.save(state, outputdir + 'last_' + file_name + str(count) + '.pth')
        else:
            count = 0


def decay_learningrate(optimizer, scale):
    # if the loss stop improving use this def
    for param in optimizer.param_groups:
        param['lr'] = param['lr'] * scale
    print("Decayed Learning Rate!! /t The new Learning Rate is %f" % (optimizer.param_groups['lr']))


def grad_clip(c_grad, optimizer):
    # if the gradient are exploding during backprop click grad
    for p_group in optimizer.param_groups:
        for param in p_group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-c_grad, c_grad)


class CalculateAvg(object):
    def __init__(self):
        self.param = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, param, n=1):
        self.param = param
        self.sum += param * n
        self.count += n
        self.avg = self.sum / self.count
