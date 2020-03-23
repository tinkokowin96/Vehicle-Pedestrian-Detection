from functions.decoding import decode
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch

label_map = {0: "dontcare", 1: "cyclist", 2: "misc", 3: "tram",
             4: "person_sitting", 5: "car", 6: "pedestrian", 7: "van", 8: "truck"}

checkpoint = 'D:/Projects/Research/Vehicle & Pedestrian Detection/Checkpoint/BEST_checkpoint.pth'
checkpoint = torch.load(checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def draw_box_label(original_image, min_score=0.20, max_overlap=0.45, top_k=200, suppress_lbl=None):
    ori_image = normalize(to_tensor(resize(original_image)))
    ori_image = ori_image.to(device)
    model = checkpoint['model']

    pre_boxes, pre_scores = model(ori_image.unsqueeze(0))
    dec_boxes, labels, _ = decode(9, min_score, max_overlap, top_k, pre_boxes, pre_scores)
    dec_boxes = torch.cat(dec_boxes, 0)
    labels = torch.cat(labels, 0).tolist()

    ori_dims = torch.FloatTensor([original_image.width, original_image.height,
                                  original_image.width, original_image.height]).unsqueeze(0).to(device)
    pre_boxes = dec_boxes * ori_dims
    print(labels)
    labels = [label_map[lbl] for ind, lbl in enumerate(labels)]
    print(labels)
    drawn_image = original_image
    draw = ImageDraw.Draw(drawn_image)
    font = ImageFont.truetype(font='D:/Projects/Research/Resources/fonts/bp.ttf', size=10)

    for i in range(pre_boxes.size(0)):
        # if you don't wanna detect specific class
        if suppress_lbl is not None:
            if labels[i] == suppress_lbl:
                continue

        box_loc = pre_boxes[i].tolist()

        # draw boxes
        draw.rectangle(xy=box_loc, outline='red')

        # draw labels
        text_size = font.getsize(labels[i])
        text_loc = [box_loc[0] + 1, box_loc[1] - text_size[1]]
        draw.text(xy=text_loc, text=labels[i].upper(), fill='red', font=font)
    del draw
    return drawn_image


if __name__ == '__main__':
    image_dir = 'D:/Resources/kitti-object-detection/testing/image_2/2.jpg'
    image = Image.open(image_dir, mode='r')
    image = image.convert('RGB')
    draw_box_label(image).show()
