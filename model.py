import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from utils.box_utils import create_prior_boxes
from utils.utility import decimate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# why inherit nn.Module is nn.Con2d inherit from it and all neural network must inherit to it
class Base_Convolution(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # stride 1 to retain the size

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6,
                               dilation=6)  # to reduce dim from 4096 to 1024 that is in tl

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layer()

    def load_pretrained_layer(self):
        state_dict = self.state_dict()
        param_name = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param = list(pretrained_state_dict.keys())

        # assign pretrained weight and bias to the current
        for i, param in enumerate(param_name[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param[i]]

        # decimate conv6 weight and bias
        fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(fc6_weight, i=[4, None, 3, 3])
        state_dict['conv6.bias'] = decimate(fc6_bias, i=[4])

        # decimate conv7 weight and bias
        fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(fc7_weight, i=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(fc7_bias, i=[4])

        self.load_state_dict(state_dict)

        print("Base Convolution is loaded and Assigned with pre-trained weights and biases")

    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_feat = out
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)

        out = F.relu(self.conv6(out))

        conv7_feat = F.relu(self.conv7(out))

        return conv4_3_feat, conv7_feat


class Auxiliary_Convolution(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.init_conv()

    # initialize convolution parameters
    def init_conv(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feat):

        out = F.relu(self.conv8_1(conv7_feat))
        out = F.relu(self.conv8_2(out))
        conv8_2_feat = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_feat = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_feat = out

        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out))
        conv11_2_feat = out

        return conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat


class Prediction_Convolution(nn.Module):
    def __init__(self, no_class):
        super().__init__()

        self.no_class = no_class

        no_loc = {'conv4_3': 4,
                  'conv7': 6,
                  'conv8_2': 6,
                  'conv9_2': 6,
                  'conv10_2': 4,
                  'conv11_2': 4}

        # for location prediction
        self.l_conv4_3 = nn.Conv2d(512, no_loc['conv4_3'] * 4, kernel_size=3, padding=1)
        self.l_conv7 = nn.Conv2d(1024, no_loc['conv7'] * 4, kernel_size=3, padding=1)
        self.l_conv8_2 = nn.Conv2d(512, no_loc['conv8_2'] * 4, kernel_size=3, padding=1)
        self.l_conv9_2 = nn.Conv2d(256, no_loc['conv9_2'] * 4, kernel_size=3, padding=1)
        self.l_conv10_2 = nn.Conv2d(256, no_loc['conv10_2'] * 4, kernel_size=3, padding=1)
        self.l_conv11_2 = nn.Conv2d(256, no_loc['conv11_2'] * 4, kernel_size=3, padding=1)

        # for class prediction
        self.cl_conv4_3 = nn.Conv2d(512, no_loc['conv4_3'] * no_class, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, no_loc['conv7'] * no_class, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, no_loc['conv8_2'] * no_class, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, no_loc['conv9_2'] * no_class, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, no_loc['conv10_2'] * no_class, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, no_loc['conv11_2'] * no_class, kernel_size=3, padding=1)

        self.init_conv()

    def init_conv(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feat, conv7_feat, conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat):
        batch_size = conv4_3_feat.size(0)
        # for location prediction
        l_conv4_3 = self.l_conv4_3(conv4_3_feat)  # (N,16,38*38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # why permute here is coz  It is to make it easier to
        # combine predictions from multiple layers.(author's answer).and contiguous make sure it is store in in
        # contiguous chunk of memory that is needed for view
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.l_conv7(conv7_feat)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.l_conv8_2(conv8_2_feat)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.l_conv9_2(conv9_2_feat)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.l_conv10_2(conv10_2_feat)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.l_conv11_2(conv11_2_feat)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        # for class prediction
        cl_conv4_3 = self.cl_conv4_3(conv4_3_feat)
        cl_conv4_3 = cl_conv4_3.permute(0, 2, 3, 1).contiguous()
        cl_conv4_3 = cl_conv4_3.view(batch_size, -1, self.no_class)

        cl_conv7 = self.cl_conv7(conv7_feat)
        cl_conv7 = cl_conv7.permute(0, 2, 3, 1).contiguous()
        cl_conv7 = cl_conv7.view(batch_size, -1, self.no_class)

        cl_conv8_2 = self.cl_conv8_2(conv8_2_feat)
        cl_conv8_2 = cl_conv8_2.permute(0, 2, 3, 1).contiguous()
        cl_conv8_2 = cl_conv8_2.view(batch_size, -1, self.no_class)

        cl_conv9_2 = self.cl_conv9_2(conv9_2_feat)
        cl_conv9_2 = cl_conv9_2.permute(0, 2, 3, 1).contiguous()
        cl_conv9_2 = cl_conv9_2.view(batch_size, -1, self.no_class)

        cl_conv10_2 = self.cl_conv10_2(conv10_2_feat)
        cl_conv10_2 = cl_conv10_2.permute(0, 2, 3, 1).contiguous()
        cl_conv10_2 = cl_conv10_2.view(batch_size, -1, self.no_class)

        cl_conv11_2 = self.cl_conv11_2(conv11_2_feat)
        cl_conv11_2 = cl_conv11_2.permute(0, 2, 3, 1).contiguous()
        cl_conv11_2 = cl_conv11_2.view(batch_size, -1, self.no_class)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # n,8732,4
        class_scores = torch.cat([cl_conv4_3, cl_conv7, cl_conv8_2, cl_conv9_2, cl_conv10_2, cl_conv11_2],
                                 dim=1)  # n,8732,no_class

        return locs, class_scores


class SSD(nn.Module):
    def __init__(self, no_class):
        super().__init__()
        self.no_class = no_class
        self.base = Base_Convolution()
        self.auxi = Auxiliary_Convolution()
        self.pred = Prediction_Convolution(self.no_class)

        # we need to normalize con4_3 coz it has considerably larger scale,I think it is because we we use 6 dilation in
        # conv4_3 and that is why the filters are much larger than original
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop      
        self.rescale_factor = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factor, 20)

        self.prior_cxcy = create_prior_boxes()

    def forward(self, image):
        conv4_3_feat, conv7_feat = self.base(image)

        # l2 norm
        norm = conv4_3_feat.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feat = conv4_3_feat / norm
        conv4_3_feat = conv4_3_feat * self.rescale_factor

        conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat = self.auxi(conv7_feat)

        locs, class_scores = self.pred(conv4_3_feat, conv7_feat, conv8_2_feat, conv9_2_feat, conv10_2_feat,
                                       conv11_2_feat)
        return locs, class_scores
