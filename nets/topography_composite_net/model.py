import torch
from torch import nn
from torchsummary import summary
from .YOLO import YOLO
import numpy as np
import SPMUtil as spmu
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CompositeNet(nn.Module):
    def __init__(self):
        super(CompositeNet, self).__init__()
        self.Net1 = ConvNet(11)
        # self.Net2 = YOLO('yolov8s.pt')
        # self.Net3 = YOLO('yolov8l-pose.pt')
        self.Net2: YOLO
        self.Net3: YOLO

        self.transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.default_device)


    def forward(self, x):
        return self.Net1(self.transforms(image=x)['image'].unsqueeze(0).to(self.default_device)), \
            self.Net2(x, verbose=False), \
            self.Net3(x, verbose=False)


    def load(self, path):
        state_dict = torch.load(path, map_location=self.default_device)
        self.Net2 = YOLO(state_dict.pop("Net2"))
        self.Net3 = YOLO(state_dict.pop("Net3"))
        self.load_state_dict(state_dict)

    def save(self, save_path, overwrite_net1_path: str=None, overwrite_net2_path: str=None, overwrite_net3_path: str=None):
        if overwrite_net1_path is not None:
            self.Net1.load_state_dict(torch.load(overwrite_net1_path))
        state_dict = self.state_dict()
        if overwrite_net2_path is not None:
            state_dict["Net2"] = torch.load(overwrite_net2_path)
        else:
            state_dict["Net2"] = self.Net2.ckpt
        if overwrite_net3_path is not None:
            state_dict["Net3"] = torch.load(overwrite_net3_path)
        else:
            state_dict["Net3"] = self.Net3.ckpt
        torch.save(state_dict, save_path)



class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64, dropout_p=0.1)
        self.layer4 = self.conv_module(64, 128, dropout_p=0.2)
        self.layer5 = self.conv_module(128, 256, dropout_p=0.2)
        self.gap = self.global_avg_pool(256, self.num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)

        return out

    def conv_module(self, in_num, out_num, dropout_p=0.0):
        if dropout_p == 0.0:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_p, inplace=True))


    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))



if __name__ == '__main__':
    pass



