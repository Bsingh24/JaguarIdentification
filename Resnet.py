import torch
from torch import nn
from collections import defaultdict
from itertools import permutations, product

def create_pairs(classes, output):
    mapping = defaultdict(list)
    for cls, output in zip(classes, output):
        mapping[int(cls)].append(output)
    
    avail_classes = list(mapping.keys())
    
    all_pairs = []
    for i in range(len(avail_classes)):
        positive_anchor_pair = list(permutations(mapping[avail_classes[i]], 2))
        positive_anchor_pair = [list(item) for item in positive_anchor_pair]
        for j in range(len(avail_classes)):
            if i != j:
                res = list(product(positive_anchor_pair, mapping[avail_classes[j]]))
                for x, y in res:
                    copy = x.copy()
                    copy.append(y)
                    all_pairs.append(copy)
                    
    return all_pairs


def create_tensors(pairs):
    anchor_embedding = []
    positive_embedding = []
    negative_embedding = []
    for anchor, positive, negative in pairs:
        anchor_embedding.append(anchor)
        positive_embedding.append(positive)
        negative_embedding.append(negative)
    
    anchor_tensor = torch.stack(anchor_embedding)
    positive_tensor = torch.stack(positive_embedding)
    negative_tensor = torch.stack(negative_embedding)

    return anchor_tensor, positive_tensor, negative_tensor


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample


    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x
        

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels):
        super().__init__()
        self.in_channels=64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)        

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.output = nn.Linear(512*4, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.output(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)