import torch
import torch.nn.functional as F
from torch import nn
from ComplexityNet.resnet_feature_extractor import resnet152_fe


class ComplexityNet(nn.Module):
    def __init__(self, use_backbone_weights=True):
        super().__init__()

        self.resnet = resnet152_fe(pretrained=use_backbone_weights)
        self.resnet = self.resnet.float()

        # Disable training for backbone.
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Sort padding at runtime,
        self.conv_score_0 = nn.Conv2d(2048, 128, (3, 3))
        self.conv_score_1 = nn.Conv2d(128, 64, (3, 3))
        self.conv_score_2 = nn.Conv2d(64, 32, (3, 3))
        self.conv_score_3 = nn.Conv2d(32, 16, (1, 1))

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16, 1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.map_conv_2048 = nn.Conv2d(2048, 32, (3, 3), padding=(1, 1))
        self.map_conv_32 = nn.Conv2d(32, 16, (3, 3), padding=(1, 1))
        self.map_conv_16 = nn.Conv2d(16, 8, (3, 3), padding=(1, 1))
        self.map_conv_8 = nn.Conv2d(8, 3, (3, 3), padding=(1, 1))

    def score_path(self, batch):
        _, s56, s28, s14, s7 = self.resnet(batch)
        x = F.relu(self.conv_score_0(s7))
        x = F.relu(self.conv_score_1(x))
        x = F.relu(self.conv_score_2(x))
        x = F.relu(self.conv_score_3(x))
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x)
        x = torch.sigmoid(self.fc1(x))

        return x

    def map_path(self, batch):
        _, s56, s28, s14, s7 = self.resnet(batch)
        x = self.upsample2(s7)
        x = F.relu(self.map_conv_2048(x))
        x = self.upsample2(x)
        x = F.relu(self.map_conv_32(x))
        x = self.upsample2(x)
        x = F.relu(self.map_conv_16(x))
        x = self.upsample4(x)
        x = torch.sigmoid(self.map_conv_8(x))

        return x

    def forward(self, batch):
        score_out = self.score_path(batch)
        map_out = self.map_path(batch)
        return map_out, score_out
