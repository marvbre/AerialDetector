import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
from torchinfo import summary
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import resnet18
from gfl_head_reduced import GFLHead
from collections import OrderedDict


class AerialDet(nn.Module):
    
    def __init__(self, num_classes=10):
        
        super(AerialDet, self).__init__()

        backbone = resnet18(weights='IMAGENET1K_V1')
        self.neck = FeaturePyramidNetwork([64, 64, 128, 256, 512], 64)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.neck = nn.Sequential(*list(neck.children())[:-2])
        self.head = GFLHead(num_classes, None, 256,feat_channels=256, stacked_convs=4,octave_base_scale=8,  strides=[8, 16, 32, 64, 128], reg_max=16)
        self.classifier = nn.Linear(64, num_classes)  # 2048 is the feature size of ResNet50's last conv layer

    def forward(self, x):
        # Extract feature maps from ResNet layers
        feature_maps = OrderedDict()

        for i, layer in enumerate([self.backbone[0:4], #Conv_1 
                                   self.backbone[4],   #Conv_2_x 
                                   self.backbone[5],   #Conv3_x
                                   self.backbone[6],   #Conv4_x
                                   self.backbone[7]]): #Conv5_x            
            x = layer(x)
            #print("Stage ",i," has output shape", x.shape)
            feature_maps[str(i)] = x
        
        # Pass through FPN and classify
        fpn_output = self.neck(feature_maps)
        print(fpn_output.keys())
        pooled = self.global_avg_pool(fpn_output['0'])
        return self.classifier(torch.flatten(pooled, 1))

# Instantiate the model
model = AerialDet(num_classes=10)

# Create dummy input tensor
input_tensor = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)

# Forward pass
output = model(input_tensor)