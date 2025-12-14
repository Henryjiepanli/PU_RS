import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import *
import numpy as np
from .resnet import *
from .Res2Net_v1b import *
from .ResNext101 import ResNeXt101


class MultiScaleDynamicAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, scale_factors=(1, 2, 4), num_samples=5):
        super(MultiScaleDynamicAttention, self).__init__()
        self.num_samples = num_samples

        # 多尺度分支
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // scale, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(in_channels // scale),
                nn.ReLU(),
                nn.Conv2d(in_channels // scale, in_channels, kernel_size=1)
            ) for scale, kernel_size in zip(scale_factors, [3, 5, 7])
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(scale_factors)), requires_grad=True)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # MC Dropout
        self.mc_dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        # 多尺度特征提取
        scale_features = [weight * scale(x) for scale, weight in zip(self.scales, self.scale_weights)]
        multi_scale_feature = sum(scale_features)

        # 通道注意力
        channel_attention = self.channel_attention(multi_scale_feature)
        x_channel = multi_scale_feature * channel_attention

        # MC Dropout 增强
        mc_features = [self.mc_dropout(x_channel) for _ in range(self.num_samples)]
        mean_feature = torch.mean(torch.stack(mc_features), dim=0)  # 均值
        variance_feature = torch.var(torch.stack(mc_features), dim=0)  # 方差

        # 综合特征
        enhanced_feature = mean_feature + F.sigmoid(variance_feature) * x_channel
        return enhanced_feature

class Contexual_Uncertainty_Guide_Unit(nn.Module):
    def __init__(self, in_channels, alpha=1.0, scales=[0.5, 0.75, 1.0, 1.25, 1.5], entropy_scale=1.0):
        super(Contexual_Uncertainty_Guide_Unit, self).__init__()
        # 输入通道数 * 多尺度数 + 边缘信息
        self.conv1 = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha  # 高频增强系数
        self.scales = scales
        self.entropy_scale = entropy_scale  # 
        self.MSDA = nn.ModuleList([MultiScaleDynamicAttention(in_channels) for _ in range(len(scales))])

        # 设计一个多尺度卷积层来替代频域操作
    def forward(self, x, seg_map):
        device = x.device  # 获取输入张量的设备信息

        seg_map = F.interpolate(seg_map, size=x.size()[2:], mode="bilinear", align_corners=True)

        # Step 1: Calculate entropy map from seg_map (binary classification)
        entropy_map = self.calculate_entropy(seg_map)

        # Step 2: Multi-scale spatial enhancement using convolutions
        multi_scale_features = []
        for i, scale in enumerate(self.scales):
            resized = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

            # 使用熵图指导高频增强
            resized_entropy_map = F.interpolate(entropy_map, size=resized.size()[2:], mode='bilinear', align_corners=False)
            enhanced_feature = resized * (1 + self.entropy_scale * self.sigmoid(resized_entropy_map))

            enhanced_feature = self.MSDA[i](enhanced_feature)

            multi_scale_features.append(F.interpolate(enhanced_feature, size=x.size()[2:], mode='bilinear', align_corners=False))

        # Step 4: Concatenate features
        combined_features = torch.cat(multi_scale_features, dim=1)

        # Step 5: Convolutional refinement
        features = self.conv1(combined_features)
        out_map = self.conv2(features)
        return features, out_map


    def calculate_entropy(self, seg_map):
        """ Calculate the entropy map based on the segmentation map for binary classification.
            seg_map: Tensor of shape [B, 1, H, W] containing the probability of foreground.
        """
        eps = 1e-6  # To avoid log(0) error
        p = self.sigmoid(seg_map)  # Probability map for foreground class
        entropy_map = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        return entropy_map


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = F.interpolate(output, size=xs[1].size()[2:], mode="bilinear", align_corners=True)
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        return output


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class dynamic_dual_uncertainty_aware_con_unit(nn.Module):
    def __init__(self, channel, sigma=0.1, learnable_sigma=False, centers=None):
        super(dynamic_dual_uncertainty_aware_con_unit, self).__init__()
        # 共享计算模块：低分辨率和高分辨率都用同一个模块
        self.UACL_1 = DynamicDualUncertaintyAwareContrastiveLearning(channel, sigma, learnable_sigma, centers)
        self.UACL_2 = DynamicDualUncertaintyAwareContrastiveLearning(channel, sigma, learnable_sigma, centers)
        self.conv_cat_1 = nn.Conv2d(2 * channel, channel, kernel_size=1)
        self.conv_cat_2 = nn.Conv2d(2 * channel, channel, kernel_size=1)
        self.fusion = FeatureFusionBlock(channel)

    def forward(self, low, high, predict):
        # 处理低分辨率和高分辨率的前景和背景
        low_fore, low_back, contrastive_loss_low = self.UACL_1(low, F.interpolate(predict, low.size()[2:], mode='bilinear', align_corners=True))
        high_fore, high_back, contrastive_loss_high = self.UACL_2(high, predict)
        
        # 合并低高分辨率特征
        low = self.conv_cat_1(torch.cat((low_fore, low_back), dim=1))
        high = self.conv_cat_2(torch.cat((high_fore, high_back), dim=1))
        
        # 融合高分辨率和低分辨率特征
        fusion = self.fusion(high, low)
        
        # 总的对比损失
        contrastive_loss = contrastive_loss_high + contrastive_loss_low
        
        return fusion, contrastive_loss


class DynamicDualUncertaintyAwareContrastiveLearning(nn.Module):
    def __init__(self, feature_dim, sigma=0.1, learnable_sigma=False, centers=None):
        super(DynamicDualUncertaintyAwareContrastiveLearning, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma)) if learnable_sigma else sigma
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(5)) if centers is None else centers  # 可学习的中心点
        self.prob = nn.Sigmoid()

    def soft_rank_gaussian(self, uncertainty_map):
        # 初始化排序分数矩阵
        ranks = torch.zeros(uncertainty_map.shape + (len(self.centers),), device=uncertainty_map.device)
        
        for i, center in enumerate(self.centers):
            ranks[..., i] = torch.exp(-((uncertainty_map - center) ** 2) / (2 * self.sigma ** 2))
        
        # 对每个像素点的分数进行归一化
        ranks = ranks / ranks.sum(dim=-1, keepdim=True)
        return ranks

    def forward(self, feature_map, pred_map):
        prob_map = self.prob(pred_map)  # 转换为概率图
        
        # 计算前景和背景的不确定性
        # 截断前景和背景的计算，避免它们重叠
        fore_uncertainty_map = torch.clamp(prob_map - 0.5, min=0)  # 前景的最大不确定性
        back_uncertainty_map = torch.clamp(0.5 - prob_map, min=0)  # 背景的最大不确定性

        # 使用高斯分布进行软排序
        fore_rank_map = self.soft_rank_gaussian(fore_uncertainty_map)
        back_rank_map = self.soft_rank_gaussian(back_uncertainty_map)

        # 加权输入特征
        # Sum across the last dimension of fore_rank_map and back_rank_map, then unsqueeze to match feature_map shape
        fore_weighted_features = feature_map * fore_rank_map.sum(dim=-1, keepdim=True).squeeze(dim=-1)  # Shape [4, 32, 128, 128]
        back_weighted_features = feature_map * back_rank_map.sum(dim=-1, keepdim=True).squeeze(dim=-1)  # Shape [4, 32, 128, 128]
        loss = self.compute_contrastive_loss(fore_weighted_features, back_weighted_features)

        return fore_weighted_features, back_weighted_features, loss

    def compute_contrastive_loss(self, fore_features, back_features, temperature=0.07):
        """
        fore_features: [B, C, H, W] -> anchor & positive pairs
        back_features: [B, C, H, W] -> negatives
        """
        B = fore_features.size(0)
        
        # Flatten and normalize features
        fore = F.normalize(fore_features.view(B, -1), dim=1)   # Anchor (also used as positive)
        back = F.normalize(back_features.view(B, -1), dim=1)   # Negatives

        # Construct cosine similarity matrix: [B, B]
        sim_matrix = torch.matmul(fore, back.T)  # similarity between fore and back
        pos_sim = torch.sum(fore * fore, dim=1, keepdim=True)  # [B, 1]

        # Concatenate positive sim + negatives sim
        logits = torch.cat([pos_sim, sim_matrix], dim=1)  # [B, 1 + B]
        logits /= temperature

        # Labels: positive is always index 0
        labels = torch.zeros(B, dtype=torch.long, device=fore.device)

        # CrossEntropyLoss over [positive + negatives]
        loss = F.cross_entropy(logits, labels)
        return loss

class PUGNet(nn.Module):
    def __init__(self, backbone_name, channel):
        super(PUGNet, self).__init__()
        self.backbone, self.encode_channels = self.get_backbone(backbone_name)

        self.conv1 = BasicConv2d(2*self.encode_channels[0], self.encode_channels[0], 1)
        self.conv2 = BasicConv2d(2*self.encode_channels[1], self.encode_channels[1], 1)
        self.conv3 = BasicConv2d(2*self.encode_channels[2], self.encode_channels[2], 1)
        self.conv4 = BasicConv2d(2*self.encode_channels[3], self.encode_channels[3], 1)

        self.RF_blocks = nn.ModuleList([
            RFB(in_channel, channel) for in_channel in (self.encode_channels)
        ])

        self.DUAM_blocks = nn.ModuleList([
            dynamic_dual_uncertainty_aware_con_unit(channel) for _ in range(len(self.encode_channels)-1)
        ])

        self.EDRM_blocks = nn.ModuleList([
            Contexual_Uncertainty_Guide_Unit(channel)for _ in range(len(self.encode_channels)-1)
        ])

        self.out_4 = nn.Sequential(BasicConv2d(channel, channel, 3, 1, 1), nn.Conv2d(channel, 2, 1))

    def get_backbone(self, backbone_name):
        if backbone_name == 'resnet':
            backbone = resnet50(backbone_path='./timm_resnet50/resnet50_b1k-532a802a.pth', pretrained=True)
            encode_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'resnext101':
            backbone = ResNeXt101(backbone_path='./timm_resnet50/resnext_101_32x4d.pth')
            encode_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'res2net':
            backbone = res2net50_v1b_26w_4s(pretrained=True)
            encode_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'PVT-v2':
            backbone = pvt_v2_b4()
            path = './pretrained_model/pvt_v2_b4.pth'
            save_model = torch.load(path)
            model_dict = backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            encode_channels = [64, 128, 320, 512]
        else:
            raise ValueError(f"Unknown backbone name: {backbone_name}")

        return backbone, encode_channels


    def forward(self, A, B):
        size = A.size()[2:]
        layer_1_A, layer_2_A, layer_3_A, layer_4_A = self.backbone(A)
        layer_1_B, layer_2_B, layer_3_B, layer_4_B = self.backbone(B)

        layer1 = self.conv1(torch.cat((layer_1_A, layer_1_B), dim=1))
        layer2 = self.conv2(torch.cat((layer_2_A, layer_2_B), dim=1))
        layer3 = self.conv3(torch.cat((layer_3_A, layer_3_B), dim=1))
        layer4 = self.conv4(torch.cat((layer_4_A, layer_4_B), dim=1))

        layer4 = self.RF_blocks[3](layer4)
        layer3 = self.RF_blocks[2](layer3)
        layer2 = self.RF_blocks[1](layer2)
        layer1 = self.RF_blocks[0](layer1)


        predict_4 = self.out_4(layer4)

        fusion, Contrastive_loss_3 = self.DUAM_blocks[2](layer3, layer4, predict_4[:,1,:,:].unsqueeze(1))

        fusion, predict_3 = self.EDRM_blocks[2](fusion, predict_4[:,1,:,:].unsqueeze(1))


        fusion, Contrastive_loss_2 = self.DUAM_blocks[1](layer2, fusion, predict_3[:,1,:,:].unsqueeze(1))

        fusion, predict_2 = self.EDRM_blocks[1](fusion, predict_3[:,1,:,:].unsqueeze(1))

        fusion, Contrastive_loss_1 = self.DUAM_blocks[0](layer1, fusion, predict_2[:,1,:,:].unsqueeze(1))

        fusion, predict_1 = self.EDRM_blocks[0](fusion, predict_2[:,1,:,:].unsqueeze(1))

        Contrastive_loss = Contrastive_loss_3 + Contrastive_loss_2 + Contrastive_loss_1


        return F.interpolate(predict_4, size, mode='bilinear', align_corners=True),F.interpolate(predict_3, size, mode='bilinear', align_corners=True),\
               F.interpolate(predict_2, size, mode='bilinear', align_corners=True),F.interpolate(predict_1, size, mode='bilinear', align_corners=True),\
               Contrastive_loss