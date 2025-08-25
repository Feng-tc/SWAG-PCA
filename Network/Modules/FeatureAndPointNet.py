import torch
import torch.nn as nn

from .BasicModules import Conv1dLayer, LinearLayer


class Transform(nn.Module):
    def __init__(self, in_channels):
        super(Transform, self).__init__()

        self.in_channels = in_channels

        self.clayer = []
        self.clayer.append(Conv1dLayer(in_channels, 64))
        self.clayer.append(Conv1dLayer(64, 128))
        self.clayer.append(Conv1dLayer(128, 1024))
        self.clayer = nn.Sequential(*self.clayer)

        self.llayer = []
        self.llayer.append(LinearLayer(1024, 512))
        self.llayer.append(LinearLayer(512, 256))
        self.llayer.append(nn.Linear(256, in_channels * in_channels))
        self.llayer = nn.Sequential(*self.llayer)

        self.iden = torch.eye(in_channels, dtype=torch.float32).view(1, in_channels * in_channels)

    def forward(self, x):
        B = x.shape[0]

        tran = self.clayer(x)

        tran = torch.max(tran, 2, keepdim=True)[0]
        tran = tran.view(-1, 1024)

        tran = self.llayer(tran)

        # iden是偏置
        iden = self.iden.repeat(B, 1)

        if x.is_cuda:
            iden = iden.cuda()

        tran = tran + iden
        tran = tran.view(-1, self.in_channels, self.in_channels)

        x = x.transpose(2, 1)
        tran = torch.bmm(x, tran)
        tran = tran.transpose(2, 1)

        return tran


class PointEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 使用ModuleList替代Python list，更好的参数管理
        self.transform1 = Transform(in_channels)
        self.layer1 = Conv1dLayer(in_channels, 64)
        self.transform2 = Transform(64)
        self.layer2 = nn.Sequential(
            Conv1dLayer(64, 128),
            Conv1dLayer(128, 1024)
        )

    def forward(self, x):
        """点云编码器的前向传播
        
        Args:
            x (torch.Tensor): 输入点云，形状为 [B, N, C]
            
        Returns:
            torch.Tensor: 编码后的特征，形状为 [B, 1088, N]
        """
        # 1. 转置输入以匹配卷积层期望的格式
        x = x.transpose(2, 1)  # [B, C, N]
        
        # 2. 第一次特征变换和提取
        x = self.transform1(x)  # [B, C, N]
        local_feature = self.layer1(x)  # [B, 64, N]
        
        # 3. 第二次特征变换和深层特征提取
        x = self.transform2(local_feature)  # [B, 64, N]
        global_feature = self.layer2(x)  # [B, 1024, N]
        
        # 4. 全局特征池化，使用keepdim避免额外的维度操作
        global_feature = global_feature.max(2, keepdim=True)[0]  # [B, 1024, 1]
        
        # 5. 特征扩展，使用expand避免内存复制
        global_feature = global_feature.expand(-1, -1, x.shape[2])  # [B, 1024, N]
        
        # 6. 特征拼接，使用torch.cat的dim参数优化性能
        return torch.cat([local_feature, global_feature], dim=1)  # [B, 1088, N]


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 使用ModuleList替代Python list，更好的参数管理
        self.transform = Transform(in_channels)
        self.layer1 = Conv1dLayer(in_channels, 64)
        self.layers = nn.Sequential(
            Conv1dLayer(64, 128),
            Conv1dLayer(128, 1024, activation=False)
        )

    def forward(self, x):
        """特征编码器的前向传播
        
        Args:
            x (torch.Tensor): 输入特征，形状为 [B, C, N]
            
        Returns:
            torch.Tensor: 编码后的特征，形状为 [B, 1088, N]
        """
        # 1. 特征变换
        xt = self.transform(x)
        
        # 2. 初始特征提取
        local_feature = self.layer1(xt)  # [B, 64, N]
        
        # 3. 深层特征提取
        global_feature = self.layers(local_feature)  # [B, 1024, N]
        
        # 4. 全局特征池化，使用keepdim避免额外的维度操作
        global_feature = global_feature.max(2, keepdim=True)[0]  # [B, 1024, 1]
        
        # 5. 特征扩展，使用expand避免内存复制
        global_feature = global_feature.expand(-1, -1, x.shape[2])  # [B, 1024, N]
        
        # 6. 特征拼接，使用torch.cat的dim参数优化性能
        return torch.cat([local_feature, global_feature], dim=1)  # [B, 1088, N]