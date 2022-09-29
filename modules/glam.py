import torch
from torch import nn
from .local_channel_attention import LocalChannelAttention
from .global_channel_attention import GlobalChannelAttention
from .local_spatial_attention import LocalSpatialAttention
from .global_spatial_attention import GlobalSpatialAttention


class GLAM(nn.Module):
    def __init__(self, in_channels, num_reduced_channels, feature_map_size, kernel_size):
        super().__init__()
        
        self.local_channel_att = LocalChannelAttention(feature_map_size, kernel_size)
        self.local_spatial_att = LocalSpatialAttention(in_channels, num_reduced_channels)
        self.global_channel_att = GlobalChannelAttention(feature_map_size, kernel_size)
        self.global_spatial_att = GlobalSpatialAttention(in_channels, num_reduced_channels)
        
        self.fusion_weights = nn.Parameter(torch.Tensor([0.333, 0.333, 0.333])) # equal intial weights
        
    def forward(self, x):
        local_channel_att = self.local_channel_att(x) # local channel
        local_att = self.local_spatial_att(x, local_channel_att) # local spatial
        global_channel_att = self.global_channel_att(x) # global channel
        global_att = self.global_spatial_att(x, global_channel_att) # global spatial
        
        local_att = local_att.unsqueeze(1) # unsqueeze to prepare for concat
        global_att = global_att.unsqueeze(1) # unsqueeze to prepare for concat
        x = x.unsqueeze(1) # unsqueeze to prepare for concat
        
        all_feature_maps = torch.cat((local_att, x, global_att), dim=1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        fused_feature_maps = (all_feature_maps * weights).sum(1)
        
        return fused_feature_maps
