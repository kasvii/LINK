import sys
sys.path.append('./lib')
sys.path.append('./data')
from core.config import cfg
import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp, Block, Attention

from models.backbones import algos
from funcs_utils import load_checkpoint

from functools import partial

from einops import rearrange, repeat
'''posenet_video'''

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        light_dim = dim // 2
        self.fc1 = nn.Linear(dim, light_dim, bias=True)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(light_dim, dim, bias=True)
        self.drop2 = nn.Dropout(drop)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(light_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.fc2(x)
        x = self.drop2(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class  HourglassNet(nn.Module):
    def __init__(self, num_frames=16, num_joints=17, embed_dim=256, depth=4, pretrained=False, num_heads=8, mlp_ratio=0.5, 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        in_dim = 2
        out_dim = 3    
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.joint_embed = nn.Linear(in_dim, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_t = norm_layer(embed_dim)

        self.regression = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.fusion = torch.nn.Conv2d(in_channels=num_frames, out_channels=num_frames, kernel_size=1)
        
        if pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        

    def SpaTemHead(self, x):
        b, t, j, c = x.shape
        x = rearrange(x, 'b t j c  -> (b j) t c')
        x = self.joint_embed(x)
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        
        x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        
        return x

    def forward(self, x):
        b, t, j, c = x.shape
        # [b t j c]
        x = self.SpaTemHead(x) # bj t c
        
        for i in range(self.depth):
            SpaAtten = self.SpatialBlocks[i]
            TemAtten = self.TemporalBlocks[i]
            x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
            x = TemAtten(x)
            x = self.norm_t(x)
            x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
            x = SpaAtten(x)
            x = self.norm_s(x)

        x = rearrange(x, '(b t) j c -> b t j c', t=t)
        x = self.fusion(x)
        x = self.regression(x)

        return x, x


def get_model(num_joint=17, embed_dim=256, depth=4, J_regressor=None, pretrained=False): 
    model = HourglassNet(num_frames=16, num_joints=num_joint, embed_dim=embed_dim, depth=depth, pretrained=pretrained)
    return model

def test_net():
    batch_size = 3
    num_joint = 17
    embed_dim = 128
    model = HourglassNet().cuda()
    model = model.eval()
    input = torch.randn(batch_size, 16, num_joint, 2).cuda()
    pred_2d, pred_2d_feat = model(input)
    print(pred_2d.shape, pred_2d_feat.shape)

if __name__ == '__main__':
    test_net()