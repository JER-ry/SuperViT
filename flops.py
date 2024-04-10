from functools import partial
import torch
from torch import nn
from deit.super_deit import VisionTransformer
from fvcore.nn import FlopCountAnalysis


img_size_list = [96, 112, 128, 160, 192, 224]


def get_model(img_size):
    return VisionTransformer(
        img_size_list=[img_size],
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).to("cpu")


print(
    {
        i: FlopCountAnalysis(
            get_model(i), torch.randn(1, 3, 224, 224, device="cpu")
        ).total()
        for i in img_size_list
    }
)

# keep_ratio=0.7
# {96: 557984640, 112: 753287424, 128: 974573952, 160: 1518407808, 192: 2198775168, 224: 3019433856}

# keep_ratio=1.0
# {96: 811149696, 112: 1102108416, 128: 1441700736, 160: 2273642880, 192: 3322901376, 224: 4608940416}