from functools import partial
from thop import profile
import torch
from torch import nn
from deit.super_deit import VisionTransformer


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
    ).to("cuda")


print(
    {
        i: profile(
            get_model(i),
            inputs=(torch.randn(512, 3, 224, 224, device="cuda"),),
            verbose=False,
        )
        for i in img_size_list
    }
)

# res: (macs, params)
# {96: (281845432320.0, 21974632.0), 112: (378803847168.0, 21974632.0), 128: (487560314880.0, 21974632.0), 160: (750451949568.0, 21974632.0), 192: (1070822326272.0, 21974632.0), 224: (1445646827520.0, 21974632.0)}