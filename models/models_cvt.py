from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from models.models_vit import VisionTransformer


class Encoder(nn.Module):
    def __init__(self,
                 in_chans=3,
                 init='trunc_norm',
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 spec=None
                 ):

        super().__init__()

        stages = []
        for i in range(spec['NUM_STAGES']):
            kwargs = {'patch_size': spec['PATCH_SIZE'][i],
                      'patch_stride': spec['PATCH_STRIDE'][i],
                      'patch_padding': spec['PATCH_PADDING'][i],
                      'embed_dim': spec['DIM_EMBED'][i],
                      'depth': spec['DEPTH'][i],
                      'num_heads': spec['NUM_HEADS'][i],
                      'mlp_ratio': spec['MLP_RATIO'][i],
                      'qkv_bias': spec['QKV_BIAS'][i],
                      'drop_rate': spec['DROP_RATE'][i],
                      'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                      'drop_path_rate': spec['DROP_PATH_RATE'][i],
                      'with_cls_token': spec['CLS_TOKEN'][i],
                      'method': spec['QKV_PROJ_METHOD'][i],
                      'kernel_size': spec['KERNEL_QKV'][i],
                      'padding_q': spec['PADDING_Q'][i],
                      'padding_kv': spec['PADDING_KV'][i],
                      'stride_kv': spec['STRIDE_KV'][i],
                      'stride_q': spec['STRIDE_Q'][i]}

            stages.append(VisionTransformer(in_chans=in_chans,
                                            init=init,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            **kwargs))

            in_chans = spec['DIM_EMBED'][i]
        self.stages = nn.ModuleList(stages)

        self.dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(self.dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]


    def forward(self, x):
        for stage in self.stages:
            x, cls_tokens = stage(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = Encoder(in_chans=in_chans,
                 init=init,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 spec=spec)

        # Classifier head
        self.head = nn.Linear(spec['DIM_EMBED'][-1], num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)


    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)

        return x


def get_cls_model(cfg, num_classes=1000):
    msvit_spec = cfg.SPEC
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=num_classes,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )


    return msvit
