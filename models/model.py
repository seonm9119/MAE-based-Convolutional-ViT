from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
from models.my_vit import VisionTransformer
from models.utils import UpsampleConcat


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
                                            update=True,
                                            **kwargs))

            in_chans = spec['DIM_EMBED'][i]

        self.stages = nn.ModuleList(stages)

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]  # TRUE

    def forward(self, x, mask):

        enc_f = []
        for stage in self.stages:
            enc_f.append(x)
            x, mask, _ = stage(x, mask)



        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)

        return x, enc_f

class Decoder(nn.Module):
    def __init__(self,
                 in_chans=384,
                 init='trunc_norm',
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 spec=None
                 ):

        super().__init__()

        decoder = []
        for i in reversed(range(spec['NUM_STAGES']-1)):
            kwargs = {'patch_size': 3,
                      'patch_stride': 1,
                      'patch_padding': 1,
                      'embed_dim': spec['DIM_EMBED'][i],
                      'depth': 2,
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


            decoder.append(VisionTransformer(in_chans=in_chans + spec['DIM_EMBED'][i],
                                             init=init,
                                             act_layer=act_layer,
                                             norm_layer=norm_layer,
                                             update=False,
                                             **kwargs))

            in_chans = spec['DIM_EMBED'][i]

        self.decoder = nn.ModuleList(decoder)
        self.upsample = UpsampleConcat()
        self.last = nn.Conv2d(spec['DIM_EMBED'][0], 3, 3, 1, 1)

    def forward(self, x, enc_f):
        # embed tokens

        h = w = x.shape[1] ** 0.5
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        for decoder in self.decoder:

            x, _ = self.upsample(x, enc_f.pop())
            x, _, _ = decoder(x, None)

        x, _ = self.upsample(x, enc_f.pop(), True)
        x = self.last(x)

        return x


class MyModel(nn.Module):

    def __init__(self,
                 in_chans=3,
                 init='trunc_norm',
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 spec=None,):

        super().__init__()

        self.encoder = Encoder(in_chans=in_chans,
                 init=init,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 spec=spec)

        self.decoder = Decoder(in_chans=spec['DIM_EMBED'][2],
                 init=init,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 spec=spec)


    def forward(self, imgs, mask):
        
        if mask is not None:
            imgs = torch.mul(imgs, mask)
        
        latent, enc_f = self.encoder(imgs, mask)
        pred = self.decoder(latent, enc_f)  # [N, L, p*p*3]
        return pred


def get_cls_model(cfg):

    msvit = MyModel(in_chans=3,
                    act_layer=nn.GELU,
                    norm_layer=partial(nn.LayerNorm, eps=1e-5),
                    init=getattr(cfg.SPEC, 'INIT', 'trunc_norm'),
                    spec=cfg.SPEC,)

    return msvit
