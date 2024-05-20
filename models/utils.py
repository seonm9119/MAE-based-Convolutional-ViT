import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConcat(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        # Define the upsampling layer with nearest neighbor
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_upsample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, dec_feature, enc_feature,last=False):
        # upsample and concat features

        if enc_feature is None and last is False:
            out = self.upsample(dec_feature)

        elif enc_feature is not None and last is False:
            out = self.upsample(dec_feature)
            out = torch.cat([out, enc_feature], dim=1)

        elif last is True:
            out = self.last_upsample(dec_feature)
            #out = torch.cat([out, enc_feature], dim=1)


        out_mask = None
        return out, out_mask


class UpdateMask(nn.Module):
    def __init__(self,
                 in_chans=3,
                 out_chans=512,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super().__init__()

        self.padding = padding
        self.stride = stride
        self.mask_kernel = torch.ones(out_chans, in_chans,
                                      kernel_size, kernel_size).cuda()

        # Define sum1 for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] * self.mask_kernel.shape[3]
        # Define the updated mask
        self.update_mask = None
        # Define the mask ratio (sum(1) / sum(M))
        self.mask_ratio = None

    def forward(self, mask):
        with torch.no_grad():
            # Create the updated mask
            # for calcurating mask ratio (sum(1) / sum(M))

            update_mask = F.conv2d(mask, self.mask_kernel,
                                   bias=None, stride=self.stride,
                                   padding=self.padding,
                                   dilation=1,
                                   groups=1)

            # calcurate mask ratio (sum(1) / sum(M))
            mask_ratio = self.sum1 / (update_mask + 1e-5)

            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        return mask_ratio, update_mask