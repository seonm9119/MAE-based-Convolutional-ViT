import torch
import torch.nn as nn

class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def dialation_holes(self, hole_mask):
        b, ch, h, w = hole_mask.shape
        dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
        torch.nn.init.constant_(dilation_conv.weight, 1.0)
        with torch.no_grad():
            output_mask = dilation_conv(hole_mask)
        updated_holes = output_mask != 0
        return updated_holes.float()

    def total_variation_loss(self, image, mask):
        hole_mask = 1 - mask
        dilated_holes = self.dialation_holes(hole_mask)
        colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
        rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]

        loss = torch.mean(torch.abs(colomns_in_Pset * (
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
                   torch.mean(torch.abs(rows_in_Pset * (
                           image[:, :, :1, :] - image[:, :, -1:, :])))

        return loss



    def forward(self, imgs, pred, mask):

        if mask is not None:
            comp = mask * imgs + (1 - mask) * pred
            tv_loss = self.total_variation_loss(comp, mask)

            hole_loss = self.l1((1 - mask) * pred, (1 - mask) * imgs)
            valid_loss = self.l1(mask * pred, mask * imgs)

            loss = valid_loss + hole_loss * 6.0 + tv_loss * 0.1
        else:
            loss = self.l1(imgs, pred)

        return loss


