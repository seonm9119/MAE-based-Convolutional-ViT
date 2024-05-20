# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from torchvision.datasets import ImageFolder
import torch
import random
import math
import numpy as np

class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class MaskDataset(ImageFolder):
    def __init__(self, *args,
                 mask_mode='rand', mask_ratio=0.75,
                 patch_size=7, input_size=224,
                 **kwargs):
        super(MaskDataset, self).__init__(*args, **kwargs)

        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.h = self.w = input_size//patch_size
        self.Length = self.h * self.w
        self.D = self.patch_size ** 2 * 3


        if mask_mode == 'block':
            num_masking_patches = int(self.Length * mask_ratio)
            self.mask_generate = MaskingGenerator((self.h, self.w), num_masking_patches, min_num_patches=patch_size)



    def __getitem__(self, index):
        output, target = super(MaskDataset, self).__getitem__(index)
        
        if self.mask_mode == 'block':
            mask = self.mask_generate()
            mask = torch.Tensor(mask)
            mask = 1 - mask
            mask = mask.unsqueeze(-1).repeat(1, 1, self.D)


        elif self.mask_mode == 'rand':


            len_keep = int(self.Length * (1 - self.mask_ratio))

            noise = torch.rand(self.Length)  # noise in [0, 1]
            # print(noise.shape)

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=0)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.zeros(self.Length)
            mask[:len_keep] = 1

            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=0, index=ids_restore)

            mask = mask.unsqueeze(-1).repeat(1,self.D)

        mask = mask.reshape(shape=(self.h, self.w, self.patch_size, self.patch_size, 3))
        mask = torch.einsum('hwpqc->chpwq', mask)
        mask = mask.reshape(shape=(3, self.h * self.patch_size, self.w * self.patch_size))

        return output, mask