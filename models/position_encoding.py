# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    $ PE_{(pos, 2i)} &=\sin \left(\operatorname{pos} / 10000^{2i / d}\right)
    PE_{(pos, 2i+1)} &=\cos \left(\operatorname{pos} / 10000^{2i / d}\right)$
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        '''
        y_embed shape (bs, h, w):
        [ [1,1,1,..,1],
        [2,2,2,..,2],
        ...
        [h,h,h,..,h] ]
        x_embed shape (bs, h, w):
        [ [1,2,3,..,w],
        [1,2,3,..,w],
        ...
         [1,2,3,..,w] ]
        '''
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # 列方向归一化 [0, 2pi] \operatorname{pos}
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # 行方向归一化 [0, 2pi] \operatorname{pos}
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 10000^{2i / d}
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # (bs, h, w, num_pos_feats) \operatorname{pos} / 10000^{2i / d}
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 偶数维使用 sin 编码，奇数维使用 cos 编码 (bs, h, w, num_pos_feats // 2, 2) -> (bs, h, w, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # (bs, h, w, 2 x num_pos_feats) -> (bs, 2 x num_pos_feats, h, w)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # 默认需要编码的特征图的行、列不超为 50 !
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        # [0, 1, ... w - 1]
        i = torch.arange(w, device=x.device)
        # [0, 1, ... h - 1]
        j = torch.arange(h, device=x.device)
        # shape: (w, num_pos_feats)
        x_emb = self.col_embed(i)
        # shape: (h, num_pos_feats)
        y_emb = self.row_embed(j)
        # (h, w, 2 x num_pos_feats) -> (2 x num_pos_feats, h, w) -> (1, 2 x num_pos_feats, h, w) -> (bs, 2 x num_pos_feats, h, w)
        pos = torch.cat([
            # (h, w, num_pos_feats)
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            # (h, w, num_pos_feats)
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # 正余弦位置编码
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        # # 可学习位置编码
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
