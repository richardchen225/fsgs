# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoder.vggt.heads.dpt_head import DPTHead


class VGGT_DPT_GS_Head(DPTHead):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 83,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ):
        super().__init__(
            dim_in,
            patch_size,
            output_dim,
            activation,
            conf_activation,
            features,
            out_channels,
            intermediate_layer_idx,
            pos_embed,
            feature_only,
            down_ratio,
        )

        head_features_1 = 128
        head_features_2 = (
            128 if output_dim > 50 else 32
        )  # sh=0, head_features_2 = 32; sh=4, head_features_2 = 128
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, head_features_1, 7, 1, 3),
            nn.ReLU(),
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        imgs,
        patch_start_idx: int = 5,
        image_size=None,
        frames_chunk_size: int = 8,
    ):
        # H, W = input_info['image_size']
        B, S, _, H, W = imgs.shape
        image_size = self.image_size if image_size is None else image_size

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, imgs, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        depth_chunks = []
        conf_chunks = []
        out_tmp_chunks = []
        # intermediate_chunks = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            chunk_output, depth, conf, out_tmp = self._forward_impl(
                encoder_tokens, imgs, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)
            depth_chunks.append(depth)
            conf_chunks.append(conf)
            out_tmp_chunks.append(out_tmp)
            # intermediate_chunks.append(intermediate_feature)
        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1), torch.cat(depth_chunks, dim=1), torch.cat(conf_chunks, dim=1), torch.cat(out_tmp_chunks, dim=1)

    def _forward_impl(
        self,
        encoder_tokens: List[torch.Tensor],
        imgs,
        patch_start_idx: int = 5,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ):

        if frames_start_idx is not None and frames_end_idx is not None:
            imgs = imgs[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = imgs.shape

        patch_h, patch_w = H // self.patch_size[0], W // self.patch_size[1]

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            # x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()

            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        out = self.scratch_forward(out)       
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        out_tmp = out.view(B, S, *out.shape[1:])
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        
        out1 = self.scratch.output_conv2(out)
        feat = out1.permute(0, 2, 3, 1)
        attr = feat[..., 0:1]
        # print(attr)
        preds = torch.exp(attr)
        # print(preds)
        preds = preds.reshape(B, S, *preds.shape[1:])
        
        conf = feat[..., 1:2]
        conf = 1 + conf.exp()
        conf = conf.reshape(B, S, *conf.shape[1:])
        
        # off = feat[..., 2:]
        # off = off.reshape(B, S, *off.shape[1:])
        
        direct_img_feat = self.input_merger(imgs.flatten(0,1))
        out = out + direct_img_feat
        out = out.view(B, S, *out.shape[1:])
        return out, preds, conf, out_tmp
        
