import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder.streamvggt.layers import Mlp
from src.model.encoder.streamvggt.layers.block import Block
from src.model.encoder.streamvggt.heads.head_act import activate_pose

class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        
        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )
    
    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4, past_key_values_camera = None, use_cache: bool = False) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]
  
        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        if use_cache:
            if past_key_values_camera is None:
                past_key_values_camera = [None] * num_iterations
            pred_pose_enc_list, past_key_values_camera = self.trunk_fn(pose_tokens, num_iterations, past_key_values_camera, use_cache)
            return pred_pose_enc_list, past_key_values_camera
        else:
            pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations, past_key_values_camera=None, use_cache=use_cache)
            return pred_pose_enc_list
        
    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int, past_key_values_camera, use_cache: bool) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        
        pred_pose_enc = None
        pred_pose_enc_list = []
        new_caches = [] if use_cache else None
        for i in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B,S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            
            is_last_iteration = (i == num_iterations - 1)
            
            if not use_cache:
                L = S * 1
                frame_ids = torch.arange(L, device=pose_tokens_modulated.device) // 1  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(pose_tokens_modulated.dtype) * torch.finfo(pose_tokens_modulated.dtype).min
                        
                for idx in range(self.trunk_depth):
                    pose_tokens_modulated = self.trunk[idx](pose_tokens_modulated, attn_mask=attn_mask)    
            else:
                # 推理逻辑修正：使用该迭代步对应的专用 Cache
                # past_key_values_camera_list[i] 存放的是之前所有帧在【第 i 次迭代】时的 KV
                current_iter_cache = past_key_values_camera[i] 

                # 如果是该迭代步的第一帧，初始化各层的 Cache 容器
                if current_iter_cache is None:
                    current_iter_cache = [None] * self.trunk_depth

                updated_iter_cache = []
                for idx in range(self.trunk_depth):
                    pose_tokens_modulated, layer_kv = self.trunk[idx](
                        pose_tokens_modulated,
                        attn_mask=None,
                        past_key_values=current_iter_cache[idx],
                        use_cache=True
                    )
                    updated_iter_cache.append(layer_kv)

                # 将更新后的【第 i 次迭代】Cache 存起来
                new_caches.append(updated_iter_cache)
                
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)
            
        if use_cache:
            return pred_pose_enc_list, new_caches
        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    return x * (1 + scale) + shift
