from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight_ctx: float
    weight_novel: float
    conf: bool = False
    mask: bool = False
    alpha: bool = False


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = (batch["context"]["image"] + 1) / 2
        pred = prediction.color
        if self.cfg.weight_ctx == 0.0 and self.cfg.weight_novel == 0.0:
            # Do not evaluate an RGB expression that may already contain a
            # non-finite renderer value and only multiply it by zero later.
            return torch.nan_to_num(pred).sum() * 0.0
        if pred.shape[:2] != gt.shape[:2]:
            raise ValueError(
                "RGB prediction/context view count mismatch: "
                f"prediction={tuple(pred.shape)}, gt={tuple(gt.shape)}."
            )

        # The encoder uses the first half of context views to construct the
        # GS map and renders the remaining half as held-out views.  Use the
        # same source/held-out split as the overfit experiment instead of
        # silently mixing both groups into one MSE.
        if depth_dict is None or "depth" not in depth_dict:
            raise ValueError("RGB MSE needs depth_dict['depth'] to locate the source split.")
        source_views = int(depth_dict["depth"].shape[1])
        total_views = int(pred.shape[1])
        if not 0 < source_views <= total_views:
            raise ValueError(
                "Invalid source/held-out split: "
                f"source_views={source_views}, total_views={total_views}."
            )

        if self.cfg.weight_ctx == 0.0:
            source_loss = pred.new_zeros(())
        else:
            source_loss = (pred[:, :source_views] - gt[:, :source_views]).square().mean()
        if source_views < total_views:
            if self.cfg.weight_novel == 0.0:
                heldout_loss = pred.new_zeros(())
            else:
                heldout_loss = (
                    pred[:, source_views:] - gt[:, source_views:]
                ).square().mean()
        else:
            heldout_loss = source_loss.new_zeros(())

        return torch.nan_to_num(
            self.cfg.weight_ctx * source_loss
            + self.cfg.weight_novel * heldout_loss,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
#         def forward(
#             self,
#             prediction: DecoderOutput,
#             batch: BatchedExample,
#             gaussians: Gaussians | None,
#             depth_dict: dict | None,
#             global_step: int,
#         ) -> Float[Tensor, ""]:

#             b, s, _, h, w = prediction.color.shape
#             num_ctx = depth_dict["depth"].shape[1]
#             num_novel = s - num_ctx

#             # -------------------------------------------------------------------
#             # 1. 预处理预测图与 GT 图 (保持维度，不提前展平)
#             # -------------------------------------------------------------------
#             # 预测图: (B, S, H, W, 3)
#             all_pred = prediction.color.permute(0, 1, 3, 4, 2)
#             # GT 图: (B, S, H, W, 3)
#             all_gt = ((batch["context"]["image"][:, batch["using_index"]] + 1) / 2).permute(0, 1, 3, 4, 2)

#             # -------------------------------------------------------------------
#             # 2. 定义时序权重函数 (Temporal Weighting)
#             # -------------------------------------------------------------------
#             def get_seq_weights(num_steps, min_weight, max_weight):
#                 """
#                 生成一个从 1.0 到 max_weight 线性递增的权重序列
#                 """
#                 if num_steps <= 1:
#                     return torch.ones(1, device='cuda')
#                 # 比如: [1.0, 2.0, 3.0] 如果 num_steps=3
#                 return torch.linspace(min_weight, max_weight, num_steps, device='cuda')

#             # -------------------------------------------------------------------
#             # 3. 计算 Context View Loss (组内权重递增)
#             # -------------------------------------------------------------------
#             pred_ctx = all_pred[:, :num_ctx, ...]
#             gt_ctx = all_gt[:, :num_ctx, ...]

#             # 计算每个视角的 MSE: (B, S_ctx, H, W, 3) -> (S_ctx)
#             mse_per_ctx_view = ((pred_ctx - gt_ctx) ** 2).mean(dim=(0, 2, 3, 4))

#             # 获取组内递增权重
#             w_ctx_seq = get_seq_weights(num_ctx, min_weight=0.5, max_weight=1.0)
#             loss_ctx = (mse_per_ctx_view * w_ctx_seq).mean()

#             # -------------------------------------------------------------------
#             # 4. 计算 Novel View Loss (组内权重递增)
#             # -------------------------------------------------------------------
#             pred_novel = all_pred[:, num_ctx:, ...]
#             gt_novel = all_gt[:, num_ctx:, ...]

#             # 计算每个视角的 MSE: (B, S_novel, H, W, 3) -> (S_novel)
#             mse_per_novel_view = ((pred_novel - gt_novel) ** 2).mean(dim=(0, 2, 3, 4))

#             # Novel 视角通常更难，我们可以让权重的斜率更陡一点 (比如最大到 5.0)
#             w_novel_seq = get_seq_weights(num_novel, min_weight=1.0, max_weight=1.5)
#             loss_novel = (mse_per_novel_view * w_novel_seq).mean()

#             # -------------------------------------------------------------------
#             # 5. 组装最终结果
#             # -------------------------------------------------------------------
#             total_loss = loss_novel + loss_ctx

#             return torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
