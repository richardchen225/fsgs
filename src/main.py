import os
from pathlib import Path
import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from safetensors.torch import load_file
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import get_model
import torch.nn.init as init
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

    
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {output_dir}.")
    
    cfg.train.output_path = output_dir
    
    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict, resolve=True)
        )
        callbacks.append(LearningRateMonitor("step", True))
        if wandb.run is not None:
            wandb.run.log_code("src", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    else:
        logger = LocalLogger()
    
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'
    
    step_tracker = StepTracker()
    
    requested_devices = cfg.trainer.devices
    use_ddp = isinstance(requested_devices, int) and requested_devices > 1

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices=requested_devices,
        strategy="ddp_find_unused_parameters_false" if use_ddp else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        inference_mode=False if cfg.mode == "train" else True,
    )
 
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    model = get_model(cfg.model.encoder, cfg.model.decoder)

    if cfg.mode == 'train':
        if cfg.checkpointing.load is not None:
            print(f"Resuming full training state from {cfg.checkpointing.load}")
            ckpt = None
        elif cfg.checkpointing.train_pretrained_weights is None:
            print("No train pretrained weights configured; training from initialized weights.")
            ckpt = None
        else:
            print(f"Loading train pretrained weights from {cfg.checkpointing.train_pretrained_weights}")
            if str(cfg.checkpointing.train_pretrained_weights).endswith(".safetensors"):
                ckpt = load_file(cfg.checkpointing.train_pretrained_weights)
            else:
                ckpt = torch.load(cfg.checkpointing.train_pretrained_weights, map_location='cpu')
                ckpt = ckpt.get("state_dict", ckpt)
        def rename_key(key: str) -> str:
            if key.startswith("gs_head"):
                key = key.replace("gs_head", "encoder.gaussian_param_head", 1)
            elif key.startswith("gs_renderer"):
                key = key.replace("gs_renderer", "encoder", 1)
            elif key.startswith("depth_head"):
                key = key.replace("depth_head", "encoder.depth_head", 1)
            elif key.startswith("visual_geometry_transformer"):
                key = key.replace("visual_geometry_transformer", "encoder.aggregator", 1)
            elif key.startswith("cam_head"):
                key = key.replace("cam_head", "encoder.camera_head", 1)

            key = key.replace("reg_token", "register_token")
            key = key.replace("cam_token", "camera_token")
            key = key.replace("refine_net", "trunk")
            key = key.replace("init_token", "empty_pose_tokens")
            key = key.replace("out_norm", "trunk_norm")
            key = key.replace("param_predictor", "pose_branch")
            key = key.replace("adapt_norm_gen", "poseLN_modulation")
            key = key.replace("param_embed", "embed_pose")
            return key
        if ckpt is not None:
            new_ckpt = {}
            for old_k, v in ckpt.items():
                new_k = rename_key(old_k)
                new_ckpt[new_k] = v
             
            ckpt = new_ckpt
            ckpt1 = {key: value for key, value in ckpt.items() if 'gs_head' in key}
            model.load_state_dict(ckpt1, strict=False)
            ckpt2 = {key: value for key, value in ckpt.items() if 'gaussian_param_head' in key}
            model.load_state_dict(ckpt2, strict=False)
                
    else:
        if cfg.checkpointing.load is None:
            raise ValueError("checkpointing.load must be set when mode=test.")
        test_checkpoint = cfg.checkpointing.load
        print(f"Loading test checkpoint from {test_checkpoint}")
        if str(test_checkpoint).endswith(".safetensors"):
            ckpt = load_file(test_checkpoint)
        else:
            ckpt = torch.load(test_checkpoint, map_location='cpu')
        ckpt = ckpt.get("state_dict", ckpt)
        ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
        ckpt = {key: value for key, value in ckpt.items() if 'gaussian_param_head' in key or 'gs_head' in key or 'cam_dec' in key or 'depth_refiner' in key or 'gs_residual_refiner' in key}

        model_state = model.state_dict()
        compatible_ckpt = {}
        skipped_keys = []
        for key, value in ckpt.items():
            remap_key = {
                "gs_residual_refiner.net.0.weight": "gs_residual_refiner.evidence_encoder.0.weight",
                "gs_residual_refiner.net.0.bias": "gs_residual_refiner.evidence_encoder.0.bias",
                "gs_residual_refiner.net.2.weight": "gs_residual_refiner.evidence_encoder.3.weight",
                "gs_residual_refiner.net.2.bias": "gs_residual_refiner.evidence_encoder.3.bias",
            }.get(key)
            if key not in model_state and remap_key in model_state:
                key = remap_key

            if key not in model_state:
                skipped_keys.append(key)
                continue
            if value.shape == model_state[key].shape:
                compatible_ckpt[key] = value
                continue

            # Older GS refiners predicted opacity/scale/SH/gate only. The current
            # head prepends mean/quat residuals, so copy old channels into the
            # matching tail and keep new geometry channels zero-initialized.
            remap_key = {
                "gs_residual_refiner.net.6.weight": "gs_residual_refiner.net.4.weight",
                "gs_residual_refiner.net.6.bias": "gs_residual_refiner.net.4.bias",
            }.get(key)
            if remap_key in model_state:
                target = model_state[remap_key]
                if (
                    value.shape[0] + 7 == target.shape[0]
                    and value.shape[1:] == target.shape[1:]
                ):
                    expanded = target.clone()
                    expanded.zero_()
                    expanded[7:] = value
                    compatible_ckpt[remap_key] = expanded
                    continue

            if (
                key in ("gs_residual_refiner.net.4.weight", "gs_residual_refiner.net.4.bias")
                and value.shape[0] + 7 == model_state[key].shape[0]
                and value.shape[1:] == model_state[key].shape[1:]
            ):
                expanded = model_state[key].clone()
                expanded.zero_()
                expanded[7:] = value
                compatible_ckpt[key] = expanded
                continue

            skipped_keys.append(key)

        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} incompatible checkpoint keys.")
        model.load_state_dict(compatible_ckpt, strict=False)
    
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        model,
        get_losses(cfg.loss),
        step_tracker
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    
    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=cfg.checkpointing.load,
        )
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
        )


if __name__ == "__main__":
    train()
