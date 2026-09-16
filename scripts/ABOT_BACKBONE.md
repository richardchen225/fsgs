# ABot backbone option

The DL3DV experiment (`+experiment=dl3dv`) defaults to
`model.encoder.reconstruction_backbone=abot`. The shared encoder default remains
`zipmap` for other experiments. `abot` replaces only the frozen feature backbone
and camera prediction. The
existing trainable DPT (`gaussian_param_head`), `gs_head`, DA3 depth loss, MoGe
intrinsics, and GIR switches are retained. ABot's point/confidence heads are
strictly loaded with the official checkpoint, then discarded; they do not
generate our geometry.

## Installation

The optional package is pinned to commit
`edbe7e153bc2a5e35e2d4cb76fb02c90d14fade8`. Do not install a floating main branch.
On the training server, in a compatible Python 3.10+ environment:

```bash
python -m pip install --no-deps -r requirements_abot.txt
```

ABot upstream specifies PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1,
einops >=0.7, OmegaConf >=2.3, safetensors >=0.4, and huggingface-hub >=0.34,<1.
`--no-deps` avoids silently replacing this project's Torch/CUDA environment.
It does not establish compatibility with another Torch version. Use a separate
environment when testing a different Torch version, and rebuild/reinstall the
matching gsplat and torch-scatter extensions if Torch changes.

This adapter uses PyTorch SDPA, including for batch size 2. FlashInfer, cuRoPE,
loop-closure assets, and a new custom CUDA kernel are not required. The existing
GS rasterizer still has its original CUDA requirements.

Download the released checkpoint from
<https://huggingface.co/acvlab/ABot-Recon>:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('acvlab/ABot-Recon', 'abot_recon.safetensors', local_dir='weights')"
```

## Train

Start a new ABot experiment with WM initialization of the trainable GS heads:

```bash
python -m src.main +experiment=dl3dv wandb.mode=offline \
  model.encoder.reconstruction_backbone=abot \
  model.encoder.abot_weights_path=weights/abot_recon.safetensors \
  checkpointing.load=null \
  checkpointing.train_pretrained_weights=weights/pre_wm.safetensors \
  optimizer.train_base_heads=true \
  trainer.max_steps=20000 \
  'hydra.run.dir=outputs/abot_dl3dv/${now:%Y-%m-%d_%H-%M-%S}'
```

This command preserves the experiment's current GIR settings, head learning
rates, batch size, and gradient accumulation. For a backbone-only baseline,
also pass `model.encoder.gir_enabled=false optimizer.train_gir=false`.
The backbone and native camera decoder/head, including its rotation refiner,
remain frozen. WM head weights are initialization, not an ABot-trained model.

Switch back with `model.encoder.reconstruction_backbone=zipmap` and a matching
ZipMap checkpoint. A true resume must use `checkpointing.load=/path/to/abot.ckpt`
with the same backbone and `abot_feature_layers`; Lightning restores optimizer,
scheduler, and step state normally. Do not resume a ZipMap run as an ABot run.

## Features and view order

The four default feature pairs are zero-based blocks `(6,7)`, `(16,17)`,
`(24,25)`, `(34,35)`. Each pair concatenates 1024 local and 1024 causal temporal
channels. Four `[B,source_views,tokens,2048]` tensors feed the existing DPT.
`model.encoder.abot_feature_layers=[7,17,25,35]` lists pair ends, not ZipMap's
stage indices. Changing these after training invalidates the learned heads.

The input image tensor stays RGB [0,1] and keeps the project's existing spatial
preprocessing. ABot's ImageNet normalization is applied internally. There is no
extra resize to the upstream demo's 504x280; H and W must be multiples of 14.

Input order remains **source prefix, then target suffix**. Training keeps the
existing source-count rule; test keeps the existing explicit target-count rule.
There is no chronological re-sort, second inference pass, or Sim(3) alignment.
All views get native cumulative cameras; only source views get DPT/GS features.
Target transitions can therefore still affect target pose quality. No future
target can affect an earlier source in this causal pass.

The adapter follows official per-frame SDPA decode and KV pruning, with batch
dimension preserved. KV and camera states are local to one call and released
after it. Calls do not retain a map or cache across separate invocations. This
is the project's existing clip-based train/test interface, not a new external
streaming API. Image encoding is currently per-frame, so throughput must be
measured rather than assumed equal to ZipMap.

## Test and checks

Use the training configuration and an ABot-trained checkpoint. For example,
with the existing DL3DV test dataset configuration:

```bash
python -m src.main +experiment=dl3dv mode=test wandb.mode=offline \
  model.encoder.reconstruction_backbone=abot \
  model.encoder.abot_weights_path=weights/abot_recon.safetensors \
  checkpointing.load=/path/to/abot_run.ckpt
```

The train-then-benchmark script copies the backbone selection into its generated
test configuration. Checkpoints record backbone and feature-layer metadata;
test/resume rejects incompatible selections. The external ABot checkpoint is
still required at model construction, as ZipMap's pretrained file is today.

Run the optional real-network CUDA checks before a full training run:

```bash
ABOT_RECON_CHECKPOINT=weights/abot_recon.safetensors \
  python -m unittest discover -s tests -p test_abot_backbone.py -v
```

These checks compare the adapter with official camera-only SDPA inference,
source-only vs source+target prefixes, batch 2 vs independent scenes, and
downstream head backward. Without the checkpoint/CUDA, only CPU contract tests
run and the real-network check is explicitly skipped. Follow this with a short
training/validation run on the actual data and then a multi-GPU run; CPU tests
do not validate gsplat, full DPT training, or DDP on the server.
