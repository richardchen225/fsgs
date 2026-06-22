# FSGS

这是一个用于训练和评估 FSGS 的项目。下面按新机器从零开始整理：装环境、下载 DL3DV、准备权重、启动训练。

## 1. 环境安装

建议使用 Python 3.10 和 CUDA 13.0：

```bash
conda create -n fsgs python=3.10 -y
conda activate fsgs
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

如果 `torch-scatter` 这类包安装失败，通常是本机 CUDA、PyTorch wheel 或编译器版本没有对齐。先确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

需要实验日志同步到 Weights & Biases，登录：

```bash
export WANDB_API_KEY="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"
wandb login --relogin "$WANDB_API_KEY"
```

## 2. 下载 DL3DV-ALL-480P

本项目训练默认读取：

```text
datasets/dl3dv
```

DL3DV 480P 数据集页面：

```text
https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P
```

先在 Hugging Face 页面申请访问权限；如果需要登录，执行：

```bash
huggingface-cli login
```

本仓库脚本会下载 `DL3DV/DL3DV-ALL-480P` 的 `images+poses`，并依次下载 `1K` 到 `11K`。下载路径必须通过参数指定：

```bash
bash /home/9/ug04729/tanyixin/fsgs/scripts/download_dl3dv.sh /gs/bs/tga-mdl/tanyixin-mdl/dataset/dl3dv
```

你也可以换成任意本机路径：

```bash
bash scripts/download_dl3dv.sh /data/DL3DV-ALL-480P
```

脚本内部调用 DL3DV 官方 `scripts/download.py`。

DL3DV 官方脚本会在下载 zip 后自动解压到输出目录，并删除 zip 文件。本仓库脚本会自动循环 `1K 2K 3K 4K 5K 6K 7K 8K 9K 10K 11K`，下载完成后生成：

```text
datasets/dl3dv/train_index.json
datasets/dl3dv/test_index.json
```

当前 `train_index.json` 里的每个训练条目会写成 `1K/scene_name` 这种形式。解压后的目录通常类似：

```text
datasets/dl3dv/
  train_index.json
  1K/
    <scene_name>/
      <scene_data_dir>/
        images_8/
        transforms.json
```

## 3. 下载预训练权重

权重建议统一放到：

```text
weights/
```

直接运行：

```bash
bash scripts/download_weights.sh
```

脚本会尝试下载这些文件：

```text
weights/pre_wm.safetensors
weights/pre_zipmap.pt
weights/model.pt
weights/pre_dav3.safetensors
```

## 4. 检查训练配置

数据路径有两处需要保持一致：

- `config/dataset/dl3dv.yaml`
- `config/experiment/dl3dv.yaml`

如果你按上面的默认路径下载，两处都保持 `datasets/dl3dv`：

```yaml
# config/dataset/dl3dv.yaml
roots: [datasets/dl3dv]
```

```yaml
# config/experiment/dl3dv.yaml
dataset:
  dl3dv:
    roots: [datasets/dl3dv]
```

如果你下载到了别的目录，比如 `/data/DL3DV-ALL-480P`，两处同步改成：

```yaml
# config/dataset/dl3dv.yaml
roots: [/data/DL3DV-ALL-480P]
```

```yaml
# config/experiment/dl3dv.yaml
dataset:
  dl3dv:
    roots: [/data/DL3DV-ALL-480P]
```

本地训练时通常还需要确认这些权重路径：

```yaml
# config/experiment/dl3dv.yaml
model:
  encoder:
    moge_weights_path: weights/model.pt
    zipmap_weights_path: weights/pre_zipmap.pt

checkpointing:
  train_pretrained_weights: weights/pre_wm.safetensors

loss:
  depth:
    dav3_weights_path: weights/pre_dav3.safetensors
```
这几个默认值已经和 `scripts/download_weights.sh` 对齐了。如果下载到别的位置，需要修改这几个路径配置。

## 5. 开始训练

单进程启动：

```bash
python -m src.main +experiment=dl3dv
```

默认 `config/experiment/dl3dv.yaml` 使用 4 张 GPU：

```yaml
trainer:
  devices: 4
```

训练启动后会在当前项目目录下创建 `output/` 文件夹。`config/experiment/dl3dv.yaml` 中默认配置为：

```yaml
hydra:
  run:
    dir: output/exp_${wandb.name}/${now:%Y-%m-%d_%H-%M-%S}
```

每次实验会生成一个新的时间戳目录，Hydra 配置、wandb 本地文件、训练日志和 checkpoint 都会保存在这个实验目录下，其中 checkpoint 位于 `checkpoints/` 子目录。

训练 batch size 在 `config/experiment/dl3dv.yaml` 里配置：

```yaml
data_loader:
  train:
    batch_size: 2
```

默认已经是 `batch_size: 2`。同一个 batch 内会共享本次随机抽到的 context view 数量，因此两个场景可以被默认 collate 到同一个 batch。验证阶段如果也把 `data_loader.val.batch_size` 改成 2，日志里的 `val/psnr`、`val/lpips`、`val/ssim` 会按 batch 内所有样本一起计算，comparison 也会逐个保存；测试阶段当前仍要求 `batch_size: 1`。

多 GPU 可以用仓库里的脚本，默认 4 卡：

```bash
bash train.sh
```


## 6. 常见检查

- 找不到数据：确认下载目录下有 `train_index.json`，并且数组里的相对路径能拼到真实目录。
- 下载无权限：先在 `DL3DV/DL3DV-ALL-480P` 页面申请访问，再执行 `huggingface-cli login`。
- 想从 checkpoint 推理或续训：显式设置 `checkpointing.load=/path/to/checkpoint.ckpt`。
- 想换数据目录：同步改 `config/dataset/dl3dv.yaml` 和 `config/experiment/dl3dv.yaml` 里的 `roots`，或在命令行覆盖 `dataset.dl3dv.roots`。



