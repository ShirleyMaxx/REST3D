# `rest3d` Environment Setup

Tested on CUDA 12.1, Python 3.11, PyTorch 2.5.1. Used for scene reconstruction (Stage 1-2).

> Please first clone the repo and set `REST3D_ROOT` as described in the main [README](../README.md):
> ```bash
> export REST3D_ROOT="/path/to/REST3D"
> ```

**1. Create the conda environment:**
```bash
conda env create -f environments/default.yml
conda activate rest3d
```

**2. Install PyTorch + Python dependencies:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r environments/requirements_rest3d.txt
```

**3. Install SAM 3:**

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3 Hugging Face [repo](https://huggingface.co/facebook/sam3).
We follow the official [SAM 3](https://github.com/facebookresearch/sam3#installation) setup for installation and checkpoints, with two differences: we pin to the commit we tested on (`5dd401d`), and apply a small patch.
```bash
cd "$REST3D_ROOT/.."
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 5dd401d                                            # the commit we developed on
git apply "$REST3D_ROOT/third_party/sam3_patch/sam3_agent.patch"
pip install -e . && pip install decord psutil
```

**4. Install SAM 3D Objects:**

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects).
We follow the official [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) setup for installation and checkpoints, with two differences: we pin to the commit we tested on (`e19b169`), and apply a small patch.
```bash
cd "$REST3D_ROOT/.."
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
git checkout e19b169                                             # the commit we developed on
git apply "$REST3D_ROOT/third_party/sam3d_objects_init.patch"
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'        # 2-step: pytorch3d's torch constraint is otherwise unsolvable
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

export SAM3D_OBJECTS_ROOT="$(pwd)"
```

Download the SAM 3D Objects inference checkpoints into `$SAM3D_OBJECTS_ROOT/checkpoints/hf/` following the [official guide](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md#2-getting-checkpoints).

<details>
<summary><b>Note:</b> <code>pip's dependency resolver</code> conflict warnings during these installs are safe to ignore.</summary>

pip may report version conflicts during the SAM 3 / SAM-3D-Objects installs — the
exact packages and versions vary, e.g.:

```
sam3 0.1.0 requires ftfy==6.1.1, but you have ftfy 6.2.0 which is incompatible.
sam3 0.1.0 requires timm>=1.0.17, but you have timm 0.9.16 which is incompatible.
hf-gradio 0.4.1 requires gradio-client<3.0,>=2.0, but you have gradio-client 1.13.3 which is incompatible.
sam3d-objects 0.0.1 requires ftfy==6.2.0, but you have ftfy 6.1.1 which is incompatible.
sam3d-objects 0.0.1 requires timm==0.9.16, but you have timm 1.0.27 which is incompatible.
```
</details>

**5. Install REST3D:**
```bash
cd "$REST3D_ROOT"
pip install -e ".[gemini]"     # default Gemini backend
```

**6. Verify:**

This should import everything without error and print something like `torch 2.5.1+cu121 cuda True`.
```bash
python -c "import torch, sam3, sam3d_objects, pytorch3d, kaolin, rest3d; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
