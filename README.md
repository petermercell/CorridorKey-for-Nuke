# CorridorKey for Nuke

A TensorRT-accelerated Nuke plugin for [CorridorKey](https://github.com/nikopueringer/CorridorKey) — the AI green screen keyer by Niko Pueringer / Corridor Digital that produces physically accurate unmixed foreground color and alpha.

This plugin runs the CorridorKey neural network directly inside Nuke as a native node (`TRTCorridorKey`), powered by NVIDIA TensorRT for real-time inference on the GPU.

## What It Does

CorridorKey takes a green screen plate and a coarse alpha hint, and produces a clean straight foreground color and linear alpha matte — preserving hair, motion blur, and semi-transparent edges that traditional keyers destroy.

This plugin wraps that model into a two-input Nuke node:

- **Input 0 (plate):** RGB green screen footage
- **Input 1 (mask):** Coarse alpha hint (rough chroma key or AI roto)
- **Output:** RGBA — unmixed FG color (sRGB) in RGB + linear alpha in A

The TensorRT engine runs at ~300ms per frame at 2048×2048 FP16 on an RTX A5000 (24 GB).

## Requirements

- Linux x86_64
- NVIDIA GPU with 24+ GB VRAM (tested on RTX A5000)
- CUDA 13.1 (or 12.9)
- [TensorRT 10.15.1](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-13.1.tar.gz)
- Nuke 17.0 (NDK)
- ~128 GB+ system RAM for the ONNX export step

## Build Guide

The full pipeline is: `.pth` → ONNX → TensorRT engine → Nuke plugin. Each step only needs to be done once.

### Step 1: Export CorridorKey to ONNX

Clone the original CorridorKey repo and set up its environment:

```bash
git clone https://github.com/nikopueringer/CorridorKey.git
cd CorridorKey
uv sync
uv pip install onnx onnxruntime onnxscript
```

Download the model weights (~300 MB) into the `weights/` folder:

```bash
mkdir -p weights
wget -O weights/CorridorKey_v1.0.pth \
    https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth
```

Copy `export_corridorkey_onnx.py` from this repo into the CorridorKey directory, then export:

```bash
source .venv/bin/activate
uv run python export_corridorkey_onnx.py --img-size 2048 --no-verify
```

> **RAM note:** The ONNX trace at 2048×2048 requires ~140 GB of memory. If you have 128 GB RAM, add temporary swap:
> ```bash
> sudo fallocate -l 64G /swapfile
> sudo chmod 600 /swapfile
> sudo mkswap /swapfile
> sudo swapon /swapfile
> sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
> uv run python export_corridorkey_onnx.py --img-size 2048 --no-verify
> sudo swapoff /swapfile
> sudo rm /swapfile
> ```
>
> Alternatively, export at 1024 (`--img-size 1024`) which needs much less RAM. The model still works well — the Nuke plugin resizes the input to model resolution anyway.

This produces `weights/CorridorKey_v1.0.onnx`.

### Step 2: Build the TensorRT Engine

Download the pre-built [TensorRT 10.15.1](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-13.1.tar.gz) and extract it to `/opt/TensorRT-10.15.1.29` (or wherever you prefer).

```bash
export LD_LIBRARY_PATH=/opt/TensorRT-10.15.1.29/lib:/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

/opt/TensorRT-10.15.1.29/bin/trtexec \
    --onnx=weights/CorridorKey_v1.0.onnx \
    --saveEngine=weights/CorridorKey_v1.0_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:2G
```

FP16 is recommended — CorridorKey already uses float16 autocast during inference. This takes about 5 minutes and produces a ~198 MB engine file.

### Step 3: Build the Nuke Plugin

```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

rm -rf build && mkdir build && cd build
cmake .. \
    -DNUKE_ROOT=/opt/Nuke17.0v1 \
    -DTENSORRT_ROOT=/opt/TensorRT-10.15.1.29
make -j8
```

Copy `TRTCorridorKey.so` to your Nuke plugin path (e.g. `~/.nuke/`).

## Usage in Nuke

1. Create a `TRTCorridorKey` node (found under AI menu)
2. Connect your green screen plate to the **plate** input
3. Connect a coarse alpha hint to the **mask** input (a rough Keylight/Primatte key or AI roto works shuffle to Alpha to RGBA)
4. Point the **Engine File** knob to your `.engine` file
5. Set **Resolution** to match your engine (2048×2048 or 1024×1024)

### Knobs

| Knob | Description |
|------|-------------|
| Engine File | Path to the TensorRT `.engine` file |
| Resolution | Must match the ONNX export resolution |
| Output | RGBA (FG + Alpha), Alpha Only, or FG Only |
| Invert Matte | Inverts the predicted alpha |
| GPU Device | GPU index for multi-GPU systems |

### Output Details

- **RGB channels:** Straight (unpremultiplied) foreground color in sRGB
- **Alpha channel:** Linear alpha matte

To composite properly in Nuke, convert the FG from sRGB to linear, then premultiply:

```
TRTCorridorKey → Colorspace (sRGB to linear) → Premult → Merge (over)
```

## Model Details

The CorridorKey engine uses:

- **Input:** `[1, 4, 2048, 2048]` — ImageNet-normalized RGB concatenated with the raw alpha hint
- **Outputs:** `alpha` `[1, 1, 2048, 2048]` and `fg` `[1, 3, 2048, 2048]`, both post-sigmoid
- **Architecture:** Hiera Base Plus backbone (from timm) with dual decoder heads and a CNN refiner
- **Native resolution:** 2048×2048 (positional embeddings are baked at export resolution)

## Files in This Repo

```
├── TRTCorridorKey.cpp          # Nuke plugin source
├── CMakeLists.txt              # Build configuration
├── export_corridorkey_onnx.py  # PyTorch → ONNX export script
├── test_trt_engine.py          # Standalone TRT engine test
├── LICENSE                     # License
└── README.md                   # This file
```

## Credits

- **CorridorKey model** by [Niko Pueringer / Corridor Digital](https://github.com/nikopueringer/CorridorKey) — CC BY-NC-SA 4.0
- **TRTCorridorKey Nuke plugin** by [Peter Mercell](https://petermercell.com)
- Built with [TensorRT](https://developer.nvidia.com/tensorrt) and the [Nuke NDK](https://learn.foundry.com/nuke/developers/latest/)

## License

This plugin code is released under the MIT License. See [LICENSE](LICENSE) for details.

The CorridorKey model weights and architecture are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by Corridor Digital. You must comply with their license when using the model. See the [CorridorKey repository](https://github.com/nikopueringer/CorridorKey) for full terms.
