#!/usr/bin/env python3
"""
Export CorridorKey GreenFormer to ONNX format.

Usage (run from the CorridorKey repo root):
    uv run python export_corridorkey_onnx.py

Or with explicit paths:
    uv run python export_corridorkey_onnx.py \
        --checkpoint weights/CorridorKey_v1.0.pth \
        --output weights/CorridorKey_v1.0.onnx \
        --img-size 2048 \
        --opset 17

Notes:
    - Exports at fixed 2048x2048 (native training resolution).
    - The pos_embed is baked at this size, so the ONNX model is fixed-resolution.
    - Input:  "input"        float32[1, 4, 2048, 2048]  (ImageNet-normed RGB + raw alpha hint)
    - Output: "alpha"        float32[1, 1, 2048, 2048]  (after sigmoid, linear alpha)
    - Output: "fg"           float32[1, 3, 2048, 2048]  (after sigmoid, sRGB straight FG)
    - Memory: ~16-20 GB RAM needed for tracing at 2048x2048. Use CPU.

Author: Peter Mercell (petermercell.com)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wrapper: dict output -> named tuple outputs for clean ONNX graph
# ---------------------------------------------------------------------------
class GreenFormerONNX(nn.Module):
    """Thin wrapper that converts dict output to (alpha, fg) tuple for ONNX."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["alpha"], out["fg"]


# ---------------------------------------------------------------------------
# Checkpoint loading (adapted from inference_engine.py)
# ---------------------------------------------------------------------------
def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load CorridorKey checkpoint, handling _orig_mod. prefix and pos_embed resize."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)

    new_state_dict = {}
    model_state = model.state_dict()

    for k, v in state_dict.items():
        # Strip torch.compile prefix
        if k.startswith("_orig_mod."):
            k = k[10:]

        # Handle pos_embed shape mismatch (bicubic interpolation)
        if "pos_embed" in k and k in model_state:
            if v.shape != model_state[k].shape:
                print(f"  Resizing {k}: {list(v.shape)} -> {list(model_state[k].shape)}")
                N_src = v.shape[1]
                C = v.shape[2]
                grid_src = int(math.sqrt(N_src))
                N_dst = model_state[k].shape[1]
                grid_dst = int(math.sqrt(N_dst))
                v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                v_resized = F.interpolate(
                    v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False
                )
                v = v_resized.flatten(2).transpose(1, 2)

        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [Warning] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    else:
        print("  All keys matched successfully.")

    return model


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
def export_onnx(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 2048,
    opset_version: int = 17,
    verify: bool = True,
) -> None:
    device = "cpu"  # CPU for export — avoids GPU memory issues during tracing

    # 1. Build model
    print(f"\n{'='*60}")
    print(f"CorridorKey ONNX Export")
    print(f"{'='*60}")
    print(f"  Resolution:  {img_size}x{img_size}")
    print(f"  Opset:       {opset_version}")
    print(f"  Output:      {output_path}")
    print()

    # Import from the repo (must run from CorridorKey root)
    from CorridorKeyModule.core.model_transformer import GreenFormer

    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=img_size,
        use_refiner=True,
    )
    model = model.to(device)
    model.eval()

    # 2. Load weights
    model = load_checkpoint(model, checkpoint_path, device)

    # 3. Wrap for ONNX (dict -> tuple)
    wrapper = GreenFormerONNX(model)
    wrapper.eval()

    # 4. Create dummy input
    print(f"\nCreating dummy input: [1, 4, {img_size}, {img_size}]...")
    dummy_input = torch.randn(1, 4, img_size, img_size, device=device)

    # 5. Test forward pass
    print("Running test forward pass...")
    with torch.no_grad():
        alpha_out, fg_out = wrapper(dummy_input)
    print(f"  alpha shape: {list(alpha_out.shape)}, range: [{alpha_out.min():.4f}, {alpha_out.max():.4f}]")
    print(f"  fg shape:    {list(fg_out.shape)}, range: [{fg_out.min():.4f}, {fg_out.max():.4f}]")

    # 6. Export to ONNX
    print(f"\nExporting to ONNX (this may take a few minutes at {img_size}x{img_size})...")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        opset_version=opset_version,
        dynamo=False,  # Legacy exporter — more reliable for vision transformers
        input_names=["input"],
        output_names=["alpha", "fg"],
        # Fixed resolution — pos_embed is baked, no dynamic axes
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")

    # 7. Verify with ONNX checker
    print("\nVerifying ONNX model...")
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: OK")

    # Print input/output info (useful for Netron / TRT)
    print("\n  ONNX Graph I/O:")
    for inp in onnx_model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    Input:  {inp.name:20s} {dims}")
    for out in onnx_model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    Output: {out.name:20s} {dims}")

    # 8. Optional: verify with ONNX Runtime
    if verify:
        print("\nVerifying with ONNX Runtime...")
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {"input": dummy_input.numpy()}
        ort_alpha, ort_fg = sess.run(None, ort_inputs)

        # Compare PyTorch vs ORT
        alpha_diff = np.abs(alpha_out.numpy() - ort_alpha).max()
        fg_diff = np.abs(fg_out.numpy() - ort_fg).max()
        print(f"  Max abs diff (alpha): {alpha_diff:.6f}")
        print(f"  Max abs diff (fg):    {fg_diff:.6f}")

        if alpha_diff < 1e-3 and fg_diff < 1e-3:
            print("  Verification: PASSED")
        else:
            print("  Verification: WARNING — diffs larger than expected (may be float precision)")

    print(f"\n{'='*60}")
    print(f"Export complete! Next steps:")
    print(f"  1. Inspect in Netron:  https://netron.app/")
    print(f"  2. Build TRT engine:")
    print(f"     trtexec --onnx={output_path} \\")
    print(f"       --saveEngine={output_path.replace('.onnx', '_fp32.engine')} \\")
    print(f"       --memPoolSize=workspace:4G")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CorridorKey to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="weights/CorridorKey_v1.0.pth",
        help="Path to .pth checkpoint",
    )
    parser.add_argument(
        "--output",
        default="weights/CorridorKey_v1.0.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=2048,
        help="Model input resolution (default: 2048)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX Runtime verification",
    )
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=args.img_size,
        opset_version=args.opset,
        verify=not args.no_verify,
    )
