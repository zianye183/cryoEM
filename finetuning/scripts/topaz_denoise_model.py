#!/usr/bin/env python3
"""Denoise micrographs with an explicit Topaz denoising model path or alias.

This wrapper avoids a Topaz CLI inference quirk in this installed version where
`topaz denoise --model /path/to/checkpoint.sav` prints the checkpoint path but
constructs `Denoise(args.arch)` internally, which reloads the pretrained `unet`
model instead of the requested checkpoint. The wrapper calls Topaz's Python API
with `Denoise(args.model, ...)` directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import topaz.denoise as dn
from topaz.cuda import set_device
from topaz.denoise import Denoise, denoise_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("micrographs", nargs="+", help="Input micrographs to denoise.")
    parser.add_argument(
        "--model", required=True, help="Topaz model alias or checkpoint path."
    )
    parser.add_argument("--device", default="0", help="CUDA device id, or -1 for CPU.")
    parser.add_argument(
        "--output", required=True, help="Directory to save denoised micrographs."
    )
    parser.add_argument("--patch-size", type=int, default=4000)
    parser.add_argument("--patch-padding", type=int, default=500)
    parser.add_argument("--format", dest="format_", default="mrc")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--lowpass", type=float, default=1)
    parser.add_argument("--pixel-cutoff", type=float, default=0)
    parser.add_argument("--gaussian", type=float, default=0)
    parser.add_argument("--inv-gaussian", type=float, default=0)
    parser.add_argument("--deconvolve", action="store_true")
    parser.add_argument("--deconv-patch", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = set_device(int(args.device))
    print(
        f"# using device={args.device} with cuda={use_cuda}",
        file=sys.stderr,
        flush=True,
    )
    print(f"# current checkpoint: {args.model}", file=sys.stderr, flush=True)

    denoiser = Denoise(args.model, use_cuda=use_cuda)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gaus = (
        dn.GaussianDenoise(args.gaussian, use_cuda=use_cuda)
        if args.gaussian > 0
        else None
    )
    inv_gaus = (
        dn.InvGaussianFilter(args.inv_gaussian, use_cuda=use_cuda)
        if args.inv_gaussian > 0
        else None
    )

    denoise_stream(
        args.micrographs,
        str(output_dir),
        format=args.format_,
        suffix=args.suffix,
        models=[denoiser],
        lowpass=args.lowpass,
        pixel_cutoff=args.pixel_cutoff,
        gaus=gaus,
        inv_gaus=inv_gaus,
        deconvolve=args.deconvolve,
        deconv_patch=args.deconv_patch,
        patch_size=args.patch_size,
        padding=args.patch_padding,
        normalize=args.normalize,
        use_cuda=use_cuda,
        return_images=False,
    )


if __name__ == "__main__":
    main()
