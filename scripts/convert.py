#!/usr/bin/env -S uv run --script

import argparse
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from converters import sam, birefnet, migan, esrgan

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert model weights (.pt/.pth/.safetensors) to GGUF format.")
    parser.add_argument("arch", choices=["sam", "birefnet", "migan", "esrgan"], help="Model architecture")
    parser.add_argument("input", type=str, help="Path to the input model file")
    parser.add_argument("--output", "-o", type=str, default="models", help="Path to the output directory or file")
    parser.add_argument("--quantize", "-q", choices=["f16"], default=None, help="Convert float weights to the specified data type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    # fmt: on
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    quant_suffix = f"-{args.quantize.upper()}" if args.quantize else ""
    if output_path.is_dir() or output_path.suffix != ".gguf":
        output_path = output_path / f"{input_path.stem}{quant_suffix}.gguf"

    print(f"Converting {args.arch}")
    print("* input: ", input_path)
    print("* output:", output_path)

    try:
        match args.arch:
            case "sam":
                sam.convert(input_path, output_path, args.quantize, args.verbose)
            case "birefnet":
                birefnet.convert(input_path, output_path, args.quantize, args.verbose)
            case "migan":
                migan.convert(input_path, output_path, args.quantize, args.verbose)
            case "esrgan":
                esrgan.convert(input_path, output_path, args.quantize, args.verbose)
            case _:
                raise ValueError(f"Unknown architecture: {args.arch}")
    except ValueError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        exit(-1)

    print("")
