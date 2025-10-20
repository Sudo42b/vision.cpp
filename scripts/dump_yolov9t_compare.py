import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Reuse the custom model and weight mapping utilities
from yolov9 import YOLOv9t_Seq
import sys, subprocess


def read_cwhn_txt_tensor(path: str) -> torch.Tensor:
    """Read a text tensor saved by C++ save_input_to_txt (C,H,W,N layout)."""
    with open(path, "r") as f:
        header1 = f.readline().strip()
        if not header1.startswith("# input shape") and not header1.startswith("# shape"):
            # The file may start directly with numbers
            # rewind to beginning
            f.seek(0)
            data = np.fromstring(f.read().strip().replace("\n", " "), sep=" ")
            return torch.from_numpy(data.astype(np.float32))

        # Parse shape line: e.g. "# input shape C,H,W,N = 3,640,640,1"
        m = re.search(r"C,H,W,N\s*=\s*([0-9]+),([0-9]+),([0-9]+),([0-9]+)", header1)
        if not m:
            raise ValueError(f"Failed to parse shape header in {path}: '{header1}'")
        C, H, W, N = map(int, m.groups())
        header2 = f.readline()  # type line
        # Remaining are whitespace-separated values
        data = np.fromstring(f.read().strip().replace("\n", " "), sep=" ")
        expected = C * H * W * N
        if data.size != expected:
            raise ValueError(f"Data size mismatch in {path}: {data.size} vs expected {expected}")
        x = torch.from_numpy(data.astype(np.float32)).view(C, H, W, N)
        return x


def write_cwhn_txt_tensor(t: torch.Tensor, path: str, *, comment_name: str = "input") -> None:
    """Write tensor to txt in the same format as C++ save_input_to_txt: C,H,W,N with header."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Ensure C,H, W,N
    if t.dim() == 4 and t.shape[-1] in (1,):
        cwhn = t
    elif t.dim() == 4:  # likely NCHW
        # N,C,H,W -> C,H,W,N
        cwhn = t.permute(1, 2, 3, 0).contiguous()
    elif t.dim() == 3:  # C,H,W -> C,H,W,1
        cwhn = t.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported tensor shape for writing: {tuple(t.shape)}")

    C, H, W, N = cwhn.shape
    with open(path, "w") as f:
        f.write(f"# {comment_name} shape C,H,W,N = {C},{H},{W},{N}\n")
        f.write("# type = f32\n")
        flat = cwhn.reshape(-1).cpu().numpy()
        # Write space-separated with newline at end
        for i, v in enumerate(flat):
            end = "\n" if i + 1 == flat.size else " "
            f.write(("%g" % float(v)) + end)


# Add: build runner reflecting build_code.sh


def run_build(build_script: str = "/mnt/e/7_RISCV/vision.cpp/build_code.sh") -> None:
    if not os.path.exists(build_script):
        raise FileNotFoundError(f"Build script not found: {build_script}")
    print(f"[build] Running: {build_script}")
    subprocess.run(["bash", build_script], check=True)


# Add: C++ vision-cli runner reflecting excute.sh


def run_cpp_vision_cli(
    cpp_bin: str,
    image_path: str,
    model_path: str,
    output_path: str,
    dump_all: bool,
    dump_keys: List[int] = None,
    extra_args: str = "",
) -> None:
    if not os.path.exists(cpp_bin):
        raise FileNotFoundError(f"vision-cli not found: {cpp_bin}")
    cmd = [
        cpp_bin,
        "yolov9t",
        "-m",
        model_path,
        "-i",
        image_path,
        "-o",
        output_path,
    ]
    if dump_all or not dump_keys:
        cmd.append("--dump-all")
    else:
        cmd.append("--dump-keys")
        cmd += [str(k) for k in (dump_keys or [])]
    if extra_args:
        cmd += extra_args.split()
    repo_root = "/mnt/e/7_RISCV/vision.cpp"
    print("[cpp] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)


def preprocess_like_cpp(img_path: str, input_size: int = 640) -> torch.Tensor:
    """Replicate the C++ yolov9t_process_input: scale longest side to input_size, paste top-left, to [0,1].

    Returns CHW float32 in [0,1].
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    s = float(input_size) / float(max(H, W))
    new_w, new_h = int(round(W * s)), int(round(H * s))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(img_resized, (0, 0))  # top-left paste to match C++ behavior
    arr = np.asarray(canvas).astype(np.float32) / 255.0  # H,W,C
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # C,H,W
    return chw


@torch.no_grad()
def run_and_dump(
    img_path: str,
    weight_path: str,
    out_path: str,
    dump_all: bool,
    dump_keys: List[int],
    use_cpp_input_txt: str = "",
) -> None:
    model = YOLOv9t_Seq(ch=3, nc=80)
    # Load Ultralytics weights with detect head mapping
    try:
        model.load_ultralytics_weights_fixed(weight_path)
    except Exception:
        # fallback
        model.load_ultralytics_weights(weight_path)
    model.eval()

    base = os.path.splitext(out_path)[0]

    # Prepare input
    if use_cpp_input_txt:
        x_cwhn = read_cwhn_txt_tensor(use_cpp_input_txt)  # C,H,W,N
        if x_cwhn.dim() != 4 or x_cwhn.shape[-1] != 1:
            raise ValueError("C++ input txt must be C,H,W,N with N=1")
        x = x_cwhn.permute(3, 0, 1, 2).contiguous()  # N,C,H,W
        # Save back what we read (for symmetry)
        write_cwhn_txt_tensor(x_cwhn, base + "_input.txt", comment_name="input")
    else:
        chw = preprocess_like_cpp(img_path, input_size=640)
        x = chw.unsqueeze(0)  # N,C,H,W
        write_cwhn_txt_tensor(
            x.permute(1, 2, 3, 0).contiguous(), base + "_input.txt", comment_name="input"
        )

    # Forward
    inf_out, raw_outs = model(x)

    # Dump features along the way by running the layers manually similar to forward, or re-run with hooks
    # We'll register forward hooks to capture layer outputs matching indices used in C++
    save_indices = [3, 6, 9, 15, 18, 21]
    features: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(module, inp, out):
            # ensure C,H,W,N layout for saving
            if isinstance(out, (list, tuple)):
                t = out[0]
            else:
                t = out
            if t.dim() == 4:  # N,C,H,W
                write_cwhn_txt_tensor(
                    t.permute(1, 2, 3, 0).contiguous(),
                    f"{base}_features_layer_{idx}.txt",
                    comment_name=f"layer_{idx}",
                )
                features[idx] = t.detach().cpu()

        return hook

    hooks = []
    try:
        for idx in save_indices:
            if idx < len(model.model):
                h = model.model[idx].register_forward_hook(make_hook(idx))
                hooks.append(h)

        # Re-run a forward pass to trigger hooks
        _ = model(x)
    finally:
        for h in hooks:
            h.remove()

    # Dump raw outputs per scale similar to C++ flat [C, H*W, 1, 1] -> treat as [C, num_anchors, 1, 1]
    for i, t in enumerate(raw_outs):
        if t.dim() == 3:  # [C, H*W, N]? Some implementations flatten already
            C, HW, N = t.shape
            cwhn = t.view(C, HW, 1).unsqueeze(-1)  # C, HW, 1, 1
        elif t.dim() == 4:  # [N, C, H, W]
            N, C, H, W = t.shape
            cwhn = t.permute(1, 2, 3, 0).contiguous()  # C,H,W,N
        else:
            # Fallback flatten
            flat = t.reshape(t.shape[0], -1, 1, 1) if t.dim() >= 2 else t.reshape(-1, 1, 1, 1)
            cwhn = flat
        write_cwhn_txt_tensor(cwhn, f"{base}_features_raw_{i}.txt", comment_name=f"raw_{i}")

    # Dump predictions (post-DFL + cls depending on Detect)
    if isinstance(inf_out, torch.Tensor):
        # Try to reshape like C++: [C, num_anchors, 1, 1]
        if inf_out.dim() == 3:  # [N, num_anchors, C]
            N, A, C = inf_out.shape
            pred = inf_out.permute(2, 1, 0).contiguous().view(C, A, 1, 1)
        elif inf_out.dim() == 2:  # [num_anchors, C]
            A, C = inf_out.shape
            pred = inf_out.t().contiguous().view(C, A, 1, 1)
        elif inf_out.dim() == 4 and inf_out.shape[0] == 1:  # [1, C, H, W]
            pred = inf_out.squeeze(0).permute(0, 1, 2).unsqueeze(-1)  # C,H,W,1
        else:
            # Fallback: flatten to [C, A, 1, 1]
            flat = (
                inf_out.reshape(inf_out.shape[-1], -1)
                if inf_out.dim() > 1
                else inf_out.reshape(1, -1)
            )
            pred = flat.t().contiguous().view(flat.shape[1], flat.shape[0], 1, 1)
        write_cwhn_txt_tensor(pred, f"{base}_features_predictions.txt", comment_name="predictions")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dump YOLOv9t PyTorch features to compare with C++")
    ap.add_argument("-i", "--input", required=True, help="Input image path")
    ap.add_argument("-w", "--weights", required=True, help="Ultralytics yolov9t.pt path")
    ap.add_argument(
        "-o", "--output", required=True, help="Output image path (base name used for dumps)"
    )
    ap.add_argument(
        "--dump-all", action="store_true", help="Dump all predefined feature layers and outputs"
    )
    ap.add_argument(
        "--dump-keys",
        type=int,
        nargs="*",
        default=[],
        help="Specific feature layer indices to dump",
    )
    ap.add_argument(
        "--use-cpp-input",
        default="",
        help="Use preprocessed input txt from C++ (path to *_input.txt)",
    )
    # Added flags to mirror shell scripts
    ap.add_argument("--build-cpp", action="store_true", help="Build C++ project via build_code.sh")
    ap.add_argument(
        "--run-cpp", action="store_true", help="Run C++ vision-cli (excute.sh-like) before Python"
    )
    ap.add_argument(
        "--cpp-bin",
        default="/mnt/e/7_RISCV/vision.cpp/build/bin/vision-cli",
        help="Path to vision-cli binary",
    )
    ap.add_argument(
        "--cpp-model", default="models/yolov9t_converted.gguf", help="GGUF model path for C++ run"
    )
    ap.add_argument(
        "--cpp-output", default=None, help="Output path for C++ run; defaults to --output"
    )
    ap.add_argument(
        "--cpp-extra", default="", help="Extra args for vision-cli (e.g., --backend gpu)"
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    # Optionally build C++
    if getattr(args, "build_cpp", False):
        run_build()
    # Optionally run C++ to generate dumps and reuse its input
    use_cpp_input = args.use_cpp_input
    if getattr(args, "run_cpp", False):
        cpp_out = args.cpp_output or args.output
        run_cpp_vision_cli(
            cpp_bin=args.cpp_bin,
            image_path=args.input,
            model_path=args.cpp_model,
            output_path=cpp_out,
            dump_all=args.dump_all or (not args.dump_keys),
            dump_keys=args.dump_keys if args.dump_keys else None,
            extra_args=args.cpp_extra,
        )
        base_cpp = os.path.splitext(cpp_out)[0]
        use_cpp_input = base_cpp + "_input.txt"
    # Proceed with Python dump side
    run_and_dump(
        img_path=args.input,
        weight_path=args.weights,
        out_path=args.output,
        dump_all=args.dump_all,
        dump_keys=args.dump_keys,
        use_cpp_input_txt=use_cpp_input,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
