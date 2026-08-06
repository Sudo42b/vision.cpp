#!/usr/bin/env python3
"""Write the decoding configuration into the GGUF, so the runner needs no sidecar file.

The compiler that produces the `.gguf` is framework-agnostic: it turns a plain PyTorch
module into a graph and a tensor file, and it has no idea what MMDetection is. Anchor
scales and head layout come from an MMDetection config, so they cannot travel through
that path -- which is why they used to be written to `<name>.postproc.json` and passed
to the runner as a separate argument.

This step closes that gap from the frontend side: take the produced `.gguf`, add the
values as metadata keys, and write it back out. The compiler stays untouched, the
framework knowledge stays in this directory, and deployment drops from three files to
two.

Usage:
    python mmdet_inject.py <model.gguf> <name.postproc.json> [-o out.gguf]

With no -o the input file is replaced.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_GGUF_PY = _HERE.parents[3] / "depend" / "llama" / "gguf-py"
if _GGUF_PY.is_dir():
    sys.path.insert(0, str(_GGUF_PY))

from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

PREFIX = "mmdet."

# JSON field -> how to write it. Anything not listed is ignored, so an unfamiliar field
# in a newer sidecar does not silently become a key nobody reads.
SCALARS_STR = ["head_type", "cls_convs_prefix", "reg_convs_prefix", "cls_head", "reg_head"]
SCALARS_INT = ["num_classes", "num_base", "stacked_convs", "feat_channels"]
SCALARS_F32 = ["octave_base_scale", "center_offset"]
SCALARS_BOOL = ["use_sigmoid", "to_rgb", "head_has_norm"]
ARRAYS_F32 = ["strides", "octave_scales", "ratios", "means", "stds", "img_mean", "img_std"]


def inject(gguf_in: Path, cfg: dict, gguf_out: Path) -> list[str]:
    reader = GGUFReader(gguf_in, "r")

    arch = "mmdet"
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = str(bytes(f.parts[f.data[0]]), encoding="utf-8")
            break

    writer = GGUFWriter(gguf_out, arch, use_temp_file=False)

    # Carry over the existing metadata unchanged. general.architecture is written by the
    # constructor, so skip it here rather than adding it twice.
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith(PREFIX):
            continue
        try:
            writer.add_key_value(field.name, field.contents(), field.types[0])
        except Exception:  # a field shape gguf-py cannot round-trip; leave it out
            print(f"  ! skipped metadata key {field.name}")

    written = []

    def put(name, value, fn):
        fn(PREFIX + name, value)
        written.append(PREFIX + name)

    for k in SCALARS_STR:
        if k in cfg:
            put(k, str(cfg[k]), writer.add_string)
    for k in SCALARS_INT:
        if k in cfg:
            put(k, int(cfg[k]), writer.add_int32)
    for k in SCALARS_F32:
        if k in cfg:
            put(k, float(cfg[k]), writer.add_float32)
    for k in SCALARS_BOOL:
        if k in cfg:
            put(k, bool(cfg[k]), writer.add_bool)
    for k in ARRAYS_F32:
        if k in cfg:
            values = [float(v) for v in cfg[k]]
            writer.add_array(PREFIX + k, values)
            written.append(f"{PREFIX}{k}[{len(values)}]")

    for t in reader.tensors:
        writer.add_tensor(
            t.name, t.data, raw_dtype=GGMLQuantizationType(t.tensor_type))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Add MMDetection decoding parameters to a GGUF as metadata.")
    ap.add_argument("gguf", type=Path, help="model .gguf produced from the exported backbone")
    ap.add_argument("json", type=Path, help="<name>.postproc.json written by mmdet_to_pt.py")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write here instead of replacing the input")
    a = ap.parse_args(argv)

    cfg = json.loads(a.json.read_text(encoding="utf-8"))
    if cfg.get("head_type") == "raw":
        print("! head_type is 'raw' — the config's head was not recognised, "
              "so there are no decoding parameters to write.")

    in_place = a.output is None
    dst = Path(tempfile.mkstemp(suffix=".gguf", dir=a.gguf.parent)[1]) if in_place else a.output

    written = inject(a.gguf, cfg, dst)
    if in_place:
        shutil.move(str(dst), str(a.gguf))
        dst = a.gguf

    print(f"  → {dst}  ({len(written)} keys)")
    for k in written:
        print(f"      {k}")


if __name__ == "__main__":
    main()
