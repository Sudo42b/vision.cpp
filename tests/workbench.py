import ctypes
from dataclasses import dataclass
import torch
import os
import platform
from functools import reduce
from typing import Mapping
from pathlib import Path
from torch import Tensor
from PIL import Image


float_ptr = ctypes.POINTER(ctypes.c_float)
dtype_torch_to_ggml = {
    torch.float32: 0,  # GGML_TYPE_F32
    torch.float16: 1,  # GGML_TYPE_F16
    torch.int32: 26,  # GGML_TYPE_I32
    torch.int64: 27,  # GGML_TYPE_I64
}
dtype_ggml_to_torch = {v: k for k, v in dtype_torch_to_ggml.items()}


class RawTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("ne", ctypes.c_int * 4),
    ]

    @property
    def n_bytes(self):
        tsize = dtype_ggml_to_torch.get(self.type, torch.float32).itemsize
        return self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3] * tsize


class RawParam(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
        ("type", ctypes.c_int),
    ]


def torch_to_raw_tensor(name: str, tensor: torch.Tensor):

    t = tensor.contiguous()
    if tensor.dtype not in dtype_torch_to_ggml:
        print(f"Warning: tensor {name} has unsupported dtype {tensor.dtype}, converting to float")
        t = t.to(torch.float32)
    while t.dim() < 4:
        t = t.unsqueeze(0)
    assert t.dim() == 4
    raw_tensor = RawTensor()
    raw_tensor.name = name.encode()
    raw_tensor.data = ctypes.cast(t.data_ptr(), ctypes.c_void_p)
    raw_tensor.type = dtype_torch_to_ggml[t.dtype]
    raw_tensor.ne[0] = t.size(3)
    raw_tensor.ne[1] = t.size(2)
    raw_tensor.ne[2] = t.size(1)
    raw_tensor.ne[3] = t.size(0)
    return (raw_tensor, t)


def raw_to_torch_tensor(raw_tensor: RawTensor):
    dtype = dtype_ggml_to_torch.get(raw_tensor.type, torch.float32)
    shape = tuple(raw_tensor.ne[i] for i in range(3, -1, -1))
    return torch.frombuffer(
        bytearray(ctypes.string_at(raw_tensor.data, raw_tensor.n_bytes)),
        dtype=dtype,
    ).reshape(shape)


def encode_params(params: Mapping[str, str | int | float]):
    raw_params = []
    for name, value in params.items():
        ptype = 0
        if isinstance(value, int):
            value = str(value).encode()
            ptype = 0
        elif isinstance(value, float):
            value = str(value).encode()
            ptype = 1
        elif isinstance(value, str):
            value = value.encode()
            ptype = 2
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {type(value)}")
        raw_params.append(RawParam(name=name.encode(), value=value, type=ptype))
    return raw_params


# Some python lib (numpy? torch?) loads OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = Path(__file__).parent.parent


def _load_library():
    system = platform.system().lower()
    if system == "windows":
        prefix = ""
        suffix = ".dll"
        libdir = "bin"
    elif system == "darwin":
        prefix = "lib"
        suffix = ".dylib"
        libdir = "lib"
    else:  # assume Linux / Unix
        prefix = "lib"
        suffix = ".so"
        libdir = "lib"
    lib_path = root_dir / "build" / libdir / f"{prefix}vision-workbench{suffix}"
    return ctypes.CDLL(str(lib_path))


try:
    lib = _load_library()

    lib.visp_workbench.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(RawTensor),
        ctypes.c_int32,
        ctypes.POINTER(RawParam),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.POINTER(RawTensor)),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),  # capture_names
        ctypes.c_int32,  # n_captures
    ]
    lib.visp_workbench.restype = ctypes.c_int32
except OSError as e:
    print(f"Error loading vision-workbench library: {e}")


@dataclass
class Capture:
    visp_name: str
    visp_result: torch.Tensor | None = None
    torch_result: torch.Tensor | None = None

    @property
    def filename(self):
        return self.visp_name.replace(".", "_") + ".pt"

    def save(self):
        assert self.torch_result is not None
        torch.save(self.torch_result, root_dir / ".tmp" / self.filename)

    def load(self):
        self.torch_result = torch.load(root_dir / ".tmp" / self.filename)


class Captures:
    def __init__(self, names: list[str] | None = None):
        self.list: list[Capture] = []
        if names is not None:
            for name in names:
                self.add(name)

    def add(self, name: str):
        capture = Capture(visp_name=name)
        self.list.append(capture)
        return capture

    def find(self, name: str):
        return next((c for c in self.list if c.visp_name == name), None)

    def __getitem__(self, name: str):
        if capture := self.find(name):
            return capture
        raise KeyError(name)

    def hook(self, name: str, module: torch.nn.Module):
        capture = self.find(name)
        if capture is None:
            capture = self.add(name)

        def hook(module, input, output):
            capture.torch_result = output.detach().cpu()

        module.register_forward_hook(hook)

    @property
    def names(self):
        return [capture.visp_name for capture in self.list]

    def save_to_cache(self):
        for capture in self.list:
            capture.save()

    def load_from_cache(self):
        for capture in self.list:
            capture.load()

    def compare(self, print_tensors=False, rtol=0.01, atol=0.001):
        result = True
        for capture in self.list:
            assert capture.visp_result is not None and capture.torch_result is not None
            if not tensors_match(capture.visp_result, capture.torch_result, rtol=rtol, atol=atol):
                print(f"\nCapture {capture.visp_name} does not match!")
                diff = (capture.visp_result - capture.torch_result).abs()
                print(f"Diff max={diff.max().item()}, mean={diff.mean().item()}")
                if print_tensors:
                    print_results(capture.visp_result, capture.torch_result)
                result = False
        return result


def invoke_test(
    test_case: str,
    input: torch.Tensor | list[torch.Tensor],
    state: dict[str, torch.Tensor],
    params: Mapping[str, str | int | float] = {},
    backend: str = "cpu",
    captures: Captures | None = None,
):
    input = input if isinstance(input, list) else [input]
    raw_inputs = [torch_to_raw_tensor(f"input{i}", tensor) for i, tensor in enumerate(input)]
    raw_inputs += [torch_to_raw_tensor(name, tensor) for name, tensor in state.items()]
    input_tensors = [t for _, t in raw_inputs]
    input_tensors  # keep the tensors alive
    raw_inputs = [t for t, _ in raw_inputs]
    raw_params = encode_params(params)
    raw_output = ctypes.POINTER(RawTensor)()
    output_size = ctypes.c_int32(0)
    capture = captures.names if captures else []
    raw_capture = (ctypes.c_char_p * len(capture))(*[name.encode() for name in capture])

    result = lib.visp_workbench(
        test_case.encode(),
        (RawTensor * len(raw_inputs))(*raw_inputs),
        len(raw_inputs),
        (RawParam * len(raw_params))(*raw_params) if len(raw_params) > 0 else None,
        len(raw_params),
        ctypes.byref(raw_output),
        ctypes.byref(output_size),
        1 if backend == "cpu" else 2,
        raw_capture if capture else None,
        len(capture),
    )

    assert result == 0, f"Test case {test_case} failed"
    if output_size.value == 0:
        return None

    outputs = [raw_to_torch_tensor(raw_output[i]) for i in range(output_size.value)]
    if captures is not None:
        offset = len(outputs) - len(capture)
        assert offset >= 0, "More captures requested than outputs returned"
        for i, name in enumerate(capture):
            captures.list[i].visp_result = outputs[i + offset]
        outputs = outputs[:offset]

    output = outputs[0] if len(outputs) == 1 else outputs
    return output


def input_tensor(*shape: int):
    end = reduce(lambda x, y: x * y, shape, 1)
    return torch.arange(0, end).reshape(*shape) / end


def input_like(tensor: torch.Tensor):
    if tensor.dim() == 0:
        return torch.tensor(0.5, dtype=tensor.dtype)
    return input_tensor(*tensor.shape)


def generate_state(state_dict: dict[str, torch.Tensor]):
    return {k: input_like(v) if v.dtype.is_floating_point else v for k, v in state_dict.items()}


def randomize(state_dict: dict[str, torch.Tensor]):
    return {
        k: torch.rand_like(v) if v.dtype.is_floating_point else v for k, v in state_dict.items()
    }


def to_nhwc(tensor: torch.Tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()


def to_nchw(tensor: Tensor | list[Tensor] | None):
    assert tensor is not None
    if isinstance(tensor, list):
        return [t.permute(0, 3, 1, 2).contiguous() for t in tensor]
    return tensor.permute(0, 3, 1, 2).contiguous()


def convert_to_nhwc(state: dict[str, Tensor], key=""):
    for k, v in state.items():
        is_conv = (
            v.ndim == 4
            and v.shape[2] == v.shape[3]
            and v.shape[2] in (1, 3, 4, 7)
            and k.endswith("weight")
        )
        if is_conv and (key == "" or key in k):
            if v.shape[1] == 1:  # depthwise
                state[k] = v.permute(2, 3, 1, 0).contiguous()
            else:
                state[k] = v.permute(0, 2, 3, 1).contiguous()
    return state


def fuse_batch_norm(model: dict[str, torch.Tensor], key: str, key_bn: str):
    suffix_weight = f"{key_bn}.weight"
    suffix_bias = f"{key_bn}.bias"

    if key.endswith(suffix_weight):
        base = key.removesuffix(suffix_weight)
        weight = model[key]
        var = model[f"{base}{key_bn}.running_var"]
        return weight / torch.sqrt(var + 1e-5)

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        bias = model[key]
        weight = model[f"{base}{key_bn}.weight"]
        mean = model[f"{base}{key_bn}.running_mean"]
        var = model[f"{base}{key_bn}.running_var"]
        return bias - mean * weight / torch.sqrt(var + 1e-5)

    elif key.endswith(f"{key_bn}.running_mean") or key.endswith(f"{key_bn}.running_var"):
        return None

    return model[key]


def fuse_conv_2d_batch_norm(
    model: dict[str, torch.Tensor],
    key: str,
    key_module: str,
    key_conv: str,
    key_norm: str,
    out: dict[str, torch.Tensor],
):
    suffix_conv = f"{key_module}{key_conv}.weight"
    suffix_norm = f"{key_module}{key_norm}."

    if key.endswith(suffix_conv):
        base = key.removesuffix(suffix_conv)
        conv_weight = model[key]
        bn_weight = model[f"{base}{suffix_norm}weight"]
        bn_bias = model[f"{base}{suffix_norm}bias"]
        bn_mean = model[f"{base}{suffix_norm}running_mean"]
        bn_var = model[f"{base}{suffix_norm}running_var"]
        conv_bias = model.get(f"{base}{key_module}{key_conv}.bias", torch.zeros_like(bn_bias))

        bn_weight_var = bn_weight / torch.sqrt(bn_var + 1e-5)
        fused_weight = conv_weight * bn_weight_var[:, None, None, None]
        fused_bias = conv_bias
        fused_bias -= bn_mean
        fused_bias *= bn_weight_var
        fused_bias += bn_bias
        # fused_bias = (conv_bias - bn_mean) * bn_weight_var + bn_bias

        out[key] = fused_weight
        out[key.replace("weight", "bias")] = fused_bias
        return True

    elif suffix_norm in key:
        return True  # batch norm was fused above

    return False  # no match


def print_results(result: Tensor, expected: Tensor):
    torch.set_printoptions(precision=3, linewidth=100, sci_mode=False)
    print("\nresult -----", result, sep="\n")
    print("\nexpected ---", expected, sep="\n")


def tensors_match(
    result: Tensor | list[Tensor] | None,
    expected: Tensor | list[Tensor],
    rtol=1e-3,
    atol=1e-5,
    show=False,
):
    assert result is not None, "No result returned"
    if isinstance(expected, list):
        assert isinstance(result, list), "Result is not a list"
        assert len(result) == len(expected), f"Expected {len(expected)} tensors, got {len(result)}"
        return all(tensors_match(r, e, rtol, atol, show) for r, e in zip(result, expected))
    assert isinstance(result, Tensor), "Result is not a tensor"
    if show:
        print_results(result, expected)
    return torch.allclose(result, expected, rtol=rtol, atol=atol)


def assert_images_match(
    result: Tensor | list[Tensor] | None, expected: Tensor | list[Tensor], tol=0.001
):
    assert result is not None, "No result returned"
    if isinstance(expected, list):
        assert isinstance(result, list), "Result is not a list"
        assert len(result) == len(expected), f"Expected {len(expected)} tensors, got {len(result)}"
        for r, e in zip(result, expected):
            assert_images_match(r, e, tol)
    else:
        assert isinstance(result, Tensor), "Result is not a tensor"
        rmse = torch.sqrt(torch.mean((result - expected) ** 2))
        assert rmse.item() < tol, f"RMSE {rmse.item()} exceeds tolerance {tol}"


def dump_image(t: Tensor, filepath: str):
    image = Image.fromarray((t.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")[0])
    image.save(filepath)


def dump_images(result: Tensor | list[Tensor], expected: Tensor | list[Tensor], prefix="result"):
    if isinstance(expected, list):
        assert isinstance(result, list), "Result is not a list"
        assert len(result) == len(expected), f"Expected {len(expected)} tensors, got {len(result)}"
        for i, (r, e) in enumerate(zip(result, expected)):
            dump_images(r, e, prefix=f"{prefix}_{i}")
    else:
        assert isinstance(result, Tensor), "Result is not a tensor"
        dump_image(result, f"{prefix}_result.png")
        dump_image(expected, f"{prefix}_expected.png")
