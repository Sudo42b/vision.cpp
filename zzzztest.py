from gguf import GGUFReader

path = "models/yolov9t_converted.gguf"
r = GGUFReader(path)

try:
    # 최신 버전 (>=0.7.0)
    names = r.get_tensor_names()
except AttributeError:
    # 구버전 (<0.7.0)
    names = [t.name for t in r.tensors]

for i, name in enumerate(names):
    if "bias" in name:
        print(i, name)
