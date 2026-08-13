"""vconfig.py — `verify.toml` 을 읽는다. 검증 도구들이 공유한다.

**왜 환경변수를 걷어냈나.** env 로 설정을 받으면 어떤 값으로 잰 숫자인지 로그에 안 남는다.
같은 명령을 쳐도 셸마다 다른 결과가 나오고, 재현이 안 되면 그 숫자는 근거가 못 된다.
그래서 기본값은 파일에 두고, 덮어쓰려면 `--set key=value` 로만 — 그것도 실행 시 찍는다.
"""
import os
import sys

try:
    import tomllib                      # py3.11+
except ModuleNotFoundError:             # pragma: no cover
    import tomli as tomllib             # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "verify.toml")


def _resolve(v):
    """상대경로는 **이 파일 위치 기준**으로 절대화한다. `~` 도 편다."""
    v = os.path.expanduser(v)
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(HERE, v))


class Config:
    def __init__(self, path=None, overrides=()):
        self.path = path or os.environ.get("VERIFY_CONFIG", DEFAULT)
        with open(self.path, "rb") as f:
            self.d = tomllib.load(f)
        self.overrides = []
        for kv in overrides:
            k, _, v = kv.partition("=")
            sec, _, key = k.strip().partition(".")
            if not key or sec not in self.d or key not in self.d[sec]:
                raise SystemExit(f"--set {kv}: verify.toml 에 없는 키다 (예: run.workers=2)")
            cur = self.d[sec][key]
            self.d[sec][key] = type(cur)(v) if not isinstance(cur, bool) else v == "true"
            self.overrides.append(f"{sec}.{key}={self.d[sec][key]}")

    # ── 경로 ──
    @property
    def mmdet(self):    return _resolve(self.d["paths"]["mmdet"])
    @property
    def configs(self):  return os.path.join(self.mmdet, "configs")
    @property
    def ckpt(self):     return os.path.join(self.mmdet, "checkpoints")
    @property
    def g2c(self):      return _resolve(self.d["paths"]["g2c"])
    @property
    def unwrap(self):   return _resolve(self.d["paths"]["unwrap"])
    @property
    def workdir(self):  return _resolve(self.d["paths"]["workdir"])
    @property
    def probe_workdir(self): return _resolve(self.d["paths"]["probe_workdir"])

    # ── 실행 ──
    @property
    def workers(self):     return int(self.d["run"]["workers"])
    @property
    def opt(self):         return self.d["run"]["opt"]
    @property
    def size(self):        return int(self.d["run"]["size"])
    @property
    def min_free_mb(self): return int(self.d["run"]["min_free_mb"])
    @property
    def l1(self):          return float(self.d["tolerance"]["l1"])
    @property
    def l2(self):          return float(self.d["tolerance"]["l2"])

    def banner(self):
        """무엇으로 쟀는지 로그 맨 위에 남긴다 — 이게 없으면 숫자가 근거가 못 된다."""
        s = (f"config {self.path}\n"
             f"  mmdet={self.mmdet}\n  g2c={self.g2c}\n  workdir={self.workdir}\n"
             f"  workers={self.workers} size={self.size} opt={self.opt}"
             f" tol=L1{self.l1}/L2{self.l2}")
        if self.overrides:
            s += "\n  --set " + " ".join(self.overrides)
        return s


def load(argv=None):
    """`--config PATH` 와 `--set a.b=v` 를 걷어내고 (Config, 남은 인자) 를 돌려준다."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path, sets, rest = None, [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config" and i + 1 < len(argv):
            path = argv[i + 1]; i += 2
        elif a == "--set" and i + 1 < len(argv):
            sets.append(argv[i + 1]); i += 2
        else:
            rest.append(a); i += 1
    return Config(path, sets), rest
