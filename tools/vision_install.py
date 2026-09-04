#!/usr/bin/env python3
"""vision_install.py — g2c 생성물을 등록하고 vision.cpp 를 다시 빌드한다.

    vision-install Yolo26m
    vision-install Seg --mean 123.675,116.28,103.53 --std 58.395,57.12,57.375

install_arch.py 로 등록한 뒤 CMake configure·build 까지 수행하고, 설치된 vision-cli 가
있으면 그것도 새로 빌드한 것으로 덮어쓴다.

빌드가 필요한지는 여기서 판단하지 않는다. arch/*.cpp glob 에 CONFIGURE_DEPENDS 가 붙어
있어 CMake 가 스스로 판단하고, 바뀐 것이 없으면 빌드 시스템이 넘어간다. 직접 판정하면
바이너리는 있는데 새 모델이 들어가지 않은 상태를 놓친다.

등록 옵션은 모델 폴더의 install.json 에 남는다. 값이 틀려도 오류 없이 결과만 달라지므로
기록이 없으면 같은 값을 재현할 수 없다. 옵션을 생략하면 그 파일의 값을 사용한다.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VCPP = os.path.dirname(HERE)
INSTALL_ARCH = os.path.join(HERE, "install_arch.py")


def find_source(name, explicit):
    """생성물 폴더를 찾는다. 기본 규칙은 output/<Name>/ 이다."""
    cands = [explicit] if explicit else [os.path.join("output", name), "output", "out"]
    for d in cands:
        if os.path.isfile(os.path.join(d, name + ".cpp")):
            return d
    sys.exit(
        "error: {}.cpp 를 찾지 못했다. 찾아본 곳: {}\n"
        "       g2c 의 --output 경로를 --source 로 지정한다.".format(name, ", ".join(cands))
    )


def load_saved_args(source):
    path = os.path.join(source, "install.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("install_args", [])


def save_args(source, extra):
    with open(os.path.join(source, "install.json"), "w", encoding="utf-8") as f:
        json.dump({"install_args": extra}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def cached_archs(build_dir):
    """빌드 디렉터리에 선택 목록이 걸려 있으면 그 값을 돌려준다."""
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return None
    for line in open(cache, encoding="utf-8", errors="replace"):
        if line.startswith("VISP_ARCHS:"):
            value = line.split("=", 1)[1].strip()
            return [v for v in value.split(";") if v] or None
    return None


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(rc)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="vision-install",
        description="g2c 생성물을 vision.cpp 에 등록하고 다시 빌드한다.",
        epilog="여기서 인식하지 않는 옵션은 install_arch.py 로 그대로 넘어간다 "
               "(--detect-yolo, --nms, --mean, --std, --classes, --size 등).",
    )
    ap.add_argument("name", help="아키텍처 이름. g2c 의 --name 값이다")
    ap.add_argument("--source", default=None, help="생성물 폴더 (기본: output/<name>)")
    ap.add_argument("--vision-dir", default=VCPP, help="vision.cpp 체크아웃")
    ap.add_argument("--build-dir", default=None, help="빌드 디렉터리 (기본: <vision-dir>/build)")
    ap.add_argument("--jobs", "-j", type=int, default=4, help="빌드 병렬 수")
    ap.add_argument("--prefix", default=None,
                    help="vision-cli 를 설치할 위치. 기본은 이미 설치되어 있는 자리를 갱신한다")
    ap.add_argument("--no-build", action="store_true", help="등록만 하고 빌드하지 않는다")
    args, extra = ap.parse_known_args(argv)

    build_dir = args.build_dir or os.path.join(args.vision_dir, "build")
    source = find_source(args.name, args.source)
    if not extra:
        extra = load_saved_args(source)

    run([sys.executable, INSTALL_ARCH, source, "--name", args.name] + extra)
    save_args(source, extra)

    if args.no_build:
        print("\n등록만 수행했다. 빌드: cmake --build {} -j{}".format(build_dir, args.jobs))
        return

    selected = cached_archs(build_dir)
    run(["cmake", "-S", args.vision_dir, "-B", build_dir])
    run(["cmake", "--build", build_dir, "-j{}".format(args.jobs)])

    built = os.path.join(build_dir, "bin", "vision-cli")
    print()
    print("등록: " + os.path.join(args.vision_dir, "src/visp/arch", args.name + ".cpp"))
    print("빌드: " + built)

    installed = None
    if args.prefix:
        installed = os.path.join(args.prefix, "bin", "vision-cli")
        os.makedirs(os.path.dirname(installed), exist_ok=True)
    else:
        found = shutil.which("vision-cli")
        if found and os.path.abspath(found) != os.path.abspath(built):
            installed = found
    if installed:
        shutil.copy2(built, installed)
        print("갱신: " + installed)

    if selected and args.name not in selected:
        joined = ";".join(selected)
        print()
        print('주의: 이 빌드 디렉터리는 VISP_ARCHS="{}" 로 설정되어 있어'.format(joined))
        print("      {} 은 실행 파일에 들어가지 않았다. 포함하려면 다시 설정한다:".format(args.name))
        print('      cmake -S {} -B {} -DVISP_ARCHS="{}"'.format(
            args.vision_dir, build_dir, ";".join(selected + [args.name])))


if __name__ == "__main__":
    main()
