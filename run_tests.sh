#!/bin/bash

# 현재 프로젝트의 라이브러리 경로를 우선으로 설정
export LD_LIBRARY_PATH="/mnt/e/7_RISCV/vision.cpp/build/lib:$LD_LIBRARY_PATH"
source ./.venv/bin/activate
# pytest 실행
uv run pytest "$@"