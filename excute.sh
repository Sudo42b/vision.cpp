#!/bin/bash

cd /mnt/e/7_RISCV/vision.cpp && ./build/bin/vision-cli yolov9t -m models/yolov9t_converted.gguf -i tests/input/cat-and-hat.jpg -o ./yolov9t_detections.jpg 2>&1

