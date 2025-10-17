#!/bin/bash

cd /mnt/e/7_RISCV/vision.cpp/build && make vision-cli -j8
cd /mnt/e/7_RISCV/vision.cpp && ./build/bin/vision-cli yolov9t -m scripts/yolov9t_converted.pth -i tests/input/cat-and-hat.jpg -o yolov9t_detection_output.png
cd /mnt/e/7_RISCV/vision.cpp && ./build/bin/vision-cli yolov9t -m models/yolov9t_converted-F16.gguf -i tests/input/cat-and-hat.jpg -o yolov9t_detection_output.png

cd scripts
uv run ./scripts/convert.py --arch yolov9t --input ./scripts/yolov9t_converted.pth --model-name yolov9t --output ./models/yolov9t_converted.gguf --layout cwhn  --bn-fuse --verbose


uv run ./scripts/dump_yolov9t_compare.py -w ./scripts/yolov9t_converted.pth -i tests/input/cat-and-hat.jpg -o ./yolov9t_detections.jpg --dump-all --use-cpp-input ./yolov9t_detections_input.txt 2>&1