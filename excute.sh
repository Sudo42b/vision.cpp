#!/bin/bash

# ./build/bin/vision-cli yolov9t -m models/yolov9t_converted.gguf -i tests/input/vase-and-bowl.jpg -o ./yolov9t_detections.jpg 2>&1
./build/bin/vision-cli yolov9t -m models/yolov9t_converted.gguf -i tests/input/cat-and-hat.jpg -o ./yolov9t_detections.jpg 2>&1
# ./build/bin/vision-cli yolov9t -m models/yolov9t_converted.gguf -i tests/input/wardrobe.jpg -o ./yolov9t_detections.jpg 2>&1




# ./wsl_build/bin/vision-cli yolov9t -m models/yolov9t_converted.gguf -i tests/input/cat-and-hat.jpg -o ./yolov9t_detections.jpg 2>&1