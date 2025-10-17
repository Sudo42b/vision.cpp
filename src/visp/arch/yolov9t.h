#pragma once

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/util.h"
#include "visp/vision.h"

#include <map>
#include <array>
#include <vector>

namespace visp::yolov9t {

// YOLOv9t Parameters
struct yolov9t_params {
    float scale = 1.0f;
    float offset = 0.0f;
    int num_classes = 80;
    int input_size = 640;
    std::string variant = "tiny";
    
    static constexpr int stride = 32;
};

struct detected_obj {
    float x1, y1, x2, y2;        // Bounding box (x1, y1, x2, y2)
    float confidence;        // Object confidence
    int class_id;           // Class ID
    float class_confidence; // Class confidence
};

// Complete Detect head forward pass
struct DetectOutput {
    std::vector<tensor> raw_outputs;
    std::vector<tensor> features;   // Selected backbone/neck features exposed for dumping
    std::map<int, tensor> features_map;
    tensor predictions_cls;  // [nc, num_anchors, 1, bs]
    tensor predictions_bbox;  // [4, num_anchors, 1, bs]
};
struct NMSParams {
    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    int max_det = 1000;
    int max_nms = 30000;
    int max_wh = 7680;
};

// functions

float resize_longest_side(i32x2 extent, int target_longest_side);
image_data yolov9t_process_input(image_data image, yolov9t_params const& p);
image_data yolov9t_process_input2(image_view image, yolov9t_params const& p);
void sync_detect_outputs(DetectOutput& outputs, backend_device const& backend);
// Detection parameters
yolov9t_params yolov9t_detect_params(model_file const& file);
// Core modules - actual layer implementations
// Conv function matching Python Conv class parameters
tensor Conv(model_ref m, tensor x, std::string const& name, int c1, int c2, int k=1, int s=1, int p=-1, bool act=true, bool debug=false, bool bn=false);
tensor ELAN1(model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int c4, bool debug=false);
tensor AConv(model_ref m, tensor x, std::string const& name, int c1, int c2, bool debug=false);
tensor RepNCSPELAN4(model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int c4, int n=1, bool debug=false);
tensor SPPELAN(model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int k=5, bool debug=false);
tensor RepBottleneck(model_ref m, tensor x, std::string const& name, int c1, int c2, bool shortcut=true, int g=1, int k=3, float e=0.5, bool debug=false);
tensor RepCSP(model_ref m, tensor x, std::string const& name, int c1, int c2, int n, bool shortcut=true, int g=1, float e=0.5, bool debug=false);
tensor Upsample(model_ref m, tensor x, int scale_factor, bool debug=false);
tensor Concat(model_ref m, tensor a, tensor b, int axis=2, bool debug=false);
tensor C3(model_ref m,tensor x,std::string const& name,int c1,int c2,int n = 1,bool shortcut = true,int g = 1,float e = 0.5,bool debug = false);
tensor RepConv(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int k = 3,
    int s = 1,
    int p = 1,
    int g = 1,
    int d = 1,
    bool act = true,
    bool bn = false,
    bool deploy = false,
    bool debug = false);

tensor Bottleneck(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    bool shortcut = true,
    int g = 1,
    float e = 0.5,
    bool debug = false);
// Backbone network
// std::vector<tensor> yolov9t_backbone(model_ref m, tensor x);
std::map<int, tensor> yolov9t_backbone(model_ref m, tensor x);

// Detection head components
tensor dfl_forward(model_ref m, tensor x, int reg_max, tensor& proj, bool debug=false);

std::pair<tensor, tensor> make_anchors(
    model_ref m,
    DetectOutput const& out,
    std::vector<float>& anchor_host,
    std::vector<float>& stride_host,
    std::vector<float> const& strides,
    float grid_cell_offset=0.5f);
tensor dist2bbox(model_ref m, tensor dists, tensor anchors, bool xywh);
// Detect head over multi-scale features
DetectOutput detect_forward(model_ref m, std::vector<tensor> features, std::vector<int> ch, int nc, bool training);
// Main forward pass
DetectOutput yolov9t_forward(model_ref m, tensor x);

//Process outputs to detections
std::vector<DetectOutput> process_outputs(
    detected_obj const& outputs,
    yolov9t_params const& params,
    i32x2 input_size,
    float conf_threshold = 0.25f
);

// Post-processing
std::vector<detected_obj> non_max_suppression(
    DetectOutput const& outputs, 
    float conf_thres=0.25f, 
    float iou_thres=0.45f, 
    int max_det=300, 
    int max_nms=30000,
    int max_wh=7680);

image_data hwc_to_chw_f32(image_data const& hwc);

// Class names
std::vector<std::string> const& get_coco_class_names();


int check_img_size(int imgsz, int s=32, int floor=0);
image_data letterbox(image_data im, i32x2 new_shape={640, 640}, u8x3 color={114, 114, 114}, 
    bool _auto=true, bool scaleFill=false, bool scaleup=true, int stride=32);
image_data linear_image_resize(image_data im, i32x2 new_shape);
image_data image_add_border(image_data im, int top, int bottom, int left, int right, u8x3 color);
void scale_boxes(
    std::vector<detected_obj>& detections,
    i32x2 model_shape,  // [height, width] of model input
    i32x2 img_shape);

std::vector<int> nms(
            std::vector<detected_obj> const& detections,
            float iou_threshold,
            bool agnostic,
            int max_wh = 7680);
void xywh2xyxy(float* boxes, int n);
void draw_detections(
    image_data& img,
    std::vector<detected_obj> const& detections,
    std::vector<std::string> const& class_names);

// Utilities: save selected feature maps to text files using base path prefix
void save_features_to_txt(DetectOutput const& out, char const* base_path, std::vector<int> const& keys = {});

// Save preprocessed input (float32 RGB, CWHN tensor) to a text file
void save_input_to_txt(tensor input, char const* filepath);
} // namespace visp::yolov9t
