#pragma once

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/util.h"
#include "visp/vision.h"

#include <array>
#include <map>
#include <vector>

namespace visp::yolov9t {
struct LetterboxResult {
    image_data img;
    float gain;
    float pad_w;
    float pad_h;
};

struct yolov9t_params {
    float scale = 1.0f;
    float offset = 0.0f;
    int num_classes = 80;
    int input_size = 640;
    std::string variant = "tiny";

    static constexpr int stride = 32;
};

struct detected_obj {
    float x1, y1, x2, y2;   
    float confidence;       
    int class_id;           
    float class_confidence; 
};

struct DetectOutput {
    std::vector<tensor> raw_outputs;
    std::vector<tensor> features; 
    std::map<int, tensor> features_map;
    tensor predictions_cls;  
    tensor predictions_bbox;  
    tensor anchor_points;
    tensor strides_points;
    tensor dfl_proj;     
    int reg_max = 0;     
    std::vector<float> anchor_host_data;  
    std::vector<float> stride_host_data;  
    std::vector<float> dfl_proj_host_data; 
    std::vector<tensor> dbg_reg_logits_1x1; 
    std::vector<tensor> dbg_cls_logits_1x1; 
    std::vector<tensor> dbg_reg_mid;        
    std::vector<tensor> dbg_cls_mid;        
    tensor dbg_cls_logits_cat;              
    tensor dbg_dfl_softmax;                 
    tensor dbg_box_view;                    
    tensor debug_cls_logits = nullptr;
};

struct PreprocessResult {
    tensor input_tensor;
    float scale;
    int pad_w;
    int pad_h;
    int orig_w;
    int orig_h;
};
struct NMSParams {
    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    int max_det = 1000;
    int max_nms = 30000;
    int max_wh = 7680;
};



float resize_longest_side(i32x2 extent, int target_longest_side);

LetterboxResult yolov9t_process_input(image_data image, yolov9t_params const& p);

image_data yolov9t_process_input2(image_view image, yolov9t_params const& p);
void sync_detect_outputs(DetectOutput& outputs, backend_device const& backend);

yolov9t_params yolov9t_detect_params(model_file const& file);


tensor Conv(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int k = 1,
    int s = 1,
    int p = -1,
    bool act = true,
    bool debug = false,
    bool bn = false);
tensor ELAN1(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int c3,
    int c4,
    bool debug = false);
tensor AConv(model_ref m, tensor x, std::string const& name, int c1, int c2, bool debug = false);
tensor RepNCSPELAN4(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int c3,
    int c4,
    int n = 1,
    bool debug = false);
tensor SPPELAN(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int c3,
    int k = 5,
    bool debug = false);
tensor RepBottleneck(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    bool shortcut = true,
    int g = 1,
    int k = 3,
    float e = 0.5,
    bool debug = false);
tensor RepCSP(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int n,
    bool shortcut = true,
    int g = 1,
    float e = 0.5,
    bool debug = false);
tensor Upsample(model_ref m, tensor x, int scale_factor, bool debug = false);
tensor Concat(model_ref m, tensor a, tensor b, int axis = 2, bool debug = false);
tensor C3(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int n = 1,
    bool shortcut = true,
    int g = 1,
    float e = 0.5,
    bool debug = false);
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


std::map<int, tensor> yolov9t_backbone(model_ref m, tensor x);


tensor dfl_forward(model_ref m, tensor weight, tensor x, int reg_max, bool debug=false);

std::pair<tensor, tensor> make_anchors(
    model_ref m,
    DetectOutput const& out,
    std::vector<float>& anchor_host,
    std::vector<float>& stride_host,
    std::vector<float> const& strides,
    float grid_cell_offset = 0.5f);
tensor dist2bbox(model_ref m, tensor dists, tensor anchors, bool xywh);

DetectOutput detect_forward(
    model_ref m, std::vector<tensor> features, std::vector<int> ch, int nc, bool training);

DetectOutput yolov9t_forward(model_ref m, tensor x);


std::vector<DetectOutput> process_outputs(
    detected_obj const& outputs,
    yolov9t_params const& params,
    i32x2 input_size,
    float conf_threshold = 0.25f);


std::vector<detected_obj> non_max_suppression(
    DetectOutput const& outputs,
    float conf_thres = 0.25f,
    float iou_thres = 0.45f,
    int max_det = 300,
    int max_nms = 30000,
    int max_wh = 7680);

image_data hwc_to_chw_f32(image_data const& hwc);


std::vector<std::string> const& get_coco_class_names();

int check_img_size(int imgsz, int s = 32, int floor = 0);

LetterboxResult letterbox(image_data im, i32x2 new_shape, u8x3 color,
    bool _auto, bool scaleFill, bool scaleup, int stride);

image_data linear_image_resize(image_data im, i32x2 new_shape);
image_data image_add_border(image_data im, int top, int bottom, int left, int right, u8x3 color);

void scale_boxes(std::vector<detected_obj>& boxes,
    i32x2 from_shape,
    i32x2 to_shape,
    float gain,
    float pad_w,
    float pad_h);

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


void save_features_to_txt(
    DetectOutput const& out, char const* base_path, std::vector<int> const& keys = {});


void save_input_to_txt(tensor input, char const* filepath);
} 
