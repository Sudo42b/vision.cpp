#include "config.h"
#include <stdexcept>
#include <unordered_map>

namespace visp::mmlab {
// MMDetection CocoDataset.METAINFO["classes"] 순서.
static std::vector<std::string> const& coco_class_names() {
    static std::vector<std::string> const names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};
    return names;
}
Json read_config(model_file const& f) {
    auto text = f.get_string("mmlab.config");
    if (text.empty() || text.size() > 16*1024*1024)
        throw std::runtime_error("mmlab-config: missing or oversized configuration");
    auto j = Json::parse(text);
    if (j.at("version") != 1 || j.contains("instructions"))
        throw std::runtime_error("mmlab-config: unsupported schema");
    return j;
}
Json const& child(Json const& n, std::string const& key) { return n.at("children").at(key); }
model_ref weights(model_ref m, Json const& n) {
    return m.with_prefix(tensor_name(n.at("prefix").get<std::string>().c_str()));
}
tensor parameter(model_ref m,Json const& n,char const* name) {
    if(n.contains("tensors")&&n.at("tensors").contains(name))return m.weights(n.at("tensors").at(name).get<std::string>().c_str());
    return weights(m,n).weights(name);
}
Tensors run(model_ref m, Json const& n, Tensors const& x) {
    static std::unordered_map<std::string, Builder> const builders = {
#define MMLAB_TYPE(name, group) {name, &group},
#include "types.inc"
#undef MMLAB_TYPE
    };
    auto type = n.at("type").get<std::string>();
    auto it = builders.find(type);
    if (it == builders.end()) throw std::runtime_error("mmlab-config: unimplemented module " + type);
    try { return it->second(m, n, x); }
    catch (std::exception const& e) {
        throw std::runtime_error(type + "(" + n.at("prefix").get<std::string>() + "): " + e.what());
    }
}
tensor one(model_ref m, Json const& n, tensor x) {
    auto out = run(m,n,{x});
    if (out.size()!=1) throw std::runtime_error("expected a single tensor");
    return out[0];
}
tensor layer(model_ref m, Json const& n, std::string const& key, tensor x) {
    return one(m,child(n,key),x);
}
tensor cat(model_ref m, Tensors const& xs, int axis) {
    if(xs.empty()) throw std::runtime_error("empty concatenation");
    tensor y=xs[0];
    for(size_t i=1;i<xs.size();++i) y=ggml_concat(m,y,xs[i],axis);
    return y;
}
tensor resize(model_ref m, tensor x, int64_t w, int64_t h, bool bilinear, bool align) {
    int mode=bilinear ? GGML_SCALE_MODE_BILINEAR : GGML_SCALE_MODE_NEAREST;
    if (align) mode |= GGML_SCALE_FLAG_ALIGN_CORNERS;
    return interpolate(m,x,{w,h},mode);
}
tensor sample_tokens(model_ref m,tensor values,int64_t width,int64_t height,tensor xp,tensor yp) {
    // Pixel coordinates, zero padding and align_corners=False grid sampling.
    // Values: [channels, width*height, groups]; coordinates: [points, groups].
    auto x0=ggml_floor(m,xp),y0=ggml_floor(m,yp),dx=ggml_sub(m,xp,x0),dy=ggml_sub(m,yp,y0);
    tensor result=nullptr;
    for(int h=0;h<2;++h)for(int w=0;w<2;++w) {
        auto x=ggml_scale_bias(m,x0,1,float(w)),y=ggml_scale_bias(m,y0,1,float(h));
        auto valid_x=ggml_mul(m,ggml_step(m,ggml_scale_bias(m,x,1,.5f)),ggml_step(m,ggml_scale_bias(m,x,-1,float(width)-.5f)));
        auto valid_y=ggml_mul(m,ggml_step(m,ggml_scale_bias(m,y,1,.5f)),ggml_step(m,ggml_scale_bias(m,y,-1,float(height)-.5f)));
        auto weight=ggml_mul(m,ggml_mul(m,w?dx:ggml_scale_bias(m,dx,-1,1),h?dy:ggml_scale_bias(m,dy,-1,1)),ggml_mul(m,valid_x,valid_y));
        // ggml_clamp aliases its input. Keep the unbounded coordinates alive
        // for the validity masks; otherwise clamping turns zero padding into
        // border padding depending on graph traversal order.
        auto index=ggml_add(m,ggml_clamp(m,ggml_cont(m,x),0,float(width-1)),ggml_scale(m,ggml_clamp(m,ggml_cont(m,y),0,float(height-1)),float(width)));
        index=ggml_cast(m,index,GGML_TYPE_I32);auto sampled=ggml_get_rows(m,values,index);
        sampled=ggml_mul(m,sampled,ggml_reshape_3d(m,weight,1,weight->ne[0],weight->ne[1]));
        result=result?ggml_add(m,result,sampled):sampled;
    }return result;
}
i64x4 config_input_shape(model_file const& f) {
    auto s=read_config(f).at("input_shape").get<std::vector<int64_t>>();
    if(s.size()<2 || s.size()>4 || s[0]!=1) throw std::runtime_error("mmlab-config: batch 1, rank 2..4 required");
    for(auto d:s) if(d<=0 || d>INT32_MAX) throw std::runtime_error("mmlab-config: invalid input shape");
    if(s.size()==4) return {s[1],s[3],s[2],s[0]};
    i64x4 out{1,1,1,1};
    for(size_t i=0;i<s.size();++i) out[int(i)]=s[s.size()-1-i];
    return out;
}
arch_task task_from_metadata(Json const& cfg, i64x4 shape) {
    arch_task task;
    task.input_w=int(shape[1]);task.input_h=int(shape[2]);
    if(cfg.contains("mean")) task.mean=cfg.at("mean").get<std::array<float,3>>();
    if(cfg.contains("std")) task.stdv=cfg.at("std").get<std::array<float,3>>();
    task.to_rgb=cfg.value("to_rgb",false);
    // 이 키가 없는 옛 GGUF 는 예전처럼 가로·세로를 따로 늘린다.
    task.keep_ratio=cfg.value("keep_ratio",false);
    if(cfg.contains("pad_value")) task.pad_value=cfg.at("pad_value").get<std::array<float,3>>();
    if(cfg.value("ecosystem",std::string{})=="mmdet" && cfg.value("mmdet_head_type",std::string{"raw"})!="raw") {
        auto const& j=cfg.at("mmdet");
        auto& c=task.mmdet;
        static std::unordered_map<std::string,head_kind> const kinds={
            {"anchor",head_kind::anchor},
            {"yolox",head_kind::yolox},
            {"rpn",head_kind::rpn},
            {"fcos",head_kind::fcos},
            {"gfl",head_kind::gfl},
            {"vfnet",head_kind::vfnet},
            {"reppoints",head_kind::reppoints},
            {"sabl",head_kind::sabl},
            {"tood",head_kind::tood},
            {"centernet",head_kind::centernet},
            {"yolof",head_kind::yolof},
            {"yolo",head_kind::yolo},
            {"cornernet",head_kind::cornernet},
            {"detr",head_kind::detr},
            {"conditional_detr",head_kind::conditional_detr},
            {"dab_detr",head_kind::dab_detr},
            {"deformable_detr",head_kind::deformable_detr},
            {"dino",head_kind::dino},
            {"ddq",head_kind::ddq},
        };
        c.head.kind=kinds.at(j.at("head").at("kind").get<std::string>());
#include "../runtime_mmdet.inc"
        c.det.input_w=task.input_w;c.det.input_h=task.input_h;
        std::copy(task.mean.begin(),task.mean.end(),c.img_mean);
        std::copy(task.stdv.begin(),task.stdv.end(),c.img_std);
        c.to_rgb=task.to_rgb;
        task.kind=arch_kind::detect_mmdet;
        // 이름이 없으면 결과 그림이 "class N" 으로 나온다. config 에 classes 가 있으면 그것을,
        // 없고 80 클래스면 COCO 이름을 붙인다.
        if(cfg.contains("classes")) task.class_names=cfg.at("classes").get<std::vector<std::string>>();
        else if(c.det.num_classes==80) task.class_names=coco_class_names();
    }
    return task;
}
} // namespace visp::mmlab
