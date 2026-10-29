#include "config.h"
#include <stdexcept>
#include <unordered_map>
#include <cstdlib>

namespace visp::mmlab {
Tensors build_architecture(model_ref m, std::string const& name, Json const& spec, Tensors const& inputs) {
    // Architecture factories are compiled into the runner. GGUF selects one;
    // it cannot supply code or change the architecture's forward connections.
    static std::unordered_map<std::string,std::string> const architectures = {
#define MMLAB_ARCH(scope, name) {scope "." name, scope},
#include "architectures.inc"
#undef MMLAB_ARCH
    };
    auto it=architectures.find(name);
    if(it==architectures.end())throw std::runtime_error("unregistered MMLab architecture: "+name);
    auto expected=name=="mmdet.GLIP"?"MMDetTextCond":it->second=="mmdet"?"MMDetBackbone":it->second=="mmseg"?"MMSegWrap":"MMPoseWrap";
    if(spec.at("type")!=expected)throw std::runtime_error("architecture/module protocol mismatch");
    return models(m,spec,inputs);
}
Tensors models(model_ref m, Json const& n, Tensors const& inputs) {
    auto type=n.at("type").get<std::string>();
    auto const& c=n.at("children");
    if(type=="MMSegWrap"&&n.at("attrs").value("multimodal",false)) {
        auto image=inputs.at(0),encoder_input=image;
        if(n.at("attrs").value("asymetric_input",false)) {
            double scale=n.at("attrs").at("encoder_resolution").get<double>();
            encoder_input=resize(m,image,int64_t(image->ne[0]*scale),int64_t(image->ne[1]*scale),true);
        }
        auto features=run(m,child(n,"image_encoder"),{encoder_input});features.insert(features.begin(),image);
        features.push_back(run(m,child(n,"text_encoder"),inputs).at(0));
        return run(m,child(n,"decode_head"),features);
    }
    auto out=run(m,child(n,"backbone"),inputs);
    if(c.contains("neck")) out=run(m,child(n,"neck"),out);
    if(type=="MMDetBackbone") return out;
    if(type=="MMDetTextCond") {
        out.push_back(parameter(m,n,"text_embedded"));
        return run(m,child(n,"bbox_head"),out);
    }
    if(type=="MMSegWrap") {
        if(n.at("attrs").value("multimodal",false))
            throw std::runtime_error("special segmentation predict protocol not implemented");
        if(n.at("attrs").value("cascade",false)) {
            auto const& heads=child(n,"decode_head");Tensors result;
            for(auto const& key:heads.at("order")) {
                auto features=out;
                if(!result.empty())features.insert(features.end(),result.begin(),result.end());
                auto head=child(heads,key.get<std::string>());
                if(head.at("type")=="PointHead") {
                    head["attrs"]["native_test_cfg"]=n.at("attrs").at("test_cfg");
                    head["attrs"]["native_input_w"]=inputs.at(0)->ne[0];head["attrs"]["native_input_h"]=inputs.at(0)->ne[1];
                }
                result=run(m,head,features);
            }
            return result;
        }
        auto head=child(n,"decode_head");head["attrs"]["native_input_w"]=inputs.at(0)->ne[0];head["attrs"]["native_input_h"]=inputs.at(0)->ne[1];
        if(auto probe=std::getenv("VISP_SEG_PROBE")) {
            std::string name=probe;
            if(name=="pix"||name=="pix_ms") {
                auto pixels=run(m,child(head,"pixel_decoder"),out);
                if(name=="pix")return {pixels.front()};pixels.erase(pixels.begin());return pixels;
            }
        }
        return run(m,head,out);
    }
    if(type=="MMPoseWrap") return run(m,child(n,"head"),out);
    throw std::runtime_error("missing model architecture");
}
tensor config_forward(model_ref m, tensor x, model_file const& f) {
    auto j=read_config(f);
    bool image=j.at("input_shape").size()==4;
    if(image) x=cwhn_to_contiguous_2d(m,x);
    auto out=build_architecture(m,j.at("architecture").get<std::string>(),j.at("module"),{x});
    if(out.empty()) throw std::runtime_error("mmlab-config: empty output");
    tensor result=nullptr;
    for(size_t i=0;i<out.size();++i) {
        bool token_output=j.at("module").at("type")=="MMDetTextCond" && i<out.size()/3;
        auto y=image&&!token_output?contiguous_2d_to_cwhn(m,out[i]):out[i];
        result=compute_graph_output(m,y,tensor_name(("out_"+std::to_string(i)).c_str()));
    }
    return result;
}
arch_task config_task(model_file const& f) {
    auto j=read_config(f);
    return task_from_metadata(j.at("config"),config_input_shape(f));
}
} // namespace visp::mmlab
namespace visp {
void register_mmlab_config() {
    static arch_registrar registration("mmlab-config",mmlab::config_forward);
}
}
