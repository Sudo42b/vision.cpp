#include "config.h"

#include <stdexcept>

namespace visp::mmlab {
namespace {
tensor channels(model_ref m, tensor x, int64_t start, int64_t count) {
    return ggml_cont(m,ggml_view_4d(m,x,x->ne[0],x->ne[1],count,x->ne[3],
        x->nb[1],x->nb[2],x->nb[3],size_t(start)*x->nb[2]));
}
}

Tensors text_fusion(model_ref m, Json const& n, Tensors const& inputs) {
    auto type=n.at("type").get<std::string>();
    auto const& a=n.at("attrs");
    if(type=="Conv3x3Norm") {
        auto x=run(m,child(n,"conv"),inputs).at(0);
        return {layer(m,n,"bn",x)};
    }
    if(type=="GLIPDyReLU") {
        auto x=inputs.at(0),pooled=layer(m,n,"avg_pool",x);
        auto coeff=layer(m,n,"fc",ggml_reshape_2d(m,pooled,x->ne[2],x->ne[3]));
        coeff=ggml_reshape_4d(m,coeff,1,1,coeff->ne[0],x->ne[3]);
        auto first=ggml_add(m,ggml_mul(m,x,ggml_scale(m,channels(m,coeff,0,x->ne[2]),2)),
            ggml_scale_bias(m,channels(m,coeff,x->ne[2],x->ne[2]),1,-.5f));
        auto second=ggml_add(m,ggml_mul(m,x,ggml_scale_bias(m,channels(m,coeff,2*x->ne[2],x->ne[2]),2,-1)),
            ggml_scale_bias(m,channels(m,coeff,3*x->ne[2],x->ne[2]),1,-.5f));
        return {ggml_add(m,second,ggml_relu(m,ggml_sub(m,first,second)))};
    }
    if(type=="DyConv") {
        Tensors out;
        auto const& c=n.at("children");
        bool deform=c.at("offset").at("type")!="Identity";
        bool attention=c.at("attnconv").at("type")!="Identity";
        for(size_t i=0;i<inputs.size();++i) {
            auto x=inputs[i];tensor offset=nullptr,mask=nullptr;
            if(deform) {
                auto packed=layer(m,n,"offset",x);
                offset=channels(m,packed,0,18);
                mask=ggml_sigmoid(m,channels(m,packed,18,9));
            }
            auto branch=[&](char const* key,tensor feature,bool up) {
                Tensors args{feature};
                if(deform){args.push_back(offset);args.push_back(mask);}
                auto y=run(m,child(child(n,"dyconvs"),key),args).at(0);
                if(up)y=resize(m,y,x->ne[0],x->ne[1],true,true);
                if(attention)y=ggml_mul(m,y,layer(m,n,"h_sigmoid",layer(m,n,"attnconv",y)));
                return y;
            };
            auto sum=branch("1",x,false);int count=1;
            if(i>0){sum=ggml_add(m,sum,branch("2",inputs[i-1],false));++count;}
            if(i+1<inputs.size()){sum=ggml_add(m,sum,branch("0",inputs[i+1],true));++count;}
            out.push_back(layer(m,n,"relu",ggml_scale(m,sum,1.f/count)));
        }
        return out;
    }
    if(type=="ATSSVLFusionHead")return run(m,child(n,"head"),inputs);
    if(type=="VLFusionModule") {
        if(a.value("early_fuse",false))throw std::runtime_error("early language encoder fusion unsupported");
        if(inputs.size()<2)throw std::runtime_error("text fusion requires image features and text embeddings");
        auto priors=a.at("num_base_priors").get<int>();
        if(priors!=1)throw std::runtime_error("text fusion requires one prior per pixel");
        if(inputs.back()->ne[0]!=a.at("lang_dim").get<int>() || inputs.back()->ne[2]!=1 || inputs.back()->ne[3]!=1)
            throw std::runtime_error("text fusion requires batch-one token embeddings");
        Tensors features(inputs.begin(),inputs.end()-1);
        auto embedding=ggml_l2_norm(m,inputs.back(),1e-12f);
        auto tokens=layer(m,n,"dot_product_projection_text",ggml_scale(m,embedding,.5f));
        auto bias=ggml_mul_mat(m,ggml_reshape_2d(m,parameter(m,n,"bias_lang"),embedding->ne[0],1),embedding);
        bias=ggml_cont(m,ggml_transpose(m,ggml_add(m,bias,parameter(m,n,"bias0"))));
        auto visual=run(m,child(n,"dyhead_tower"),features);
        Tensors scores,boxes,centers;
        auto const& scales=child(n,"scales");
        if(visual.size()!=scales.at("order").size())throw std::runtime_error("text fusion level/scale mismatch");
        for(size_t i=0;i<visual.size();++i) {
            auto x=visual[i];
            auto query=ggml_cont(m,ggml_transpose(m,ggml_reshape_2d(m,x,x->ne[0]*x->ne[1],x->ne[2])));
            auto score=ggml_add(m,ggml_div(m,ggml_mul_mat(m,tokens,query),ggml_exp(m,parameter(m,n,"log_scale"))),bias);
            scores.push_back(ggml_clamp(m,score,-50000,50000));
            boxes.push_back(one(m,child(scales,scales.at("order").at(i).get<std::string>()),layer(m,n,"bbox_pred",x)));
            centers.push_back(layer(m,n,"centerness",x));
        }
        scores.insert(scores.end(),boxes.begin(),boxes.end());
        scores.insert(scores.end(),centers.begin(),centers.end());
        return scores;
    }
    throw std::runtime_error("unimplemented text fusion module: "+type);
}
} // namespace visp::mmlab
