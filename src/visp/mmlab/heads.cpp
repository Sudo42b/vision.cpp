#include "config.h"
#include <stdexcept>
#include <cmath>

namespace visp::mmlab {
namespace {
tensor select(Tensors const& inputs, Json const& value) {
    int64_t index=value.get<int64_t>();
    if(index<0)index+=int64_t(inputs.size());
    if(index<0||index>=int64_t(inputs.size()))throw std::runtime_error("feature index out of range");
    return inputs.at(size_t(index));
}
Tensors transform(model_ref m, Json const& a, Tensors const& inputs) {
    auto index=a.at("in_index");
    if(!index.is_array()) return {select(inputs,index)};
    Tensors out;
    for(auto const& i:index) out.push_back(select(inputs,i));
    auto mode=a.value("input_transform",Json{});
    if(mode=="resize_concat") {
        for(size_t i=1;i<out.size();++i) out[i]=resize(m,out[i],out[0]->ne[0],out[0]->ne[1],true,a.value("align_corners",false));
        return {cat(m,out)};
    }
    if(mode!="multiple_select") throw std::runtime_error("unknown input transform");
    return out;
}
}
Tensors heads(model_ref m, Json const& n, Tensors const& inputs) {
    auto type=n.at("type").get<std::string>();
    auto const& a=n.at("attrs"); auto const& c=n.at("children");
    if(type=="CLIPTextEncoder")return {parameter(m,n,"native_class_embeddings")};
    if(type=="MLP") {
        auto x=inputs.at(0);auto const& layers=child(n,"layers");int i=0;
        for(auto const& key:layers.at("order")) {x=layer(m,layers,key.get<std::string>(),x);if(++i<int(layers.at("order").size()))x=ggml_relu(m,x);}
        return {x};
    }
    if(type=="BaseTransformerLayer") {
        if(a.at("operation_order")!=Json::array({"norm","self_attn","norm","ffn"}))throw std::runtime_error("SAN recognition requires pre-norm self-attention");
        auto x=inputs.at(0),normalized=layer(m,child(n,"norms"),"0",x);
        x=run(m,child(child(n,"attentions"),"0"),{normalized,nullptr,x,nullptr,nullptr,nullptr,inputs.size()>1?inputs[1]:nullptr}).at(0);
        return run(m,child(child(n,"ffns"),"0"),{layer(m,child(n,"norms"),"1",x),x});
    }
    if(type=="MLPMaskDecoder") {
        auto query=layer(m,n,"query_mlp",inputs.at(0)),pixel=layer(m,n,"pix_mlp",inputs.at(1));
        auto tokens=[&](tensor x){return ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],x->ne[0]*x->ne[1]);};
        auto mask=ggml_cont(m,ggml_transpose(m,ggml_mul_mat(m,query,tokens(pixel))));
        Tensors out{ggml_reshape_4d(m,mask,pixel->ne[0],pixel->ne[1],query->ne[1],1)};
        auto attention=layer(m,n,"attn_mlp",inputs.at(1));int heads=a.at("total_heads").get<int>(),layers=a.at("total_layers").get<int>();
        for(int l=0;l<layers;++l) {
            Tensors biases;
            for(int h=0;h<heads;++h) {
                auto values=ggml_cont(m,ggml_view_4d(m,attention,attention->ne[0],attention->ne[1],query->ne[0],1,attention->nb[1],attention->nb[2],attention->nb[3],size_t((l*heads+h)*query->ne[0])*attention->nb[2]));
                biases.push_back(ggml_cont(m,ggml_transpose(m,ggml_mul_mat(m,query,tokens(values)))));
            }
            auto bias=cat(m,biases,2);auto shape=i64x4{bias->ne[0],bias->ne[1],bias->ne[2],bias->ne[3]};
            bias=layer(m,n,"bias_scaling",ggml_reshape_2d(m,bias,1,ggml_nelements(bias)));
            bias=ggml_reshape_4d(m,bias,pixel->ne[0],pixel->ne[1],shape[1],shape[2]);out.push_back(bias);
        }return out;
    }
    if(type=="SideAdapterNetwork") {
        auto image=inputs.at(0),feature=layer(m,n,"patch_embed",image);auto width=feature->ne[0],height=feature->ne[1],channels=feature->ne[2];
        auto tokens=[&](tensor x){return ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],x->ne[0]*x->ne[1]);};
        auto position=parameter(m,n,"pos_embed");auto size=child(n,"patch_embed").at("attrs").at("init_out_size");
        if(position->ne[1]!=width*height) {
            position=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,position,channels,size.at(1).get<int>(),size.at(0).get<int>(),1),2,0,1,3));
            position=tokens(interpolate(m,position,{width,height},GGML_SCALE_MODE_BICUBIC));
        }
        position=ggml_concat(m,parameter(m,n,"query_pos_embed"),position,1);
        auto x=ggml_add(m,ggml_concat(m,parameter(m,n,"query_embed"),tokens(feature),1),position);
        auto count=parameter(m,n,"query_embed")->ne[1];int fusion=0;auto indices=a.at("fusion_index").get<std::vector<int>>();
        auto fuse=[&]() {
            auto clip=layer(m,child(n,"conv_clips"),std::to_string(fusion),inputs.at(1+2*fusion));
            clip=tokens(resize(m,clip,width,height,true));
            auto queries=ggml_cont(m,ggml_view_2d(m,x,channels,count,x->nb[1],0));
            auto pixels=ggml_cont(m,ggml_view_2d(m,x,channels,width*height,x->nb[1],size_t(count)*x->nb[1]));
            x=ggml_concat(m,queries,ggml_add(m,pixels,clip),1);++fusion;
        };
        if(indices.at(0)==0)fuse();auto const& layers=child(n,"encode_layers");int i=0;
        for(auto const& key:layers.at("order")) {
            x=layer(m,layers,key.get<std::string>(),x);++i;
            if(i<int(indices.size())&&fusion<int(indices.size())&&i==indices[fusion])fuse();
            if(i<int(layers.at("order").size()))x=ggml_add(m,x,position);
        }
        auto query=ggml_cont(m,ggml_view_2d(m,x,channels,count,x->nb[1],0));
        auto pixel=ggml_cont(m,ggml_view_2d(m,x,channels,width*height,x->nb[1],size_t(count)*x->nb[1]));
        pixel=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,pixel,channels,width,height,1),2,0,1,3));
        return run(m,child(n,"mask_decoder"),{query,pixel});
    }
    if(type=="RecWithAttnbias") {
        if(a.value("cross_attn",false)||a.at("sos_token_format")!="cls_token")throw std::runtime_error("SAN recognition requires self-attention and cls tokens");
        auto feature=inputs.at(0),cls=inputs.at(1);int count=a.at("sos_token_num").get<int>(),heads=a.at("num_heads").get<int>();
        auto width=feature->ne[0],height=feature->ne[1],spatial=width*height,channels=feature->ne[2];
        auto tokens=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,feature,1,2,0,3)),channels,spatial);
        auto sos=ggml_repeat(m,cls,ggml_new_tensor_2d(m,GGML_TYPE_F32,channels,count));
        auto x=ggml_concat(m,sos,ggml_concat(m,cls,tokens,1),1);int i=0;auto const& layers=child(n,"layers");
        for(auto const& key:layers.at("order")) {
            auto bias=adaptive_pool(m,inputs.at(inputs.size()==3?2:2+i),int(width),int(height),true);
            bias=ggml_reshape_3d(m,bias,spatial,count,bias->ne[3]);
            if(bias->ne[2]==1)bias=ggml_repeat(m,bias,ggml_new_tensor_3d(m,GGML_TYPE_F32,spatial,count,heads));
            auto positions=ggml_arange(m,0,float(count),1),left=ggml_reshape_2d(m,positions,count,1),right=ggml_reshape_2d(m,positions,1,count);
            auto square=ggml_new_tensor_2d(m,GGML_TYPE_F32,count,count);
            auto diagonal=ggml_scale(m,ggml_step(m,ggml_abs(m,ggml_sub(m,ggml_repeat(m,left,square),right))),-100);
            auto cls_column=ggml_scale_bias(m,ggml_sum_rows(m,diagonal),0,-100);
            auto top=ggml_concat(m,diagonal,cls_column,0);top=ggml_repeat(m,top,ggml_new_tensor_3d(m,GGML_TYPE_F32,count+1,count,heads));
            top=ggml_concat(m,top,bias,0);
            auto zero=ggml_scale(m,ggml_arange(m,0,float(count+1+spatial),1),0);
            auto lower=ggml_repeat(m,zero,ggml_new_tensor_3d(m,GGML_TYPE_F32,count+1+spatial,spatial+1,heads));
            auto blocked=ggml_scale_bias(m,ggml_repeat(m,ggml_reshape_2d(m,ggml_step(m,ggml_scale_bias(m,ggml_arange(m,0,float(count+1+spatial),1),-1,float(count)-.5f)),count+1+spatial,1),lower),-100,0);
            auto mask=ggml_concat(m,top,blocked,1);
            x=run(m,child(layers,key.get<std::string>()),{x,mask}).at(0);++i;
        }
        x=ggml_cont(m,ggml_view_2d(m,x,channels,count,x->nb[1],0));x=layer(m,n,"proj",layer(m,n,"ln_post",x));
        if(a.value("final_norm",true))x=ggml_div(m,x,ggml_sqrt(m,ggml_sum_rows(m,ggml_sqr(m,x))));return {x};
    }
    if(type=="SideAdapterCLIPHead") {
        auto features=inputs;auto classes=features.back();features.pop_back();auto image=features.front();features.erase(features.begin());
        auto masks_input=features;masks_input.insert(masks_input.begin(),image);auto masks=run(m,child(n,"side_adapter_network"),masks_input);
        Tensors recognition{features.at(features.size()-2),features.back()};recognition.insert(recognition.end(),masks.begin()+1,masks.end());
        auto embedding=run(m,child(n,"rec_with_attnbias"),recognition).at(0);
        auto scores=ggml_soft_max(m,ggml_mul_mat(m,classes,embedding));int64_t count=a.at("num_classes").get<int>();
        scores=ggml_cont(m,ggml_view_2d(m,scores,count,scores->ne[1],scores->nb[1],0));
        auto mask=ggml_sigmoid(m,resize(m,masks.front(),image->ne[0],image->ne[1],true));
        auto spatial=mask->ne[0]*mask->ne[1];mask=ggml_reshape_2d(m,mask,spatial,mask->ne[2]);
        auto result=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,mask)),ggml_cont(m,ggml_transpose(m,scores)));
        return {ggml_reshape_4d(m,result,image->ne[0],image->ne[1],count,1)};
    }
    if(type=="MultiScaleDeformableAttention") {
        auto query=inputs.at(0),position=inputs.at(1),refx=inputs.at(2),refy=inputs.at(3);
        int heads=a.at("num_heads").get<int>(),levels=a.at("num_levels").get<int>(),points=a.at("num_points").get<int>();
        auto shapes=a.at("native_shapes");auto value=layer(m,n,"value_proj",query);auto channels=value->ne[0],count=query->ne[1],dim=channels/heads;
        value=ggml_cont(m,ggml_permute(m,ggml_reshape_3d(m,value,dim,heads,count),0,2,1,3));
        auto q=ggml_add(m,query,position),offsets=layer(m,n,"sampling_offsets",q),attention=layer(m,n,"attention_weights",q);
        attention=ggml_soft_max(m,ggml_reshape_3d(m,attention,points*levels,heads,count));
        auto extract=[&](tensor t,int64_t index,int64_t stride,int64_t row_stride) {
            auto v=ggml_view_3d(m,t,1,heads,count,stride,row_stride,size_t(index)*t->nb[0]);
            return ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,v,0,2,1,3)),count,heads);
        };
        tensor total=nullptr;int64_t begin=0;
        for(int l=0;l<levels;++l) {
            int64_t width=shapes.at(l).at(0).get<int>(),height=shapes.at(l).at(1).get<int>();
            auto features=ggml_cont(m,ggml_view_3d(m,value,dim,width*height,heads,value->nb[1],value->nb[2],size_t(begin)*value->nb[1]));begin+=width*height;
            for(int p=0;p<points;++p) {
                int index=l*points+p;
                auto ox=extract(offsets,2*index,2*levels*points*offsets->nb[0],offsets->nb[1]),oy=extract(offsets,2*index+1,2*levels*points*offsets->nb[0],offsets->nb[1]);
                auto target=ggml_new_tensor_2d(m,GGML_TYPE_F32,count,heads);
                auto xp=ggml_add(m,ox,ggml_repeat(m,ggml_scale_bias(m,refx,float(width),-.5f),target));
                auto yp=ggml_add(m,oy,ggml_repeat(m,ggml_scale_bias(m,refy,float(height),-.5f),target));
                auto sampled=sample_tokens(m,features,width,height,xp,yp);
                auto weight=extract(attention,index,attention->nb[1],attention->nb[2]);
                sampled=ggml_mul(m,sampled,ggml_reshape_3d(m,weight,1,count,heads));total=total?ggml_add(m,total,sampled):sampled;
            }
        }
        total=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,total,0,2,1,3)),channels,count);
        return {ggml_add(m,query,layer(m,n,"output_proj",total))};
    }
    if(type=="Mask2FormerTransformerEncoder") {
        auto x=inputs.at(0);auto const& layers=child(n,"layers");
        for(auto const& key:layers.at("order")) {
            auto node=child(layers,key.get<std::string>());node["attrs"]["native_shapes"]=a.at("native_shapes");
            x=run(m,node,{x,inputs.at(1),inputs.at(2),inputs.at(3)}).at(0);
        }return {x};
    }
    if(type=="DeformableDetrTransformerEncoderLayer") {
        auto attention=child(n,"self_attn");attention["attrs"]["native_shapes"]=a.at("native_shapes");
        auto x=run(m,attention,inputs).at(0);auto const& norms=child(n,"norms");x=layer(m,norms,"0",x);
        return {layer(m,norms,"1",layer(m,n,"ffn",x))};
    }
    if(type=="MSDeformAttnPixelDecoder") {
        int levels=a.at("num_encoder_levels").get<int>();Tensors projected,positions,rx,ry;Json shapes=Json::array();
        auto encoding=parameter(m,child(n,"level_encoding"),"weight");
        for(int i=0;i<levels;++i) {
            auto x=layer(m,child(n,"input_convs"),std::to_string(i),inputs.at(inputs.size()-1-i));auto width=x->ne[0],height=x->ne[1],channels=x->ne[2];
            auto position=layer(m,n,"postional_encoding",x),level=ggml_view_1d(m,encoding,channels,size_t(i)*encoding->nb[1]);
            positions.push_back(ggml_add(m,position,level));shapes.push_back({width,height});
            projected.push_back(ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),channels,width*height));
            auto nx=ggml_reshape_2d(m,ggml_scale_bias(m,ggml_arange(m,0,float(width),1),1.f/width,.5f/width),width,1);
            auto ny=ggml_reshape_2d(m,ggml_scale_bias(m,ggml_arange(m,0,float(height),1),1.f/height,.5f/height),1,height);
            auto shape=ggml_new_tensor_2d(m,GGML_TYPE_F32,width,height);
            rx.push_back(ggml_reshape_1d(m,ggml_repeat(m,nx,shape),width*height));ry.push_back(ggml_reshape_1d(m,ggml_repeat(m,ny,shape),width*height));
        }
        auto encoder=child(n,"encoder");encoder["attrs"]["native_shapes"]=shapes;
        auto memory=run(m,encoder,{cat(m,projected,1),cat(m,positions,1),cat(m,rx,0),cat(m,ry,0)}).at(0);Tensors out;int64_t begin=0;
        for(auto const& shape:shapes) {
            int64_t width=shape.at(0).get<int>(),height=shape.at(1).get<int>();
            auto t=ggml_cont(m,ggml_view_2d(m,memory,memory->ne[0],width*height,memory->nb[1],size_t(begin)*memory->nb[1]));begin+=width*height;
            out.push_back(ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,t,t->ne[0],width,height,1),2,0,1,3)));
        }
        for(int i=int(inputs.size())-levels-1;i>=0;--i) {
            auto lateral=layer(m,child(n,"lateral_convs"),std::to_string(int(inputs.size())-levels-1-i),inputs.at(i));
            out.push_back(layer(m,child(n,"output_convs"),std::to_string(int(inputs.size())-levels-1-i),ggml_add(m,lateral,resize(m,out.back(),lateral->ne[0],lateral->ne[1],true))));
        }
        Tensors result{layer(m,n,"mask_feature",out.back())};for(int i=0;i<a.at("num_outs").get<int>();++i)result.push_back(out.at(i));return result;
    }
    if(type=="PointHead") {
        auto features=inputs;auto previous=features.back();features.pop_back();features=transform(m,a,features);
        auto refined=previous;auto const& cfg=a.at("native_test_cfg");bool align=a.value("align_corners",false);
        if(align)throw std::runtime_error("PointRend requires align_corners=False");
        auto tokens=[&](tensor x){return ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],x->ne[0]*x->ne[1]);};
        for(int iteration=0;iteration<cfg.at("subdivision_steps").get<int>();++iteration) {
            int scale=cfg.at("scale_factor").get<int>();refined=resize(m,refined,refined->ne[0]*scale,refined->ne[1]*scale,true,align);
            auto logits=tokens(refined);auto spatial=refined->ne[0]*refined->ne[1];
            auto top=ggml_cont(m,ggml_argsort_top_k(m,logits,2));
            auto offset=ggml_reshape_2d(m,ggml_scale(m,ggml_arange(m,0,float(spatial),1),float(logits->ne[0])),1,spatial);
            auto indices=ggml_cast(m,ggml_add(m,ggml_cast(m,top,GGML_TYPE_F32),offset),GGML_TYPE_I32);
            auto scores=ggml_get_rows(m,ggml_reshape_2d(m,logits,1,ggml_nelements(logits)),ggml_reshape_1d(m,indices,2*spatial));
            scores=ggml_reshape_2d(m,scores,2,spatial);
            auto first=ggml_view_2d(m,scores,1,spatial,scores->nb[1],0),second=ggml_view_2d(m,scores,1,spatial,scores->nb[1],scores->nb[0]);
            auto uncertainty=ggml_reshape_1d(m,ggml_cont(m,ggml_sub(m,second,first)),spatial);
            int count=int(std::min<int64_t>(spatial,cfg.at("subdivision_num_points").get<int>()));
            auto points=ggml_cont(m,ggml_argsort_top_k(m,uncertainty,count));
            auto linear=ggml_cast(m,points,GGML_TYPE_F32),row=ggml_floor(m,ggml_scale(m,linear,1.f/refined->ne[0]));
            auto col=ggml_sub(m,linear,ggml_scale(m,row,float(refined->ne[0])));
            auto nx=ggml_scale_bias(m,col,1.f/refined->ne[0],.5f/refined->ne[0]),ny=ggml_scale_bias(m,row,1.f/refined->ne[1],.5f/refined->ne[1]);
            auto sample=[&](tensor x) {return sample_tokens(m,tokens(x),x->ne[0],x->ne[1],ggml_scale_bias(m,nx,float(x->ne[0]),-.5f),ggml_scale_bias(m,ny,float(x->ne[1]),-.5f));};
            Tensors fine;for(auto x:features)fine.push_back(sample(x));auto coarse=sample(previous);
            auto x=ggml_cont(m,ggml_transpose(m,cat(m,fine,0)));coarse=ggml_cont(m,ggml_transpose(m,coarse));x=ggml_concat(m,x,coarse,1);
            auto const& layers=child(n,"fcs");
            for(auto const& key:layers.at("order")) {x=layer(m,layers,key.get<std::string>(),x);if(a.value("coarse_pred_each_layer",true))x=ggml_concat(m,x,coarse,1);}
            x=ggml_cont(m,ggml_transpose(m,layer(m,n,"fc_seg",x)));
            auto output=ggml_set_rows(m,logits,x,points);
            refined=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,output,output->ne[0],refined->ne[0],refined->ne[1],1),2,0,1,3));
        }
        return {resize(m,refined,a.at("native_input_w").get<int>(),a.at("native_input_h").get<int>(),true,align)};
    }
    if(type=="PixelDecoder"||type=="TransformerEncoderPixelDecoder") {
        auto memory=inputs.back(),x=memory;
        if(type=="PixelDecoder")x=layer(m,n,"last_feat_conv",x);
        else {
            x=layer(m,n,"encoder_in_proj",x);auto width=x->ne[0],height=x->ne[1],channels=x->ne[2];
            auto position=layer(m,n,"positional_encoding",x);
            x=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),channels,width*height);
            x=run(m,child(n,"encoder"),{x,position}).at(0);
            memory=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,x,channels,width,height,1),2,0,1,3));
            x=layer(m,n,"encoder_out_proj",memory);
        }
        for(int i=int(inputs.size())-2;i>=0;--i) {
            auto lateral=layer(m,child(n,"lateral_convs"),std::to_string(i),inputs.at(i));
            x=layer(m,child(n,"output_convs"),std::to_string(i),ggml_add(m,lateral,resize(m,x,lateral->ne[0],lateral->ne[1],false)));
        }return {layer(m,n,"mask_feature",x),memory};
    }
    if(type=="DetrTransformerDecoderLayer"||type=="Mask2FormerTransformerDecoderLayer") {
        auto x=inputs.at(0);auto const& norms=child(n,"norms");
        auto self=[&](tensor t){return run(m,child(n,"self_attn"),{t,inputs.at(1),t}).at(0);};
        auto cross=[&](tensor t){return run(m,child(n,"cross_attn"),{t,inputs.at(1),t,inputs.at(2),inputs.at(2),inputs.at(3),inputs.size()>4?inputs[4]:nullptr}).at(0);};
        x=layer(m,norms,"0",type=="DetrTransformerDecoderLayer"?self(x):cross(x));
        x=layer(m,norms,"1",type=="DetrTransformerDecoderLayer"?cross(x):self(x));
        return {layer(m,norms,"2",layer(m,n,"ffn",x))};
    }
    if(type=="DetrTransformerDecoder") {
        auto x=inputs.at(0);auto const& layers=child(n,"layers");
        for(auto const& key:layers.at("order"))x=run(m,child(layers,key.get<std::string>()),{x,inputs.at(1),inputs.at(2),inputs.at(3)}).at(0);
        return {layer(m,n,"post_norm",x)};
    }
    if(type=="MaskFormerHead"||type=="Mask2FormerHead") {
        auto decoded=run(m,child(n,"pixel_decoder"),inputs);auto feature=decoded.at(0);
        auto query=parameter(m,child(n,"query_embed"),"weight");tensor output;
        if(type=="MaskFormerHead") {
            auto memory=layer(m,n,"decoder_input_proj",decoded.at(1)),position=layer(m,n,"decoder_pe",decoded.at(1));
            memory=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,memory,1,2,0,3)),memory->ne[2],memory->ne[0]*memory->ne[1]);
            output=run(m,child(n,"transformer_decoder"),{ggml_scale(m,query,0),query,memory,position}).at(0);
        } else {
            auto const& decoder=child(n,"transformer_decoder"),layers=child(decoder,"layers");
            Tensors memories,positions;auto encoding=parameter(m,child(n,"level_embed"),"weight");
            int levels=a.at("num_transformer_feat_level").get<int>();
            for(int i=0;i<levels;++i) {
                auto x=layer(m,child(n,"decoder_input_projs"),std::to_string(i),decoded.at(i+1));
                auto memory=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],x->ne[0]*x->ne[1]);
                memories.push_back(ggml_add(m,memory,ggml_view_1d(m,encoding,encoding->ne[0],size_t(i)*encoding->nb[1])));
                positions.push_back(layer(m,n,"decoder_positional_encoding",decoded.at(i+1)));
            }
            auto state=parameter(m,child(n,"query_feat"),"weight");int i=0;
            auto pixels=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,feature,1,2,0,3)),feature->ne[2],feature->ne[0]*feature->ne[1]);
            for(auto const& key:layers.at("order")) {
                int level=i++%levels;auto embed=layer(m,n,"mask_embed",layer(m,decoder,"post_norm",state));
                auto masks=ggml_cont(m,ggml_transpose(m,ggml_mul_mat(m,embed,pixels)));
                masks=ggml_reshape_4d(m,masks,feature->ne[0],feature->ne[1],embed->ne[1],1);
                auto target=decoded.at(level+1);masks=resize(m,masks,target->ne[0],target->ne[1],true);
                auto blocked=ggml_step(m,ggml_scale(m,ggml_reshape_2d(m,masks,target->ne[0]*target->ne[1],embed->ne[1]),-1));
                auto available=ggml_step(m,ggml_scale_bias(m,ggml_sum_rows(m,blocked),-1,float(blocked->ne[0])-.5f));
                auto full=ggml_new_tensor_2d(m,GGML_TYPE_F32,blocked->ne[0],blocked->ne[1]);
                auto mask=ggml_scale(m,ggml_mul(m,blocked,ggml_repeat(m,available,full)),-1e9f);
                state=run(m,child(layers,key.get<std::string>()),{state,query,memories[level],positions[level],mask}).at(0);
            }output=layer(m,decoder,"post_norm",state);
        }
        auto classes=ggml_soft_max(m,layer(m,n,"cls_embed",output));int64_t count=a.at("num_classes").get<int>();
        classes=ggml_cont(m,ggml_view_2d(m,classes,count,classes->ne[1],classes->nb[1],0));
        auto embed=layer(m,n,"mask_embed",output);
        auto pixels=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,feature,1,2,0,3)),feature->ne[2],feature->ne[0]*feature->ne[1]);
        auto masks=ggml_mul_mat(m,embed,pixels);
        masks=ggml_cont(m,ggml_transpose(m,masks));masks=ggml_reshape_4d(m,masks,feature->ne[0],feature->ne[1],embed->ne[1],1);
        masks=ggml_sigmoid(m,resize(m,masks,a.at("native_input_w").get<int>(),a.at("native_input_h").get<int>(),true));
        auto spatial=masks->ne[0]*masks->ne[1];masks=ggml_reshape_2d(m,masks,spatial,masks->ne[2]);
        auto scores=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,masks)),ggml_cont(m,ggml_transpose(m,classes)));
        return {ggml_reshape_4d(m,scores,a.at("native_input_w").get<int>(),a.at("native_input_h").get<int>(),count,1)};
    }
    if(type=="KernelUpdator") {
        auto update=inputs.at(0),input=inputs.at(1);int width=a.at("in_channels").get<int>();
        auto params=layer(m,n,"dynamic_layer",ggml_reshape_2d(m,update,width,ggml_nelements(update)/width));
        auto features=layer(m,n,"input_layer",ggml_reshape_2d(m,input,width,ggml_nelements(input)/width));
        int64_t channels=a.at("feat_channels").get<int>();
        auto part=[&](tensor t,int index){return ggml_cont(m,ggml_view_3d(m,t,channels,t->ne[1],t->ne[2],t->nb[1],t->nb[2],size_t(index*channels)*t->nb[0]));};
        auto gate=ggml_mul(m,part(features,0),part(params,0));
        if(a.value("gate_norm_act",false))gate=layer(m,n,"activation",layer(m,n,"gate_norm",gate));
        auto input_gate=layer(m,n,"input_norm_in",layer(m,n,"input_gate",gate));
        auto update_gate=layer(m,n,"norm_in",layer(m,n,"update_gate",gate));
        if(a.value("gate_sigmoid",true)){input_gate=ggml_sigmoid(m,input_gate);update_gate=ggml_sigmoid(m,update_gate);}
        auto param_out=layer(m,n,"norm_out",part(params,1)),input_out=layer(m,n,"input_norm_out",part(features,1));
        if(a.value("activate_out",false)){param_out=layer(m,n,"activation",param_out);input_out=layer(m,n,"activation",input_out);}
        auto y=ggml_add(m,ggml_mul(m,update_gate,param_out),ggml_mul(m,input_gate,input_out));
        y=layer(m,n,"activation",layer(m,n,"fc_norm",layer(m,n,"fc_layer",y)));
        return {ggml_reshape_2d(m,y,input->ne[0],input->ne[1])};
    }
    if(type=="KernelUpdateHead") {
        if(a.at("conv_kernel_size")!=1)throw std::runtime_error("K-Net kernel update requires 1x1 kernels");
        auto x=inputs.at(0),proposal=inputs.at(1),mask=inputs.at(2);
        if(child(n,"feat_transform").at("type")!="Identity")x=layer(m,n,"feat_transform",x);
        auto gather=resize(m,mask,x->ne[0],x->ne[1],true);
        auto logits=ggml_cont(m,ggml_permute(m,gather,1,2,0,3));logits=ggml_soft_max(m,logits);
        logits=ggml_reshape_2d(m,ggml_cont(m,ggml_permute(m,logits,2,0,1,3)),x->ne[0]*x->ne[1],mask->ne[2]);
        auto features=ggml_mul_mat(m,ggml_reshape_2d(m,x,x->ne[0]*x->ne[1],x->ne[2]),logits);
        auto object=run(m,child(n,"kernel_update_conv"),{features,proposal}).at(0);
        object=layer(m,n,"attention_norm",layer(m,n,"attention",object));
        if(a.value("with_ffn",true))object=layer(m,n,"ffn_norm",layer(m,n,"ffn",object));
        auto kernel=layer(m,n,"fc_mask",layer(m,n,"mask_fcs",object));
        int scale=a.at("mask_transform_stride").get<int>();
        if(scale==2&&a.at("feat_gather_stride")==1)x=resize(m,x,x->ne[0]/2,x->ne[1]/2,true);
        kernel=ggml_cast(m,ggml_reshape_4d(m,kernel,1,1,kernel->ne[0],kernel->ne[1]),GGML_TYPE_F16);
        auto prediction=ggml_conv_2d(m,kernel,x,1,1,0,0,1,1);
        if(scale==2)prediction=resize(m,prediction,prediction->ne[0]*2,prediction->ne[1]*2,true);
        return {prediction,object};
    }
    if(type=="IterativeDecodeHead") {
        auto head=child(n,"kernel_generate_head");auto classifier=child(head,"conv_seg");
        head["children"]["conv_seg"]={{"type","Identity"},{"prefix",""},{"attrs",Json::object()},{"children",Json::object()}};
        auto feature=run(m,head,inputs).at(0),prediction=one(m,classifier,feature),kernel=parameter(m,classifier,"weight");
        kernel=ggml_reshape_2d(m,kernel,kernel->ne[2],kernel->ne[3]);auto const& updates=child(n,"kernel_update_head");
        for(auto const& key:updates.at("order")) {
            auto result=run(m,child(updates,key.get<std::string>()),{feature,kernel,prediction});prediction=result.at(0);kernel=result.at(1);
        }return {prediction};
    }
    if(type=="DyHead")return run(m,child(n,"dyhead_blocks"),inputs);
    if(type=="DyDCNv2") {
        auto y=run(m,child(n,"conv"),inputs).at(0);return {a.value("with_norm",false)?layer(m,n,"norm",y):y};
    }
    if(type=="DyHeadBlock") {
        Tensors out;
        for(size_t i=0;i<inputs.size();++i) {
            auto x=inputs[i],packed=layer(m,n,"spatial_conv_offset",x);
            auto offset=ggml_cont(m,ggml_view_4d(m,packed,packed->ne[0],packed->ne[1],18,packed->ne[3],packed->nb[1],packed->nb[2],packed->nb[3],0));
            auto mask=ggml_sigmoid(m,ggml_cont(m,ggml_view_4d(m,packed,packed->ne[0],packed->ne[1],9,packed->ne[3],packed->nb[1],packed->nb[2],packed->nb[3],18*packed->nb[2])));
            auto branch=[&](char const* name,tensor input,bool up) {
                auto y=run(m,child(n,name),{input,offset,mask}).at(0);
                if(up)y=resize(m,y,x->ne[0],x->ne[1],true,true);
                return ggml_mul(m,y,layer(m,n,"scale_attn_module",y));
            };
            auto sum=branch("spatial_conv_mid",x,false);int count=1;
            if(i>0){sum=ggml_add(m,sum,branch("spatial_conv_low",inputs[i-1],false));++count;}
            if(i+1<inputs.size()){sum=ggml_add(m,sum,branch("spatial_conv_high",inputs[i+1],true));++count;}
            out.push_back(layer(m,n,"task_attn_module",ggml_scale(m,sum,1.f/count)));
        }return out;
    }
    if(type=="DyReLU") {
        auto x=inputs.at(0),coeff=ggml_scale_bias(m,layer(m,n,"conv2",layer(m,n,"conv1",adaptive_pool(m,x,1,1))),1,-.5f);
        auto channels=x->ne[2];
        auto part=[&](int index){return ggml_cont(m,ggml_view_4d(m,coeff,1,1,channels,x->ne[3],coeff->nb[1],coeff->nb[2],coeff->nb[3],size_t(index*channels)*coeff->nb[2]));};
        auto first=ggml_add(m,ggml_mul(m,x,ggml_scale_bias(m,part(0),2,1)),part(1));
        auto second=ggml_add(m,ggml_mul(m,x,ggml_scale(m,part(2),2)),part(3));
        return {ggml_add(m,second,ggml_relu(m,ggml_sub(m,first,second)))};
    }
    if(type=="NMF2D") {
        auto feature=inputs.at(0);int groups=a.at("S").get<int>(),rank=a.at("R").get<int>();
        if(feature->ne[3]!=1||feature->ne[2]%groups)throw std::runtime_error("NMF requires batch one and divisible channels");
        auto x=ggml_reshape_3d(m,feature,feature->ne[0]*feature->ne[1],feature->ne[2]/groups,groups);
        auto bases=parameter(m,n,"visp_bases");
        if(bases->ne[0]!=rank||bases->ne[1]!=x->ne[1]||bases->ne[2]!=groups)throw std::runtime_error("NMF fixed basis shape mismatch");
        auto transpose=[&](tensor t){return ggml_cont(m,ggml_transpose(m,t));};
        auto numerator_coef=[&](tensor b){return transpose(ggml_mul_mat(m,transpose(x),transpose(b)));};
        auto coefficient=ggml_soft_max(m,ggml_scale(m,numerator_coef(bases),a.at("inv_t").get<float>()));
        auto update_coef=[&]() {
            auto gram=ggml_mul_mat(m,transpose(bases),transpose(bases));
            auto denominator=ggml_mul_mat(m,gram,coefficient);
            coefficient=ggml_div(m,ggml_mul(m,coefficient,numerator_coef(bases)),ggml_scale_bias(m,denominator,1,1e-6f));
        };
        for(int i=0;i<a.at("eval_steps").get<int>();++i) {
            update_coef();
            auto numerator=transpose(ggml_mul_mat(m,x,transpose(coefficient)));
            auto gram=ggml_mul_mat(m,transpose(coefficient),transpose(coefficient));
            auto denominator=ggml_mul_mat(m,gram,bases);
            bases=ggml_div(m,ggml_mul(m,bases,numerator),ggml_scale_bias(m,denominator,1,1e-6f));
        }
        update_coef();auto y=transpose(ggml_mul_mat(m,bases,coefficient));
        return {ggml_reshape_4d(m,y,feature->ne[0],feature->ne[1],feature->ne[2],1)};
    }
    if(type=="Hamburger") {
        auto x=inputs.at(0),y=ggml_relu(m,layer(m,n,"ham_in",x));
        return {ggml_relu(m,ggml_add(m,x,layer(m,n,"ham_out",layer(m,n,"ham",y))))};
    }
    if(type=="LightHamHead") {
        auto features=transform(m,a,inputs);
        for(size_t i=1;i<features.size();++i)features[i]=resize(m,features[i],features[0]->ne[0],features[0]->ne[1],true,a.value("align_corners",false));
        return {layer(m,n,"conv_seg",layer(m,n,"align",layer(m,n,"hamburger",layer(m,n,"squeeze",cat(m,features)))))};
    }
    if(type=="ConcatCell") {
        auto x=layer(m,n,"input1_conv",inputs.at(0)),y=layer(m,n,"input2_conv",inputs.at(1));
        auto target=std::make_pair(x->ne[1],x->ne[0])>std::make_pair(y->ne[1],y->ne[0])?x:y;
        auto scale=[&](tensor z) {
            if(z->ne[0]==target->ne[0]&&z->ne[1]==target->ne[1])return z;
            if(std::make_pair(z->ne[1],z->ne[0])<std::make_pair(target->ne[1],target->ne[0]))
                return resize(m,z,target->ne[0],target->ne[1],a.at("upsample_mode")=="bilinear");
            int kw=int((z->ne[0]+target->ne[0]-1)/target->ne[0]),kh=int((z->ne[1]+target->ne[1]-1)/target->ne[1]);
            int pw=int(target->ne[0]*kw-z->ne[0]),ph=int(target->ne[1]*kh-z->ne[1]);
            if(pw||ph)z=ggml_pad_ext(m,z,pw/2,pw-pw/2,ph/2,ph-ph/2,0,0,0,0);
            return ggml_pool_2d(m,z,GGML_OP_POOL_MAX,kw,kh,kw,kh,0,0);
        };
        x=cat(m,{scale(x),scale(y)});return {a.value("with_out_conv",true)?layer(m,n,"out_conv",x):x};
    }
    if(type=="CARAFEPack") {
        auto x=inputs.at(0);int scale=a.at("scale_factor").get<int>(),kernel=a.at("up_kernel").get<int>(),groups=a.at("up_group").get<int>();
        auto mask=layer(m,n,"content_encoder",layer(m,n,"channel_compressor",x));
        Json shuffle={{"type","PixelShuffle"},{"attrs",{{"upscale_factor",scale}}},{"prefix",""},{"children",Json::object()}};
        mask=one(m,shuffle,mask);auto ow=mask->ne[0],oh=mask->ne[1];int area=kernel*kernel;
        mask=ggml_reshape_4d(m,ggml_cont(m,ggml_permute(m,mask,1,2,0,3)),area,groups,ow*oh,x->ne[3]);
        mask=ggml_soft_max(m,mask);mask=ggml_cont(m,ggml_permute(m,mask,1,2,0,3));
        mask=ggml_reshape_4d(m,mask,ow,oh,area*groups,x->ne[3]);
        if(x->ne[2]%groups)throw std::runtime_error("CARAFE group/channel mismatch");
        int half=kernel/2;auto padded=ggml_pad_ext(m,x,half,half,half,half,0,0,0,0);Tensors out;
        for(int g=0;g<groups;++g) {
            tensor total=nullptr;auto channels=x->ne[2]/groups;
            for(int h=0;h<kernel;++h)for(int w=0;w<kernel;++w) {
                auto feature=ggml_cont(m,ggml_view_4d(m,padded,x->ne[0],x->ne[1],channels,x->ne[3],padded->nb[1],padded->nb[2],padded->nb[3],size_t(w)*padded->nb[0]+size_t(h)*padded->nb[1]+size_t(g*channels)*padded->nb[2]));
                feature=resize(m,feature,ow,oh,false);
                auto weight=ggml_view_4d(m,mask,ow,oh,1,x->ne[3],mask->nb[1],mask->nb[2],mask->nb[3],size_t(g*area+h*kernel+w)*mask->nb[2]);
                auto y=ggml_mul(m,feature,weight);total=total?ggml_add(m,total,y):y;
            }out.push_back(total);
        }return {cat(m,out)};
    }
    if(type=="FPN_CARAFE") {
        Tensors laterals;auto const& convs=child(n,"lateral_convs");int i=0;
        for(auto const& key:convs.at("order")) {
            auto x=i<=a.at("backbone_end_level").get<int>()-a.at("start_level").get<int>()?inputs.at(std::min(i+a.at("start_level").get<int>(),int(inputs.size())-1)):laterals.back();
            laterals.push_back(layer(m,convs,key.get<std::string>(),x));++i;
        }
        for(int level=int(laterals.size())-1;level>0;--level) {
            auto x=laterals[level];if(!a.at("upsample").is_null())x=layer(m,child(n,"upsample_modules"),std::to_string(level-1),x);
            auto target=laterals[level-1];
            if(x->ne[0]<target->ne[0]||x->ne[1]<target->ne[1])throw std::runtime_error("CARAFE top-down output too small");
            x=ggml_cont(m,ggml_view_4d(m,x,target->ne[0],target->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],0));
            laterals[level-1]=ggml_add(m,target,x);
        }
        Tensors out;i=0;auto const& outputs=child(n,"fpn_convs");
        for(auto const& key:outputs.at("order"))out.push_back(layer(m,outputs,key.get<std::string>(),laterals.at(i++)));
        return out;
    }
    if(type=="UpInterpolationConv") {
        auto x=inputs.at(0);int scale=a.at("scale_factor").get<int>();auto align=a.value("align_corners",Json{});
        return {layer(m,n,"conv",resize(m,x,x->ne[0]*scale,x->ne[1]*scale,a.at("mode")=="bilinear",align.is_boolean()&&align.get<bool>()))};
    }
    if(type=="LastConv")return {layer(m,n,"conv_out",inputs.back())};
    if(type=="FPG") {
        Tensors feats;auto const& lateral=child(n,"lateral_convs");int level=a.at("start_level").get<int>();
        for(auto const& key:lateral.at("order"))feats.push_back(layer(m,lateral,key.get<std::string>(),inputs.at(level++)));
        auto const& extras=child(n,"extra_downsamples");
        for(auto const& key:extras.at("order"))feats.push_back(layer(m,extras,key.get<std::string>(),feats.back()));
        std::vector<Tensors> outs{feats};int levels=a.at("num_outs").get<int>();
        for(int i=0;i<a.at("stack_times").get<int>();++i) {
            Tensors next;bool reverse=a.at("paths").at(i)=="td";
            auto const& stage=child(child(n,"fpn_transitions"),std::to_string(i));
            for(int j=0;j<levels;++j) {
                auto skip=a.at("skip_inds").at(j).get<std::vector<int>>();
                if(std::find(skip.begin(),skip.end(),i)!=skip.end()){next.push_back(outs.back().at(j));continue;}
                int lvl=reverse?levels-j-1:j;auto const& trans=child(stage,std::to_string(lvl));tensor y=nullptr;
                auto add=[&](char const* key,tensor x) {
                    auto const& node=child(trans,key);if(node.at("type")=="Identity")return;
                    auto t=one(m,node,x);y=y?ggml_add(m,y,t):t;
                };
                auto same=reverse?"same_down":"same_up";
                if(child(trans,same).at("type")!="Identity")add(same,next.at(next.size()-1));
                add("across_lateral",outs.back().at(lvl));
                if(lvl>0)add("across_up",outs.back().at(lvl-1));
                if(lvl<levels-1)add("across_down",outs.back().at(lvl+1));
                add("across_skip",outs.front().at(lvl));
                if(!y)throw std::runtime_error("FPG empty transition");next.push_back(y);
            }
            if(reverse)std::reverse(next.begin(),next.end());outs.push_back(next);
        }
        Tensors final;auto const& transitions=child(n,"output_transition");
        for(int i=0;i<levels;++i) {Tensors levels;for(auto const& stage:outs)levels.push_back(stage.at(i));final.push_back(run(m,child(transitions,std::to_string(i)),levels).at(0));}
        return final;
    }
    if(type=="NASFCOS_FPN") {
        Tensors feats;auto const& adapts=child(n,"adapt_convs");int i=a.at("start_level").get<int>();
        for(auto const& key:adapts.at("order"))feats.push_back(layer(m,adapts,key.get<std::string>(),inputs.at(i++)));
        auto const& cells=child(n,"fpn");
        for(auto const& key:cells.at("order")) {
            auto name=key.get<std::string>();feats.push_back(run(m,child(cells,name),{feats.at(name.at(1)-'0'),feats.at(name.at(2)-'0')}).at(0));
        }
        Tensors out;int level=1;
        for(int index:{9,8,7}) {
            auto x=feats.at(index);x=ggml_add(m,x,resize(m,feats.at(5),x->ne[0],x->ne[1],true));
            auto target=inputs.at(level++);out.push_back(resize(m,x,target->ne[0],target->ne[1],true));
        }
        auto const& extras=child(n,"extra_downsamples");
        for(auto const& key:extras.at("order"))out.push_back(layer(m,extras,key.get<std::string>(),out.back()));
        return out;
    }
    if(type=="ISAHead") {
        auto x=layer(m,n,"in_conv",transform(m,a,inputs).at(0)),residual=x;
        int64_t lw=a.at("down_factor").at(1).get<int>(),lh=a.at("down_factor").at(0).get<int>(),width=x->ne[0],height=x->ne[1],channels=x->ne[2];
        if(x->ne[3]!=1)throw std::runtime_error("ISA requires batch one");
        int64_t gw=(width+lw-1)/lw,gh=(height+lh-1)/lh,pw=gw*lw-width,ph=gh*lh-height;
        if(pw||ph)x=ggml_pad_ext(m,x,int(pw/2),int(pw-pw/2),int(ph/2),int(ph-ph/2),0,0,0,0);
        x=ggml_reshape_4d(m,x,lw,gw,lh*gh,channels);
        x=ggml_cont(m,ggml_permute(m,x,1,0,2,3));x=ggml_reshape_4d(m,x,gw,lw*lh,gh,channels);
        x=ggml_cont(m,ggml_permute(m,x,0,3,1,2));x=layer(m,n,"global_relation",x);
        x=ggml_reshape_4d(m,x,gw,gh*channels,lw,lh);x=ggml_cont(m,ggml_permute(m,x,2,1,0,3));
        x=ggml_reshape_4d(m,x,lw,gh,channels*gw,lh);x=ggml_cont(m,ggml_permute(m,x,0,3,2,1));
        x=layer(m,n,"local_relation",ggml_reshape_4d(m,x,lw,lh,channels,gw*gh));
        x=ggml_reshape_4d(m,x,lw,lh*channels,gw,gh);x=ggml_cont(m,ggml_permute(m,x,0,2,1,3));
        x=ggml_reshape_4d(m,x,lw*gw,lh,channels,gh);x=ggml_cont(m,ggml_permute(m,x,0,1,3,2));
        x=ggml_reshape_4d(m,x,lw*gw,lh*gh,channels,1);
        if(pw||ph)x=ggml_cont(m,ggml_view_4d(m,x,width,height,channels,1,x->nb[1],x->nb[2],x->nb[3],size_t(pw/2)*x->nb[0]+size_t(ph/2)*x->nb[1]));
        return {layer(m,n,"conv_seg",layer(m,n,"out_conv",cat(m,{x,residual})))};
    }
    if(type=="CrissCrossAttention") {
        auto x=inputs.at(0),query=layer(m,n,"query_conv",x),key=layer(m,n,"key_conv",x),value=layer(m,n,"value_conv",x);
        auto width=x->ne[0],height=x->ne[1];if(x->ne[3]!=1)throw std::runtime_error("criss-cross attention requires batch one");
        auto vertical=[&](tensor y){return ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,y,2,1,0,3)),y->ne[2],height,width);};
        auto horizontal=[&](tensor y){return ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,y,1,2,0,3)),y->ne[2],width,height);};
        auto energy_h=ggml_mul_mat(m,vertical(key),vertical(query));
        auto indices=ggml_arange(m,0,float(height),1),lhs=ggml_reshape_2d(m,indices,height,1),rhs=ggml_reshape_2d(m,indices,1,height);
        auto square=ggml_new_tensor_2d(m,GGML_TYPE_F32,height,height);
        auto diagonal=ggml_scale_bias(m,ggml_step(m,ggml_abs(m,ggml_sub(m,ggml_repeat(m,lhs,square),rhs))),1e9f,-1e9f);
        energy_h=ggml_add(m,energy_h,diagonal);energy_h=ggml_cont(m,ggml_permute(m,energy_h,0,2,1,3));
        auto energy_w=ggml_mul_mat(m,horizontal(key),horizontal(query));
        auto attention=ggml_soft_max(m,cat(m,{energy_h,energy_w},0));
        auto part=[&](int64_t start,int64_t count){return ggml_cont(m,ggml_view_3d(m,attention,count,width,height,attention->nb[1],attention->nb[2],size_t(start)*attention->nb[0]));};
        auto ah=ggml_cont(m,ggml_permute(m,part(0,height),0,2,1,3)),aw=part(height,width);
        auto vh=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,value,2,0,1,3)),height,value->ne[2],width);
        auto vw=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,value,0,2,1,3)),width,value->ne[2],height);
        auto oh=ggml_cont(m,ggml_permute(m,ggml_mul_mat(m,vh,ah),2,1,0,3));
        auto ow=ggml_cont(m,ggml_permute(m,ggml_mul_mat(m,vw,aw),2,0,1,3));
        return {ggml_add(m,x,layer(m,n,"gamma",ggml_add(m,oh,ow)))};
    }
    if(type=="CCHead") {
        auto x=transform(m,a,inputs).at(0);auto const& convs=child(n,"convs");auto y=layer(m,convs,"0",x);
        for(int i=0;i<a.at("recurrence").get<int>();++i)y=layer(m,n,"cca",y);
        y=layer(m,convs,"1",y);if(a.value("concat_input",false))y=layer(m,n,"conv_cat",cat(m,{x,y}));return {layer(m,n,"conv_seg",y)};
    }
    if(type=="SegformerHead") {
        auto xs=transform(m,a,inputs);Tensors out;auto const& convs=child(n,"convs");
        bool bilinear=a.value("interpolate_mode",std::string{"bilinear"})=="bilinear";
        for(size_t i=0;i<xs.size();++i)out.push_back(resize(m,layer(m,convs,std::to_string(i),xs[i]),xs[0]->ne[0],xs[0]->ne[1],bilinear,a.value("align_corners",false)));
        return {layer(m,n,"conv_seg",layer(m,n,"fusion_conv",cat(m,out)))};
    }
    if(type=="BasePIDHead")return {layer(m,n,"act",layer(m,n,"norm",layer(m,n,"conv",inputs.at(0))))};
    if(type=="PIDHead")return {layer(m,n,"conv_seg",layer(m,n,"i_head",inputs.at(0)))};
    if(type=="DDRHead")return {layer(m,n,"conv_seg",layer(m,n,"head",inputs.at(0)))};
    if(type=="DAPPM"||type=="PAPPM") {
        auto x=inputs.at(0);auto const& scales=child(n,"scales");auto base=layer(m,scales,"0",x);Tensors branches;
        bool bilinear=a.value("unsample_mode",std::string{"bilinear"})=="bilinear";
        if(type=="DAPPM")branches.push_back(base);
        for(size_t i=1;i<scales.at("order").size();++i) {
            auto y=resize(m,layer(m,scales,std::to_string(i),x),x->ne[0],x->ne[1],bilinear);
            y=ggml_add(m,y,type=="DAPPM"?branches.back():base);
            if(type=="DAPPM")y=layer(m,child(n,"processes"),std::to_string(i-1),y);branches.push_back(y);
        }
        auto merged=type=="DAPPM"?cat(m,branches):cat(m,{base,layer(m,n,"processes",cat(m,branches))});
        return {ggml_add(m,layer(m,n,"compression",merged),layer(m,n,"shortcut",x))};
    }
    if(type=="ReassembleBlocks") {
        Tensors out;auto mode=a.at("readout_type").get<std::string>();
        if(inputs.size()%2)throw std::runtime_error("DPT requires feature/class-token pairs");
        for(size_t i=0;i<inputs.size()/2;++i) {
            auto x=inputs[2*i],cls=inputs[2*i+1];auto width=x->ne[0],height=x->ne[1],channels=x->ne[2];auto key=std::to_string(i);
            if(mode!="ignore") {
                auto t=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),channels,width*height,x->ne[3]);
                cls=ggml_reshape_3d(m,cls,channels,1,x->ne[3]);
                if(mode=="project")t=layer(m,child(n,"readout_projects"),key,cat(m,{t,ggml_repeat(m,cls,t)},0));
                else if(mode=="add")t=ggml_add(m,t,cls);else throw std::runtime_error("DPT readout unsupported");
                x=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,t,channels,width,height,x->ne[3]),2,0,1,3));
            }
            out.push_back(layer(m,child(n,"resize_layers"),key,layer(m,child(n,"projects"),key,x)));
        }return out;
    }
    if(type=="PreActResidualConvUnit") {auto x=inputs.at(0);return {ggml_add(m,x,layer(m,n,"conv2",layer(m,n,"conv1",x)))};}
    if(type=="FeatureFusionBlock") {
        auto x=inputs.at(0);if(inputs.size()==2) {
            auto residual=inputs[1];if(residual->ne[0]!=x->ne[0]||residual->ne[1]!=x->ne[1])residual=resize(m,residual,x->ne[0],x->ne[1],true);
            x=ggml_add(m,x,layer(m,n,"res_conv_unit1",residual));
        }
        x=layer(m,n,"res_conv_unit2",x);return {layer(m,n,"project",resize(m,x,2*x->ne[0],2*x->ne[1],true,a.value("align_corners",true)))};
    }
    if(type=="DPTHead") {
        Tensors pairs;for(auto const& index:a.at("in_index")){auto i=index.get<size_t>();pairs.push_back(inputs.at(2*i));pairs.push_back(inputs.at(2*i+1));}
        auto xs=run(m,child(n,"reassemble_blocks"),pairs);auto const& convs=child(n,"convs"),fusion=child(n,"fusion_blocks");
        for(size_t i=0;i<xs.size();++i)xs[i]=layer(m,convs,std::to_string(i),xs[i]);
        auto x=layer(m,fusion,"0",xs.back());
        for(size_t i=1;i<xs.size();++i)x=run(m,child(fusion,std::to_string(i)),{x,xs[xs.size()-1-i]}).at(0);
        return {layer(m,n,"conv_seg",layer(m,n,"project",x))};
    }
    if(type=="CascadeFeatureFusion") {
        auto high=inputs.at(1),low=inputs.at(0);low=resize(m,low,high->ne[0],high->ne[1],true,a.value("align_corners",false));
        low=layer(m,n,"conv_low",low);return {ggml_relu(m,ggml_add(m,low,layer(m,n,"conv_high",high))),low};
    }
    if(type=="ICNeck") {
        auto middle=run(m,child(n,"cff_24"),{inputs.at(2),inputs.at(1)});
        auto high=run(m,child(n,"cff_12"),{middle.at(0),inputs.at(0)});return {middle.at(1),high.at(1),high.at(0)};
    }
    if(type=="SETRUPHead") {
        auto x=transform(m,a,inputs).at(0);auto width=x->ne[0],height=x->ne[1],channels=x->ne[2];
        auto t=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),channels,width*height,x->ne[3]);
        t=layer(m,n,"norm",t);x=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,t,channels,width,height,x->ne[3]),2,0,1,3));
        x=layer(m,n,"up_convs",x);return {layer(m,n,"conv_seg",x)};
    }
    if(type=="SegmenterMaskTransformerHead") {
        auto x=transform(m,a,inputs).at(0);auto width=x->ne[0],height=x->ne[1],count=width*height,batch=x->ne[3];
        auto t=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],count,batch);
        t=layer(m,n,"dec_proj",t);auto cls=weights(m,n).weights("cls_emb");int64_t classes=cls->ne[1];
        t=layer(m,n,"decoder_norm",layer(m,n,"layers",ggml_concat(m,t,cls,1)));auto channels=t->ne[0];
        auto patches=ggml_cont(m,ggml_view_3d(m,t,channels,count,batch,t->nb[1],t->nb[2],0));
        cls=ggml_cont(m,ggml_view_3d(m,t,channels,classes,batch,t->nb[1],t->nb[2],size_t(count)*t->nb[1]));
        auto normalize=[&](tensor y){auto norm=ggml_clamp(m,ggml_sqrt(m,ggml_sum_rows(m,ggml_sqr(m,y))),1e-12f,INFINITY);return ggml_div(m,y,norm);};
        patches=normalize(layer(m,n,"patch_proj",patches));cls=normalize(layer(m,n,"classes_proj",cls));
        auto mask=layer(m,n,"mask_norm",ggml_mul_mat(m,cls,patches));
        return {ggml_reshape_4d(m,ggml_cont(m,ggml_transpose(m,mask)),width,height,classes,batch)};
    }
    if(type=="Encoding") {
        auto x=inputs.at(0);auto channels=x->ne[2],spatial=x->ne[0]*x->ne[1];auto wm=weights(m,n);
        auto tokens=ggml_cont(m,ggml_transpose(m,ggml_reshape_3d(m,x,spatial,channels,x->ne[3])));
        auto codes=wm.weights("codewords"),scale=wm.weights("scale");
        auto input_norm=ggml_sum_rows(m,ggml_sqr(m,tokens));
        auto code_norm=ggml_cont(m,ggml_transpose(m,ggml_sum_rows(m,ggml_sqr(m,codes))));
        auto scores=ggml_mul_mat(m,codes,tokens);
        scores=ggml_add(m,ggml_add(m,ggml_scale(m,scores,-2),input_norm),code_norm);
        auto assignment=ggml_soft_max(m,ggml_mul(m,scores,scale));
        assignment=ggml_cont(m,ggml_transpose(m,assignment));
        auto sums=ggml_sum_rows(m,assignment);
        auto aggregate=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,tokens)),assignment);
        auto weighted=ggml_mul(m,codes,sums);
        return {ggml_sub(m,aggregate,weighted)};
    }
    if(type=="EncModule") {
        auto x=inputs.at(0),encoded=layer(m,n,"encoding_project",x);auto const& blocks=child(n,"encoding");
        for(auto const& key:blocks.at("order")) {
            auto const& block=child(blocks,key.get<std::string>());auto kind=block.at("type").get<std::string>();
            if(kind=="SyncBatchNorm"||kind=="BatchNorm1d") {
                auto wm=weights(m,block);auto gamma=wm.weights("weight"),beta=wm.weights("bias");
                encoded=ggml_add(m,ggml_mul(m,encoded,ggml_reshape_3d(m,gamma,1,gamma->ne[0],1)),ggml_reshape_3d(m,beta,1,beta->ne[0],1));
            } else encoded=one(m,block,encoded);
        }
        auto feature=ggml_cont(m,ggml_transpose(m,ggml_mean(m,ggml_cont(m,ggml_transpose(m,encoded)))));
        feature=ggml_reshape_2d(m,feature,x->ne[2],x->ne[3]);
        auto gamma=layer(m,n,"fc",feature);gamma=ggml_reshape_4d(m,gamma,1,1,x->ne[2],x->ne[3]);
        return {feature,ggml_relu(m,ggml_add(m,x,ggml_mul(m,x,gamma)))};
    }
    if(type=="EncHead") {
        auto xs=transform(m,a,inputs);auto x=layer(m,n,"bottleneck",xs.back());
        if(a.value("add_lateral",false)) {
            Tensors branches{x};auto const& lateral=child(n,"lateral_convs");
            for(size_t i=0;i+1<xs.size();++i)branches.push_back(resize(m,layer(m,lateral,std::to_string(i),xs[i]),x->ne[0],x->ne[1],true,a.value("align_corners",false)));
            x=layer(m,n,"fusion",cat(m,branches));
        }
        auto encoded=run(m,child(n,"enc_module"),{x});Tensors out{layer(m,n,"conv_seg",encoded.at(1))};
        if(a.value("use_se_loss",false))out.push_back(layer(m,n,"se_layer",encoded.at(0)));return out;
    }
    if(type=="PPMConcat") {
        auto x=inputs.at(0);Tensors out;
        for(auto const& key:n.at("order")) {auto y=layer(m,n,key.get<std::string>(),x);
            out.push_back(ggml_reshape_4d(m,ggml_cont(m,y),y->ne[0]*y->ne[1],1,y->ne[2],y->ne[3]));}
        return {cat(m,out,0)};
    }
    if(type=="AFNB"||type=="APNB") {
        auto high=inputs.at(type=="AFNB"?1:0),low=inputs.at(0);tensor context=nullptr;
        auto const& stages=child(n,"stages");
        for(auto const& key:stages.at("order")) {auto y=run(m,child(stages,key.get<std::string>()),{high,low}).at(0);context=context?ggml_add(m,context,y):y;}
        return {layer(m,n,"bottleneck",cat(m,{context,high}))};
    }
    if(type=="ANNHead") {
        auto xs=transform(m,a,inputs);auto x=run(m,child(n,"fusion"),xs).at(0);
        return {layer(m,n,"conv_seg",layer(m,n,"context",layer(m,n,"bottleneck",x)))};
    }
    if(type=="ACM") {
        auto original=inputs.at(0);int scale=a.at("pool_scale").get<int>();
        auto pooled=layer(m,n,"pooled_redu_conv",adaptive_pool(m,original,scale,scale));
        auto x=layer(m,n,"input_redu_conv",original);
        auto context=layer(m,n,"global_info",adaptive_pool(m,x,1,1));
        auto affinity=ggml_sigmoid(m,layer(m,n,"gla",ggml_add(m,x,context)));
        affinity=ggml_cont(m,ggml_transpose(m,ggml_reshape_3d(m,affinity,x->ne[0]*x->ne[1],scale*scale,x->ne[3])));
        auto values=ggml_reshape_3d(m,ggml_cont(m,pooled),scale*scale,pooled->ne[2],pooled->ne[3]);
        auto y=ggml_cont(m,ggml_transpose(m,ggml_mul_mat(m,values,affinity)));
        y=ggml_reshape_4d(m,y,x->ne[0],x->ne[1],x->ne[2],x->ne[3]);
        y=ggml_relu(m,ggml_add(m,layer(m,n,"residual_conv",y),x));
        if(a.value("fusion",false))y=layer(m,n,"fusion_conv",y);return {y};
    }
    if(type=="DCM") {
        auto original=inputs.at(0);int size=a.at("filter_size").get<int>();
        auto kernel=layer(m,n,"filter_gen_conv",adaptive_pool(m,original,size,size));
        auto x=layer(m,n,"input_redu_conv",original);
        if(x->ne[3]!=1)throw std::runtime_error("dynamic convolution requires batch one");
        kernel=ggml_reshape_4d(m,ggml_cont(m,kernel),size,size,1,x->ne[2]);
        int pad=(size-1)/2,extra=(size-1)%2;
        x=ggml_pad_ext(m,x,pad+extra,pad,pad+extra,pad,0,0,0,0);
        auto y=conv_2d_wt_dw(m,x,kernel,nullptr,1,0,1);
        if(c.contains("norm"))y=layer(m,n,"norm",y);y=layer(m,n,"activate",y);
        if(a.value("fusion",false))y=layer(m,n,"fusion_conv",y);return {y};
    }
    if(type=="APCHead"||type=="DMHead") {
        auto x=transform(m,a,inputs).at(0);Tensors branches{x};
        auto const& blocks=child(n,type=="APCHead"?"acm_modules":"dcm_modules");
        for(auto const& key:blocks.at("order"))branches.push_back(layer(m,blocks,key.get<std::string>(),x));
        return {layer(m,n,"conv_seg",layer(m,n,"bottleneck",cat(m,branches)))};
    }
    if(type=="PAM") {
        auto base=n;base["type"]="SelfAttentionBlock";auto x=inputs.at(0);
        auto y=heads(m,base,{x,x}).at(0);return {ggml_add(m,x,layer(m,n,"gamma",y))};
    }
    if(type=="CAM") {
        auto x=inputs.at(0);auto features=ggml_reshape_3d(m,ggml_cont(m,x),x->ne[0]*x->ne[1],x->ne[2],x->ne[3]);
        auto energy=ggml_mul_mat(m,features,features);
        auto maximum=ggml_pool_1d(m,energy,GGML_OP_POOL_MAX,int(energy->ne[0]),int(energy->ne[0]),0);
        auto attention=ggml_soft_max(m,ggml_sub(m,ggml_repeat(m,maximum,energy),energy));
        auto y=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,features)),attention);
        y=ggml_reshape_4d(m,y,x->ne[0],x->ne[1],x->ne[2],x->ne[3]);
        return {ggml_add(m,x,layer(m,n,"gamma",y))};
    }
    if(type=="DAHead") {
        auto x=transform(m,a,inputs).at(0);
        auto pam=layer(m,n,"pam_out_conv",layer(m,n,"pam",layer(m,n,"pam_in_conv",x)));
        auto cam=layer(m,n,"cam_out_conv",layer(m,n,"cam",layer(m,n,"cam_in_conv",x)));
        return {layer(m,n,"conv_seg",ggml_add(m,pam,cam)),layer(m,n,"pam_conv_seg",pam),layer(m,n,"cam_conv_seg",cam)};
    }
    if(type=="Feature2Pyramid") {
        Tensors out;bool four=c.contains("upsample_4x")&&child(n,"upsample_4x").at("type")!="Identity";
        std::vector<std::string> ops=four?std::vector<std::string>{"upsample_4x","upsample_2x","identity","downsample_2x"}:
            std::vector<std::string>{"upsample_2x","identity","downsample_2x","downsample_4x"};
        for(size_t i=0;i<inputs.size();++i)out.push_back(layer(m,n,ops.at(i),inputs[i]));return out;
    }
    if(type=="MultiLevelNeck") {
        Tensors xs;auto const& lateral=child(n,"lateral_convs");
        for(size_t i=0;i<inputs.size();++i)xs.push_back(layer(m,lateral,std::to_string(i),inputs[i]));
        int count=a.at("num_outs").get<int>();if(xs.size()==1)xs.resize(count,xs[0]);Tensors out;
        for(int i=0;i<count;++i) {
            float scale=a.at("scales").at(i).get<float>();auto x=xs.at(i);
            x=resize(m,x,int64_t(x->ne[0]*scale),int64_t(x->ne[1]*scale),true);
            out.push_back(layer(m,child(n,"convs"),std::to_string(i),x));
        }
        return out;
    }
    if(type=="JPU") {
        int start=a.at("start_level").get<int>(),end=a.at("backbone_end_level").get<int>();Tensors xs;
        auto const& convs=child(n,"conv_layers");
        for(int i=start;i<end;++i)xs.push_back(layer(m,convs,std::to_string(i-start),inputs.at(i)));
        for(size_t i=1;i<xs.size();++i)xs[i]=resize(m,xs[i],xs[0]->ne[0],xs[0]->ne[1],true,a.value("align_corners",false));
        auto x=cat(m,xs);Tensors branches;auto const& dilations=child(n,"dilation_layers");
        for(auto const& key:dilations.at("order"))branches.push_back(layer(m,dilations,key.get<std::string>(),x));
        Tensors out;for(int i=start;i<end-1;++i)out.push_back(inputs.at(i));out.push_back(cat(m,branches));return out;
    }
    if(type=="SSDNeck") {
        auto out=inputs;if(c.contains("l2_norm"))out[0]=layer(m,n,"l2_norm",out[0]);
        auto x=out.back();auto const& extra=child(n,"extra_layers");
        for(auto const& key:extra.at("order")){x=layer(m,extra,key.get<std::string>(),x);out.push_back(x);}return out;
    }
    if(type=="L2Norm") {
        auto x=inputs.at(0),y=ggml_cont(m,ggml_permute(m,x,1,2,0,3));
        auto norm=ggml_scale_bias(m,ggml_sqrt(m,ggml_sum_rows(m,ggml_sqr(m,y))),1,a.value("eps",1e-10f));
        y=ggml_mul(m,ggml_div(m,y,norm),weights(m,n).weights("weight"));
        return {ggml_cont(m,ggml_permute(m,y,2,0,1,3))};
    }
    if(type=="CTResNetNeck")return {layer(m,n,"deconv_layers",inputs.back())};
    if(type=="BFP") {
        int level=a.at("refine_level").get<int>();auto center=inputs.at(level);tensor sum=nullptr;
        for(size_t i=0;i<inputs.size();++i) {
            auto y=int(i)<level?adaptive_pool(m,inputs[i],int(center->ne[0]),int(center->ne[1]),true):resize(m,inputs[i],center->ne[0],center->ne[1],false);
            sum=sum?ggml_add(m,sum,y):y;
        }
        auto x=ggml_scale(m,sum,1.f/float(inputs.size()));
        if(c.contains("refine"))x=layer(m,n,"refine",x);
        Tensors out;
        for(size_t i=0;i<inputs.size();++i) {
            auto y=int(i)<level?resize(m,x,inputs[i]->ne[0],inputs[i]->ne[1],false):adaptive_pool(m,x,int(inputs[i]->ne[0]),int(inputs[i]->ne[1]),true);
            out.push_back(ggml_add(m,y,inputs[i]));
        }
        return out;
    }
    if(type=="ContextBlock") {
        auto x=inputs.at(0);tensor context;
        if(a.at("pooling_type")=="avg")context=layer(m,n,"avg_pool",x);
        else {
            auto count=x->ne[0]*x->ne[1];auto mask=layer(m,n,"conv_mask",x);
            mask=ggml_soft_max(m,ggml_reshape_3d(m,mask,count,1,x->ne[3]));
            auto features=ggml_reshape_3d(m,ggml_cont(m,x),count,x->ne[2],x->ne[3]);
            context=ggml_reshape_4d(m,ggml_mul_mat(m,mask,features),1,1,x->ne[2],x->ne[3]);
        }
        if(c.contains("channel_mul_conv")&&child(n,"channel_mul_conv").at("type")!="Identity")x=ggml_mul(m,x,ggml_sigmoid(m,layer(m,n,"channel_mul_conv",context)));
        if(c.contains("channel_add_conv")&&child(n,"channel_add_conv").at("type")!="Identity")x=ggml_add(m,x,layer(m,n,"channel_add_conv",context));
        return {x};
    }
    if(type=="NonLocal2d" || type=="DisentangledNonLocal2d") {
        auto x=inputs.at(0);auto spatial=x->ne[0]*x->ne[1];
        auto flat=[&](tensor t){return ggml_reshape_3d(m,ggml_cont(m,t),t->ne[0]*t->ne[1],t->ne[2],t->ne[3]);};
        auto mode=a.value("mode",std::string{"embedded_gaussian"});
        if(mode!="gaussian"&&mode!="embedded_gaussian"&&mode!="dot_product")throw std::runtime_error("nonlocal mode unsupported");
        auto q=flat(mode=="gaussian"?x:layer(m,n,"theta",x));
        auto k=flat(mode=="gaussian"&&!a.value("sub_sample",false)?x:layer(m,n,"phi",x));
        if(type=="DisentangledNonLocal2d") {q=ggml_sub(m,q,ggml_mean(m,q));k=ggml_sub(m,k,ggml_mean(m,k));}
        auto keys=ggml_cont(m,ggml_transpose(m,k)),queries=ggml_cont(m,ggml_transpose(m,q));
        auto score=ggml_mul_mat(m,keys,queries);
        if(mode=="dot_product")score=ggml_scale(m,score,1.f/float(k->ne[0]));
        else {
            float scale=mode=="embedded_gaussian"&&a.value("use_scale",true)?1.f/std::sqrt(float(k->ne[1])):1.f;
            if(type=="DisentangledNonLocal2d")scale/=a.at("temperature").get<float>();
            score=ggml_soft_max(m,ggml_scale(m,score,scale));
        }
        auto values=flat(layer(m,n,"g",x));auto y=ggml_mul_mat(m,values,score);
        y=ggml_reshape_4d(m,ggml_cont(m,ggml_transpose(m,y)),x->ne[0],x->ne[1],values->ne[1],x->ne[3]);
        if(type=="DisentangledNonLocal2d") {
            auto mask=ggml_soft_max(m,ggml_reshape_3d(m,layer(m,n,"conv_mask",x),spatial,1,x->ne[3]));
            auto unary=ggml_reshape_4d(m,ggml_mul_mat(m,mask,values),1,1,values->ne[1],x->ne[3]);
            y=ggml_add(m,y,unary);
        }
        return {ggml_add(m,x,layer(m,n,"conv_out",y))};
    }
    if(type=="NLHead" || type=="DNLHead" || type=="GCHead") {
        auto x=transform(m,a,inputs).at(0);auto const& convs=child(n,"convs");
        auto y=layer(m,convs,"0",x);
        y=layer(m,n,type=="NLHead"?"nl_block":type=="DNLHead"?"dnl_block":"gc_block",y);
        y=layer(m,convs,"1",y);
        if(a.value("concat_input",false))y=layer(m,n,"conv_cat",cat(m,{x,y}));
        return {layer(m,n,"conv_seg",y)};
    }
    if(type=="DetectionBlock") {
        auto x=inputs.at(0);for(int i=1;i<=5;++i)x=layer(m,n,"conv"+std::to_string(i),x);
        return {x};
    }
    if(type=="YOLOV3Neck") {
        auto x=layer(m,n,"detect1",inputs.back());Tensors out{x};
        for(size_t i=1;i<inputs.size();++i) {
            auto tmp=layer(m,n,"conv"+std::to_string(i),x);
            tmp=resize(m,tmp,tmp->ne[0]*2,tmp->ne[1]*2,false);
            x=layer(m,n,"detect"+std::to_string(i+1),cat(m,{tmp,inputs[inputs.size()-1-i]}));out.push_back(x);
        }
        return out;
    }
    if(type=="DilatedEncoder") {
        auto x=layer(m,n,"lateral_norm",layer(m,n,"lateral_conv",inputs.back()));
        x=layer(m,n,"fpn_norm",layer(m,n,"fpn_conv",x));return {layer(m,n,"dilated_encoder_blocks",x)};
    }
    if(type=="HRFPN") {
        Tensors xs{inputs.at(0)};
        for(size_t i=1;i<inputs.size();++i)xs.push_back(resize(m,inputs[i],inputs[i]->ne[0]*(1LL<<i),inputs[i]->ne[1]*(1LL<<i),true));
        auto x=layer(m,n,"reduction_conv",cat(m,xs));Tensors out;
        auto op=a.value("pooling_type",std::string{"AVG"})=="MAX"?GGML_OP_POOL_MAX:GGML_OP_POOL_AVG;
        for(int i=0;i<a.at("num_outs").get<int>();++i) {
            auto y=i?ggml_pool_2d(m,x,op,1<<i,1<<i,1<<i,1<<i,0,0):x;
            out.push_back(layer(m,child(n,"fpn_convs"),std::to_string(i),y));
        }
        return out;
    }
    if(type=="RTMOHead")return run(m,child(n,"head_module"),inputs);
    if(type=="RTMOHeadModule") {
        Tensors cls, boxes, vis, poses;
        for(size_t i=0;i<inputs.size();++i) {
            auto x=inputs[i];auto half=x->ne[2]/2;auto key=std::to_string(i);
            auto channel=[&](int64_t start){return ggml_cont(m,ggml_view_4d(m,x,x->ne[0],x->ne[1],half,x->ne[3],x->nb[1],x->nb[2],x->nb[3],size_t(start)*x->nb[2]));};
            auto cf=layer(m,child(n,"conv_cls"),key,channel(0));
            auto pf=layer(m,child(n,"conv_pose"),key,channel(half));
            cls.push_back(layer(m,child(n,"out_cls"),key,cf));
            boxes.push_back(layer(m,child(n,"out_bbox"),key,pf));
            vis.push_back(layer(m,child(n,"out_kpt_vis"),key,pf));
            poses.push_back(a.value("pose_vec_channels",0)>0?layer(m,child(n,"out_pose"),key,pf):pf);
        }
        cls.insert(cls.end(),boxes.begin(),boxes.end());cls.insert(cls.end(),vis.begin(),vis.end());cls.insert(cls.end(),poses.begin(),poses.end());
        return cls;
    }
    if(type=="CSPRepLayer") {
        auto x=inputs.at(0);auto y=layer(m,n,"bottlenecks",layer(m,n,"conv1",x));
        return {layer(m,n,"conv3",ggml_add(m,y,layer(m,n,"conv2",x)))};
    }
    if(type=="RepVGGBlock") {
        auto x=inputs.at(0);auto y=ggml_add(m,layer(m,n,"branch_3x3",x),layer(m,n,"branch_1x1",x));
        if(c.contains("branch_norm")&&child(n,"branch_norm").at("type")!="Identity")y=ggml_add(m,y,layer(m,n,"branch_norm",x));
        return {layer(m,n,"act",y)};
    }
    if(type=="FFN") {
        auto x=inputs.at(0);auto y=layer(m,n,"layers",x);
        return {a.value("add_identity",true)?ggml_add(m,inputs.size()>1?inputs[1]:x,y):y};
    }
    if(type=="MultiheadAttention") {
        auto x=inputs.at(0);
        if(c.contains("attn")) {
            auto q=inputs.size()>1&&inputs[1]?ggml_add(m,x,inputs[1]):x;
            auto key=inputs.size()>3&&inputs[3]?inputs[3]:x,value=inputs.size()>4&&inputs[4]?inputs[4]:key;
            auto position=inputs.size()>5?inputs[5]:inputs.size()>1?inputs[1]:nullptr;
            if(position)key=ggml_add(m,key,position);
            auto y=run(m,child(n,"attn"),{q,key,value,inputs.size()>6?inputs[6]:nullptr}).at(0);
            return {ggml_add(m,inputs.size()>2?inputs[2]:x,y)};
        }
        auto w=weights(m,n);auto kernel=w.weights("in_proj_weight");auto bias=w.find("in_proj_bias");
        int64_t channels=x->ne[0],heads=a.at("num_heads").get<int>(),dim=channels/heads;
        auto project=[&](tensor t,int index) {
            auto k=ggml_cont(m,ggml_view_2d(m,kernel,channels,channels,kernel->nb[1],size_t(index*channels)*kernel->nb[1]));
            auto y=ggml_mul_mat(m,k,t);
            if(bias)y=ggml_add(m,y,ggml_view_1d(m,bias,channels,size_t(index*channels)*bias->nb[0]));
            y=ggml_reshape_4d(m,y,dim,heads,t->ne[1],t->ne[2]);
            return ggml_cont(m,ggml_permute(m,y,0,2,1,3));
        };
        auto q=project(inputs.at(0),0),k=project(inputs.at(1),1),v=project(inputs.at(2),2);
        auto score=ggml_scale(m,ggml_mul_mat(m,k,q),1.f/std::sqrt(float(dim)));
        if(inputs.size()>3&&inputs[3])score=ggml_add(m,score,inputs[3]);score=ggml_soft_max(m,score);
        auto y=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,v)),score);
        y=ggml_cont(m,ggml_permute(m,y,0,2,1,3));
        y=ggml_reshape_3d(m,y,channels,x->ne[1],x->ne[2]);
        return {layer(m,n,"out_proj",y)};
    }
    if(type=="DetrTransformerEncoder") {
        auto x=inputs.at(0);auto const& layers=child(n,"layers");
        for(auto const& key:layers.at("order"))x=run(m,child(layers,key.get<std::string>()),{x,inputs.at(1)}).at(0);
        return {x};
    }
    if(type=="DetrTransformerEncoderLayer") {
        auto x=run(m,child(n,"self_attn"),inputs).at(0);auto const& norms=child(n,"norms");
        x=layer(m,norms,"0",x);x=layer(m,n,"ffn",x);return {layer(m,norms,"1",x)};
    }
    if(type=="SinePositionalEncoding") {
        if(!a.contains("spatial_dim")) {
            auto x=inputs.at(0);int features=a.at("num_feats").get<int>();
            if(features%2)throw std::runtime_error("sine position encoding requires even features");
            auto freq=ggml_exp(m,ggml_scale(m,ggml_arange(m,0,float(features/2),1),-2.f*std::log(a.value("temperature",10000.f))/features));
            auto encode=[&](int64_t extent,bool horizontal) {
                auto coord=ggml_arange(m,1,float(extent+1),1);
                if(a.value("normalize",false))coord=ggml_scale_bias(m,coord,a.value("scale",2.f*float(M_PI))/float(extent+a.value("eps",1e-6f)),a.value("offset",0.f)*a.value("scale",2.f*float(M_PI))/float(extent+a.value("eps",1e-6f)));
                coord=ggml_reshape_3d(m,coord,1,horizontal?extent:1,horizontal?1:extent);
                auto shape=ggml_new_tensor_3d(m,GGML_TYPE_F32,1,x->ne[0],x->ne[1]);coord=ggml_reshape_2d(m,ggml_repeat(m,coord,shape),1,x->ne[0]*x->ne[1]);
                auto full=ggml_new_tensor_2d(m,GGML_TYPE_F32,features/2,x->ne[0]*x->ne[1]);auto angle=ggml_mul(m,ggml_repeat(m,coord,full),freq);
                auto sin=ggml_reshape_3d(m,ggml_sin(m,angle),1,features/2,angle->ne[1]),cos=ggml_reshape_3d(m,ggml_cos(m,angle),1,features/2,angle->ne[1]);
                return ggml_reshape_2d(m,ggml_concat(m,sin,cos,0),features,angle->ne[1]);
            };return {ggml_concat(m,encode(x->ne[1],false),encode(x->ne[0],true),0)};
        }
        if(a.at("spatial_dim")!=2)throw std::runtime_error("positional encoding protocol unsupported");
        auto x=inputs.at(0);auto dim=weights(m,n).weights("dim_t");
        auto gx=ggml_reshape_2d(m,ggml_arange(m,0,float(x->ne[0]),1),x->ne[0],1);
        auto gy=ggml_reshape_2d(m,ggml_arange(m,0,float(x->ne[1]),1),1,x->ne[1]);
        auto shape=ggml_new_tensor_2d(m,GGML_TYPE_F32,x->ne[0],x->ne[1]);
        auto encode=[&](tensor grid) {
            auto p=ggml_reshape_2d(m,ggml_repeat(m,grid,shape),1,x->ne[0]*x->ne[1]);
            auto full=ggml_new_tensor_2d(m,GGML_TYPE_F32,dim->ne[0],x->ne[0]*x->ne[1]);
            auto frequency=ggml_div(m,ggml_repeat(m,p,full),dim);
            return ggml_concat(m,ggml_cos(m,frequency),ggml_sin(m,frequency),0);
        };
        // HybridEncoder transposes [token, feature, h/w] before flattening:
        // each token contains all h features followed by all w features.
        return {ggml_concat(m,encode(gy),encode(gx),0)};
    }
    if(type=="HybridEncoder") {
        Tensors features;auto const& proj=child(n,"input_proj");
        for(size_t i=0;i<inputs.size();++i)features.push_back(layer(m,proj,std::to_string(i),inputs[i]));
        if(a.value("num_encoder_layers",0)>0) {
            auto const& encoder=child(n,"encoder");int i=0;
            for(auto const& index:a.at("use_encoder_idx")) {
                auto x=features.at(index.get<size_t>());auto position=layer(m,n,"sincos_pos_enc",x);
                auto tokens=ggml_cont(m,ggml_transpose(m,ggml_reshape_3d(m,x,x->ne[0]*x->ne[1],x->ne[2],x->ne[3])));
                tokens=run(m,child(encoder,std::to_string(i++)),{tokens,position}).at(0);
                features[index.get<size_t>()]=ggml_reshape_4d(m,ggml_cont(m,ggml_transpose(m,tokens)),x->ne[0],x->ne[1],x->ne[2],x->ne[3]);
            }
        }
        Tensors inner{features.back()};
        for(size_t i=features.size()-1;i>0;--i) {
            auto key=std::to_string(features.size()-1-i);auto high=layer(m,child(n,"lateral_convs"),key,inner.front());inner.front()=high;
            auto up=resize(m,high,high->ne[0]*2,high->ne[1]*2,false);
            inner.insert(inner.begin(),layer(m,child(n,"fpn_blocks"),key,cat(m,{up,features[i-1]})));
        }
        Tensors out{inner.front()};
        for(size_t i=0;i+1<inner.size();++i) {
            auto key=std::to_string(i);auto down=layer(m,child(n,"downsample_convs"),key,out.back());
            out.push_back(layer(m,child(n,"pan_blocks"),key,cat(m,{down,inner[i+1]})));
        }
        auto indices=a.value("output_indices",Json{});
        if(!indices.is_null()) {Tensors selected;for(auto const& i:indices)selected.push_back(out.at(i.get<size_t>()));out=selected;}
        if(c.contains("projector"))out=run(m,child(n,"projector"),out);
        return out;
    }
    if(type=="FPNHead") {
        auto features=transform(m,a,inputs);auto const& scales=child(n,"scale_heads");
        auto y=layer(m,scales,"0",features[0]);
        for(size_t i=1;i<features.size();++i) {
            auto z=layer(m,scales,std::to_string(i),features[i]);
            y=ggml_add(m,y,resize(m,z,y->ne[0],y->ne[1],true,a.value("align_corners",false)));
        }
        return {layer(m,n,"conv_seg",y)};
    }
    if(type=="EMAModule") {
        auto x=inputs.at(0);auto spatial=x->ne[0]*x->ne[1];
        auto features=ggml_reshape_3d(m,ggml_cont(m,x),spatial,x->ne[2],x->ne[3]);
        auto vectors=ggml_cont(m,ggml_transpose(m,features));
        auto bases=weights(m,n).weights("bases");tensor attention=nullptr;
        for(int i=0;i<a.at("num_stages").get<int>();++i) {
            attention=ggml_soft_max(m,ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,bases)),vectors));
            auto normalized=ggml_cont(m,ggml_transpose(m,attention));
            auto divisor=ggml_clamp(m,ggml_sum_rows(m,ggml_abs(m,normalized)),1e-12f,INFINITY);
            normalized=ggml_div(m,normalized,divisor);
            auto next=ggml_mul_mat(m,features,normalized);
            auto length=ggml_sqrt(m,ggml_sum_rows(m,ggml_sqr(m,next)));
            next=ggml_div(m,next,ggml_clamp(m,length,1e-12f,INFINITY));
            bases=ggml_cont(m,ggml_transpose(m,next));
        }
        auto y=ggml_mul_mat(m,bases,attention);
        y=ggml_cont(m,ggml_transpose(m,y));
        return {ggml_reshape_4d(m,y,x->ne[0],x->ne[1],x->ne[2],x->ne[3])};
    }
    if(type=="EMAHead") {
        auto x=transform(m,a,inputs).at(0);auto identity=layer(m,n,"ema_in_conv",x);
        auto y=layer(m,n,"ema_module",layer(m,n,"ema_mid_conv",identity));
        y=layer(m,n,"ema_out_conv",ggml_relu(m,y));
        y=layer(m,n,"bottleneck",ggml_relu(m,ggml_add(m,identity,y)));
        if(a.value("concat_input",false))y=layer(m,n,"conv_cat",cat(m,{x,y}));
        return {layer(m,n,"conv_seg",y)};
    }
    if(type=="SpatialGatherModule") {
        auto x=inputs.at(0),probs=inputs.at(1);auto spatial=x->ne[0]*x->ne[1];
        probs=ggml_soft_max(m,ggml_scale(m,ggml_reshape_3d(m,ggml_cont(m,probs),spatial,probs->ne[2],probs->ne[3]),a.value("scale",1.f)));
        auto features=ggml_reshape_3d(m,ggml_cont(m,x),spatial,x->ne[2],x->ne[3]);
        auto y=ggml_mul_mat(m,probs,features);
        return {ggml_reshape_4d(m,y,y->ne[0],1,y->ne[1],y->ne[2])};
    }
    if(type=="SelfAttentionBlock" || type=="ObjectAttentionBlock") {
        auto original=inputs.at(0), key_input=inputs.size()>1?inputs.at(1):original;
        auto q=layer(m,n,"query_project",original), k=layer(m,n,"key_project",key_input), v=layer(m,n,"value_project",key_input);
        if(c.contains("query_downsample"))q=layer(m,n,"query_downsample",q);
        if(c.contains("key_downsample")) {k=layer(m,n,"key_downsample",k);v=layer(m,n,"key_downsample",v);}
        auto qw=q->ne[0],qh=q->ne[1];
        auto tokens=[&](tensor t){return ggml_cont(m,ggml_transpose(m,ggml_reshape_3d(m,ggml_cont(m,t),t->ne[0]*t->ne[1],t->ne[2],t->ne[3])));};
        auto scores=ggml_mul_mat(m,tokens(k),tokens(q));
        if(a.value("matmul_norm",false))scores=ggml_scale(m,scores,1.f/std::sqrt(float(q->ne[2])));
        auto values=ggml_reshape_3d(m,ggml_cont(m,v),v->ne[0]*v->ne[1],v->ne[2],v->ne[3]);
        auto y=ggml_cont(m,ggml_transpose(m,ggml_mul_mat(m,values,ggml_soft_max(m,scores))));
        y=ggml_reshape_4d(m,y,qw,qh,v->ne[2],v->ne[3]);
        if(c.contains("out_project"))y=layer(m,n,"out_project",y);
        if(c.contains("output_project"))y=layer(m,n,"output_project",y);
        if(type=="ObjectAttentionBlock")y=layer(m,n,"bottleneck",cat(m,{y,original}));
        return {y};
    }
    if(type=="OCRHead") {
        auto features=inputs;auto previous=features.back();features.pop_back();
        auto x=layer(m,n,"bottleneck",transform(m,a,features).at(0));
        auto context=run(m,child(n,"spatial_gather_module"),{x,previous}).at(0);
        auto y=run(m,child(n,"object_context_block"),{x,context}).at(0);
        return {layer(m,n,"conv_seg",y)};
    }
    if(type=="LRASPPHead") {
        auto features=transform(m,a,inputs);auto x=features.back();bool align=a.value("align_corners",false);
        auto pooled=layer(m,n,"image_pool",x);
        x=ggml_mul(m,layer(m,n,"aspp_conv",x),resize(m,pooled,x->ne[0],x->ne[1],true,align));
        x=layer(m,n,"conv_up_input",x);
        for(int i=int(features.size())-2;i>=0;--i) {
            x=resize(m,x,features[i]->ne[0],features[i]->ne[1],true,align);
            x=layer(m,child(n,"conv_ups"),"conv_up"+std::to_string(i),cat(m,{x,layer(m,child(n,"convs"),"conv"+std::to_string(i),features[i])}));
        }
        return {layer(m,n,"conv_seg",x)};
    }
    if(type=="ChannelMapper") {
        Tensors out;auto const& convs=child(n,"convs");
        for(size_t i=0;i<inputs.size();++i)out.push_back(layer(m,convs,std::to_string(i),inputs[i]));
        if(c.contains("extra_convs")) {
            auto const& extra=child(n,"extra_convs");
            for(size_t i=0;i<extra.at("order").size();++i)
                out.push_back(layer(m,extra,std::to_string(i),i==0?inputs.back():out.back()));
        }
        return out;
    }
    if(type=="DEKRHead") {
        auto x=inputs.back();
        return {layer(m,n,"heatmap_conv_layers",x),layer(m,n,"displacement_conv_layers",x)};
    }
    if(type=="AdaptiveActivationBlock") {
        auto x=inputs.at(0), affine=layer(m,n,"transform_matrix_conv",x);
        auto spatial=x->ne[0]*x->ne[1];int groups=a.at("groups").get<int>();
        affine=ggml_cont(m,ggml_permute(m,affine,1,2,0,3));
        affine=ggml_reshape_4d(m,affine,3,2,spatial*groups,x->ne[3]);
        auto regular=ggml_cont(m,ggml_transpose(m,weights(m,n).weights("regular_matrix")));
        auto offset=ggml_mul_mat(m,regular,affine);
        offset=ggml_cont(m,ggml_permute(m,offset,1,0,2,3));
        offset=ggml_reshape_4d(m,offset,18*groups,x->ne[0],x->ne[1],x->ne[3]);
        offset=ggml_cont(m,ggml_permute(m,offset,2,0,1,3));
        auto y=run(m,child(n,"adapt_conv"),{x,offset}).at(0);
        return {layer(m,n,"act",ggml_add(m,layer(m,n,"norm",y),x))};
    }
    if(type=="FeatureMapProcessor") {
        Tensors out;
        auto idx=a.value("select_index",Json{});
        if(idx.is_null())out=inputs;
        else if(idx.is_array())for(auto const& i:idx)out.push_back(select(inputs,i));
        else out.push_back(select(inputs,idx));
        bool align=a.value("align_corners",false);
        if(a.value("concat",false)) {
            for(size_t i=1;i<out.size();++i)out[i]=resize(m,out[i],out[0]->ne[0],out[0]->ne[1],true,align);
            out={cat(m,out)};
        }
        float scale=a.value("scale_factor",1.f);
        for(auto& x:out) {
            if(a.value("apply_relu",false))x=ggml_relu(m,x);
            if(scale!=1.f)x=resize(m,x,int64_t(x->ne[0]*scale),int64_t(x->ne[1]*scale),true,align);
        }
        return out;
    }
    if(type=="GlobalAveragePooling") {
        Tensors out;
        for(auto x:inputs) {
            auto y=layer(m,n,"gap",x);
            // MMDet ReID may use fixed-kernel pooling with spatial output
            // larger than 1x1. Its forward flattens all non-batch dimensions.
            out.push_back(ggml_reshape_2d(m,ggml_cont(m,y),y->ne[0]*y->ne[1]*y->ne[2],y->ne[3]));
        }
        return out;
    }
    if(type=="Heatmap1DHead") {
        auto x=ggml_soft_max(m,layer(m,n,"fc",inputs[0]));
        auto coords=ggml_arange(m,0,float(a.at("heatmap_size").get<int>()),1);
        return {ggml_sum_rows(m,ggml_mul(m,x,coords))};
    }
    if(type=="MultilabelClassificationHead")return {layer(m,n,"fc",inputs[0])};
    if(type=="InternetHead") {
        auto x=inputs.back();
        auto heat=cat(m,{layer(m,n,"right_hand_head",x),layer(m,n,"left_hand_head",x)},3);
        x=layer(m,n,"neck",x);
        return {heat,layer(m,n,"root_head",x),layer(m,n,"hand_type_head",x)};
    }
    if(type=="CPMHead") {
        Tensors out;
        auto const& dec=child(n,"multi_deconv_layers"), final=child(n,"multi_final_layers");
        for(size_t i=0;i<inputs.size();++i) {
            auto key=std::to_string(i);
            out.push_back(layer(m,final,key,layer(m,dec,key,inputs[i])));
        }
        return out;
    }
    if(type=="CSPNeXtPAFPN" || type=="YOLOXPAFPN") {
        Tensors inner{inputs.back()};
        auto const& reduce=child(n,"reduce_layers"), top=child(n,"top_down_blocks"), down=child(n,"downsamples"), bottom=child(n,"bottom_up_blocks");
        for(size_t i=inputs.size()-1;i>0;--i) {
            auto key=std::to_string(inputs.size()-1-i);
            auto high=layer(m,reduce,key,inner.front());inner.front()=high;
            auto up=layer(m,n,"upsample",high);
            inner.insert(inner.begin(),layer(m,top,key,cat(m,{up,inputs[i-1]})));
        }
        Tensors out{inner.front()};
        for(size_t i=0;i+1<inner.size();++i) {
            auto key=std::to_string(i);
            out.push_back(layer(m,bottom,key,cat(m,{layer(m,down,key,out.back()),inner[i+1]})));
        }
        Tensors result;
        auto indices=a.value("out_indices",Json::array({0,1,2}));
        for(auto const& v:indices) {
            auto i=v.get<size_t>();
            result.push_back(c.contains("out_convs")?layer(m,child(n,"out_convs"),std::to_string(i),out.at(i)):out.at(i));
        }
        return result;
    }
    if(type=="Scale") return {ggml_mul(m,inputs[0],weights(m,n).weights("scale"))};
    if(type=="ScaleNorm") {
        auto x=inputs[0];
        auto norm=ggml_sqrt(m,ggml_sum_rows(m,ggml_sqr(m,x)));
        norm=ggml_clamp(m,ggml_scale(m,norm,a.at("scale").get<float>()),a.at("eps").get<float>(),INFINITY);
        return {ggml_mul(m,ggml_div(m,x,norm),weights(m,n).weights("g"))};
    }
    if(type=="RTMCCBlock") {
        if(a.value("attn_type",std::string{"self-attn"})!="self-attn" || a.value("pos_enc",false) || a.value("use_rel_bias",false))
            throw std::runtime_error("RTMCC positional/cross attention not implemented");
        auto x=inputs[0];auto w=weights(m,n);
        auto uv=layer(m,n,"act_fn",layer(m,n,"uv",layer(m,n,"ln",x)));
        int e=a.at("e").get<int>(), s=a.at("s").get<int>();
        auto part=[&](int start,int size) {return ggml_cont(m,ggml_view_2d(m,uv,size,uv->ne[1],uv->nb[1],size_t(start)*uv->nb[0]));};
        auto u=part(0,e), v=part(e,e), base=part(2*e,s);
        auto gamma=w.weights("gamma"), beta=w.weights("beta");
        auto affine=[&](int i) {
            auto g=ggml_view_1d(m,gamma,s,size_t(i)*gamma->nb[1]);
            auto b=ggml_view_1d(m,beta,s,size_t(i)*beta->nb[1]);
            return ggml_add(m,ggml_mul(m,base,g),b);
        };
        auto q=affine(0), k=affine(1);
        auto scores=ggml_sqr(m,ggml_relu(m,ggml_scale(m,ggml_mul_mat(m,k,q),1.f/a.at("sqrt_s").get<float>())));
        auto values=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,v)),scores);
        auto y=layer(m,n,"o",ggml_mul(m,u,values));
        if(a.value("shortcut",false)) y=ggml_add(m,layer(m,n,"res_scale",x),y);
        return {y};
    }
    if(type=="RTMCCHead") {
        auto x=layer(m,n,"final_layer",inputs.back());
        x=ggml_reshape_3d(m,ggml_cont(m,x),x->ne[0]*x->ne[1],x->ne[2],x->ne[3]);
        x=layer(m,n,"gau",layer(m,n,"mlp",x));
        return {layer(m,n,"cls_x",x),layer(m,n,"cls_y",x)};
    }
    if(type=="RTMWHead") {
        auto flatten=[](model_ref mm,tensor x){return ggml_reshape_3d(mm,ggml_cont(mm,x),x->ne[0]*x->ne[1],x->ne[2],x->ne[3]);};
        auto top=layer(m,n,"mlp",flatten(m,layer(m,n,"final_layer",inputs.at(1))));
        auto bottom=cat(m,{layer(m,n,"conv_dec",layer(m,n,"ps",inputs.at(1))),inputs.at(0)});
        bottom=layer(m,n,"mlp2",flatten(m,layer(m,n,"final_layer2",bottom)));
        auto x=layer(m,n,"gau",cat(m,{top,bottom},0));
        return {layer(m,n,"cls_x",x),layer(m,n,"cls_y",x)};
    }
    if(type=="TemporalRegressionHead") {
        auto x=layer(m,n,"conv",inputs.back());
        int joints=a.at("num_joints").get<int>();
        return {ggml_reshape_3d(m,ggml_cont(m,x),3,joints,ggml_nelements(x)/(3*joints))};
    }
    if(type=="FPN" || type=="PAFPN") {
        int start=a.value("start_level",0);
        auto const& lateral=child(n,"lateral_convs");
        auto const& fpn=child(n,"fpn_convs");
        Tensors xs;
        for(size_t i=0;i<lateral.at("order").size();++i)
            xs.push_back(layer(m,lateral,lateral.at("order").at(i).get<std::string>(),inputs.at(start+i)));
        auto up=a.value("upsample_cfg",Json::object());
        bool bilinear=up.value("mode",std::string{"nearest"})=="bilinear";
        for(size_t i=xs.size()-1;i>0;--i) {
            xs[i-1]=ggml_add(m,xs[i-1],resize(m,xs[i],xs[i-1]->ne[0],xs[i-1]->ne[1],bilinear,up.value("align_corners",false)));
        }
        Tensors out;
        for(size_t i=0;i<xs.size();++i) out.push_back(layer(m,fpn,fpn.at("order").at(i).get<std::string>(),xs[i]));
        if(type=="PAFPN") {
            for(size_t i=0;i+1<out.size();++i)out[i+1]=ggml_add(m,out[i+1],layer(m,child(n,"downsample_convs"),std::to_string(i),out[i]));
            for(size_t i=1;i<out.size();++i)out[i]=layer(m,child(n,"pafpn_convs"),std::to_string(i-1),out[i]);
        }
        auto extra=a.value("add_extra_convs",Json(false));
        int count=a.at("num_outs").get<int>();
        if(extra==false || extra.is_null()) {
            while(int(out.size())<count) out.push_back(ggml_pool_2d(m,out.back(),GGML_OP_POOL_MAX,1,1,2,2,0,0));
        } else {
            tensor x;
            if(extra=="on_input") x=inputs.at(a.at("backbone_end_level").get<size_t>()-1);
            else if(extra=="on_lateral") x=xs.back();
            else if(extra=="on_output") x=out.back();
            else throw std::runtime_error("unknown extra conv source");
            for(size_t i=out.size();int(i)<count;++i) {
                if(i>xs.size() && a.value("relu_before_extra_convs",false)) x=ggml_relu(m,x);
                x=layer(m,fpn,fpn.at("order").at(i).get<std::string>(),x);out.push_back(x);
            }
        }
        return out;
    }
    if(type=="PPM" || type=="ASPPModule" || type=="DepthwiseSeparableASPPModule") {
        Tensors out;
        for(auto const& key:n.at("order")) {
            tensor y=layer(m,n,key.get<std::string>(),inputs.at(0));
            out.push_back(type=="PPM"?resize(m,y,inputs[0]->ne[0],inputs[0]->ne[1],true,a.value("align_corners",false)):y);
        }
        return out;
    }
    if(type=="FCNHead" || type=="DepthwiseSeparableFCNHead") {
        auto x=transform(m,a,inputs).at(0);
        auto y=layer(m,n,"convs",x);
        if(a.value("concat_input",false)) y=layer(m,n,"conv_cat",cat(m,{x,y}));
        return {layer(m,n,"conv_seg",y)};
    }
    if(type=="PSPHead") {
        auto x=transform(m,a,inputs).at(0);
        auto ys=run(m,child(n,"psp_modules"),{x});
        ys.insert(ys.begin(),x);
        x=layer(m,n,"bottleneck",cat(m,ys));
        return {layer(m,n,"conv_seg",x)};
    }
    if(type=="ASPPHead" || type=="DepthwiseSeparableASPPHead") {
        auto x=transform(m,a,inputs).at(0);
        auto ys=run(m,child(n,"aspp_modules"),{x});
        auto pooled=layer(m,n,"image_pool",x);
        ys.insert(ys.begin(),resize(m,pooled,x->ne[0],x->ne[1],true,a.value("align_corners",false)));
        x=layer(m,n,"bottleneck",cat(m,ys));
        if(type=="DepthwiseSeparableASPPHead") {
            if(c.contains("c1_bottleneck")) {
                auto c1=layer(m,n,"c1_bottleneck",inputs[0]);
                x=cat(m,{resize(m,x,c1->ne[0],c1->ne[1],true,a.value("align_corners",false)),c1});
            }
            x=layer(m,n,"sep_bottleneck",x);
        }
        return {layer(m,n,"conv_seg",x)};
    }
    if(type=="UPerHead") {
        auto xs=transform(m,a,inputs);
        auto ys=run(m,child(n,"psp_modules"),{xs.back()});
        ys.insert(ys.begin(),xs.back());
        auto psp=layer(m,n,"bottleneck",cat(m,ys));
        Tensors lateral;
        auto const& lc=child(n,"lateral_convs"), fc=child(n,"fpn_convs");
        for(size_t i=0;i<lc.at("order").size();++i)
            lateral.push_back(layer(m,lc,lc.at("order")[i].get<std::string>(),xs.at(i)));
        lateral.push_back(psp);
        bool align=a.value("align_corners",false);
        for(size_t i=lateral.size()-1;i>0;--i)
            lateral[i-1]=ggml_add(m,lateral[i-1],resize(m,lateral[i],lateral[i-1]->ne[0],lateral[i-1]->ne[1],true,align));
        Tensors out;
        for(size_t i=0;i<fc.at("order").size();++i)
            out.push_back(layer(m,fc,fc.at("order")[i].get<std::string>(),lateral.at(i)));
        out.push_back(lateral.back());
        for(size_t i=1;i<out.size();++i)out[i]=resize(m,out[i],out[0]->ne[0],out[0]->ne[1],true,align);
        auto x=layer(m,n,"fpn_bottleneck",cat(m,out));
        return {layer(m,n,"conv_seg",x)};
    }
    if(type=="HeatmapHead" || type=="Heatmap3DHead" || type=="AssociativeEmbeddingHead") {
        tensor x=inputs.back();
        for(auto const& key:{"deconv_layers","conv_layers","final_layer"})
            if(c.contains(key)) x=layer(m,n,key,x);
        if(type=="Heatmap3DHead") {
            int depth=a.at("depth_size").get<int>();
            return {ggml_reshape_4d(m,x,x->ne[0],x->ne[1],depth,x->ne[2]/depth*x->ne[3])};
        }
        if(type=="AssociativeEmbeddingHead") {
            int channels=a.at("num_keypoints").get<int>();
            auto part=[&](int start,int count){return ggml_cont(m,ggml_view_4d(m,x,x->ne[0],x->ne[1],count,x->ne[3],x->nb[1],x->nb[2],x->nb[3],size_t(start)*x->nb[2]));};
            return {part(0,channels),part(channels,int(x->ne[2])-channels)};
        }
        return {x};
    }
    throw std::runtime_error("missing neck/head forward");
}
} // namespace visp::mmlab
