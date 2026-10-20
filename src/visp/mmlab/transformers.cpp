#include "config.h"
#include <stdexcept>
#include <cmath>

namespace visp::mmlab {
namespace {
tensor tokens(model_ref m,tensor x) {
    return ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],x->ne[0]*x->ne[1],x->ne[3]);
}
tensor spatial(model_ref m,tensor x,int64_t w,int64_t h) {
    return ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,x,x->ne[0],w,h,x->ne[2]),2,0,1,3));
}
tensor norm_spatial(model_ref m,Json const& n,std::string const& name,tensor x) {
    return spatial(m,layer(m,n,name,tokens(m,x)),x->ne[0],x->ne[1]);
}
int pair(Json const& j,int index){return j.is_array()?j.at(index).get<int>():j.get<int>();}
tensor window_partition(model_ref m,tensor x,int window) {
    auto channels=x->ne[0],width=x->ne[1],height=x->ne[2],batch=x->ne[3];
    x=ggml_reshape_4d(m,x,channels*window,width/window,window,height/window*batch);
    x=ggml_cont(m,ggml_permute(m,x,0,2,1,3));
    return ggml_reshape_3d(m,x,channels,window*window,width/window*height/window*batch);
}
tensor window_reverse(model_ref m,tensor x,int64_t width,int64_t height,int window) {
    auto channels=x->ne[0],batch=x->ne[2]/(width/window)/(height/window);
    x=ggml_reshape_4d(m,x,channels*window,window,width/window,height/window*batch);
    x=ggml_cont(m,ggml_permute(m,x,0,2,1,3));return ggml_reshape_4d(m,x,channels,width,height,batch);
}
}
Tensors transformers(model_ref m,Json const& n,Tensors const& inputs) {
    auto type=n.at("type").get<std::string>();auto const& a=n.at("attrs");auto const& c=n.at("children");auto x=inputs.at(0);
    if(type=="GeneralizedAttention") {
        if(a.at("spatial_range").get<int>()>=0||x->ne[3]!=1)throw std::runtime_error("GeneralizedAttention requires global batch-one attention");
        auto flags=a.at("attention_type").get<std::vector<bool>>();int heads=a.at("num_heads").get<int>(),dim=a.at("qk_embed_dim").get<int>();
        auto query_input=c.contains("q_downsample")?layer(m,n,"q_downsample",x):x,key_input=c.contains("kv_downsample")?layer(m,n,"kv_downsample",x):x;
        auto qw=query_input->ne[0],qh=query_input->ne[1],kw=key_input->ne[0],kh=key_input->ne[1];
        auto split=[&](tensor t){return ggml_cont(m,ggml_permute(m,ggml_reshape_3d(m,t,dim,heads,t->ne[1]),0,2,1,3));};
        tensor query=nullptr,key=nullptr,energy=nullptr;
        if(flags[0]||flags[1])query=split(tokens(m,layer(m,n,"query_conv",query_input)));
        if(flags[0]||flags[2])key=split(tokens(m,layer(m,n,"key_conv",key_input)));
        auto bias=[&](char const* name){return ggml_reshape_3d(m,parameter(m,n,name),dim,1,heads);};
        bool saliency=flags[2]&&!flags[0]&&!flags[1]&&!flags[3];
        if(flags[0]||flags[2]) {
            auto q=flags[0]?query:bias("appr_bias");if(flags[0]&&flags[2])q=ggml_add(m,q,bias("appr_bias"));
            energy=ggml_mul_mat(m,key,q);
        }
        if(flags[1]||flags[3]) {
            int dimensions=a.at("position_embedding_dim").get<int>();
            auto position=[&](int64_t queries,int64_t keys,char const* projection) {
                auto q=ggml_reshape_3d(m,ggml_scale(m,ggml_arange(m,0,float(queries),1),a.at("q_stride").get<float>()),1,1,queries);
                auto k=ggml_reshape_3d(m,ggml_scale(m,ggml_arange(m,0,float(keys),1),a.at("kv_stride").get<float>()),1,keys,1);
                auto shape=ggml_new_tensor_3d(m,GGML_TYPE_F32,1,keys,queries);
                auto difference=ggml_scale(m,ggml_sub(m,ggml_repeat(m,q,shape),k),a.at("position_magnitude").get<float>());
                auto frequency=ggml_exp(m,ggml_scale(m,ggml_arange(m,0,float(dimensions/4),1),-4.f*std::log(1000.f)/dimensions));
                auto full=ggml_new_tensor_3d(m,GGML_TYPE_F32,dimensions/4,keys,queries);
                auto angle=ggml_mul(m,ggml_repeat(m,difference,full),frequency);
                auto embedding=ggml_concat(m,ggml_sin(m,angle),ggml_cos(m,angle),0);
                auto p=ggml_scale(m,layer(m,n,projection,embedding),1.f/std::sqrt(2.f));
                return ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,p,dim,heads,keys,queries),0,2,1,3));
            };
            auto px=position(qw,kw,"appr_geom_fc_x"),py=position(qh,kh,"appr_geom_fc_y");
            auto geometry_query=flags[1]?query:bias("geom_bias");
            if(flags[1]&&flags[3])geometry_query=ggml_add(m,geometry_query,bias("geom_bias"));
            Tensors columns;
            for(int64_t h=0;h<qh;++h)for(int64_t w=0;w<qw;++w) {
                auto q=flags[1]?ggml_view_3d(m,geometry_query,dim,1,heads,geometry_query->nb[1],geometry_query->nb[2],size_t(h*qw+w)*geometry_query->nb[1]):geometry_query;
                auto xposition=ggml_view_3d(m,px,dim,kw,heads,px->nb[1],px->nb[2],size_t(w)*px->nb[3]);
                auto yposition=ggml_view_3d(m,py,dim,kh,heads,py->nb[1],py->nb[2],size_t(h)*py->nb[3]);
                auto ex=ggml_reshape_3d(m,ggml_mul_mat(m,xposition,q),kw,1,heads),ey=ggml_reshape_3d(m,ggml_mul_mat(m,yposition,q),1,kh,heads);
                auto square=ggml_new_tensor_3d(m,GGML_TYPE_F32,kw,kh,heads);
                columns.push_back(ggml_reshape_3d(m,ggml_add(m,ggml_repeat(m,ex,square),ey),kw*kh,1,heads));
            }
            auto geometry=cat(m,columns,1);energy=energy?ggml_add(m,geometry,energy):geometry;
        }
        if(!energy)throw std::runtime_error("GeneralizedAttention requires an attention term");
        auto value=split(tokens(m,layer(m,n,"value_conv",key_input)));
        auto output=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,value)),ggml_soft_max(m,energy));
        output=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,output,0,2,1,3)),dim*heads,saliency?1:qw*qh,1);
        output=layer(m,n,"proj_conv",spatial(m,output,saliency?1:qw,saliency?1:qh));
        if(a.at("q_stride").get<int>()>1)output=resize(m,output,x->ne[0],x->ne[1],true);
        return {ggml_add(m,x,ggml_mul(m,output,parameter(m,n,"gamma")))};
    }
    if(type=="MixFFN")return {ggml_add(m,inputs.size()>1?inputs[1]:x,layer(m,n,"layers",x))};
    if(type=="EfficientMultiheadAttention"||type=="SpatialReductionAttention"||type=="GlobalSubsampledAttention") {
        auto query=tokens(m,x),kv=query;
        if(a.value("sr_ratio",1)>1)kv=layer(m,n,"norm",tokens(m,layer(m,n,"sr",x)));
        auto y=run(m,child(n,"attn"),{query,kv,kv}).at(0);
        return {ggml_add(m,inputs.size()>1?inputs[1]:x,spatial(m,y,x->ne[0],x->ne[1]))};
    }
    if(type=="ConditionalPositionEncoding") {
        auto y=layer(m,n,"proj",x);
        return {a.value("stride",1)==1?ggml_add(m,x,y):y};
    }
    if(type=="GSAEncoderLayer") {
        auto normalized=norm_spatial(m,n,"norm1",x);
        auto zero=ggml_scale(m,x,0);
        x=ggml_add(m,x,run(m,child(n,"attn"),{normalized,zero}).at(0));
        auto y=layer(m,n,"ffn",layer(m,n,"norm2",tokens(m,x)));
        return {ggml_add(m,x,spatial(m,y,x->ne[0],x->ne[1]))};
    }
    if(type=="PCPVT") {
        Tensors out;auto indices=a.at("out_indices").get<std::vector<int>>();
        auto const& stages=child(n,"layers");int i=0;
        for(auto const& key:stages.at("order")) {
            auto index=key.get<std::string>();x=layer(m,child(n,"patch_embeds"),index,x);
            auto const& blocks=child(stages,index);int j=0;
            for(auto const& block:blocks.at("order")) {
                x=layer(m,blocks,block.get<std::string>(),x);
                if(j++==0)x=layer(m,child(n,"position_encodings"),index,x);
            }
            if(a.value("norm_after_stage",false))x=norm_spatial(m,child(n,"norm_list"),index,x);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end())out.push_back(x);++i;
        }return out;
    }
    if(type=="AbsolutePositionEmbedding") {
        auto position=parameter(m,n,"pos_embed");auto h=a.at("pos_shape").at(0).get<int>(),w=a.at("pos_shape").at(1).get<int>();
        position=ggml_cont(m,ggml_view_3d(m,position,position->ne[0],h*w,1,position->nb[1],position->nb[2],size_t(position->ne[1]-h*w)*position->nb[1]));
        position=resize(m,spatial(m,position,w,h),x->ne[0],x->ne[1],true);return {ggml_add(m,x,position)};
    }
    if(type=="TransformerEncoderLayer"||type=="PVTEncoderLayer") {
        auto y=run(m,child(n,"attn"),{norm_spatial(m,n,"norm1",x),x}).at(0);
        return run(m,child(n,"ffn"),{norm_spatial(m,n,"norm2",y),y});
    }
    if(type=="MixVisionTransformer"||type=="PyramidVisionTransformer"||type=="PyramidVisionTransformerV2") {
        auto const& layers=child(n,"layers");auto indices=a.at("out_indices").get<std::vector<int>>();Tensors out;int i=0;
        for(auto const& key:layers.at("order")) {
            auto const& stage=child(layers,key.get<std::string>());x=layer(m,stage,"0",x);x=layer(m,stage,"1",x);x=norm_spatial(m,stage,"2",x);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end())out.push_back(x);++i;
        }return out;
    }
    if(type=="BEiTTransformerEncoderLayer"||type=="MAETransformerEncoderLayer") {
        auto wm=weights(m,n);auto y=layer(m,n,"attn",layer(m,n,a.at("norm1_name").get<std::string>(),x));
        x=ggml_add(m,x,ggml_mul(m,y,wm.weights("gamma_1")));
        y=layer(m,n,"ffn",layer(m,n,a.at("norm2_name").get<std::string>(),x));
        return {ggml_add(m,x,ggml_mul(m,y,wm.weights("gamma_2")))};
    }
    if(type=="BEiT"||type=="MAE") {
        auto feature=layer(m,n,"patch_embed",x);auto width=feature->ne[0],height=feature->ne[1];auto wm=weights(m,n);
        x=ggml_concat(m,wm.weights("cls_token"),tokens(m,feature),1);
        if(type=="MAE")x=ggml_add(m,x,wm.weights("pos_embed"));
        Tensors out;auto indices=a.at("out_indices").get<std::vector<int>>();auto const& layers=child(n,"layers");int i=0;
        for(auto const& key:layers.at("order")) {
            x=layer(m,layers,key.get<std::string>(),x);
            if(i+1==int(layers.at("order").size())&&a.value("final_norm",false))x=layer(m,n,a.at("norm1_name").get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end()) {
                auto t=ggml_cont(m,ggml_view_3d(m,x,x->ne[0],width*height,x->ne[2],x->nb[1],x->nb[2],x->nb[1]));out.push_back(spatial(m,t,width,height));
            }++i;
        }return out;
    }
    if(type=="WindowMSA"||type=="BEiTAttention"||type=="MAEAttention") {
        auto wm=weights(m,n);int heads=a.at("num_heads").get<int>();auto channels=x->ne[0],count=x->ne[1],batch=x->ne[2],dim=channels/heads;
        auto qkv=layer(m,n,"qkv",x);
        bool qv=a.value("bias",Json{})=="qv_bias";
        auto part=[&](int index) {
            auto t=ggml_cont(m,ggml_view_3d(m,qkv,channels,count,batch,qkv->nb[1],qkv->nb[2],size_t(index*channels)*qkv->nb[0]));
            if(qv&&index!=1)t=ggml_add(m,t,wm.weights(index==0?"q_bias":"v_bias"));
            return ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,t,dim,heads,count,batch),0,2,1,3));
        };
        auto q=part(0),k=part(1),v=part(2);
        auto scores=ggml_scale(m,ggml_mul_mat(m,k,q),a.value("scale",1.f/std::sqrt(float(dim))));
        if(n.at("tensors").contains("relative_position_bias_table")) {
            auto indices=parameter(m,n,"relative_position_index");indices=ggml_reshape_1d(m,indices,ggml_nelements(indices));
            if(ggml_nelements(indices)!=count*count)throw std::runtime_error("relative position window must match token count");
            auto bias=ggml_get_rows(m,parameter(m,n,"relative_position_bias_table"),indices);
            bias=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,bias,heads,count,count,1),2,0,1,3));scores=ggml_add(m,scores,bias);
        }
        if(inputs.size()>1&&inputs[1])scores=ggml_add(m,scores,inputs[1]);
        auto y=ggml_mul_mat(m,ggml_cont(m,ggml_transpose(m,v)),ggml_soft_max(m,scores));
        y=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,y,0,2,1,3)),channels,count,batch);
        return {layer(m,n,"proj",y)};
    }
    if(type=="ShiftWindowMSA") {
        int window=a.at("window_size").get<int>(),shift=a.at("shift_size").get<int>();auto width=x->ne[0],height=x->ne[1];
        int pw=(window-width%window)%window,ph=(window-height%window)%window;
        auto y=ggml_cont(m,ggml_permute(m,x,1,2,0,3));if(pw||ph)y=ggml_pad(m,y,0,pw,ph,0);
        auto padded_w=width+pw,padded_h=height+ph;tensor mask=nullptr;
        if(shift) {
            y=ggml_roll(m,y,0,-shift,-shift,0);
            auto labels=[&](int64_t extent) {
                auto index=ggml_arange(m,0,float(extent),1);
                return ggml_add(m,ggml_step(m,ggml_scale_bias(m,index,1,float(-extent+window)+.5f)),
                    ggml_step(m,ggml_scale_bias(m,index,1,float(-extent+shift)+.5f)));
            };
            auto row=ggml_reshape_4d(m,ggml_scale(m,labels(padded_h),3),1,1,padded_h,1);
            auto col=ggml_reshape_4d(m,labels(padded_w),1,padded_w,1,1);
            auto target=ggml_new_tensor_4d(m,GGML_TYPE_F32,1,padded_w,padded_h,1);
            auto grid=ggml_add(m,ggml_repeat(m,col,target),row);
            grid=window_partition(m,grid,window);int64_t count=window*window,windows=grid->ne[2];
            auto lhs=ggml_reshape_4d(m,grid,count,1,1,windows),rhs=ggml_reshape_4d(m,grid,1,count,1,windows);
            auto square=ggml_new_tensor_4d(m,GGML_TYPE_F32,count,count,1,windows);
            mask=ggml_scale(m,ggml_clamp(m,ggml_abs(m,ggml_sub(m,ggml_repeat(m,lhs,square),rhs)),0,1),-100);
        }
        y=window_partition(m,y,window);y=run(m,child(n,"w_msa"),{y,mask}).at(0);
        y=window_reverse(m,y,padded_w,padded_h,window);if(shift)y=ggml_roll(m,y,0,shift,shift,0);
        if(pw||ph)y=ggml_cont(m,ggml_view_4d(m,y,y->ne[0],width,height,y->ne[3],y->nb[1],y->nb[2],y->nb[3],0));
        return {ggml_cont(m,ggml_permute(m,y,2,0,1,3))};
    }
    if(type=="SwinBlock") {
        auto y=layer(m,n,"attn",norm_spatial(m,n,"norm1",x));x=ggml_add(m,x,y);
        auto t=tokens(m,x);auto normalized=layer(m,n,"norm2",t);
        return {spatial(m,run(m,child(n,"ffn"),{normalized,t}).at(0),x->ne[0],x->ne[1])};
    }
    if(type=="PatchMerging") {
        if(c.contains("adap_padding"))x=layer(m,n,"adap_padding",x);
        auto const& sample=child(n,"sampler").at("attrs");
        auto kernel=sample.at("kernel_size"),stride=sample.at("stride"),padding=sample.at("padding"),dilation=sample.at("dilation");
        if(pair(kernel,0)!=2||pair(kernel,1)!=2||pair(stride,0)!=2||pair(stride,1)!=2||pair(padding,0)||pair(padding,1)||pair(dilation,0)!=1||pair(dilation,1)!=1)
            throw std::runtime_error("PatchMerging requires 2x2 stride-two sampling");
        Tensors patches;auto width=x->ne[0]/2,height=x->ne[1]/2,channels=x->ne[2];
        auto channel_first=ggml_cont(m,ggml_permute(m,x,1,2,0,3));
        for(int h=0;h<2;++h)for(int w=0;w<2;++w) {
            auto y=ggml_cont(m,slice(m,channel_first,{}, {w,x->ne[0],2},{h,x->ne[1],2},{}));
            y=ggml_reshape_3d(m,y,channels,width*height,1);
            patches.push_back(ggml_cont(m,ggml_transpose(m,y)));
        }
        auto y=ggml_cont(m,ggml_permute(m,cat(m,patches,2),2,1,0,3));
        y=ggml_reshape_3d(m,y,4*channels,width*height,1);
        if(c.contains("norm"))y=layer(m,n,"norm",y);y=layer(m,n,"reduction",y);
        return {spatial(m,y,width,height)};
    }
    if(type=="SwinBlockSequence") {
        x=layer(m,n,"blocks",x);auto y=c.contains("downsample")?layer(m,n,"downsample",x):x;return {y,x};
    }
    if(type=="SwinTransformer") {
        x=layer(m,n,"patch_embed",x);
        if(a.value("use_abs_pos_embed",false))x=spatial(m,ggml_add(m,tokens(m,x),weights(m,n).weights("absolute_pos_embed")),x->ne[0],x->ne[1]);
        auto const& stages=child(n,"stages");auto indices=a.at("out_indices").get<std::vector<int>>();Tensors out;int i=0;
        for(auto const& key:stages.at("order")) {auto ys=run(m,child(stages,key.get<std::string>()),{x});x=ys.at(0);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end())out.push_back(norm_spatial(m,n,"norm"+std::to_string(i),ys.at(1)));++i;}
        return out;
    }
    throw std::runtime_error("native Transformer protocol unsupported: "+type);
}
}
