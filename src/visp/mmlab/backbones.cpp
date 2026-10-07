#include "config.h"
#include <stdexcept>

namespace visp::mmlab {
Tensors backbones(model_ref m, Json const& n, Tensors const& inputs) {
    auto type=n.at("type").get<std::string>();
    auto const& a=n.at("attrs");
    auto const& c=n.at("children");
    tensor x=inputs.at(0);
    if(type=="StemConv")return run(m,child(n,"proj"),{x});
    if(type=="OverlapPatchEmbed")return {layer(m,n,"norm",layer(m,n,"proj",x))};
    if(type=="MSCAAttention") {
        auto base=layer(m,n,"conv0",x),attention=base;
        for(int i=0;i<3;++i) {
            auto key="conv"+std::to_string(i);attention=ggml_add(m,attention,layer(m,n,key+"_2",layer(m,n,key+"_1",base)));
        }
        return {ggml_mul(m,x,layer(m,n,"conv3",attention))};
    }
    if(type=="MSCASpatialAttention")return {ggml_add(m,x,layer(m,n,"proj_2",layer(m,n,"spatial_gating_unit",layer(m,n,"activation",layer(m,n,"proj_1",x)))))};
    if(type=="MSCABlock") {
        auto scale=[&](tensor y,char const* key){auto g=parameter(m,n,key);return ggml_mul(m,y,ggml_reshape_4d(m,g,1,1,g->ne[0],1));};
        x=ggml_add(m,x,scale(layer(m,n,"attn",layer(m,n,"norm1",x)),"layer_scale_1"));
        return {ggml_add(m,x,scale(layer(m,n,"mlp",layer(m,n,"norm2",x)),"layer_scale_2"))};
    }
    if(type=="MSCAN") {
        Tensors out;
        for(int i=1;i<=a.at("num_stages").get<int>();++i) {
            auto key=std::to_string(i);x=layer(m,n,"block"+key,layer(m,n,"patch_embed"+key,x));
            auto t=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),x->ne[2],x->ne[0]*x->ne[1],x->ne[3]);
            t=layer(m,n,"norm"+key,t);
            x=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,t,t->ne[0],x->ne[0],x->ne[1],x->ne[3]),2,0,1,3));out.push_back(x);
        }return out;
    }
    if(type=="EfficientNet") {
        Tensors out;auto indices=a.at("out_indices").get<std::vector<int>>();auto const& blocks=child(n,"layers");int i=0;
        for(auto const& key:blocks.at("order")) {x=layer(m,blocks,key.get<std::string>(),x);if(std::find(indices.begin(),indices.end(),i)!=indices.end())out.push_back(x);++i;}return out;
    }
    if(type=="EdgeResidual") {
        auto y=layer(m,n,"conv1",x);if(a.value("with_se",false))y=layer(m,n,"se",y);y=layer(m,n,"conv2",y);
        return {a.value("with_residual",false)?ggml_add(m,x,y):y};
    }
    if(type=="DDRNet") {
        auto width=x->ne[0]/8,height=x->ne[1]/8;bool align=a.value("align_corners",false);x=layer(m,n,"stem",x);
        auto const& context=child(n,"context_branch_layers"),spatial=child(n,"spatial_branch_layers");tensor low=x,high=x;
        for(int i=0;i<2;++i) {
            auto key=std::to_string(i),suffix=std::to_string(i+1);
            low=layer(m,context,key,i?ggml_relu(m,low):low);high=layer(m,spatial,key,i?ggml_relu(m,high):high);
            auto compressed=layer(m,n,"compression_"+suffix,ggml_relu(m,low));
            low=ggml_add(m,low,layer(m,n,"down_"+suffix,ggml_relu(m,high)));
            high=ggml_add(m,high,resize(m,compressed,width,height,true,align));
        }
        high=layer(m,spatial,"2",ggml_relu(m,high));low=layer(m,n,"spp",layer(m,context,"2",ggml_relu(m,low)));
        return {ggml_add(m,high,resize(m,low,width,height,true,align))};
    }
    if(type=="PagFM") {
        auto high=x,low=inputs.at(1);if(a.value("after_relu",false)){high=ggml_relu(m,high);low=ggml_relu(m,low);}
        bool bilinear=a.value("upsample_mode",std::string{"bilinear"})=="bilinear";
        auto product=ggml_mul(m,layer(m,n,"f_p",high),resize(m,layer(m,n,"f_i",low),high->ne[0],high->ne[1],bilinear));tensor sigma;
        if(a.value("with_channel",false))sigma=layer(m,n,"up",product);
        else {auto t=ggml_cont(m,ggml_permute(m,product,1,2,0,3));sigma=ggml_cont(m,ggml_permute(m,ggml_sum_rows(m,t),2,0,1,3));}
        sigma=ggml_sigmoid(m,sigma);low=resize(m,low,high->ne[0],high->ne[1],bilinear);
        return {ggml_add(m,ggml_mul(m,low,sigma),ggml_mul(m,high,ggml_scale_bias(m,sigma,-1,1)))};
    }
    if(type=="LightBag"||type=="Bag") {
        auto high=x,low=inputs.at(1),sigma=ggml_sigmoid(m,inputs.at(2));
        if(type=="Bag")return {layer(m,n,"conv",ggml_add(m,ggml_mul(m,high,sigma),ggml_mul(m,low,ggml_scale_bias(m,sigma,-1,1))))};
        return {ggml_add(m,layer(m,n,"f_p",ggml_add(m,high,ggml_mul(m,low,ggml_scale_bias(m,sigma,-1,1)))),layer(m,n,"f_i",ggml_add(m,low,ggml_mul(m,high,sigma))))};
    }
    if(type=="PIDNet") {
        auto width=x->ne[0]/8,height=x->ne[1]/8;bool align=a.value("align_corners",false);x=layer(m,n,"stem",x);
        auto const& ib=child(n,"i_branch_layers"),pb=child(n,"p_branch_layers"),db=child(n,"d_branch_layers");tensor low=x,high=x,detail=x;
        for(int i=0;i<2;++i) {
            auto key=std::to_string(i),suffix=std::to_string(i+1);low=ggml_relu(m,layer(m,ib,key,low));
            high=layer(m,pb,key,i?ggml_relu(m,high):high);detail=layer(m,db,key,i?ggml_relu(m,detail):detail);
            high=run(m,child(n,"pag_"+suffix),{high,layer(m,n,"compression_"+suffix,low)}).at(0);
            detail=ggml_add(m,detail,resize(m,layer(m,n,"diff_"+suffix,low),width,height,true,align));
        }
        high=layer(m,pb,"2",ggml_relu(m,high));detail=layer(m,db,"2",ggml_relu(m,detail));low=layer(m,n,"spp",layer(m,ib,"2",low));
        return run(m,child(n,"dfm"),{high,resize(m,low,width,height,true,align),detail});
    }
    if(type=="ICNet") {
        Tensors out{layer(m,n,"conv_sub1",x)};bool align=a.value("align_corners",false);auto const& backbone=child(n,"backbone");
        auto half=[&](tensor t) {
            auto width=t->ne[0]/2,height=t->ne[1]/2;
            if(!align&&(t->ne[0]%2||t->ne[1]%2))t=ggml_cont(m,ggml_view_4d(m,t,2*width,2*height,t->ne[2],t->ne[3],t->nb[1],t->nb[2],t->nb[3],0));
            return resize(m,t,width,height,true,align);
        };
        x=half(x);
        x=layer(m,backbone,"layer2",layer(m,backbone,"layer1",layer(m,backbone,"maxpool",layer(m,backbone,"stem",x))));out.push_back(layer(m,n,"conv_sub2",x));
        x=half(x);x=layer(m,backbone,"layer4",layer(m,backbone,"layer3",x));
        auto pooled=run(m,child(n,"psp_modules"),{x});pooled.push_back(x);
        out.push_back(layer(m,n,"conv_sub4",layer(m,n,"psp_bottleneck",cat(m,pooled))));return out;
    }
    if(type=="DetailBranch")return run(m,child(n,"detail_branch"),{x});
    if(type=="StemBlock") {x=layer(m,n,"conv_first",x);return {layer(m,n,"fuse_last",cat(m,{layer(m,n,"convs",x),layer(m,n,"pool",x)}))};}
    if(type=="GELayer")return {layer(m,n,"act",ggml_add(m,layer(m,n,"conv2",layer(m,n,"dwconv",layer(m,n,"conv1",x))),c.contains("shortcut")?layer(m,n,"shortcut",x):x))};
    if(type=="CEBlock")return {layer(m,n,"conv_last",ggml_add(m,x,layer(m,n,"conv_gap",layer(m,n,"gap",x))))};
    if(type=="SemanticBranch") {Tensors out;for(auto const& name:a.at("semantic_stages")){x=layer(m,n,name.get<std::string>(),x);out.push_back(x);}return out;}
    if(type=="BGALayer") {
        auto detail=layer(m,n,"detail_dwconv",x),down=layer(m,n,"detail_down",x);
        auto semantic=layer(m,n,"semantic_conv",inputs.at(1)),sd=layer(m,n,"semantic_dwconv",inputs.at(1));bool align=a.value("align_corners",false);
        auto first=ggml_mul(m,detail,ggml_sigmoid(m,resize(m,semantic,detail->ne[0],detail->ne[1],true,align)));
        auto second=ggml_mul(m,down,ggml_sigmoid(m,sd));
        return {layer(m,n,"conv",ggml_add(m,first,resize(m,second,first->ne[0],first->ne[1],true,align)))};
    }
    if(type=="BiSeNetV2") {
        auto xs=run(m,child(n,"semantic"),{x});auto y=run(m,child(n,"bga"),{layer(m,n,"detail",x),xs.back()}).at(0);
        Tensors all{y},out;all.insert(all.end(),xs.begin(),xs.end()-1);
        for(auto const& index:a.at("out_indices"))out.push_back(all.at(index.get<size_t>()));return out;
    }
    if(type=="STDCModule") {
        auto original=x;auto const& blocks=child(n,"layers");Tensors out;bool add=a.at("fusion_type")=="add",down=a.value("with_downsample",false);int i=0;
        for(auto const& key:blocks.at("order")) {
            if(i==1&&down&&!add)x=layer(m,n,"downsample",x);
            x=layer(m,blocks,key.get<std::string>(),x);out.push_back(x);++i;
        }
        if(down&&!add)out[0]=layer(m,n,"skip",out[0]);
        auto y=cat(m,out);if(add)y=ggml_add(m,y,down?layer(m,n,"skip",original):original);return {y};
    }
    if(type=="STDCNet") {
        Tensors out;auto const& stages=child(n,"stages");
        for(auto const& key:stages.at("order")){x=layer(m,stages,key.get<std::string>(),x);out.push_back(x);}
        if(a.value("with_final_conv",false))out.back()=layer(m,n,"final_conv",out.back());
        out.erase(out.begin(),out.begin()+a.at("num_shallow_features").get<int>());return out;
    }
    if(type=="STDCContextPathNet") {
        auto xs=run(m,child(n,"backbone"),{x});auto up=layer(m,n,"conv_avg",adaptive_pool(m,xs.back(),1,1));Tensors arms;
        auto const& blocks=child(n,"arms");bool bilinear=a.value("upsample_mode",std::string{"nearest"})=="bilinear",align=a.contains("align_corners")&&!a.at("align_corners").is_null()&&a.at("align_corners").get<bool>();int i=0;
        for(auto const& key:blocks.at("order")) {
            auto source=xs.at(xs.size()-1-i),target=xs.at(xs.size()-2-i);
            up=resize(m,up,source->ne[0],source->ne[1],bilinear,align);
            up=ggml_add(m,layer(m,blocks,key.get<std::string>(),source),up);
            up=layer(m,child(n,"convs"),key.get<std::string>(),resize(m,up,target->ne[0],target->ne[1],bilinear,align));arms.push_back(up);++i;
        }
        auto fused=run(m,child(n,"ffm"),{xs[0],arms.at(1)}).at(0);Tensors out{xs[0]};out.insert(out.end(),arms.begin(),arms.end());out.push_back(fused);return out;
    }
    if(type=="LayerNorm2d") {
        auto y=ggml_cont(m,ggml_permute(m,x,1,2,0,3));
        y=ggml_norm(m,y,a.value("eps",1e-6f));auto wm=weights(m,n);
        if(auto weight=wm.find("weight"))y=ggml_mul(m,y,weight);
        if(auto bias=wm.find("bias"))y=ggml_add(m,y,bias);
        return {ggml_cont(m,ggml_permute(m,y,2,0,1,3))};
    }
    if(type=="ConvNeXtBlock") {
        auto y=layer(m,n,"depthwise_conv",x);bool linear=a.value("linear_pw_conv",true);
        if(linear) {
            y=ggml_cont(m,ggml_permute(m,y,1,2,0,3));auto wm=weights(m,child(n,"norm"));
            y=ggml_norm(m,y,child(n,"norm").at("attrs").value("eps",1e-6f));
            if(auto weight=wm.find("weight"))y=ggml_mul(m,y,weight);
            if(auto bias=wm.find("bias"))y=ggml_add(m,y,bias);
        } else y=layer(m,n,"norm",y);
        y=layer(m,n,"act",layer(m,n,"pointwise_conv1",y));
        if(c.contains("grn")&&child(n,"grn").at("type")!="Identity")throw std::runtime_error("ConvNeXt GRN not implemented");
        y=layer(m,n,"pointwise_conv2",y);
        if(linear)y=ggml_cont(m,ggml_permute(m,y,2,0,1,3));
        if(auto gamma=weights(m,n).find("gamma"))y=ggml_mul(m,y,ggml_reshape_4d(m,gamma,1,1,gamma->ne[0],1));
        return {ggml_add(m,x,y)};
    }
    if(type=="ConvNeXt") {
        Tensors out;auto indices=a.at("out_indices").get<std::vector<int>>();
        auto const& stages=child(n,"stages");int i=0;
        for(auto const& key:stages.at("order")) {
            auto name=key.get<std::string>();x=layer(m,stages,name,layer(m,child(n,"downsample_layers"),name,x));
            if(std::find(indices.begin(),indices.end(),i)!=indices.end()) {
                auto y=a.value("gap_before_final_norm",false)?adaptive_pool(m,x,1,1):x;
                y=layer(m,n,"norm"+std::to_string(i),y);
                if(a.value("gap_before_final_norm",false))y=ggml_reshape_2d(m,y,y->ne[2],y->ne[3]);out.push_back(y);
            }++i;
        }return out;
    }
    if(type=="Pooling")return {ggml_sub(m,layer(m,n,"pool",x),x)};
    if(type=="Mlp") {
        x=layer(m,n,"fc1",x);if(c.contains("dwconv"))x=layer(m,n,"dwconv",x);
        return {layer(m,n,"drop",layer(m,n,"fc2",layer(m,n,"drop",layer(m,n,"act",x))))};
    }
    if(type=="PoolFormerBlock") {
        auto wm=weights(m,n);auto scale=[&](tensor y,char const* key){auto g=wm.weights(key);return ggml_mul(m,y,ggml_reshape_4d(m,g,1,1,g->ne[0],1));};
        x=ggml_add(m,x,scale(layer(m,n,"token_mixer",layer(m,n,"norm1",x)),"layer_scale_1"));
        return {ggml_add(m,x,scale(layer(m,n,"mlp",layer(m,n,"norm2",x)),"layer_scale_2"))};
    }
    if(type=="PoolFormer") {
        x=layer(m,n,"patch_embed",x);Tensors out;auto const& network=child(n,"network");
        auto indices=a.at("out_indices").get<std::vector<int>>();int i=0;
        for(auto const& key:network.at("order")) {x=layer(m,network,key.get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end())out.push_back(layer(m,n,"norm"+std::to_string(i),x));++i;}
        return out;
    }
    if(type=="AdaptivePadding") {
        auto k=a.at("kernel_size"),s=a.at("stride"),d=a.at("dilation");
        auto ph=std::max<int64_t>(0,((x->ne[1]+s.at(0).get<int>()-1)/s.at(0).get<int>()-1)*s.at(0).get<int>()+(k.at(0).get<int>()-1)*d.at(0).get<int>()+1-x->ne[1]);
        auto pw=std::max<int64_t>(0,((x->ne[0]+s.at(1).get<int>()-1)/s.at(1).get<int>()-1)*s.at(1).get<int>()+(k.at(1).get<int>()-1)*d.at(1).get<int>()+1-x->ne[0]);
        bool same=a.at("padding")=="same";int lw=same?int(pw/2):0,lh=same?int(ph/2):0;
        if(pw||ph)x=ggml_pad_ext(m,x,lw,int(pw)-lw,lh,int(ph)-lh,0,0,0,0);return {x};
    }
    if(type=="PatchEmbed") {
        if(c.contains("proj"))return {layer(m,n,"norm",layer(m,n,"proj",x))};
        if(c.contains("adap_padding"))x=layer(m,n,"adap_padding",x);
        x=layer(m,n,"projection",x);
        if(c.contains("norm")) {
            auto w=x->ne[0],h=x->ne[1],channels=x->ne[2],batch=x->ne[3];
            auto tokens=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,x,1,2,0,3)),channels,w*h,batch);
            tokens=layer(m,n,"norm",tokens);
            x=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,tokens,channels,w,h,batch),2,0,1,3));
        }
        return {x};
    }
    if(type=="TransformerEncoderLayer") {
        if(child(n,"attn").at("type")=="EfficientMultiheadAttention")return transformers(m,n,inputs);
        auto normalized=layer(m,n,a.at("norm1_name").get<std::string>(),x);
        x=run(m,child(n,"attn"),{normalized,nullptr,x}).at(0);
        normalized=layer(m,n,a.at("norm2_name").get<std::string>(),x);
        return run(m,child(n,"ffn"),{normalized,x});
    }
    if(type=="VisionTransformer") {
        auto features=layer(m,n,"patch_embed",x);auto w=features->ne[0],h=features->ne[1],channels=features->ne[2];
        auto tokens=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,features,1,2,0,3)),channels,w*h,features->ne[3]);
        auto wm=weights(m,n);tokens=ggml_concat(m,wm.weights("cls_token"),tokens,1);
        auto position=wm.weights("pos_embed");
        if(position->ne[1]!=tokens->ne[1]) {
            int ph=a.at("img_size").at(0).get<int>()/a.at("patch_size").get<int>();
            int pw=a.at("img_size").at(1).get<int>()/a.at("patch_size").get<int>();
            auto cls=ggml_cont(m,ggml_view_2d(m,position,channels,1,position->nb[1],0));
            auto grid=ggml_cont(m,ggml_view_2d(m,position,channels,pw*ph,position->nb[1],position->nb[1]));
            grid=ggml_cont(m,ggml_permute(m,ggml_reshape_4d(m,grid,channels,pw,ph,1),2,0,1,3));
            auto mode=a.value("interpolate_mode",std::string{"bicubic"});
            int interpolation=mode=="bicubic"?GGML_SCALE_MODE_BICUBIC:mode=="bilinear"?GGML_SCALE_MODE_BILINEAR:GGML_SCALE_MODE_NEAREST;
            grid=interpolate(m,grid,{w,h},interpolation);
            grid=ggml_reshape_3d(m,ggml_cont(m,ggml_permute(m,grid,1,2,0,3)),channels,w*h,1);
            position=ggml_concat(m,cls,grid,1);
        }
        tokens=ggml_add(m,tokens,position);bool cls=a.value("with_cls_token",true);
        auto strip=[&](tensor t){return ggml_cont(m,ggml_view_3d(m,t,channels,w*h,t->ne[2],t->nb[1],t->nb[2],cls?t->nb[1]:0));};
        if(!cls)tokens=ggml_cont(m,ggml_view_3d(m,tokens,channels,w*h,tokens->ne[2],tokens->nb[1],tokens->nb[2],tokens->nb[1]));
        if(a.value("pre_norm",false))tokens=layer(m,n,a.at("pre_ln_name").get<std::string>(),tokens);
        Tensors out;
        auto append=[&](tensor t) {
            auto spatial=ggml_cont(m,ggml_transpose(m,strip(t)));
            out.push_back(ggml_reshape_4d(m,spatial,w,h,channels,t->ne[2]));
            if(a.value("output_cls_token",false))out.push_back(ggml_cont(m,ggml_view_2d(m,t,channels,1,t->nb[1],0)));
        };
        if(a.value("out_origin",false))append(tokens);
        auto indices=a.at("out_indices").get<std::vector<int>>();auto const& layers=child(n,"layers");int i=0;
        for(auto const& key:layers.at("order")) {
            tokens=layer(m,layers,key.get<std::string>(),tokens);
            if(i+1==int(layers.at("order").size())&&a.value("final_norm",false))tokens=layer(m,n,a.at("norm1_name").get<std::string>(),tokens);
            if(std::find(indices.begin(),indices.end(),i++)!=indices.end())append(tokens);
        }
        return out;
    }
    if(type=="SpatialPath") {
        for(auto const& name:a.at("layers"))x=layer(m,n,name.get<std::string>(),x);return {x};
    }
    if(type=="AttentionRefinementModule") {
        x=layer(m,n,"conv_layer",x);return {ggml_mul(m,x,layer(m,n,"atten_conv_layer",x))};
    }
    if(type=="ContextPath") {
        auto features=run(m,child(n,"backbone"),{x});
        auto gap=layer(m,n,"gap_conv",features.at(3));
        auto low=ggml_add(m,layer(m,n,"arm32",features.at(3)),gap);
        low=layer(m,n,"conv_head32",resize(m,low,features[2]->ne[0],features[2]->ne[1],false));
        auto high=ggml_add(m,layer(m,n,"arm16",features.at(2)),low);
        high=layer(m,n,"conv_head16",resize(m,high,features[1]->ne[0],features[1]->ne[1],false));
        return {high,low};
    }
    if(type=="BiSeNetV1") {
        auto context=run(m,child(n,"context_path"),{x});auto spatial=layer(m,n,"spatial_path",x);
        auto fused=run(m,child(n,"ffm"),{spatial,context.at(0)}).at(0);
        Tensors all{fused,context.at(0),context.at(1)},out;
        for(auto const& index:a.at("out_indices"))out.push_back(all.at(index.get<size_t>()));return out;
    }
    if(type=="HourglassModule") {
        auto up=layer(m,n,"up1",x),low=layer(m,n,"low3",layer(m,n,"low2",layer(m,n,"low1",x)));
        auto cfg=a.at("upsample_cfg");auto w=up->ne[0],h=up->ne[1];
        if(cfg.contains("scale_factor")){float scale=cfg.at("scale_factor").get<float>();w=int64_t(low->ne[0]*scale);h=int64_t(low->ne[1]*scale);}
        return {ggml_add(m,up,resize(m,low,w,h,cfg.value("mode",std::string{"nearest"})=="bilinear",cfg.value("align_corners",false)))};
    }
    if(type=="HourglassNet") {
        x=layer(m,n,"stem",x);Tensors out;int stacks=a.at("num_stacks").get<int>();
        for(int i=0;i<stacks;++i) {
            auto key=std::to_string(i);auto y=layer(m,child(n,"out_convs"),key,layer(m,child(n,"hourglass_modules"),key,x));out.push_back(y);
            if(i+1<stacks) {
                x=ggml_add(m,layer(m,child(n,"conv1x1s"),key,x),layer(m,child(n,"remap_convs"),key,y));
                x=layer(m,child(n,"inters"),key,layer(m,n,"relu",x));
            }
        }
        return out;
    }
    if(type=="RSoftmax") {
        int radix=a.at("radix").get<int>(),groups=a.at("groups").get<int>();auto count=x->ne[2];
        if(radix==1)return {ggml_reshape_2d(m,ggml_sigmoid(m,x),count,x->ne[3])};
        x=ggml_reshape_4d(m,ggml_cont(m,x),count/(radix*groups),radix,groups,x->ne[3]);
        x=ggml_soft_max(m,ggml_cont(m,ggml_permute(m,x,1,0,2,3)));
        x=ggml_cont(m,ggml_permute(m,x,2,0,1,3));
        return {ggml_reshape_2d(m,x,count,x->ne[3])};
    }
    if(type=="SplitAttentionConv2d") {
        x=layer(m,n,"relu",layer(m,n,a.at("norm0_name").get<std::string>(),layer(m,n,"conv",x)));
        int radix=a.at("radix").get<int>();auto channels=x->ne[2]/radix;
        auto part=[&](int i){return ggml_cont(m,ggml_view_4d(m,x,x->ne[0],x->ne[1],channels,x->ne[3],x->nb[1],x->nb[2],x->nb[3],size_t(i*channels)*x->nb[2]));};
        auto gap=part(0);for(int i=1;i<radix;++i)gap=ggml_add(m,gap,part(i));
        gap=adaptive_pool(m,gap,1,1);
        gap=layer(m,n,"relu",layer(m,n,a.at("norm1_name").get<std::string>(),layer(m,n,"fc1",gap)));
        auto attention=layer(m,n,"rsoftmax",layer(m,n,"fc2",gap));
        tensor sum=nullptr;
        for(int i=0;i<radix;++i) {
            auto weights=ggml_reshape_4d(m,ggml_view_1d(m,attention,channels,size_t(i*channels)*attention->nb[0]),1,1,channels,1);
            auto y=ggml_mul(m,part(i),weights);sum=sum?ggml_add(m,sum,y):y;
        }
        return {sum};
    }
    if(type=="SSDVGG") {
        Tensors out;auto indices=a.at("out_feature_indices").get<std::vector<int>>();auto const& features=child(n,"features");int i=0;
        for(auto const& key:features.at("order")) {
            x=layer(m,features,key.get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i++)!=indices.end())out.push_back(x);
        }
        return out;
    }
    if(type=="ERFNet")return {layer(m,n,"decoder",layer(m,n,"encoder",x))};
    if(type=="NonBottleneck1d")return {layer(m,n,"act",ggml_add(m,x,layer(m,n,"convs_layers",x)))};
    if(type=="UpsamplerBlock")return {layer(m,n,"act",layer(m,n,"bn",layer(m,n,"conv",x)))};
    if(type=="DownsamplerBlock") {
        auto conv=layer(m,n,"conv",x),pool=layer(m,n,"pool",x);
        pool=resize(m,pool,conv->ne[0],conv->ne[1],true);
        return {layer(m,n,"act",layer(m,n,"bn",cat(m,{conv,pool})))};
    }
    if(type=="Bottle2neck") {
        auto identity=c.contains("downsample")?layer(m,n,"downsample",x):x;
        auto plugin=[&](tensor t,int i){auto key="after_conv"+std::to_string(i)+"_plugin_names";if(a.contains(key))for(auto const& k:a.at(key))t=layer(m,n,k.get<std::string>(),t);return t;};
        x=plugin(layer(m,n,"relu",layer(m,n,a.at("norm1_name").get<std::string>(),layer(m,n,"conv1",x))),1);
        int width=a.at("width").get<int>(),scales=a.at("scales").get<int>();
        auto piece=[&](int i){return ggml_cont(m,ggml_view_4d(m,x,x->ne[0],x->ne[1],width,x->ne[3],x->nb[1],x->nb[2],x->nb[3],size_t(i*width)*x->nb[2]));};
        auto const& convs=child(n,"convs"),bns=child(n,"bns");
        tensor previous=nullptr;Tensors parts;
        for(int i=0;i<scales-1;++i) {
            auto p=piece(i);if(i && a.at("stage_type")=="normal")p=ggml_add(m,p,previous);
            previous=layer(m,n,"relu",layer(m,bns,std::to_string(i),layer(m,convs,std::to_string(i),p)));parts.push_back(previous);
        }
        auto last=piece(scales-1);
        if(a.at("stage_type")=="stage"&&a.at("conv2_stride")!=1)last=layer(m,n,"pool",last);
        parts.push_back(last);x=plugin(cat(m,parts),2);
        x=plugin(layer(m,n,a.at("norm3_name").get<std::string>(),layer(m,n,"conv3",x)),3);
        return {layer(m,n,"relu",ggml_add(m,x,identity))};
    }
    if(type=="BasicConvBlock")return {layer(m,n,"convs",x)};
    if(type=="InterpConv")return {layer(m,n,"interp_upsample",x)};
    if(type=="DeconvModule")return {layer(m,n,"deconv_upsamping",x)};
    if(type=="UpConvBlock")return {layer(m,n,"conv_block",cat(m,{x,layer(m,n,"upsample",inputs.at(1))}))};
    if(type=="UNet") {
        Tensors encoded;auto const& encoder=child(n,"encoder");
        for(auto const& key:encoder.at("order")) {x=layer(m,encoder,key.get<std::string>(),x);encoded.push_back(x);}
        Tensors out{x};auto const& decoder=child(n,"decoder");
        for(int i=int(decoder.at("order").size())-1;i>=0;--i) {
            x=run(m,child(decoder,std::to_string(i)),{encoded[i],x}).at(0);out.push_back(x);
        }
        return out;
    }
    if(type=="LearningToDownsample")return {layer(m,n,"dsconv2",layer(m,n,"dsconv1",layer(m,n,"conv",x)))};
    if(type=="GlobalFeatureExtractor") {
        x=layer(m,n,"bottleneck3",layer(m,n,"bottleneck2",layer(m,n,"bottleneck1",x)));
        auto pooled=run(m,child(n,"ppm"),{x});pooled.insert(pooled.begin(),x);
        return {layer(m,n,"out",cat(m,pooled))};
    }
    if(type=="FeatureFusionModule") {
        if(c.contains("conv0")) {auto y=layer(m,n,"conv0",cat(m,inputs));return {ggml_add(m,y,ggml_mul(m,y,layer(m,n,"attention",y)))};}
        if(c.contains("conv1")) {
            auto y=layer(m,n,"conv1",cat(m,inputs));auto atten=layer(m,n,"conv_atten",layer(m,n,"gap",y));
            return {ggml_add(m,y,ggml_mul(m,y,atten))};
        }
        if(!c.contains("dwconv"))throw std::runtime_error("unknown feature fusion protocol");
        auto low=inputs.at(1);low=resize(m,low,x->ne[0],x->ne[1],true,a.value("align_corners",false));
        low=layer(m,n,"conv_lower_res",layer(m,n,"dwconv",low));
        return {layer(m,n,"relu",ggml_add(m,layer(m,n,"conv_higher_res",x),low))};
    }
    if(type=="FastSCNN") {
        auto high=layer(m,n,"learning_to_downsample",x),low=layer(m,n,"global_feature_extractor",high);
        Tensors all{high,low,run(m,child(n,"feature_fusion"),{high,low}).at(0)},out;
        for(auto const& index:a.at("out_indices"))out.push_back(all.at(index.get<size_t>()));return out;
    }
    if(type=="ResBlock")return {ggml_add(m,x,layer(m,n,"conv2",layer(m,n,"conv1",x)))};
    if(type=="Darknet") {
        Tensors out;auto indices=a.at("out_indices").get<std::vector<int>>();int i=0;
        for(auto const& name:a.at("cr_blocks")) {
            x=layer(m,n,name.get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i++)!=indices.end())out.push_back(x);
        }
        return out;
    }
    if(type=="InvertedResidual") {
        if(c.contains("depthwise_conv")) {auto variant=n;variant["type"]="InvertedResidualV3";return backbones(m,variant,inputs);}
        auto y=layer(m,n,"conv",x);
        return {a.value("use_res_connect",false)?ggml_add(m,x,y):y};
    }
    if(type=="InvertedResidualV3") {
        auto y=x;
        if(a.value("with_expand_conv",false))y=layer(m,n,"expand_conv",y);
        y=layer(m,n,"depthwise_conv",y);
        if(a.value("with_se",false))y=layer(m,n,"se",y);
        y=layer(m,n,"linear_conv",y);
        return {a.value("with_res_shortcut",false)?ggml_add(m,x,y):y};
    }
    if(type=="SELayer") {
        auto y=layer(m,n,"conv2",layer(m,n,"conv1",layer(m,n,"global_avgpool",x)));
        return {ggml_mul(m,x,y)};
    }
    if(type=="MobileNetV2" || type=="MobileNetV3") {
        if(type=="MobileNetV2")x=layer(m,n,"conv1",x);
        Tensors out;auto indices=a.at("out_indices").get<std::vector<int>>();int i=0;
        for(auto const& name:a.at("layers")) {
            x=layer(m,n,name.get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i++)!=indices.end())out.push_back(x);
        }
        return out;
    }
    if(type=="CpmBlock") return {layer(m,n,"model",x)};
    if(type=="CPM") {
        Tensors out{layer(m,n,"stem",x)};
        auto middle=layer(m,n,"middle",x);
        auto const& stages=child(n,"cpm_stages"), convs=child(n,"out_convs"), mids=child(n,"middle_conv");
        for(auto const& k:stages.at("order")) {
            auto key=k.get<std::string>();
            auto y=cat(m,{out.back(),layer(m,mids,key,middle)});
            out.push_back(layer(m,convs,key,layer(m,stages,key,y)));
        }
        return out;
    }
    if(type=="HRModule") {
        auto const& branches=child(n,"branches");
        Tensors xs;
        for(size_t i=0;i<branches.at("order").size();++i)
            xs.push_back(layer(m,branches,branches.at("order")[i].get<std::string>(),inputs.at(i)));
        if(xs.size()==1)return xs;
        Tensors out;
        auto const& fuse=child(n,"fuse_layers");
        for(size_t i=0;i<fuse.at("order").size();++i) {
            auto const& row=child(fuse,fuse.at("order")[i].get<std::string>());
            tensor y=nullptr;
            for(size_t j=0;j<xs.size();++j) {
                auto v=i==j?xs[j]:layer(m,row,std::to_string(j),xs[j]);
                y=y?ggml_add(m,y,v):v;
            }
            out.push_back(layer(m,n,"relu",y));
        }
        return out;
    }
    if(type=="HRNet") {
        x=layer(m,n,"relu",layer(m,n,a.value("norm1_name",std::string{"bn1"}),layer(m,n,"conv1",x)));
        x=layer(m,n,"relu",layer(m,n,a.value("norm2_name",std::string{"bn2"}),layer(m,n,"conv2",x)));
        x=layer(m,n,"layer1",x);
        Tensors ys{x};
        for(int stage=2;stage<=4;++stage) {
            auto const& transition=child(n,"transition"+std::to_string(stage-1));
            Tensors next;
            for(size_t i=0;i<transition.at("order").size();++i) {
                auto const& t=child(transition,std::to_string(i));
                next.push_back(t.at("type")=="Identity" ? ys.at(i) : one(m,t,ys.back()));
            }
            ys=run(m,child(n,"stage"+std::to_string(stage)),next);
        }
        return ys;
    }
    if(type=="CSPNeXt" || type=="CSPDarknet") {
        Tensors out;
        auto indices=a.at("out_indices").get<std::vector<int>>();
        int i=0;
        for(auto const& name:a.at("layers")) {
            x=layer(m,n,name.get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end())out.push_back(x);
            ++i;
        }
        return out;
    }
    if(type=="Focus") {
        // Original order: top-left, bottom-left, top-right, bottom-right.
        x=cat(m,{space_to_depth_quad(m,x,0,0),space_to_depth_quad(m,x,0,1),
                 space_to_depth_quad(m,x,1,0),space_to_depth_quad(m,x,1,1)});
        return {layer(m,n,"conv",x)};
    }
    if(type=="CSPNeXtBlock" || type=="DarknetBottleneck") {
        auto y=layer(m,n,"conv2",layer(m,n,"conv1",x));
        return {a.value("add_identity",false)?ggml_add(m,x,y):y};
    }
    if(type=="CSPLayer") {
        auto shortcut=layer(m,n,"short_conv",x);
        x=layer(m,n,"blocks",layer(m,n,"main_conv",x));
        x=cat(m,{x,shortcut});
        if(a.value("channel_attention",false))x=layer(m,n,"attention",x);
        return {layer(m,n,"final_conv",x)};
    }
    if(type=="SPPBottleneck") {
        x=layer(m,n,"conv1",x);
        Tensors xs{x};
        auto const& pools=child(n,"poolings");
        for(auto const& k:pools.at("order")) xs.push_back(layer(m,pools,k.get<std::string>(),x));
        return {layer(m,n,"conv2",cat(m,xs))};
    }
    if(type=="ChannelAttention") {
        auto y=layer(m,n,"act",layer(m,n,"fc",layer(m,n,"global_avgpool",x)));
        return {ggml_mul(m,x,y)};
    }
    if(type=="TCN") {
        x=layer(m,n,"expand_conv",x);
        Tensors out;
        auto const& blocks=child(n,"tcn_blocks");
        for(auto const& k:blocks.at("order")) {
            x=layer(m,blocks,k.get<std::string>(),x);out.push_back(x);
        }
        return out;
    }
    if(type=="BasicTemporalBlock") {
        auto y=layer(m,n,"conv2",layer(m,n,"conv1",x));
        if(a.value("residual",true)) {
            int64_t start=a.value("use_stride_conv",false)
                ? a.value("causal_shift",0)+a.at("kernel_size").get<int>()/2
                : a.at("pad").get<int>()+a.value("causal_shift",0);
            int64_t stride=a.value("use_stride_conv",false)?a.at("kernel_size").get<int>():1;
            // ne0 strides require gathering rows after transpose (views cannot
            // represent a strided innermost axis).
            auto indices=ggml_cast(m,ggml_arange(m,float(start),float(start+y->ne[0]*stride),float(stride)),GGML_TYPE_I32);
            auto r=ggml_cont(m,ggml_transpose(m,ggml_get_rows(m,ggml_cont(m,ggml_transpose(m,x)),indices)));
            if(c.contains("short_cut")) r=layer(m,n,"short_cut",r);
            y=ggml_add(m,y,r);
        }
        return {y};
    }
    if(type=="BasicBlock" || type=="Bottleneck") {
        // YOLOF's residual Bottleneck has ConvModules and no post-add ReLU.
        if(child(n,"conv1").at("type")=="ConvModule") {
            auto y=layer(m,n,"conv2",layer(m,n,"conv1",x));if(type=="Bottleneck")y=layer(m,n,"conv3",y);
            y=ggml_add(m,y,c.contains("downsample")?layer(m,n,"downsample",x):x);
            return {c.contains("act")?layer(m,n,"act",y):y};
        }
        auto plugins=[&](tensor t,int conv) {
            auto key="after_conv"+std::to_string(conv)+"_plugin_names";
            if(a.contains(key))for(auto const& name:a.at(key))t=layer(m,n,name.get<std::string>(),t);
            return t;
        };
        tensor identity=c.contains("downsample")?layer(m,n,"downsample",x):x;
        x=layer(m,n,"conv1",x);
        x=layer(m,n,a.value("norm1_name",std::string{"bn1"}),x);
        x=layer(m,n,"relu",x);
        x=plugins(x,1);
        x=layer(m,n,"conv2",x);
        bool split=child(n,"conv2").at("type")=="SplitAttentionConv2d";
        if(split) {if(a.value("avg_down_stride",false))x=layer(m,n,"avd_layer",x);}
        else x=layer(m,n,a.value("norm2_name",std::string{"bn2"}),x);
        if(type=="Bottleneck") {
            if(!split)x=layer(m,n,"relu",x);
            x=plugins(x,2);
            x=layer(m,n,"conv3",x);
            x=layer(m,n,a.value("norm3_name",std::string{"bn3"}),x);
            x=plugins(x,3);
        }
        return {layer(m,n,"relu",ggml_add(m,x,identity))};
    }
    if(type=="TridentBottleneck") {
        auto residual=c.contains("downsample")?layer(m,n,"downsample",x):x;
        x=layer(m,n,"relu",layer(m,n,a.at("norm1_name").get<std::string>(),layer(m,n,"conv1",x)));
        x=layer(m,n,"relu",layer(m,n,a.at("norm2_name").get<std::string>(),layer(m,n,"conv2",x)));
        x=layer(m,n,a.at("norm3_name").get<std::string>(),layer(m,n,"conv3",x));
        return {layer(m,n,"relu",ggml_add(m,x,residual))};
    }
    if(type=="ResNet" || type=="ResNetV1c" || type=="ResNetV1d" || type=="Res2Net" || type=="RegNet" || type=="ResNeSt"||type=="TridentResNet"||type=="DetectoRS_ResNet") {
        if(a.value("deep_stem",false)) x=layer(m,n,"stem",x);
        else {
            x=layer(m,n,"conv1",x);
            x=layer(m,n,a.value("norm1_name",std::string{"bn1"}),x);
            x=layer(m,n,"relu",x);
        }
        if(type!="RegNet")x=layer(m,n,"maxpool",x);
        Tensors out;
        if(type=="DetectoRS_ResNet"&&a.value("output_img",false))out.push_back(inputs.at(0));
        auto indices=a.at("out_indices").get<std::vector<int>>();
        int i=0;
        for(auto const& stage:a.at("res_layers")) {
            x=layer(m,n,stage.get<std::string>(),x);
            if(std::find(indices.begin(),indices.end(),i)!=indices.end()) out.push_back(x);
            ++i;
        }
        return out;
    }
    if(type=="InputInjection") return {layer(m,n,"pool",x)};
    if(type=="GlobalContextExtractor") {
        tensor y=layer(m,n,"avg_pool",x);
        y=ggml_reshape_2d(m,ggml_cont(m,y),x->ne[2],x->ne[3]);
        y=layer(m,n,"fc",y);
        y=ggml_reshape_4d(m,y,1,1,x->ne[2],x->ne[3]);
        return {ggml_mul(m,x,y)};
    }
    if(type=="ContextGuidedBlock") {
        tensor orig=x;
        x=layer(m,n,"conv1x1",x);
        x=cat(m,{layer(m,n,"f_loc",x),layer(m,n,"f_sur",x)});
        x=layer(m,n,"activate",layer(m,n,"bn",x));
        if(a.value("downsample",false)) x=layer(m,n,"bottleneck",x);
        x=layer(m,n,"f_glo",x);
        if(a.value("skip_connect",false)) x=ggml_add(m,orig,x);
        return {x};
    }
    if(type=="CGNet") {
        auto p2=layer(m,n,"inject_2x",x), p4=layer(m,n,"inject_4x",x);
        x=layer(m,n,"stem",x);
        x=layer(m,n,"norm_prelu_0",cat(m,{x,p2}));
        Tensors out{x};
        for(int level=1;level<=2;++level) {
            auto const& stage=child(n,"level"+std::to_string(level));
            tensor down=nullptr;
            for(auto const& k:stage.at("order")) {
                x=layer(m,stage,k.get<std::string>(),x);
                if(!down) down=x;
            }
            if(!down) throw std::runtime_error("CGNet stage must contain blocks");
            x=cat(m,level==1?Tensors{x,down,p4}:Tensors{down,x});
            x=layer(m,n,"norm_prelu_"+std::to_string(level),x);
            out.push_back(x);
        }
        return out;
    }
    throw std::runtime_error("missing backbone forward");
}
} // namespace visp::mmlab
