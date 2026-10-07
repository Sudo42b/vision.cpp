#include "config.h"
#include <stdexcept>
#include <cmath>

namespace visp::mmlab {
namespace {
int pair(Json const& j, int i) { return j.is_array()?j.at(i).get<int>():j.get<int>(); }
tensor slice(model_ref m, tensor x, int axis, int64_t start, int64_t count) {
    auto shape=i64x4{x->ne[0],x->ne[1],x->ne[2],x->ne[3]};shape[axis]=count;
    return ggml_cont(m,ggml_view_4d(m,x,shape[0],shape[1],shape[2],shape[3],
        x->nb[1],x->nb[2],x->nb[3],size_t(start)*x->nb[axis]));
}
tensor deform(model_ref m, Json const& n, Tensors const& inputs) {
    auto const& a=n.at("attrs");auto type=n.at("type").get<std::string>();
    auto w=weights(m,n);auto kernel=w.weights("weight"), x=inputs.at(0);
    auto s=a.at("stride"),p=a.at("padding"),d=a.at("dilation");
    if(pair(s,0)!=pair(s,1)||pair(p,0)!=pair(p,1)||pair(d,0)!=1||pair(d,1)!=1)
        throw std::runtime_error("asymmetric/dilated deform convolution unsupported");
    int stride=pair(s,0),pad=pair(p,0),groups=a.value("groups",1),dg=a.value("deform_groups",1);
    int64_t area=kernel->ne[0]*kernel->ne[1];
    bool modulated=type.find("Modulated")!=std::string::npos;
    tensor offset, mask=nullptr;
    if(type.find("Pack")!=std::string::npos) {
        auto packed=layer(m,n,"conv_offset",x);
        offset=slice(m,packed,2,0,2*dg*area);
        if(modulated)mask=ggml_sigmoid(m,slice(m,packed,2,2*dg*area,dg*area));
    } else {
        offset=inputs.at(1);
        if(modulated)mask=inputs.at(2);
    }
    auto ow=(x->ne[0]+2*pad-kernel->ne[0])/stride+1;
    auto oh=(x->ne[1]+2*pad-kernel->ne[1])/stride+1;
    auto fit=[&](tensor t) {
        if(t->ne[0]<ow||t->ne[1]<oh)throw std::runtime_error("deform offset too small");
        t=ggml_cont(m,t);
        // mmcv indexes the contiguous buffer using OUTPUT spatial strides.
        return ggml_view_4d(m,t,ow,oh,t->ne[2],1,ow*t->nb[0],ow*oh*t->nb[0],ow*oh*t->ne[2]*t->nb[0],0);
    };
    offset=fit(offset);if(mask)mask=fit(mask);
    if(dg<=0||x->ne[2]%dg || (groups!=1&&groups!=dg))
        throw std::runtime_error("unsupported deform/convolution group combination");
    Tensors parts;auto channels=x->ne[2]/dg;
    for(int g=0;g<dg;++g) {
        auto kg=groups==1?slice(m,kernel,2,g*channels,channels):
            slice(m,kernel,3,g*(kernel->ne[3]/groups),kernel->ne[3]/groups);
        parts.push_back(conv_2d_deform(m,slice(m,x,2,g*channels,channels),kg,
            slice(m,offset,2,g*2*area,2*area),mask?slice(m,mask,2,g*area,area):nullptr,stride,pad));
    }
    auto y=parts[0];
    for(size_t i=1;i<parts.size();++i)y=groups==1?ggml_add(m,y,parts[i]):ggml_concat(m,y,parts[i],2);
    if(auto bias=w.find("bias"))y=ggml_add(m,y,ggml_reshape_4d(m,bias,1,1,bias->ne[0],1));
    return y;
}
tensor average_pool(model_ref m, Json const& a, tensor x) {
    auto k=a.at("kernel_size"),s=a.at("stride"),p=a.at("padding");if(s.is_null())s=k;
    int kw=pair(k,1),kh=pair(k,0),sw=pair(s,1),sh=pair(s,0),pw=pair(p,1),ph=pair(p,0);
    auto width=x->ne[0],height=x->ne[1];bool ceil=a.value("ceil_mode",false);
    auto count=[&](int64_t n,int kernel,int stride,int padding) {
        auto y=int64_t(std::floor(double(n+2*padding-kernel+(ceil?stride-1:0))/stride))+1;
        if(ceil&&(y-1)*stride>=n+padding)--y;
        if(y<=0)throw std::runtime_error("invalid average pool output");return y;
    };
    auto ow=count(width,kw,sw,pw),oh=count(height,kh,sh,ph);
    // Extra ceil windows must contain zeros, but those zeros do not enlarge
    // PyTorch's divisor. Account for the divisor separately below.
    auto ew=std::max<int64_t>(0,(ow-1)*sw+kw-2*pw-width),eh=std::max<int64_t>(0,(oh-1)*sh+kh-2*ph-height);
    if(ew||eh)x=ggml_pad_ext(m,x,0,int(ew),0,int(eh),0,0,0,0);
    auto y=ggml_pool_2d(m,x,GGML_OP_POOL_AVG,kw,kh,sw,sh,pw,ph);
    auto override=a.value("divisor_override",Json{});
    if(!override.is_null()) {
        float divisor=override.get<float>();if(divisor<=0)throw std::runtime_error("invalid pool divisor");
        return ggml_scale(m,y,float(kw*kh)/divisor);
    }
    if(!ceil&&a.value("count_include_pad",true))return y;
    bool include=a.value("count_include_pad",true);
    auto extent=[&](int64_t input,int64_t output,int kernel,int stride,int padding) {
        auto start=ggml_scale_bias(m,ggml_arange(m,0,float(output),1),float(stride),float(-padding));
        auto end=ggml_scale_bias(m,start,1,float(kernel));
        end=ggml_clamp(m,end,-INFINITY,float(input+(include?padding:0)));
        start=ggml_clamp(m,start,float(include?-padding:0),INFINITY);
        return ggml_sub(m,end,start);
    };
    auto cw=ggml_reshape_2d(m,extent(width,ow,kw,sw,pw),ow,1),ch=ggml_reshape_2d(m,extent(height,oh,kh,sh,ph),1,oh);
    auto shape=ggml_new_tensor_2d(m,GGML_TYPE_F32,ow,oh);
    auto divisor=ggml_mul(m,ggml_repeat(m,cw,shape),ch);
    return ggml_div(m,ggml_scale(m,y,float(kw*kh)),divisor);
}
float pool_padding(int64_t size, int kernel, int stride, int padding, bool ceil) {
    if(!ceil) return float(padding);
    int64_t count=int64_t(std::ceil(double(size+2*padding-kernel)/stride))+1;
    if((count-1)*stride>=size+padding) --count;
    float adjusted=float((count-1)*stride+kernel-size)/2;
    if(int(adjusted)!=padding) throw std::runtime_error("ceil pool cannot preserve kernel padding for this shape");
    return adjusted;
}
}
tensor adaptive_pool(model_ref m, tensor x, int w, int h, bool maximum) {
    if(w<=0 || h<=0) throw std::runtime_error("invalid adaptive pool output");
    Tensors rows;
    for(int iy=0;iy<h;++iy) {
        Tensors cols;
        auto y0=iy*x->ne[1]/h, y1=((iy+1)*x->ne[1]+h-1)/h;
        for(int ix=0;ix<w;++ix) {
            auto x0=ix*x->ne[0]/w, x1=((ix+1)*x->ne[0]+w-1)/w;
            tensor crop=ggml_cont(m,ggml_view_4d(m,x,x1-x0,y1-y0,x->ne[2],x->ne[3],
                x->nb[1],x->nb[2],x->nb[3],size_t(y0)*x->nb[1]+size_t(x0)*x->nb[0]));
            cols.push_back(ggml_pool_2d(m,crop,maximum?GGML_OP_POOL_MAX:GGML_OP_POOL_AVG,int(x1-x0),int(y1-y0),
                int(x1-x0),int(y1-y0),0,0));
        }
        rows.push_back(cat(m,cols,0));
    }
    return cat(m,rows,1);
}
Tensors modules(model_ref m, Json const& n, Tensors const& inputs) {
    auto type=n.at("type").get<std::string>();
    auto const& a=n.at("attrs");
    tensor x=inputs.at(0);
    model_ref w=weights(m,n);
    if(type=="Sequential" || type=="ResLayer" || type=="Res2Layer" || type=="ModuleList") {
        Tensors out=inputs;
        for(auto const& key:n.at("order")) out=run(m,child(n,key.get<std::string>()),out);
        return out;
    } else if(type=="Identity" || type=="Dropout" || type=="Dropout2d" || type=="DropPath") {
        // All exported models are eval-only.
    } else if(type=="DeformConv2d" || type=="ModulatedDeformConv2d" || type=="DeformConv2dPack" || type=="ModulatedDeformConv2dPack") {
        x=deform(m,n,inputs);
    } else if(type=="SAConv2d") {
        if(a.value("groups",1)!=1)throw std::runtime_error("SAC grouped convolution unsupported");
        auto stride=a.at("stride"),padding=a.at("padding"),dilation=a.at("dilation");
        if(pair(stride,0)!=pair(stride,1)||pair(padding,0)!=pair(padding,1)||pair(dilation,0)!=pair(dilation,1))throw std::runtime_error("SAC asymmetric parameters unsupported");
        x=ggml_add(m,x,layer(m,n,"pre_context",adaptive_pool(m,x,1,1)));
        auto average=ggml_pool_2d(m,pad_reflect_ext(m,x,2,2,2,2),GGML_OP_POOL_AVG,5,5,1,1,0,0);
        auto gate=layer(m,n,"switch",average),kernel=parameter(m,n,"weight");
        auto convolution=[&](tensor weight,char const* offset_name,int factor) {
            int pad=pair(padding,0)*factor,dil=pair(dilation,0)*factor,step=pair(stride,0);
            if(!a.value("use_deform",false))return ggml_conv_2d(m,ggml_cast(m,weight,GGML_TYPE_F16),x,step,step,pad,pad,dil,dil);
            auto offset=layer(m,n,offset_name,average);int kw=int(weight->ne[0]),kh=int(weight->ne[1]);
            int unit_pad=pad-(kw-1)*(dil-1)/2;
            if(kw!=kh||kw%2==0||unit_pad<0)throw std::runtime_error("SAC dilated sampling requires odd square kernels");
            if(dil!=1) {
                Tensors channels;
                for(int h=0;h<kh;++h)for(int k=0;k<kw;++k) {
                    int index=2*(h*kw+k);
                    channels.push_back(ggml_scale_bias(m,slice(m,offset,2,index,1),1,float(h*(dil-1)+unit_pad-pad)));
                    channels.push_back(ggml_scale_bias(m,slice(m,offset,2,index+1,1),1,float(k*(dil-1)+unit_pad-pad)));
                }offset=cat(m,channels);
            }
            return conv_2d_deform(m,x,weight,offset,nullptr,step,unit_pad);
        };
        auto small=convolution(kernel,"offset_s",1),large=convolution(ggml_add(m,kernel,parameter(m,n,"weight_diff")),"offset_l",3);
        x=ggml_add(m,ggml_mul(m,small,gate),ggml_mul(m,large,ggml_scale_bias(m,gate,-1,1)));
        x=ggml_add(m,x,layer(m,n,"post_context",adaptive_pool(m,x,1,1)));
    } else if(type=="QuickGELU") {
        x=ggml_mul(m,x,ggml_sigmoid(m,ggml_scale(m,x,1.702f)));
    } else if(type=="_ScalarAffine") {
        x=ggml_add(m,ggml_mul(m,x,parameter(m,n,"w")),parameter(m,n,"b"));
    } else if(type=="Conv1d") {
        if(a.value("padding_mode",std::string{"zeros"})!="zeros") throw std::runtime_error("nonzero convolution padding mode unsupported");
        if(a.value("groups",1)!=1) throw std::runtime_error("grouped Conv1d unsupported");
        tensor kernel=w.weights("weight");
        if(kernel->type!=GGML_TYPE_F16) kernel=ggml_cast(m,kernel,GGML_TYPE_F16);
        x=ggml_conv_1d(m,kernel,x,pair(a.at("stride"),0),pair(a.at("padding"),0),pair(a.at("dilation"),0));
        if(auto bias=w.find("bias")) x=ggml_add(m,x,ggml_reshape_3d(m,bias,1,bias->ne[0],1));
    } else if(type=="Conv2d" || type=="Conv2dAdaptivePadding" || type=="ConvWS2d" || type=="ConvAWS2d") {
        if(a.value("padding_mode",std::string{"zeros"})!="zeros") throw std::runtime_error("nonzero convolution padding mode unsupported");
        auto stride=a.at("stride"), pad=a.at("padding"), dilation=a.at("dilation");
        if(type=="Conv2dAdaptivePadding") {
            auto kernel=w.weights("weight");
            auto pw=std::max<int64_t>(0,((x->ne[0]+pair(stride,1)-1)/pair(stride,1)-1)*pair(stride,1)+(kernel->ne[0]-1)*pair(dilation,1)+1-x->ne[0]);
            auto ph=std::max<int64_t>(0,((x->ne[1]+pair(stride,0)-1)/pair(stride,0)-1)*pair(stride,0)+(kernel->ne[1]-1)*pair(dilation,0)+1-x->ne[1]);
            if(pw||ph)x=ggml_pad_ext(m,x,int(pw/2),int(pw-pw/2),int(ph/2),int(ph-ph/2),0,0,0,0);
        }
        int groups=a.value("groups",1);
        if(groups==1) x=conv_2d_ex(w,x,pair(stride,1),pair(stride,0),pair(pad,1),pair(pad,0),pair(dilation,1),pair(dilation,0));
        else if(groups==a.at("in_channels").get<int>() && groups==a.at("out_channels").get<int>())
            x=conv_2d_depthwise_ex(w,x,pair(stride,1),pair(stride,0),pair(pad,1),pair(pad,0),pair(dilation,1),pair(dilation,0));
        else {
            if(pair(stride,0)!=pair(stride,1)||pair(pad,0)!=pair(pad,1)||pair(dilation,0)!=pair(dilation,1))
                throw std::runtime_error("asymmetric grouped convolution unsupported");
            x=conv_2d_grouped(w,x,pair(stride,0),pair(pad,0),pair(dilation,0),groups);
        }
    } else if(type=="ConvTranspose2d") {
        if(pair(a.at("dilation"),0)!=1 || pair(a.at("dilation"),1)!=1) throw std::runtime_error("dilated transposed convolution unsupported");
        if(pair(a.at("stride"),0)!=pair(a.at("stride"),1) || pair(a.at("padding"),0)!=pair(a.at("padding"),1)
           || pair(a.at("output_padding"),0)!=pair(a.at("output_padding"),1))
            throw std::runtime_error("asymmetric transposed convolution unsupported");
        x=conv_transpose_2d(w,x,pair(a.at("stride"),0),pair(a.at("padding"),0),a.value("groups",1),pair(a.at("output_padding"),0));
    } else if(type=="BatchNorm1d") {
        auto scale=w.weights("weight"), bias=w.weights("bias");
        x=ggml_add(m,ggml_mul(m,x,ggml_reshape_3d(m,scale,1,scale->ne[0],1)),ggml_reshape_3d(m,bias,1,bias->ne[0],1));
    } else if(type=="BatchNorm2d" || type=="SyncBatchNorm" || type=="_BatchNormXd") x=batch_norm_2d(w,x);
    else if(type=="GroupNorm") x=group_norm(w,x,a.at("num_groups").get<int>(),a.value("eps",1e-5f));
    else if(type=="LayerNorm") {
        auto shape=a.at("normalized_shape");
        if(shape.size()==3) {
            if(x->ne[0]!=1||x->ne[1]!=1)throw std::runtime_error("channel LayerNorm requires spatial 1x1");
            x=ggml_reshape_4d(m,ggml_norm(m,ggml_reshape_2d(m,ggml_cont(m,x),x->ne[2],x->ne[3]),a.value("eps",1e-5f)),1,1,x->ne[2],x->ne[3]);
        } else x=ggml_norm(m,x,a.value("eps",1e-5f));
        if(auto weight=w.find("weight")) x=ggml_mul(m,x,weight);
        if(auto bias=w.find("bias")) x=ggml_add(m,x,bias);
    }
    else if(type=="Linear" || type=="NonDynamicallyQuantizableLinear") x=linear(w,x);
    else if(type=="ReLU") x=ggml_relu(m,x);
    else if(type=="ReLU6") x=ggml_clamp(m,x,0,6);
    else if(type=="PReLU") x=prelu(w,x);
    else if(type=="Sigmoid") x=ggml_sigmoid(m,x);
    else if(type=="Softmax") {
        if(a.at("dim")!=2 && a.at("dim")!=-1)throw std::runtime_error("softmax dimension protocol unsupported");
        x=ggml_soft_max(m,x);
    }
    else if(type=="SiLU" || type=="Swish") x=ggml_silu(m,x);
    else if(type=="Hardsigmoid") x=ggml_hardsigmoid(m,x);
    else if(type=="HSigmoid") {
        float divisor=a.value("divisor",6.f);
        if(divisor==0)throw std::runtime_error("hard sigmoid divisor is zero");
        x=ggml_clamp(m,ggml_scale_bias(m,x,1.f/divisor,a.value("bias",3.f)/divisor),a.value("min_value",0.f),a.value("max_value",1.f));
    }
    else if(type=="Hardswish" || type=="HSwish") x=ggml_hardswish(m,x);
    else if(type=="GELU") x=a.value("approximate",std::string{"none"})=="tanh" ? ggml_gelu(m,x):ggml_gelu_erf(m,x);
    else if(type=="LeakyReLU") x=ggml_leaky_relu(m,x,a.value("negative_slope",0.01f),false);
    else if(type=="MaxPool2d" || type=="AvgPool2d") {
        if(type=="AvgPool2d")return {average_pool(m,a,x)};
        auto k=a.at("kernel_size"), s=a.at("stride"), p=a.at("padding");
        if(s.is_null()) s=k;
        if(type=="MaxPool2d") {
            auto d=a.value("dilation",Json(1));
            if(pair(d,0)!=1 || pair(d,1)!=1) throw std::runtime_error("dilated max pooling unsupported");
        }
        bool ceil=a.value("ceil_mode",false);
        x=ggml_pool_2d(m,x,type=="MaxPool2d"?GGML_OP_POOL_MAX:GGML_OP_POOL_AVG,
            pair(k,1),pair(k,0),pair(s,1),pair(s,0),
            pool_padding(x->ne[0],pair(k,1),pair(s,1),pair(p,1),ceil),
            pool_padding(x->ne[1],pair(k,0),pair(s,0),pair(p,0),ceil));
    } else if(type=="PixelShuffle") {
        int r=a.at("upscale_factor").get<int>();
        if(x->ne[2]%(r*r))throw std::runtime_error("pixel shuffle channel mismatch");
        auto w0=x->ne[0], h0=x->ne[1], c0=x->ne[2]/(r*r), batch=x->ne[3];
        x=ggml_reshape_4d(m,ggml_cont(m,x),w0,h0,r,r*c0*batch);
        x=ggml_cont(m,ggml_permute(m,x,1,2,0,3));
        x=ggml_reshape_4d(m,x,w0*r,h0,r,c0*batch);
        x=ggml_cont(m,ggml_permute(m,x,0,2,1,3));
        x=ggml_reshape_4d(m,x,w0*r,h0*r,c0,batch);
    } else if(type=="Upsample") {
        auto size=a.value("size",Json{}), scale=a.value("scale_factor",Json{});
        int64_t w,h;
        if(!size.is_null()) {w=pair(size,1);h=pair(size,0);}
        else {
            double sh=scale.is_array()?scale.at(0).get<double>():scale.get<double>();
            double sw=scale.is_array()?scale.at(1).get<double>():scale.get<double>();
            w=int64_t(x->ne[0]*sw);h=int64_t(x->ne[1]*sh);
        }
        auto mode=a.value("mode",std::string{"nearest"});
        if(mode!="nearest" && mode!="bilinear") throw std::runtime_error("unsupported resize mode");
        auto align=a.value("align_corners",Json{});
        x=resize(m,x,w,h,mode=="bilinear",!align.is_null()&&align.get<bool>());
    } else if(type=="AdaptiveAvgPool2d" || type=="AdaptiveMaxPool2d") {
        auto size=a.at("output_size");
        x=adaptive_pool(m,x,pair(size,1),pair(size,0),type=="AdaptiveMaxPool2d");
    } else if(type=="TridentConv") {
        int branch=a.at("test_branch_idx").get<int>();
        if(branch<0)throw std::runtime_error("Trident all-branch inference requires a multi-batch protocol");
        int dilation=a.at("dilations").at(branch).get<int>(),padding=a.at("paddings").at(branch).get<int>();
        auto stride=a.at("stride");auto wm=weights(m,n);
        x=ggml_conv_2d(m,ggml_cast(m,wm.weights("weight"),GGML_TYPE_F16),x,pair(stride,1),pair(stride,0),padding,padding,dilation,dilation);
        if(auto bias=wm.find("bias"))x=ggml_add(m,x,ggml_reshape_4d(m,bias,1,1,bias->ne[0],1));
    } else if(type=="ConvModule") {
        for(auto const& item:a.value("order",Json::array({"conv","norm","act"}))) {
            auto op=item.get<std::string>();
            if(op=="conv") x=layer(m,n,"conv",x);
            else if(op=="norm" && a.value("with_norm",false)) x=layer(m,n,a.at("norm_name").get<std::string>(),x);
            else if(op=="act" && a.value("with_activation",false)) x=layer(m,n,"activate",x);
        }
    } else if(type=="DepthwiseSeparableConvModule") {
        x=layer(m,n,"depthwise_conv",x); x=layer(m,n,"pointwise_conv",x);
    } else throw std::runtime_error("missing primitive forward");
    return {x};
}
} // namespace visp::mmlab
