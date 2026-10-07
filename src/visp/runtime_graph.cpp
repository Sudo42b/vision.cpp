#include "runtime_graph.h"
#include "nn.h"
#include "nlohmann/json.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace visp {
namespace {
using Json = nlohmann::json;
// --- 1D conv: ggml_conv_1d + optional bias ---
static tensor conv_1d(model_ref m, tensor w, tensor b, tensor x,
                      int stride, int pad, int dilation) {
    // ⚠️ **g2c 의 conv1d 가중치는 [OC, K, IC] 로 실린다**(2D 의 OIHW 와 축 순서가 다르다).
    //    GGUF ne 로는 [IC, K, OC]. ggml_conv_1d 는 커널 ne=[K, IC, OC] 를 요구하므로
    //    ne0/ne1 을 맞바꾼다. 안 바꾸면 IC 가 1 로 읽혀 im2col 이 "b too small" 로 죽는다.
    w = ggml_cont(m, ggml_permute(m, w, 1, 0, 2, 3));
    if (w->type != GGML_TYPE_F16) w = ggml_cast(m, w, GGML_TYPE_F16);
    // 그래프 입력은 이미지 관례(whcn)로 들어와 ne=[1,1,C,1] 이 될 수 있다.
    // ggml_conv_1d 는 data ne=[L,IC,N] 를 요구하므로 커널의 IC 로 다시 접는다.
    // ⚠️ **배치 1 전제다**(하네스가 N=1 로만 잰다). N>1 이면 여기가 틀린다.
    int64_t ic = w->ne[1];
    if (x->ne[1] != ic) {
        int64_t L = ggml_nelements(x) / ic;
        x = ggml_reshape_3d(m, ggml_cont(m, x), L, ic, 1);
    }
    tensor out = ggml_conv_1d(m, w, x, stride, pad, dilation);
    if (b) out = ggml_add(m, out, ggml_reshape_3d(m, b, 1, b->ne[0], 1));
    return out;
}
// --- grid_sample_2d: ggml 에 대응 커널이 없어 custom op 으로 구현 ---
// a: WHCN [W,H,C,N].  g: [2,GW,GH,N] (torch (N,GH,GW,2) 의 ggml 역순), 정규화 좌표 [-1,1].
// dst: [GW,GH,C,N].  flags: bit0=nearest, bit1=align_corners.  범위 밖은 0 (zeros padding).
#define GTX_GS_AT(t, i0, i1, i2, i3) \
    (*(float const *)((char const *)(t)->data + (i0)*(t)->nb[0] + (i1)*(t)->nb[1] \
                      + (i2)*(t)->nb[2] + (i3)*(t)->nb[3]))

static void grid_sample_2d_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    ggml_tensor const * a = dst->src[0];
    ggml_tensor const * g = dst->src[1];
    int const flags = (int)(intptr_t)userdata;
    bool const nearest = (flags & 1) != 0;
    bool const align   = (flags & 2) != 0;
    int64_t const W = a->ne[0], H = a->ne[1];
    int64_t const GW = dst->ne[0], GH = dst->ne[1], C = dst->ne[2], N = dst->ne[3];
    int64_t const rows = GH * C * N;                 // (gh, c, n) 행 단위로 스레드 분할
    for (int64_t r = ith; r < rows; r += nth) {
        int64_t const gh = r % GH, c = (r / GH) % C, n = r / (GH * C);
        for (int64_t gw = 0; gw < GW; ++gw) {
            float const * gp = (float const *)((char const *)g->data
                + gw * g->nb[1] + gh * g->nb[2] + n * g->nb[3]);
            float fx = gp[0], fy = gp[1];            // torch grid 마지막 축 = (x, y)
            fx = align ? (fx + 1.f) * (W - 1) * 0.5f : ((fx + 1.f) * W - 1.f) * 0.5f;
            fy = align ? (fy + 1.f) * (H - 1) * 0.5f : ((fy + 1.f) * H - 1.f) * 0.5f;
            float v = 0.f;
            if (nearest) {
                int64_t const xi = (int64_t)std::nearbyint(fx);
                int64_t const yi = (int64_t)std::nearbyint(fy);
                if (xi >= 0 && xi < W && yi >= 0 && yi < H) v = GTX_GS_AT(a, xi, yi, c, n);
            } else {
                int64_t const x0 = (int64_t)std::floor(fx), y0 = (int64_t)std::floor(fy);
                float const dx = fx - (float)x0, dy = fy - (float)y0;
                for (int64_t j = 0; j < 2; ++j) {
                    for (int64_t i = 0; i < 2; ++i) {
                        int64_t const xi = x0 + i, yi = y0 + j;
                        if (xi < 0 || xi >= W || yi < 0 || yi >= H) continue;
                        v += GTX_GS_AT(a, xi, yi, c, n)
                             * (i ? dx : 1.f - dx) * (j ? dy : 1.f - dy);
                    }
                }
            }
            *(float *)((char *)dst->data + gw * dst->nb[0] + gh * dst->nb[1]
                       + c * dst->nb[2] + n * dst->nb[3]) = v;
        }
    }
}

static tensor grid_sample_2d(model_ref m, tensor a, tensor g, int flags) {
    ggml_tensor * args[2] = { a, g };
    return ggml_custom_4d(m, GGML_TYPE_F32, g->ne[1], g->ne[2], a->ne[2], a->ne[3],
                          args, 2, grid_sample_2d_op, GGML_N_TASKS_MAX,
                          (void *)(intptr_t)flags);
}
// --- mmcv (Modulated)DeformConv2d 의 bias (라이브러리 conv_2d_deform 은 안 받는다) ---
static tensor deform_bias(model_ref m, tensor x, tensor bias) {
    if (!bias) {
        return x;
    }
    if (!(m.flags & model_build_flag::cwhn)) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);   // WHCN 은 채널이 ne[2]
    }
    return ggml_add(m, x, bias);
}
// --- deformable conv: ggml_conv_2d_deform + optional bias ---
// weight(GGUF, OIHW) → ggml ne [KW,KH,IC,OC].  x: WHCN.
// offset: [OW,OH,2*KH*KW,N],  mask: [OW,OH,KH*KW,N] (v1 이면 nullptr).
static tensor runtime_deform_conv(model_ref m, tensor x, tensor offset, tensor mask,
                             int stride, int pad) {
    tensor out = ggml_conv_2d_deform(m, m.weights("weight"), x, offset, mask,
                                     stride, stride, pad, pad);
    if (tensor bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        out = ggml_add_inplace(m, out, bias);
    }
    return out;
}

struct Value {
    using Array = std::vector<Value>;
    std::variant<std::monostate, int64_t, double, bool, std::string, tensor, model_ref, Array> data;
    Value() = default;
    Value(int x) : data(int64_t(x)) {}
    Value(int64_t x) : data(x) {}
    Value(double x) : data(x) {}
    Value(bool x) : data(x) {}
    Value(std::string x) : data(std::move(x)) {}
    Value(tensor x) : data(x) {}
    Value(model_ref x) : data(x) {}
    static Value array(Array x) { Value v; v.data = std::move(x); return v; }
    double number() const {
        if (auto x = std::get_if<int64_t>(&data)) return double(*x);
        if (auto x = std::get_if<double>(&data)) return *x;
        if (auto x = std::get_if<bool>(&data)) return *x;
        throw std::runtime_error("recipe: expected a number");
    }
    int64_t integer() const {
        if (auto x = std::get_if<int64_t>(&data)) return *x;
        double n = number();
        if (!std::isfinite(n) || n < double(INT64_MIN) || n >= double(INT64_MAX))
            throw std::runtime_error("recipe: integer is out of range");
        return int64_t(n);
    }
    bool truth() const {
        if (std::holds_alternative<std::monostate>(data)) return false;
        if (auto x = std::get_if<tensor>(&data)) return *x != nullptr;
        return number() != 0;
    }
    Array const& array() const { return std::get<Array>(data); }
};

template<class T> auto convert(Value const& v) {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<U, model_ref>) return std::get<model_ref>(v.data);
    else if constexpr ((std::is_same_v<U, ggml_context*> || std::is_same_v<U, ggml_context const*>)) return static_cast<ggml_context*>(convert<model_ref>(v));
    else if constexpr (std::is_same_v<U, tensor> || std::is_same_v<U, ggml_tensor const*>) {
        if (std::holds_alternative<std::monostate>(v.data)) return U(nullptr);
        return U(std::get<tensor>(v.data));
    } else if constexpr (std::is_same_v<U, char const*>) return std::get<std::string>(v.data).c_str();
    else if constexpr (std::is_same_v<U, tensor_name>) {
        auto const& s = std::get<std::string>(v.data);
        if (s.size() >= GGML_MAX_NAME) throw std::runtime_error("recipe: tensor name too long");
        return tensor_name(s.c_str());
    } else if constexpr (std::is_same_v<U, bool>) return v.truth();
    else if constexpr (std::is_integral_v<U> || std::is_enum_v<U>) return U(v.integer());
    else if constexpr (std::is_floating_point_v<U>) return U(v.number());
    else if constexpr (std::is_same_v<U, i64x2> || std::is_same_v<U, i64x4>) {
        U result{};
        auto const& a = v.array();
        if (a.size() != (std::is_same_v<U, i64x2> ? 2u : 4u)) throw std::runtime_error("recipe: wrong dimension count");
        for (size_t i=0; i<a.size(); ++i) result[int(i)] = a[i].integer();
        return result;
    } else if constexpr (std::is_same_v<U, slice_t>) {
        if (!std::holds_alternative<Value::Array>(v.data)) return slice_t(v.integer());
        auto const& a = v.array();
        if (a.empty()) return slice_t{};
        if (a.size() == 1) return slice_t(a[0].integer());
        if (a.size() > 3) throw std::runtime_error("recipe: invalid slice");
        return slice_t(a[0].integer(), a[1].integer(), a.size()==3 ? a[2].integer() : 1);
    } else if constexpr (std::is_same_v<U, std::array<tensor, GGML_MAX_SRC>>) {
        U result{};
        auto const& a = v.array();
        if (a.size()>result.size()) throw std::runtime_error("recipe: too many concat inputs");
        for (size_t i=0;i<a.size();++i) result[i]=convert<tensor>(a[i]);
        return result;
    } else static_assert(!sizeof(U), "Unregistered recipe argument type");
}

struct Operations {
    using Function = std::function<Value(std::vector<Value> const&)>;
    std::unordered_map<std::string, Function> calls;
    std::unordered_map<std::string, Value> constants;
    template<class R, class... A, size_t... I>
    static Value invoke(R(*fn)(A...), std::vector<Value> const& args, std::index_sequence<I...>) {
        if constexpr (std::is_void_v<R>) { fn(convert<A>(args[I])...); return {}; }
        else if constexpr (std::is_integral_v<R> && !std::is_same_v<R,bool>)
            return Value(int64_t(fn(convert<A>(args[I])...)));
        else return Value(fn(convert<A>(args[I])...));
    }
    template<class R, class... A>
    void add(char const* name, R(*fn)(A...), std::vector<Value> defaults = {}) {
        calls.emplace(name, [fn,defaults,name](std::vector<Value> const& supplied) {
            constexpr auto count=sizeof...(A);
            if (supplied.size()>count || supplied.size()+defaults.size()<count)
                throw std::runtime_error(std::string("recipe: wrong argument count for ")+name);
            auto args=supplied;
            for (size_t i=args.size();i<count;++i) args.push_back(defaults[i-(count-defaults.size())]);
            return invoke(fn,args,std::index_sequence_for<A...>{});
        });
    }
    Operations() {
#include "runtime_ops.inc"
#include "runtime_constants.inc"
        add("conv_1d", &conv_1d);
        add("grid_sample_2d", &grid_sample_2d);
        add("deform_bias", &deform_bias);
        auto full_deform=calls.at("conv_2d_deform");
        add("runtime_deform_conv", &runtime_deform_conv);
        auto short_deform=calls.at("runtime_deform_conv");
        calls["conv_2d_deform"]=[full_deform,short_deform](auto const& a) {
            return a.size()==6?short_deform(a):full_deform(a);
        };
        constants.emplace("slice_t::max", Value(INT64_MAX));
        constants.emplace("INFINITY", Value(std::numeric_limits<double>::infinity()));
        calls.emplace("model_find", [](auto const& a) {
            if(a.size()!=2) throw std::runtime_error("recipe: invalid model_find");
            return Value(convert<model_ref>(a[0]).find(convert<char const*>(a[1])));
        });
        calls.emplace("model_weights", [](auto const& a) {
            if(a.size()!=2) throw std::runtime_error("recipe: invalid model_weights");
            auto t=convert<model_ref>(a[0]).find(convert<char const*>(a[1]));
            if(!t) throw std::runtime_error(std::string("recipe: missing weight ")+convert<char const*>(a[1]));
            return Value(t);
        });
    }
};
Operations const& operations() { static Operations const value; return value; }

struct Evaluator {
    model_ref model;
    std::unordered_map<std::string, Value> variables;
    size_t budget=10000000;
    Value eval(Json const& j, int depth=0) {
        if (!budget-- || depth>256) throw std::runtime_error("recipe: expression budget exceeded");
        auto child=[&](Json const& x) {return eval(x,depth+1);};
        if(j.is_null()) return {};
        if(j.is_boolean()) return Value(j.get<bool>());
        if(j.is_number_integer()) return Value(j.get<int64_t>());
        if(j.is_number()) return Value(j.get<double>());
        if(j.is_string()) return Value(j.get<std::string>());
        if(!j.is_object()) throw std::runtime_error("recipe: invalid expression");
        if(j.contains("ref")) {
            auto name=j.at("ref").get<std::string>();
            if(auto it=variables.find(name);it!=variables.end()) return it->second;
            if(auto it=operations().constants.find(name);it!=operations().constants.end()) return it->second;
            throw std::runtime_error("recipe: undefined reference "+name);
        }
        if(j.contains("model")) {
            auto prefix=j.at("model").get<std::string>();
            if(prefix.size()>=GGML_MAX_NAME) throw std::runtime_error("recipe: prefix too long");
            return Value(prefix.empty()?model:model.with_prefix(prefix.c_str()));
        }
        if(j.contains("array")) {
            Value::Array a; for(auto const& x:j.at("array")) a.push_back(child(x));
            return Value::array(std::move(a));
        }
        if(j.contains("call")) {
            auto name=j.at("call").get<std::string>();
            auto it=operations().calls.find(name);
            if(it==operations().calls.end()) throw std::runtime_error("recipe: unregistered operation "+name);
            std::vector<Value> args;for(auto const& x:j.at("args"))args.push_back(child(x));
            return it->second(args);
        }
        if(j.contains("field")) {
            tensor t=convert<tensor>(child(j.at("value")));
            if(!t) throw std::runtime_error("recipe: null tensor field");
            auto field=j.at("field").get<std::string>();
            if(field=="type")return Value(int(t->type));
            Value::Array a;
            for(int i=0;i<4;++i) {
                if(field=="ne") a.emplace_back(t->ne[i]);
                else if(field=="nb")a.emplace_back(int64_t(t->nb[i]));
                else throw std::runtime_error("recipe: invalid tensor field");
            }
            return Value::array(std::move(a));
        }
        if(j.contains("index")) {
            auto a=child(j.at("index"));auto i=child(j.at("at")).integer();
            if(i<0 || size_t(i)>=a.array().size())throw std::runtime_error("recipe: index out of range");
            return a.array()[size_t(i)];
        }
        if(j.contains("if"))return child(j.at("if")).truth()?child(j.at("then")):child(j.at("else"));
        if(j.contains("cast")) {
            auto x=child(j.at("value"));auto kind=j.at("cast").get<std::string>();
            return kind=="float"||kind=="double" ? Value(x.number()):Value(x.integer());
        }
        if(j.contains("unary")) {
            auto x=child(j.at("value"));auto op=j.at("unary").get<std::string>();
            if(op=="!")return Value(!x.truth());
            if(op=="+")return x;
            if(op=="-")return std::holds_alternative<int64_t>(x.data)?Value(-x.integer()):Value(-x.number());
        }
        if(j.contains("binary")) {
            auto op=j.at("binary").get<std::string>();auto a=child(j.at("left"));
            if(op=="&&")return Value(a.truth()&&child(j.at("right")).truth());
            if(op=="||")return Value(a.truth()||child(j.at("right")).truth());
            auto b=child(j.at("right"));
            if(op=="=="||op=="!=") {
                bool eq;
                if(std::holds_alternative<tensor>(a.data)||std::holds_alternative<std::monostate>(a.data))
                    eq=convert<tensor>(a)==convert<tensor>(b);
                else eq=a.number()==b.number();
                return Value(op=="=="?eq:!eq);
            }
            double x=a.number(),y=b.number();
            if(op=="<")return Value(x<y);if(op==">")return Value(x>y);
            if(op=="<=")return Value(x<=y);if(op==">=")return Value(x>=y);
            bool ints=std::holds_alternative<int64_t>(a.data)&&std::holds_alternative<int64_t>(b.data);
            if(op=="+")return ints?Value(a.integer()+b.integer()):Value(x+y);
            if(op=="-")return ints?Value(a.integer()-b.integer()):Value(x-y);
            if(op=="*")return ints?Value(a.integer()*b.integer()):Value(x*y);
            if(op=="/"||op=="%") {
                if(y==0)throw std::runtime_error("recipe: division by zero");
                if(op=="%")return Value(a.integer()%b.integer());
                return ints?Value(a.integer()/b.integer()):Value(x/y);
            }
        }
        throw std::runtime_error("recipe: unknown expression");
    }
};
Json read_program(model_file const& file) {
    auto text=file.get_string("mmlab.graph");
    if(text.empty() || text.size()>64*1024*1024)throw std::runtime_error("recipe: missing or oversized mmlab.graph");
    auto j=Json::parse(text);
    if(j.at("version")!=1 || !j.at("instructions").is_array() || j.at("instructions").size()>262144)
        throw std::runtime_error("recipe: unsupported version or instruction count");
    return j;
}
} // namespace

tensor runtime_graph_forward(model_ref m, tensor x, model_file const& file) {
    auto program=read_program(file);
    Evaluator evaluator{m,{{"x",Value(x)}}};
    tensor result=nullptr;
    size_t index=0;
    for(auto const& instruction:program.at("instructions")) {
        try {
            if(instruction.contains("set")) {
                auto v=evaluator.eval(instruction.at("value"));
                if(auto t=std::get_if<tensor>(&v.data))result=*t;
                evaluator.variables.insert_or_assign(instruction.at("set").get<std::string>(),std::move(v));
            } else {
                auto v=evaluator.eval(instruction.at("eval"));
                if(auto t=std::get_if<tensor>(&v.data))result=*t;
            }
        } catch(std::exception const& e) {
            throw std::runtime_error("recipe instruction "+std::to_string(index)+": "+e.what());
        }
        ++index;
    }
    if(!result)throw std::runtime_error("recipe: no output tensor");
    return result;
}
i64x4 runtime_graph_input_shape(model_file const& file) {
    auto program=read_program(file);
    auto shape=program.at("input_shape").get<std::vector<int64_t>>();
    if(shape.empty()||shape.size()>4)throw std::runtime_error("recipe: input rank must be 1..4");
    i64x4 result{1,1,1,1};
    // ggml reverses PyTorch axes, image runners use CWHN at the public boundary.
    if(shape.size()==4)result={shape[1],shape[3],shape[2],shape[0]};
    else if(shape.size()==3 && program.value("input_layout",std::string{"cwhn"})=="cwhn")
        result={shape[1],shape[2],1,shape[0]};
    else for(size_t i=0;i<shape.size();++i)result[int(i)]=shape[shape.size()-1-i];
    for(int i=0;i<4;++i)if(result[i]<=0 || result[i]>INT32_MAX)throw std::runtime_error("recipe: invalid input dimension");
    return result;
}
tensor_data_layout runtime_graph_weight_layout(model_file const& file, tensor_data_layout preferred) {
    // Conv1d/BatchNorm1d renderers use [L,C,N], rather than the CPU image layout.
    // Preserve the layout used by the existing non-image verification harness.
    auto program=read_program(file);
    return program.at("input_shape").size()==4 ? preferred : tensor_data_layout::whcn;
}
arch_task runtime_graph_task(model_file const& file) {
    arch_task task;
    auto program=read_program(file);
    auto shape=runtime_graph_input_shape(file);
    task.input_w=int(shape[1]);task.input_h=int(shape[2]);
    auto const& cfg=program.at("config");
    if(cfg.contains("mean")) task.mean=cfg.at("mean").get<std::array<float,3>>();
    if(cfg.contains("std")) task.stdv=cfg.at("std").get<std::array<float,3>>();
    task.to_rgb=cfg.value("to_rgb",false);
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
#include "runtime_mmdet.inc"
        c.det.input_w=task.input_w;c.det.input_h=task.input_h;
        std::copy(task.mean.begin(),task.mean.end(),c.img_mean);
        std::copy(task.stdv.begin(),task.stdv.end(),c.img_std);
        c.to_rgb=task.to_rgb;
        task.kind=arch_kind::detect_mmdet;
    }
    return task;
}
// Referenced explicitly from arch_registry.cpp, so static and shared builds retain it.
void register_runtime_graph() { static arch_registrar registration("mmlab", runtime_graph_forward); }
} // namespace visp
