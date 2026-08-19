// run_frcnn.cpp — two-stage 검출기(Faster/Mask R-CNN 계열)를 **2패스**로 돌린다.
//
// dense head 는 한 그래프로 끝나지만 two-stage 는 안 된다. RPN 이 낸 후보에서 NMS 로
// proposal 을 고르고(개수·좌표가 실행 중에 정해진다), 그 좌표에서 feature 를 잘라내
// (RoIAlign) 두 번째 head 에 넣는다. 둘 다 **값에 따라 달라지는 동작**이라 정적 그래프로
// 표현할 수 없다 — `ddq` 와 같은 이유다.
//
//   패스 0: SubA(백본+FPN+RPN)  →  호스트 rpn_proposals() → roi_align()
//   패스 1: SubB(RoI head)      →  호스트 detect_roi()
//
// 호스트 부품 셋(`rpn_proposals`·`roi_align`·`detect_roi`)은 `postproc.h` 에 이미 있다.
// 이 파일은 그것들을 잇는 배선이다.
//
// 컴파일: -DARCH_A=<SubA클래스> -DARCH_B=<SubB클래스>
//         -DVISP_ARCH_HEADER_A='"..."' -DVISP_ARCH_HEADER_B='"..."'
#include VISP_ARCH_HEADER_A
#include VISP_ARCH_HEADER_B
#ifdef VISP_ARCH_HEADER_C
#include VISP_ARCH_HEADER_C
#endif
#ifdef VISP_ARCH_HEADER_D
#include VISP_ARCH_HEADER_D
#endif
#ifdef VISP_ARCH_HEADER_E
#include VISP_ARCH_HEADER_E
#endif

#include "visp/ml.h"
#include "visp/postproc.h"

#include <ggml.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <span>
#include <string>
#include <vector>

using namespace visp;

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define FWD_A CAT(ARCH_A, _forward)
#define PRM_A CAT(ARCH_A, _detect_params)
#define FWD_B CAT(ARCH_B, _forward)
#define PRM_B CAT(ARCH_B, _detect_params)
#ifdef ARCH_C
#define FWD_C CAT(ARCH_C, _forward)
#define PRM_C CAT(ARCH_C, _detect_params)
#endif
#ifdef ARCH_D
#define FWD_D CAT(ARCH_D, _forward)
#define PRM_D CAT(ARCH_D, _detect_params)
#endif
#ifdef ARCH_E
#define FWD_E CAT(ARCH_E, _forward)
#define PRM_E CAT(ARCH_E, _detect_params)
#endif

static std::vector<float> load_bin(const char* path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

static void dump_bin(std::string const& path, std::vector<float> const& v) {
    if (FILE* f = fopen(path.c_str(), "wb")) {
        fwrite(v.data(), sizeof(float), v.size(), f);
        fclose(f);
    }
}

// 아주 작은 JSON 스칼라/배열 리더. 프론트엔드가 낸 `frcnn.json` 만 읽으면 되므로
// 완전한 파서가 필요 없다 — 키를 찾아 그 뒤 숫자를 읽는다.
struct tiny_json {
    std::string s;
    explicit tiny_json(const char* path) {
        std::ifstream f(path);
        s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    size_t find_key(const std::string& k) const {
        size_t p = s.find("\"" + k + "\"");
        return p == std::string::npos ? p : s.find(':', p) + 1;
    }
    float num(const std::string& k, float dflt) const {
        size_t p = find_key(k);
        if (p == std::string::npos) return dflt;
        if (s.compare(p, 5, "true") == 0 || s.compare(p + 1, 4, "true") == 0) return 1.0f;
        if (s.find("false", p) == p || s.find("false", p + 1) == p + 1) return 0.0f;
        return strtof(s.c_str() + p, nullptr);
    }
    // ⚠️ **중첩 배열을 평탄화해서 읽는다.** `stage_stds` 는 단계마다 4개씩 담은
    //    `[[...],[...],[...]]` 꼴이다. 닫는 대괄호를 `find(']')` 로 찾으면 **첫 안쪽 배열
    //    끝**에서 멈춰 4개만 읽힌다. 그러면 `st_stds.size() >= (st+1)*4` 가 st=0 만
    //    통과하고, **2단계부터는 stds 가 조용히 안 걸린다**(캐스케이드 실측: 0→1 정제는
    //    맞고 1→2 만 틀렸다 — 그래서 마지막 단계 텐서가 상대 L1 1.685 로 깨졌다).
    //    괄호 깊이를 세어 **짝이 맞는** 닫는 괄호까지 읽는다.
    std::vector<float> arr(const std::string& k) const {
        std::vector<float> out;
        size_t p = find_key(k);
        if (p == std::string::npos) return out;
        size_t l = s.find('[', p);
        if (l == std::string::npos) return out;
        size_t r = l;
        for (int depth = 0; r < s.size(); ++r) {
            if (s[r] == '[') ++depth;
            else if (s[r] == ']' && --depth == 0) break;
        }
        const char* c = s.c_str() + l + 1;
        while (c < s.c_str() + r) {
            char* e = nullptr;
            float v = strtof(c, &e);
            if (e == c) { ++c; continue; }
            out.push_back(v);
            c = e;
        }
        return out;
    }
};

// 그래프 출력 out_0..out_{n-1} 을 호스트로 읽는다.
static std::vector<std::vector<float>> read_outputs(compute_graph const& g, int n,
                                                    std::vector<std::pair<int, int>>* hw) {
    std::vector<std::vector<float>> out;
    for (int i = 0; i < n; ++i) {
        tensor t = ggml_graph_get_tensor(g.graph, ("out_" + std::to_string(i)).c_str());
        if (!t) break;
        std::vector<float> d(ggml_nelements(t));
        transfer_from_backend(t, std::span<float>(d.data(), d.size()));
        if (hw) hw->push_back({(int)t->ne[2], (int)t->ne[1]});   // cwhn: ne1=W, ne2=H
        out.push_back(std::move(d));
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <subA.gguf> <subB.gguf> <frcnn.json> <in_cwhn.bin> <out_prefix>"
                " [size] [subC.gguf] [subD.gguf] [subE.gguf]\n",
                argv[0]);
        return 1;
    }
    const char* ga = argv[1];
    const char* gb = argv[2];
    tiny_json J(argv[3]);
    const char* inb = argv[4];
    const std::string pref = argv[5];
    const int SZ = argc > 6 ? atoi(argv[6]) : 512;
    // Mask Scoring R-CNN 전용. 안 주면 마스크 경로를 아예 안 탄다(다른 계열은 그대로다).
    const char* gc = argc > 7 ? argv[7] : nullptr;
    const char* gd = argc > 8 ? argv[8] : nullptr;
    const char* ge = argc > 9 ? argv[9] : nullptr;   // Grid R-CNN 의 격자 head

    backend_device backend = backend_init();

    // ── 패스 0: SubA = 백본 + FPN + RPN head ─────────────────────────────────
    model_file fa = model_load(ga);
    model_weights wa = model_init(fa.n_tensors());
    model_transfer(fa, wa, backend, backend.preferred_float_type(), fa.tensor_layout());
    compute_graph g0 = compute_graph_init(262144);
    model_ref ma(wa, g0);
    tensor input = compute_graph_input(ma, GGML_TYPE_F32, {3, SZ, SZ, 1}, "x");
    ggml_build_forward_expand(g0, input);
    ggml_build_forward_expand(g0, FWD_A(ma, input, PRM_A(fa)));
    compute_graph_allocate(g0, backend);
    auto in = load_bin(inb, (size_t)3 * SZ * SZ);
    transfer_to_backend(input, std::span<const float>(in.data(), in.size()));
    compute(g0, backend);

    // SubA 출력 규약(frcnn_wrap.FRCNN_SubA): RoI 레벨 feats(NF) + rpn_cls×L + rpn_bbox×L
    // ⚠️ **NF 를 4 로 박으면 안 된다.** FPN 계열은 P2..P5 로 4개지만, C4 계열(TridentNet)은
    //    neck 이 없어 백본 C4 **하나**만 온다. `roi_strides` 길이가 정답이다
    //    (프론트엔드의 `_n_roi_levels` 와 같은 근거를 쓴다).
    std::vector<std::pair<int, int>> all_hw;
    auto outs = read_outputs(g0, 64, &all_hw);
    // ⚠️ **RPN 이 없는 계열이 있다.** `FastRCNN` 은 proposal 을 밖에서 받는 것이 정의라
    //    `rpn_head` 도 `test_cfg.rpn` 도 없다 — SubA 는 feats 만 낸다. 미지원이 아니라
    //    **다른 계약**이므로 proposal 을 파일에서 읽고 RPN 계산을 건너뛴다.
    const bool ext_props = J.num("external_proposals", 0.0f) != 0.0f;
    const int L = ext_props ? 0 : (int)J.arr("rpn_strides").size();
    const int NF = (int)J.arr("roi_strides").size();
    if ((int)outs.size() < NF + 2 * L) {
        fprintf(stderr, "SubA 출력 %zu 개 < 기대 %d (feats %d + rpn %d×2)\n",
                outs.size(), NF + 2 * L, NF, L);
        return 3;
    }
    // HTC 의 시맨틱 융합 feature 는 rpn 출력 **뒤에** 하나 더 붙어 온다.
    const bool has_sem = J.num("has_semantic", 0.0f) != 0.0f;
    std::vector<float> sem_feat;
    std::pair<int, int> sem_hw{0, 0};
    if (has_sem && (int)outs.size() > NF + 2 * L) {
        sem_feat = outs[NF + 2 * L];
        sem_hw = all_hw[NF + 2 * L];
    }
    // SCNet 의 전역 컨텍스트는 시맨틱 **다음** 자리에 온다(둘 다 있으면 +1, +2).
    const bool has_glb = J.num("has_glbctx", 0.0f) != 0.0f;
    std::vector<float> glb_feat;
    if (has_glb) {
        const int gi = NF + 2 * L + (has_sem ? 1 : 0);
        if ((int)outs.size() > gi) glb_feat = outs[gi];
    }
    std::vector<std::vector<float>> feats(outs.begin(), outs.begin() + NF);
    std::vector<std::vector<float>> rpn_cls(outs.begin() + NF, outs.begin() + NF + L);
    std::vector<std::vector<float>> rpn_box(outs.begin() + NF + L, outs.begin() + NF + 2 * L);
    std::vector<std::pair<int, int>> feat_hw(all_hw.begin(), all_hw.begin() + NF);
    std::vector<std::pair<int, int>> rpn_hw(all_hw.begin() + NF, all_hw.begin() + NF + L);

    // ── 호스트: RPN proposal (동적 — NMS) ────────────────────────────────────
    rpn_params rp;
    rp.strides = J.arr("rpn_strides");
    rp.octave_base_scale = J.num("rpn_scale", 8.0f);
    // 한 레벨에 스케일이 여럿인 계열(C4)이 있다. 목록이 있으면 그게 정본이다.
    if (std::vector<float> sc = J.arr("rpn_scales"); !sc.empty()) rp.octave_scales = sc;
    rp.ratios = J.arr("rpn_ratios");
    rp.nms_pre = (int)J.num("rpn_nms_pre", 1000);
    rp.nms_thr = J.num("rpn_nms_thr", 0.7f);
    rp.max_per_img = (int)J.num("rpn_max", 1000);
    rp.centers = J.arr("rpn_centers");                       // 비어 있으면 재현식
    rp.clip_border = J.num("rpn_clip_border", 1.0f) != 0.0f;
    rp.min_bbox_size = J.num("rpn_min_bbox_size", 0.0f);
    rp.cls_out_channels = (int)J.num("rpn_cls_out_channels", 1.0f);
    rp.input_w = rp.input_h = SZ;
    std::vector<float> props;
    if (ext_props) {
        // proposal 파일: float32 x1,y1,x2,y2 반복. 기준값 쪽과 **같은 파일**을 읽어야
        // 비교가 성립한다 — 각자 만들면 proposal 생성기를 재게 된다.
        const char* pp = getenv("FRCNN_PROPOSALS");
        if (!pp) { fprintf(stderr, "external_proposals 인데 FRCNN_PROPOSALS 가 없다\n"); return 3; }
        std::ifstream f(pp, std::ios::binary);
        if (!f) { fprintf(stderr, "proposal 파일을 못 연다: %s\n", pp); return 3; }
        f.seekg(0, std::ios::end);
        const size_t n = (size_t)f.tellg() / sizeof(float);
        f.seekg(0);
        props.resize(n);
        f.read(reinterpret_cast<char*>(props.data()), (std::streamsize)(n * sizeof(float)));
        rp.max_per_img = (int)(n / 4);
    } else {
        props = rpn_proposals(rpn_cls, rpn_box, rpn_hw, rp);
    }
    const int M_real = (int)(props.size() / 4);
    // ⚠️ **proposal 이 상한보다 적게 나오는 계열이 있다.** SubB 는 상한(rpn_max)으로
    //    구워지고 그 안의 `flatten` 이 `ggml_reshape_2d(…, 2048, 1000)` 처럼 **행 수를
    //    상수로 박는다** — 적게 넣으면 `ggml_nelements(a) == ne0*ne1` 로 죽는다.
    //    FPN 계열은 후보가 많아 항상 상한을 채우지만, C4 계열(TridentNet)은 레벨이
    //    하나뿐이라 NMS 뒤 107개만 남았다. 상한까지 0 으로 채우고 **디코드는 앞
    //    `M_real` 행만** 쓴다(0 박스는 RoIAlign 이 레벨 0 의 (0,0) 을 읽을 뿐 무해하다).
    const int M = std::max(M_real, rp.max_per_img);
    props.resize((size_t)M * 4, 0.0f);
    fprintf(stderr, "[frcnn] proposal %d 개 (상한 %d 까지 채움, nms %.2f)\n",
            M_real, M, rp.nms_thr);
    if (M_real == 0) {
        fprintf(stderr, "proposal 이 0 개다 — RPN 출력/규약 확인\n");
        return 4;
    }

    // ── 호스트: RoIAlign (동적 — 좌표가 실행 중에 정해진다) ──────────────────
    roi_align_params ap;
    ap.output_size = (int)J.num("roi_out", 7);
    // ⚠️ C4 계열은 FPN 이 없어 채널이 256 이 아니다(TridentNet=1024).
    ap.channels = (int)J.num("roi_channels", 256.0f);
    ap.strides = J.arr("roi_strides");
    ap.finest_scale = J.num("roi_finest_scale", 56.0f);
    ap.sampling_ratio = (int)J.num("roi_sampling_ratio", 0);
    ap.aligned = J.num("roi_aligned", 1.0f) != 0.0f;
    // ⚠️ **groie 는 레벨을 고르지 않는다.** GenericRoIExtractor 는 전 레벨에 RoIAlign 을
    //    걸고, 레벨마다 5x5 conv+ReLU 를 태운 뒤 더한다(conv+ReLU 가 비선형이라 합과
    //    교환이 안 되므로 "합쳐서 한 번" 으로 못 접는다). 그래프 입력이 하나뿐이라
    //    Double-Head 와 같은 수법으로 **레벨별 결과를 배치로 이어붙여** 넘기고,
    //    pre→합산→post 는 SubB 가 그래프 안에서 한다.
    const int groie_lv = (int)J.num("groie_levels", 0.0f);
    std::vector<float> roi;
    if (groie_lv > 0) {
        roi.reserve((size_t)groie_lv * M * ap.channels * ap.output_size * ap.output_size);
        for (int l = 0; l < groie_lv; ++l) {
            roi_align_params lp = ap;
            lp.force_level = l;
            std::vector<float> one = roi_align(feats, feat_hw, props.data(), M, lp);
            roi.insert(roi.end(), one.begin(), one.end());
        }
    } else {
        roi = roi_align(feats, feat_hw, props.data(), M, ap);
    }

    // ── HTC: 시맨틱 feature 를 RoIAlign 해서 bbox_feats 에 **더한다** ──────────
    // mmdet `htc_roi_head.py:99-104`. 안 더하면 크래시 없이 박스만 밀린다
    // (실측: 이걸 뺀 torch 가 우리 결과와 0.07px 로 일치했다 — 유일한 차이였다).
    // ⚠️ 시맨틱 RoI 출력 크기가 **14** 라 bbox 쪽 7 과 다르다. mmdet 은 그때
    //    `adaptive_avg_pool2d` 로 줄인다. 14→7 은 정수배라 2×2 평균이면 정확하다.
    auto fuse_semantic = [&](std::vector<float>& rf, float const* boxes, int m) {
        if (!has_sem || sem_feat.empty()) return;
        roi_align_params sp;
        sp.output_size = (int)J.num("sem_roi_out", 14.0f);
        sp.channels = (int)J.num("sem_channels", 256.0f);
        sp.strides = {J.num("sem_stride", 8.0f)};
        sp.finest_scale = ap.finest_scale;
        sp.sampling_ratio = (int)J.num("sem_sampling_ratio", 0.0f);
        sp.aligned = J.num("sem_aligned", 1.0f) != 0.0f;
        std::vector<std::vector<float>> sf{sem_feat};
        std::vector<std::pair<int, int>> shw{sem_hw};
        std::vector<float> s = roi_align(sf, shw, boxes, m, sp);
        const int C = sp.channels, SO = sp.output_size, O = ap.output_size;
        const int k = SO / O;                       // 14/7 = 2
        if (k < 1 || SO != k * O) return;           // 정수배가 아니면 건너뛴다
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < C; ++c)
                for (int y = 0; y < O; ++y)
                    for (int x = 0; x < O; ++x) {
                        float acc = 0.0f;
                        for (int dy = 0; dy < k; ++dy)
                            for (int dx = 0; dx < k; ++dx)
                                acc += s[(((size_t)i * C + c) * SO + y * k + dy) * SO + x * k + dx];
                        rf[(((size_t)i * C + c) * O + y) * O + x] += acc / (k * k);
                    }
    };
    // ── SCNet: 전역 컨텍스트를 모든 RoI feature 에 **채널별로 더한다** ────────
    // `scnet_roi_head._fuse_glbctx` — glbctx 는 AdaptiveAvgPool(1) 을 거쳐 이미지당
    // 채널 벡터 하나다(B,C,1,1). 배치가 1 이므로 그 벡터를 전 RoI·전 위치에 더한다.
    // 안 더하면 크래시 없이 박스만 밀린다(실측: 이걸 뺀 torch 가 우리 결과와 0.45px).
    auto fuse_glbctx = [&](std::vector<float>& rf, int m) {
        if (!has_glb || glb_feat.empty()) return;
        const int C = ap.channels, O = ap.output_size;
        if ((int)glb_feat.size() < C) return;
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < C; ++c) {
                const float g = glb_feat[c];
                for (int y = 0; y < O; ++y)
                    for (int x = 0; x < O; ++x)
                        rf[(((size_t)i * C + c) * O + y) * O + x] += g;
            }
    };
    fuse_semantic(roi, props.data(), M);
    fuse_glbctx(roi, M);

    // 캐스케이드는 단계마다 박스를 정제하고 **그 박스로 RoIAlign 을 다시** 한다.
    // 단계 수와 단계별 정규화 상수는 프론트엔드가 frcnn.json 에 실어 준다.
    const int NS = (int)J.num("num_bbox_stages", 1);

    // ── 패스 1..NS: SubB = RoI head (단계마다) ─────────────────────
    // 캐스케이드 단계들은 **구조가 같고 가중치만 다르다**(Shared2FCBBoxHead x3).
    // 그래서 그래프는 매 단계 같은 함수로 짜고 gguf 만 갈아 \łłłł끼운다.
    const int O = ap.output_size, C = ap.channels;
    const std::vector<float> st_stds = J.arr("stage_stds"), st_means = J.arr("stage_means");
    std::vector<std::vector<float>> cls_st, box_st;
    std::vector<float> rois = props;

    // gguf 경로는 쉼표로 구분해 단계 수만큼 받는다.
    std::vector<std::string> gbs;
    {
        std::string s(gb), cur;
        for (char ch : s) { if (ch == ',') { gbs.push_back(cur); cur.clear(); } else cur += ch; }
        gbs.push_back(cur);
    }
    if ((int)gbs.size() != NS) {
        fprintf(stderr, "SubB gguf %zu 개 != 단계 %d\n", gbs.size(), NS);
        return 6;
    }

    // Double-Head R-CNN: 회귀용 RoI 는 상자를 중심 기준 `RSF` 배 키워 **다시** 자른다
    // (mmdet `BaseRoIExtractor.roi_rescale` — 이미지 경계로 자르지 않는다).
    // 그래프 입력은 하나뿐이라 두 벌을 **배치 방향으로 이어** 넘기고, SubB 래퍼가 가른다.
    const float RSF = J.num("reg_roi_scale_factor", 0.0f);
    auto rescale_boxes = [&](std::vector<float> const& b) {
        std::vector<float> o(b.size());
        for (size_t i = 0; i + 3 < b.size(); i += 4) {
            const float cx = (b[i] + b[i + 2]) * 0.5f, cy = (b[i + 1] + b[i + 3]) * 0.5f;
            const float w = (b[i + 2] - b[i]) * RSF * 0.5f;
            const float h = (b[i + 3] - b[i + 1]) * RSF * 0.5f;
            o[i] = cx - w; o[i + 1] = cy - h; o[i + 2] = cx + w; o[i + 3] = cy + h;
        }
        return o;
    };
    auto with_reg_half = [&](std::vector<float> rf, std::vector<float> const& boxes) {
        if (RSF <= 0.0f) return rf;
        std::vector<float> bx = rescale_boxes(boxes);
        // ⚠️ **레벨은 키우기 전 박스(`boxes`)로 고른다.** mmdet 은 `map_roi_levels` 를
        //    `roi_rescale` **앞**에서 부른다(single_level_roi_extractor.py:97-100).
        //    키운 박스로 레벨까지 고르면 경계 상자가 한 레벨 위에서 읽혀 조용히 밀린다
        //    (실측: double_heads fp32 8.56px — fp16 잡음이 이걸 가리고 있었다).
        std::vector<float> rg = roi_align(feats, feat_hw, bx.data(), M, ap, boxes.data());
        rf.insert(rf.end(), rg.begin(), rg.end());
        return rf;
    };

    for (int st = 0; st < NS; ++st) {
        // ⚠️ 시맨틱 융합은 **단계마다** 건다. mmdet 의 `_bbox_forward(stage, …, semantic_feat)`
        //    가 매 단계 부른다 — 0단계만 더하면 뒤 단계가 융합 없이 돈다.
        std::vector<float> base_st;
        if (st == 0) {
            base_st = roi;                       // 위에서 이미 융합했다
        } else {
            base_st = roi_align(feats, feat_hw, rois.data(), M, ap);
            fuse_semantic(base_st, rois.data(), M);
            fuse_glbctx(base_st, M);
        }
        std::vector<float> roi_st = with_reg_half(base_st, (st == 0) ? props : rois);
        // SubB 에 넣는 행 수. Double-Head 는 2배(cls+reg), groie 는 레벨 배다 —
        // 둘 다 "그래프 입력이 하나뿐이라 배치로 이어붙인다" 는 같은 이유다.
        const int MB = (RSF > 0.0f) ? 2 * M : (groie_lv > 0 ? groie_lv * M : M);

        model_file fb = model_load(gbs[st].c_str());
        model_weights wb = model_init(fb.n_tensors());
        model_transfer(fb, wb, backend, backend.preferred_float_type(), fb.tensor_layout());
        compute_graph g1 = compute_graph_init(65536);
        model_ref mb(wb, g1);
        tensor rin = compute_graph_input(mb, GGML_TYPE_F32, {C, O, O, MB}, "roi");
        ggml_build_forward_expand(g1, rin);
        ggml_build_forward_expand(g1, FWD_B(mb, rin, PRM_B(fb)));
        compute_graph_allocate(g1, backend);
        // roi_align 은 NCHW flat 을 낸다. 생성 코드는 cwhn 규약이라 채널을 ne0 로 돌린다.
        std::vector<float> roi_cwhn((size_t)MB * C * O * O);
        for (int n = 0; n < MB; ++n)
            for (int c = 0; c < C; ++c)
                for (int y = 0; y < O; ++y)
                    for (int x = 0; x < O; ++x)
                        roi_cwhn[(((size_t)n * O + y) * O + x) * C + c] =
                            roi_st[(((size_t)n * C + c) * O + y) * O + x];
        transfer_to_backend(rin, std::span<const float>(roi_cwhn.data(), roi_cwhn.size()));
        compute(g1, backend);

        auto bo = read_outputs(g1, 8, nullptr);
        if (bo.size() == 1) {
            // ⚠️ **회귀 분기가 없는 bbox head 가 있다.** Grid R-CNN 은 `with_reg=False`
            //    로 박스를 아예 예측하지 않고 격자 head 가 나중에 다시 낸다. mmdet 도
            //    `bbox_pred is None` 이면 `rois` 를 그대로 박스로 쓴다
            //    (`bbox_head.predict_by_feat`). 0 델타를 넣으면 같은 결과가 된다 —
            //    평균 0 · exp(0)=1 이라 디코드가 항등이 되기 때문이다.
            //    여기서 막고 "출력이 부족하다" 로 실패시키면 계열을 못 재본다.
            const int ncls = (int)(bo[0].size() / M) - 1;
            bo.push_back(std::vector<float>((size_t)M * ncls * 4, 0.0f));
            fprintf(stderr, "[frcnn] bbox head 에 회귀 분기가 없다 → RoI 를 박스로 쓴다\n");
        }
        if (bo.size() < 2) {
            fprintf(stderr, "SubB 출력이 2개 미만이다(cls_score, bbox_pred 필요)\n");
            return 5;
        }
        // ⚠️ head 가 (cls, box) **한 쌍만** 낸다고 가정하면 안 된다. CrowdDet 의
        //    `MultiInstanceBBoxHead` 는 proposal 하나가 사람 둘을 낼 수 있다고 보고
        //    쌍을 2벌 낸다(총 4텐서). 앞 두 개만 재면 **나머지 절반이 검증 안 된 채**
        //    통과한다 — 이 저장소가 계속 물린 "조용히 틀림" 부류다.
        //    쌍 단위로 전부 담고, 단계 정제(캐스케이드)에는 0번 쌍만 쓴다.
        const int NPAIR = (int)bo.size() / 2;
        for (int p = 0; p < NPAIR; ++p) {
            cls_st.push_back(bo[2 * p]);
            box_st.push_back(bo[2 * p + 1]);
        }
        // 캐스케이드 정제·덤프 인덱스는 **단계**를 세는 것이라 쌍이 여럿이면 어긋난다.
        // 지금 지원 조합은 (다단계 × 1쌍) 또는 (1단계 × 여러 쌍) 둘 중 하나다.
        if (NS > 1 && NPAIR > 1) {
            fprintf(stderr, "캐스케이드(%d단계)와 다중 인스턴스(%d쌍)를 함께 못 쓴다\n",
                    NS, NPAIR);
            return 5;
        }
        if (st == 0) roi = roi_st;              // 덤프용(1단계 RoI feature)

        if (st + 1 < NS) {
            // ── 호스트: 박스 정제 (mmdet `regress_by_class`) ──
            // 예측 클래스(배경 제외 argmax)의 delta 만 골라 디코드한다.
            // ⚠ 단계마다 stds 가 다르다(0.1 -> 0.05 -> 0.033) — 하나로 쓰면 조용히 틀린다.
            const int NC = (int)(cls_st[st].size() / M) - 1;      // 배경 제외
            const bool agn = J.num("class_agnostic", 0.0f) != 0.0f;
            const bool has_s = st_stds.size() >= (size_t)(st + 1) * 4;
            const bool has_m = st_means.size() >= (size_t)(st + 1) * 4;
            std::vector<float> nb((size_t)M * 4);
            for (int i = 0; i < M; ++i) {
                int best = 0;
                for (int c = 1; c < NC; ++c)
                    if (cls_st[st][(size_t)i * (NC + 1) + c] > cls_st[st][(size_t)i * (NC + 1) + best])
                        best = c;
                const size_t off = (size_t)i * (agn ? 4 : (size_t)NC * 4) + (agn ? 0 : (size_t)best * 4);
                float d[4];
                for (int k = 0; k < 4; ++k) {
                    d[k] = box_st[st][off + k];
                    if (has_s) d[k] *= st_stds[(size_t)st * 4 + k];
                    if (has_m) d[k] += st_means[(size_t)st * 4 + k];
                }
                // ⚠️ mmdet 은 dw/dh 를 **±|log(wh_ratio_clip)|** 로 자른다
                //    (delta_xywh_bbox_coder.py:345-350, 기본 16/1000 → 4.135).
                //    안 자르면 `exp(dw)` 가 폭주해 박스가 수백 px 로 튄다.
                const float MAXR = 4.13516655f;   // |log(16/1000)|
                d[2] = std::min(std::max(d[2], -MAXR), MAXR);
                d[3] = std::min(std::max(d[3], -MAXR), MAXR);
                const float x1 = rois[(size_t)i * 4 + 0], y1 = rois[(size_t)i * 4 + 1];
                const float x2 = rois[(size_t)i * 4 + 2], y2 = rois[(size_t)i * 4 + 3];
                const float pw = x2 - x1, ph = y2 - y1;
                const float cx = x1 + pw * 0.5f + d[0] * pw, cy = y1 + ph * 0.5f + d[1] * ph;
                const float w = pw * std::exp(d[2]), h = ph * std::exp(d[3]);
                nb[(size_t)i * 4 + 0] = std::max(0.0f, cx - w * 0.5f);
                nb[(size_t)i * 4 + 1] = std::max(0.0f, cy - h * 0.5f);
                nb[(size_t)i * 4 + 2] = std::min((float)SZ, cx + w * 0.5f);
                nb[(size_t)i * 4 + 3] = std::min((float)SZ, cy + h * 0.5f);
            }
            rois.swap(nb);
        }
    }

    // ── 최종 디코드: cls/box → 박스 ─────────────────────────────────────────
    // `detect_roi` 는 **softmax 를 마친** 점수를 기대한다(postproc.h). SubB 가 내는 것은
    // 로짓이므로(실측: 한 행의 합이 -0.41) 여기서 건다. 안 걸면 크래시 없이 점수만 틀린다.
    //
    // ⚠️ **캐스케이드는 마지막 단계만 쓰면 안 된다.** mmdet 은 전 단계의 cls 를 모아
    //    평균한 뒤 디코드한다(`cascade_roi_head.py:517`
    //    `sum([score[i] for score in ms_scores]) / float(len(ms_scores))`).
    //    마지막 단계만 쓰면 크래시 없이 점수가 틀리고, 점수가 틀리면 NMS 가 남기는 집합이
    //    달라져 박스까지 어긋난다(실측: cascade_rcnn 607px).
    //
    // ⚠️ **평균은 로짓에서 한다. softmax 를 먼저 걸면 안 된다.** mmdet 이 `ms_scores` 에
    //    담는 것은 `_bbox_forward` 의 **날 cls_score** 이고, softmax 는 그 평균 뒤
    //    `bbox_head.py:526` 에서 한 번만 걸린다. 순서를 바꾸면 값이 달라진다
    //    (softmax 는 선형이 아니다 — mean∘softmax ≠ softmax∘mean).
    //
    //    박스는 평균하지 않는다. mmdet 도 `bbox_preds` 는 마지막 단계 것을 그대로 쓴다.
    std::vector<detection> dets;
    // ── CrowdDet: proposal 하나가 사람 둘 — 디코드도 NMS 도 다르다 ──────────
    const int NI = (int)J.num("num_instance", 1.0f);
    if (NI > 1 && !cls_st.empty() && !box_st.empty()) {
        // ⚠️ **정제된 쌍을 쓴다.** head 가 (cls, box, cls_ref, box_ref) 넷을 내고
        //    mmdet 은 정제된 쪽으로 예측한다(`multi_instance_roi_head.py`).
        //    앞 쌍을 쓰면 크래시 없이 점수만 달라진다.
        const bool refine = J.num("with_refine", 0.0f) != 0.0f;
        const int use = (refine && cls_st.size() >= 2) ? 1 : 0;
        std::vector<float> const& cls = cls_st[use];
        std::vector<float> const& box = box_st[use];
        // cat(dim=1) 이라 proposal 하나에 [inst0 (C+1)값][inst1 (C+1)값] 이 붙어 있다.
        const int NCLS = (int)(cls.size() / ((size_t)M * NI)) - 1;
        const float score_thr = J.num("rcnn_score_thr", 0.01f);
        const float nms_thr = J.num("rcnn_nms_thr", 0.5f);
        const int max_img = (int)J.num("rcnn_max", 500.0f);
        const std::vector<float> rm = J.arr("rcnn_means"), rs = J.arr("rcnn_stds");
        float mean[4] = {0, 0, 0, 0}, sd[4] = {1, 1, 1, 1};
        for (int k = 0; k < 4; ++k) {
            if (rm.size() >= 4) mean[k] = rm[k];
            if (rs.size() >= 4) sd[k] = rs[k];
        }
        struct cand { float x1, y1, x2, y2, score; int roi; };
        std::vector<cand> cs;
        for (int i = 0; i < M_real; ++i) {
            const float px1 = rois[(size_t)i * 4 + 0], py1 = rois[(size_t)i * 4 + 1];
            const float px2 = rois[(size_t)i * 4 + 2], py2 = rois[(size_t)i * 4 + 3];
            const float pw = px2 - px1, ph = py2 - py1;
            for (int p = 0; p < NI; ++p) {
                float const* c = cls.data() + ((size_t)i * NI + p) * (NCLS + 1);
                float mx = c[0];
                for (int k = 1; k <= NCLS; ++k) mx = std::max(mx, c[k]);
                float sum = 0.0f;
                for (int k = 0; k <= NCLS; ++k) sum += std::exp(c[k] - mx);
                const float fg = std::exp(c[1] - mx) / sum;   // 전경 확률(클래스 1)
                if (fg <= score_thr) continue;
                float const* d = box.data() + ((size_t)i * NI + p) * 4;
                const float dx = d[0] * sd[0] + mean[0], dy = d[1] * sd[1] + mean[1];
                const float dw = d[2] * sd[2] + mean[2], dh = d[3] * sd[3] + mean[3];
                const float cx = px1 + pw * 0.5f + dx * pw, cy = py1 + ph * 0.5f + dy * ph;
                const float w = pw * std::exp(dw), h = ph * std::exp(dh);
                cs.push_back({std::max(0.0f, cx - w * 0.5f), std::max(0.0f, cy - h * 0.5f),
                              std::min((float)SZ, cx + w * 0.5f),
                              std::min((float)SZ, cy + h * 0.5f), fg, i});
            }
        }
        std::stable_sort(cs.begin(), cs.end(),
                         [](cand const& a, cand const& b) { return a.score > b.score; });
        // set-NMS — 평범한 NMS 인데 **같은 proposal 에서 나온 상자는 서로 안 누른다.**
        // 그게 이 계열의 핵심이다: 겹쳐 선 두 사람은 IoU 가 높아 보통 NMS 면 하나가
        // 지워진다. 이걸 빼면 억제가 통째로 어긋나 500건이 그대로 남는다(실측 1:500).
        std::vector<char> keep(cs.size(), 1);
        for (size_t a = 0; a < cs.size(); ++a) {
            if (!keep[a]) continue;
            for (size_t b = a + 1; b < cs.size(); ++b) {
                if (!keep[b] || cs[b].roi == cs[a].roi) continue;   // 같은 proposal 은 면제
                const float ix1 = std::max(cs[a].x1, cs[b].x1), iy1 = std::max(cs[a].y1, cs[b].y1);
                const float ix2 = std::min(cs[a].x2, cs[b].x2), iy2 = std::min(cs[a].y2, cs[b].y2);
                const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
                const float inter = iw * ih;
                const float ua = (cs[a].x2 - cs[a].x1) * (cs[a].y2 - cs[a].y1) +
                                 (cs[b].x2 - cs[b].x1) * (cs[b].y2 - cs[b].y1) - inter;
                if (ua > 0.0f && inter / ua > nms_thr) keep[b] = 0;
            }
        }
        for (size_t i = 0; i < cs.size() && (int)dets.size() < max_img; ++i) {
            if (keep[i]) dets.push_back({cs[i].x1, cs[i].y1, cs[i].x2, cs[i].y2, cs[i].score, 0});
        }
    } else if (!cls_st.empty() && !box_st.empty()) {
        const int last = (int)cls_st.size() - 1;
        // ⚠️ **채널 수를 출력 크기에서 -1 로 유추하면 안 되는 계열이 있다.** SeesawLoss 는
        //    `num_classes + 2` 를 쓴다(앞 C 개 = 클래스, 뒤 2개 = 전경/배경 objectness).
        //    -1 로 읽으면 클래스가 하나 많아지고 라벨이 통째로 한 칸 밀린다.
        const bool seesaw = J.num("seesaw_activation", 0.0f) != 0.0f;
        const int NCLS = seesaw ? (int)J.num("cls_num_classes", 0.0f)
                                : (int)(cls_st[last].size() / M) - 1;
        // 캐스케이드 단계들만 평균한다. 다중 인스턴스(CrowdDet)는 단계가 아니라 **쌍**이라
        // 평균 대상이 아니다 — 위에서 (NS>1 && NPAIR>1) 을 막아 뒀으므로 여기서는
        // `NS > 1` 일 때만 여러 원소가 단계를 뜻한다.
        const int NAVG = (NS > 1) ? (int)cls_st.size() : 1;
        std::vector<float> prob((size_t)M * (NCLS + 1));
        if (seesaw) {
            // mmdet `SeesawLoss.get_activation`: 클래스와 objectness 를 따로 softmax 하고
            // 클래스 점수에 전경 확률을 곱한 뒤 배경 확률을 뒤에 붙인다.
            const int RAW = NCLS + 2;
            for (int i = 0; i < M; ++i) {
                float const* src = cls_st[last].data() + (size_t)i * RAW;
                float* dst = prob.data() + (size_t)i * (NCLS + 1);
                float mx = src[0];
                for (int c = 1; c < NCLS; ++c) mx = std::max(mx, src[c]);
                float sum = 0.0f;
                for (int c = 0; c < NCLS; ++c) { dst[c] = std::exp(src[c] - mx); sum += dst[c]; }
                for (int c = 0; c < NCLS; ++c) dst[c] /= sum;
                const float om = std::max(src[NCLS], src[NCLS + 1]);
                const float ep = std::exp(src[NCLS] - om), en = std::exp(src[NCLS + 1] - om);
                const float pos = ep / (ep + en), neg = en / (ep + en);
                for (int c = 0; c < NCLS; ++c) dst[c] *= pos;
                dst[NCLS] = neg;                         // 배경은 맨 뒤
            }
        } else
        for (int i = 0; i < M; ++i) {
            float* dst = prob.data() + (size_t)i * (NCLS + 1);
            // ① 단계별 로짓 평균
            for (int c = 0; c <= NCLS; ++c) {
                float acc = 0.0f;
                for (int s = 0; s < NAVG; ++s)
                    acc += cls_st[(NAVG == 1) ? last : s][(size_t)i * (NCLS + 1) + c];
                dst[c] = acc / (float)NAVG;
            }
            // ② 그 다음에 softmax
            float mx = dst[0];
            for (int c = 1; c <= NCLS; ++c) mx = std::max(mx, dst[c]);
            float sum = 0.0f;
            for (int c = 0; c <= NCLS; ++c) { dst[c] = std::exp(dst[c] - mx); sum += dst[c]; }
            for (int c = 0; c <= NCLS; ++c) dst[c] /= sum;
        }
        roi_params rp2;
        const std::vector<float> rm = J.arr("rcnn_means"), rs = J.arr("rcnn_stds");
        for (int k = 0; k < 4; ++k) {
            if (rm.size() >= 4) rp2.means[k] = rm[k];
            if (rs.size() >= 4) rp2.stds[k] = rs[k];
        }
        rp2.num_classes = NCLS;
        rp2.class_agnostic = J.num("class_agnostic", 0.0f) != 0.0f;
        rp2.score_thr = J.num("rcnn_score_thr", 0.05f);
        rp2.nms_thr = J.num("rcnn_nms_thr", 0.5f);
        rp2.max_per_img = (int)J.num("rcnn_max", 100.0f);
        rp2.input_w = SZ;
        rp2.input_h = SZ;
        // 마지막 단계의 박스는 그 단계에 **들어간** RoI 기준이다. 캐스케이드에서 `rois` 는
        // 이미 다음 단계용으로 갱신되지 않으므로(마지막 단계는 정제를 건너뛴다) 그대로 쓴다.
        dets = detect_roi(prob.data(), box_st[last].data(), rois.data(), M_real, rp2);
    }

#if defined(ARCH_C) && defined(ARCH_D)
    // ── 패스 2: 마스크 IoU 로 점수를 다시 매긴다 (Mask Scoring R-CNN) ───────
    // mmdet 은 `score * mask_iou[label]` 로 점수를 낮춘다(maskiou_head.predict_by_feat).
    // 이걸 빼면 **박스는 맞는데 점수만 틀린다** — 실측 박스 0.07px / 점수 0.163.
    // 게다가 점수가 임계값을 넘나들어 개수까지 달라진다(mmdet 4건 vs C++ 6건).
    //
    // ⚠️⚠️ **두 그래프를 검출 하나씩(배치 1) 돌린다. 묶어 넣으면 안 된다.**
    //    `ggml_compute_forward_conv_transpose_2d` 는 **배치 축을 안 돈다** — src1 을
    //    풀 때 i12(채널)·i11(행)만 돌고 i13(배치)이 없고, 출력도 `dst->data + i2*nb2`
    //    로 Cout 만 인덱싱한다(ggml-cpu/ops.cpp). 그래서 배치 20으로 넣으면 **0번 행만
    //    계산되고 나머지 19행은 bias 만 남는다.** 크래시도 경고도 없다 —
    //    실측: 행0 rel_L1 6.0e-04, 행1..4 는 전부 1.01 이고 |x| 평균이 서로 똑같았다
    //    (0.0870 = bias 뿐). mask head 의 `upsample`(14→28 deconv)이 그 경로다.
    if (J.num("has_mask_iou", 0.0f) != 0.0f && gc && gd && !dets.empty()) {
        const int MD = (int)dets.size();
        // ⚠️ **박스를 원본 해상도로 되돌리지 않는다.** mmdet 은 마스크 분기가 있으면
        //    bbox 쪽 rescale 을 끈다(`base_roi_head.py`:
        //    `bbox_rescale = rescale if not self.with_mask else False`) — 마스크 쪽이
        //    박스와 마스크를 한꺼번에 되돌리기 때문이다. 여기 좌표는 이미 feature 계다.
        std::vector<float> mbox((size_t)MD * 4);
        std::vector<int> mlab(MD);
        for (int i = 0; i < MD; ++i) {
            mbox[(size_t)i * 4 + 0] = dets[i].x1; mbox[(size_t)i * 4 + 1] = dets[i].y1;
            mbox[(size_t)i * 4 + 2] = dets[i].x2; mbox[(size_t)i * 4 + 3] = dets[i].y2;
            mlab[i] = dets[i].label;
        }
        roi_align_params mp;
        mp.output_size = (int)J.num("mask_roi_out", 14.0f);
        mp.channels = C;
        mp.strides = J.arr("mask_strides");
        mp.finest_scale = J.num("mask_finest_scale", 56.0f);
        // 마스크 extractor 가 쓰는 레벨 수는 bbox 쪽과 다를 수 있다(P6 를 안 쓴다).
        const int ML = std::min((int)mp.strides.size(), (int)feats.size());
        std::vector<std::vector<float>> mfeats(feats.begin(), feats.begin() + ML);
        std::vector<std::pair<int, int>> mhw(feat_hw.begin(), feat_hw.begin() + ML);
        const int MO = mp.output_size;
        std::vector<float> mfeat = roi_align(mfeats, mhw, mbox.data(), MD, mp);

        // 모델은 한 번만 올린다. 행마다 바뀌는 것은 그래프뿐이다.
        model_file fc = model_load(gc);
        model_weights wc = model_init(fc.n_tensors());
        model_transfer(fc, wc, backend, backend.preferred_float_type(), fc.tensor_layout());
        model_file fd = model_load(gd);
        model_weights wd = model_init(fd.n_tensors());
        model_transfer(fd, wd, backend, backend.preferred_float_type(), fd.tensor_layout());

        std::vector<float> logit_all, din_all, miou(MD, 1.0f);
        int MH = 0, MW = 0, NCLS_M = 0;
        const int CD = C + 1;
        for (int n = 0; n < MD; ++n) {
            // ① SubC = mask head. (1,256,14,14) → (1, NCLS_M, 28, 28)
            std::vector<float> mo0;
            {
                compute_graph g2 = compute_graph_init(65536);
                model_ref mc(wc, g2);
                tensor min_ = compute_graph_input(mc, GGML_TYPE_F32, {C, MO, MO, 1}, "mroi");
                ggml_build_forward_expand(g2, min_);
                ggml_build_forward_expand(g2, FWD_C(mc, min_, PRM_C(fc)));
                compute_graph_allocate(g2, backend);
                // roi_align 은 NCHW flat 을 낸다. 생성 코드는 cwhn 규약이다.
                std::vector<float> cw((size_t)C * MO * MO);
                for (int c = 0; c < C; ++c)
                    for (int y = 0; y < MO; ++y)
                        for (int x = 0; x < MO; ++x)
                            cw[((size_t)y * MO + x) * C + c] =
                                mfeat[(((size_t)n * C + c) * MO + y) * MO + x];
                transfer_to_backend(min_, std::span<const float>(cw.data(), cw.size()));
                compute(g2, backend);
                std::vector<std::pair<int, int>> hw;
                auto mo = read_outputs(g2, 4, &hw);
                if (mo.empty()) {
                    fprintf(stderr, "SubC(mask head) 출력이 없다\n");
                    return 7;
                }
                mo0 = mo[0];
                MH = hw[0].first; MW = hw[0].second;
                NCLS_M = (int)(mo0.size() / ((size_t)MH * MW));
            }
            const int PH = MH / 2, PW = MW / 2;      // 28 → 14
            if (PH != MO || PW != MO) {
                fprintf(stderr, "mask pool %dx%d != roi %dx%d — 이어붙이기 불가\n",
                        PH, PW, MO, MO);
                return 7;
            }
            // 호스트: 라벨 채널 고르기 → sigmoid → maxpool(2,2) → mask_feat 뒤에 잇기.
            // ⚠️ 이 세 단계는 그래프에 못 넣는다. 라벨은 **실행 중에** 정해지는 값이라
            //    `mask_preds[range(M), labels]` 가 trace 에 안 실린다(조용히 0번 채널이 된다).
            // ⚠️ 채널 순서는 mmdet 대로 **feature 256 뒤에 mask 1** 이다
            //    (`torch.cat((mask_feat, mask_pred_pooled), 1)`).
            std::vector<float> din((size_t)CD * MO * MO);   // cwhn
            const int k = std::min(std::max(mlab[n], 0), NCLS_M - 1);
            for (int y = 0; y < MO; ++y)
                for (int x = 0; x < MO; ++x) {
                    float* dst = din.data() + ((size_t)y * MO + x) * CD;
                    for (int c = 0; c < C; ++c)
                        dst[c] = mfeat[(((size_t)n * C + c) * MO + y) * MO + x];
                    float mx = -INFINITY;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const size_t si =
                                ((size_t)(2 * y + dy) * MW + (2 * x + dx)) * NCLS_M + k;
                            mx = std::max(mx, 1.0f / (1.0f + std::exp(-mo0[si])));
                        }
                    dst[C] = mx;
                }
            // ② SubD = mask-IoU head. (1,257,14,14) → (1, NCLS)
            {
                compute_graph g3 = compute_graph_init(65536);
                model_ref md(wd, g3);
                tensor din_ = compute_graph_input(md, GGML_TYPE_F32, {CD, MO, MO, 1}, "miou_in");
                ggml_build_forward_expand(g3, din_);
                ggml_build_forward_expand(g3, FWD_D(md, din_, PRM_D(fd)));
                compute_graph_allocate(g3, backend);
                transfer_to_backend(din_, std::span<const float>(din.data(), din.size()));
                compute(g3, backend);
                auto od = read_outputs(g3, 4, nullptr);
                if (od.empty()) {
                    fprintf(stderr, "SubD(mask-IoU head) 출력이 없다\n");
                    return 7;
                }
                const int NIOU = (int)od[0].size();
                miou[n] = od[0][std::min(std::max(mlab[n], 0), NIOU - 1)];
            }
            logit_all.insert(logit_all.end(), mo0.begin(), mo0.end());
            din_all.insert(din_all.end(), din.begin(), din.end());
        }
        for (int i = 0; i < MD; ++i) {
            dets[i].score *= miou[i];
        }
        // ⚠️ **재정렬하지 않는다.** mmdet 도 보정 뒤 정렬하지 않는다
        //    (`predict_by_feat` 는 `results.scores` 를 제자리에서 곱할 뿐이다).
        //    여기서 정렬하면 양쪽 순서가 갈려 인덱스로 짝짓는 대조가 어긋난다.
        // 중간 텐서도 낸다 — 값이 틀렸을 때 **어느 단계**인지 torch 와 대조한다.
        dump_bin(pref + ".mfeat.bin", mfeat);       // 마스크 RoIAlign (NCHW)
        dump_bin(pref + ".mlogit.bin", logit_all);  // SubC 출력 (행별 cwhn)
        dump_bin(pref + ".miouin.bin", din_all);    // SubD 입력 (행별 cwhn, 257ch)
        dump_bin(pref + ".maskiou.bin", miou);      // 라벨 채널의 IoU 예측
    }
#endif

#ifdef ARCH_E
    // ── 패스 3: 격자점 히트맵으로 박스를 다시 낸다 (Grid R-CNN) ─────────────
    // bbox head 는 `with_reg=False` 라 박스를 아예 안 낸다(위에서 RoI 를 그대로 썼다).
    // 진짜 박스는 여기서 나온다 — 9개 격자점의 히트맵 최대점을 이미지 좌표로 옮기고
    // 같은 변에 놓인 점들을 **점수 가중 평균**한다(`grid_head._predict_by_feat_single`).
    if (J.num("has_grid", 0.0f) != 0.0f && ge && !dets.empty()) {
        const int MD = (int)dets.size();
        const int GP = (int)J.num("grid_points", 9.0f);
        const int GS = (int)J.num("grid_size", 3.0f);
        std::vector<float> sub = J.arr("grid_sub_xy");      // (x1,y1) x GP
        roi_align_params gp_;
        gp_.output_size = (int)J.num("grid_roi_out", 14.0f);
        gp_.channels = C;
        gp_.strides = J.arr("grid_strides");
        gp_.finest_scale = J.num("grid_finest_scale", 56.0f);
        const int GL = std::min((int)gp_.strides.size(), (int)feats.size());
        std::vector<std::vector<float>> gfeats(feats.begin(), feats.begin() + GL);
        std::vector<std::pair<int, int>> ghw(feat_hw.begin(), feat_hw.begin() + GL);
        const int GO = gp_.output_size;

        std::vector<float> gbox((size_t)MD * 4);
        for (int i = 0; i < MD; ++i) {
            gbox[(size_t)i * 4 + 0] = dets[i].x1; gbox[(size_t)i * 4 + 1] = dets[i].y1;
            gbox[(size_t)i * 4 + 2] = dets[i].x2; gbox[(size_t)i * 4 + 3] = dets[i].y2;
        }
        std::vector<float> gfeat = roi_align(gfeats, ghw, gbox.data(), MD, gp_);

        model_file fe = model_load(ge);
        model_weights we = model_init(fe.n_tensors());
        model_transfer(fe, we, backend, backend.preferred_float_type(), fe.tensor_layout());

        std::vector<float> heat_all;
        for (int n = 0; n < MD; ++n) {
            // ⚠️ 여기도 **배치 1** 이다. grid head 는 deconv 를 두 번 타는데 ggml 의
            //    conv_transpose 가 배치 축을 안 돈다(위 마스크 경로 주석 참고).
            compute_graph g4 = compute_graph_init(65536);
            model_ref me(we, g4);
            tensor gin = compute_graph_input(me, GGML_TYPE_F32, {C, GO, GO, 1}, "groi");
            ggml_build_forward_expand(g4, gin);
            ggml_build_forward_expand(g4, FWD_E(me, gin, PRM_E(fe)));
            compute_graph_allocate(g4, backend);
            std::vector<float> cw((size_t)C * GO * GO);
            for (int c = 0; c < C; ++c)
                for (int y = 0; y < GO; ++y)
                    for (int x = 0; x < GO; ++x)
                        cw[((size_t)y * GO + x) * C + c] =
                            gfeat[(((size_t)n * C + c) * GO + y) * GO + x];
            transfer_to_backend(gin, std::span<const float>(cw.data(), cw.size()));
            compute(g4, backend);
            std::vector<std::pair<int, int>> hw;
            auto go = read_outputs(g4, 4, &hw);
            if (go.empty()) {
                fprintf(stderr, "SubE(grid head) 출력이 없다\n");
                return 8;
            }
            const int GH = hw[0].first, GW = hw[0].second;     // 반쪽 히트맵(28)
            std::vector<float> const& hm = go[0];              // cwhn: ((y*GW+x)*GP+k)
            // 격자점마다 최대점을 찾는다. 점수는 sigmoid 를 거친 값이다.
            std::vector<float> sc(GP), ax(GP), ay(GP);
            for (int k = 0; k < GP; ++k) {
                float best = -INFINITY; int bx = 0, by = 0;
                for (int y = 0; y < GH; ++y)
                    for (int x = 0; x < GW; ++x) {
                        const float v = hm[((size_t)y * GW + x) * GP + k];
                        if (v > best) { best = v; bx = x; by = y; }
                    }
                sc[k] = 1.0f / (1.0f + std::exp(-best));
                // 반쪽 히트맵 좌표 → 전체 히트맵 좌표.
                const float sx = (2 * k < (int)sub.size()) ? sub[2 * k] : 0.0f;
                const float sy = (2 * k + 1 < (int)sub.size()) ? sub[2 * k + 1] : 0.0f;
                // ⚠️ **나누는 것은 전체 크기가 아니라 반쪽 크기(GW/GH)다.** mmdet 이
                //    그렇게 쓴다 — 전체 맵이 상자의 2배 영역을 덮으므로 반쪽으로 나누면
                //    확장 상자 기준이 된다. 전체 크기로 나누면 박스가 절반이 된다.
                const float w0 = dets[n].x2 - dets[n].x1, h0 = dets[n].y2 - dets[n].y1;
                ax[k] = ((float)bx + sx + 0.5f) / (float)GW * w0 + (dets[n].x1 - w0 * 0.5f);
                ay[k] = ((float)by + sy + 0.5f) / (float)GH * h0 + (dets[n].y1 - h0 * 0.5f);
            }
            // 같은 변에 놓인 격자점들을 점수로 가중 평균한다.
            auto vote = [&](std::vector<int> const& idx, std::vector<float> const& v) {
                float num = 0.0f, den = 0.0f;
                for (int i : idx) { num += v[i] * sc[i]; den += sc[i]; }
                return den > 0.0f ? num / den : 0.0f;
            };
            std::vector<int> ix1, iy1, ix2, iy2;
            for (int i = 0; i < GS; ++i) {
                ix1.push_back(i);
                iy1.push_back(i * GS);
                ix2.push_back(GP - GS + i);
                iy2.push_back((i + 1) * GS - 1);
            }
            // ⚠️ **자르지 않는다.** mmdet 에 clamp 가 있지만 **동작하지 않는다** —
            //    `bboxes[:, [0, 2]].clamp_(...)` 는 팬시 인덱싱이라 복사본을 자르고 버린다
            //    (grid_head.py:482-483). 여기서 진짜로 자르면 경계에 걸친 상자만
            //    어긋난다(실측 801.3 vs 800.0 → 1.30px). 라이브러리의 **실제 동작**이
            //    정본이지 주석이나 의도가 아니다.
            dets[n].x1 = vote(ix1, ax);
            dets[n].y1 = vote(iy1, ay);
            dets[n].x2 = vote(ix2, ax);
            dets[n].y2 = vote(iy2, ay);
            heat_all.insert(heat_all.end(), hm.begin(), hm.end());
        }
        dump_bin(pref + ".gridheat.bin", heat_all);
    }
#endif

    // ── 덤프 (torch 대조용) ─────────────────────────────────────────────────
    dump_bin(pref + ".props.bin", props);
    dump_bin(pref + ".roi.bin", roi);
    // 인덱스는 **담은 쌍의 수**로 돈다 — 캐스케이드면 단계 수, 다중 인스턴스면 쌍 수다.
    // `NS` 로 돌면 CrowdDet 처럼 1단계·2쌍인 경우 뒤쪽 쌍이 덤프되지 않아
    // torch 대조에서 **절반이 빠진 채** 통과한다.
    for (size_t k = 0; k < cls_st.size(); ++k) {
        dump_bin(pref + ".cls." + std::to_string(k) + ".bin", cls_st[k]);
        dump_bin(pref + ".box." + std::to_string(k) + ".bin", box_st[k]);
    }
    dump_bin(pref + ".rois.bin", rois);
    for (int l = 0; l < L; ++l) {
        dump_bin(pref + ".rpncls." + std::to_string(l) + ".bin", rpn_cls[l]);
        dump_bin(pref + ".rpnbox." + std::to_string(l) + ".bin", rpn_box[l]);
    }
    // 최종 박스도 낸다 — 여기까지 와서 개수만 알려주면 결과를 수치로 확인할 방법이 없다.
    // `run_mmdet` 과 같은 규약: `<prefix>.boxes.bin` 에 박스당 6값(x1,y1,x2,y2,score,label).
    {
        std::vector<float> flat;
        flat.reserve(dets.size() * 6);
        for (detection const& d : dets) {
            flat.insert(flat.end(), {d.x1, d.y1, d.x2, d.y2, d.score, (float)d.label});
        }
        dump_bin(pref + ".boxes.bin", flat);
    }
    printf("- frcnn: 2패스 (proposal %d · roi %dx%d) → %s.*.bin\n", M, O, O, pref.c_str());
    printf("  %4s %9s %9s %9s %9s %8s  class\n", "#", "x1", "y1", "x2", "y2", "score");
    for (size_t i = 0; i < dets.size() && i < 10; ++i) {
        detection const& d = dets[i];
        printf("  %4zu %9.2f %9.2f %9.2f %9.2f %8.4f  %d\n", i, d.x1, d.y1, d.x2, d.y2,
               d.score, d.label);
    }
    if (dets.size() > 10) printf("   ... (%zu more)\n", dets.size() - 10);
    return 0;
}
