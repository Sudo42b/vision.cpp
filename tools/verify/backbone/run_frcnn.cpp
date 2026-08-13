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
    std::vector<float> arr(const std::string& k) const {
        std::vector<float> out;
        size_t p = find_key(k);
        if (p == std::string::npos) return out;
        size_t l = s.find('[', p), r = s.find(']', l);
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
                "usage: %s <subA.gguf> <subB.gguf> <frcnn.json> <in_cwhn.bin> <out_prefix> [size]\n",
                argv[0]);
        return 1;
    }
    const char* ga = argv[1];
    const char* gb = argv[2];
    tiny_json J(argv[3]);
    const char* inb = argv[4];
    const std::string pref = argv[5];
    const int SZ = argc > 6 ? atoi(argv[6]) : 512;

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
    const int L = (int)J.arr("rpn_strides").size();
    const int NF = (int)J.arr("roi_strides").size();
    if ((int)outs.size() < NF + 2 * L) {
        fprintf(stderr, "SubA 출력 %zu 개 < 기대 %d (feats %d + rpn %d×2)\n",
                outs.size(), NF + 2 * L, NF, L);
        return 3;
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
    rp.ratios = J.arr("rpn_ratios");
    rp.nms_pre = (int)J.num("rpn_nms_pre", 1000);
    rp.nms_thr = J.num("rpn_nms_thr", 0.7f);
    rp.max_per_img = (int)J.num("rpn_max", 1000);
    rp.input_w = rp.input_h = SZ;
    std::vector<float> props = rpn_proposals(rpn_cls, rpn_box, rpn_hw, rp);
    const int M = (int)(props.size() / 4);
    fprintf(stderr, "[frcnn] proposal %d 개 (nms %.2f)\n", M, rp.nms_thr);
    if (M == 0) {
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
    std::vector<float> roi = roi_align(feats, feat_hw, props.data(), M, ap);

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
        std::vector<float> rg = roi_align(feats, feat_hw, bx.data(), M, ap);
        rf.insert(rf.end(), rg.begin(), rg.end());
        return rf;
    };

    for (int st = 0; st < NS; ++st) {
        std::vector<float> roi_st = with_reg_half(
            (st == 0) ? roi : roi_align(feats, feat_hw, rois.data(), M, ap),
            (st == 0) ? props : rois);
        const int MB = (RSF > 0.0f) ? 2 * M : M;   // SubB 에 넣는 행 수

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
    printf("- frcnn: 2패스 (proposal %d · roi %dx%d) → %s.*.bin\n", M, O, O, pref.c_str());
    return 0;
}
