#include "visp/mmdet_run.h"

#include "util/string.h"
#include "visp/ml.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace visp {
namespace {

std::vector<float> to_vec(tensor t) {
    int64_t ne = ggml_nelements(t);
    std::vector<float> d((size_t)ne);
    transfer_from_backend(t, std::span<float>(d.data(), d.size()));
    return d;
}

void dump_bin(std::string const& path, std::vector<float> const& v) {
    if (FILE* f = fopen(path.c_str(), "wb")) {
        fwrite(v.data(), sizeof(float), v.size(), f);
        fclose(f);
    }
}

float sigm(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// DDQ 만 그래프가 여러 벌이다. decoder 층마다 호스트 NMS 로 중복 query 를 가려 다음 층의
// self-attention 마스크를 만드는데, NMS 는 실행 중 값에 따라 반복이 정해지는 탐욕
// 알고리즘이라 정적 그래프로 표현할 수 없다(mmdet_head.h 의 DDQ 절).
std::vector<detection> run_ddq(backend_device& backend, model_weights& weights,
                               compute_graph& graph, model_ref m, tensor input,
                               std::vector<tensor> const& feats, mmdet_cfg const& cfg,
                               std::span<const float> in_data) {
    anchor_head_cfg const& hc = cfg.head;
    det_params const& dp = cfg.det;

    std::vector<msd_level> lv;
    ddq_levels(feats, lv);
    ddq_stage s0;
    ddq_encode_forward(m, feats, hc, s0);
    for (tensor x : {s0.mem, s0.om, s0.ecls, s0.ecoord, s0.qmap}) {
        ggml_build_forward_expand(graph, x);
    }
    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, in_data);
    compute(graph, backend);

    const int64_t T = s0.mem->ne[1], E = s0.mem->ne[0], NC = s0.ecls->ne[0];
    const int64_t NQ = hc.num_queries;
    std::vector<float> mem_v = to_vec(s0.mem), qm_v = to_vec(s0.qmap);
    std::vector<float> ec_v = to_vec(s0.ecls), eb_v = to_vec(s0.ecoord);

    // 호스트: 점수 = sigmoid(클래스 최댓값), 박스 = cxcywh→xyxy(sigmoid(coord)).
    std::vector<detection> dets((size_t)T);
    for (int64_t i = 0; i < T; ++i) {
        float best = -1e30f;
        for (int64_t k = 0; k < NC; ++k) best = std::max(best, ec_v[i * NC + k]);
        float cx = sigm(eb_v[i * 4 + 0]), cy = sigm(eb_v[i * 4 + 1]);
        float w = sigm(eb_v[i * 4 + 2]), h = sigm(eb_v[i * 4 + 3]);
        dets[i] = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, sigm(best), 0};
    }
    std::vector<int> keep = nms(dets, hc.ddq_iou_thr);
    if ((int64_t)keep.size() > NQ) keep.resize((size_t)NQ);
    if ((int64_t)keep.size() < NQ) {
        throw except("ddq: NMS 후 {} 개 < num_queries {} — 그래프 모양이 안 맞는다",
                     keep.size(), (long long)NQ);
    }
    // DDQ 는 query 내용을 학습 embedding 이 아니라 **선택된 위치의 feature** 로 채운다.
    std::vector<float> q_v((size_t)E * NQ), r_v((size_t)4 * NQ);
    for (int64_t i = 0; i < NQ; ++i) {
        const int64_t t0 = keep[(size_t)i];
        std::copy_n(&qm_v[(size_t)t0 * E], E, &q_v[(size_t)i * E]);
        for (int k = 0; k < 4; ++k) r_v[(size_t)i * 4 + k] = sigm(eb_v[(size_t)t0 * 4 + k]);
    }
    std::vector<float> mask_v((size_t)NQ * NQ, 0.0f);   // 가산 마스크(0 = 통과)
    std::vector<char> alive((size_t)NQ, 1);             // 0층은 전부 살아있다

    std::vector<std::vector<float>> cls_out, box_out;
    for (int li = 0; li < hc.dec_layers; ++li) {
        compute_graph g = compute_graph_init(65536);
        model_ref mi(weights, g);
        tensor memT = compute_graph_input(mi, GGML_TYPE_F32, {E, T, 1, 1}, "mem");
        tensor qT = compute_graph_input(mi, GGML_TYPE_F32, {E, NQ, 1, 1}, "q");
        tensor rT = compute_graph_input(mi, GGML_TYPE_F32, {4, NQ, 1, 1}, "ref");
        tensor kT = compute_graph_input(mi, GGML_TYPE_F32, {NQ, NQ, 1, 1}, "mask");
        for (tensor x : {memT, qT, rT, kT}) ggml_build_forward_expand(g, x);
        ddq_stage si;
        ddq_layer_forward(mi, lv, hc, li, memT, qT, rT, kT, si);
        for (tensor x : {si.query, si.ref, si.cls, si.box}) ggml_build_forward_expand(g, x);
        compute_graph_allocate(g, backend);
        transfer_to_backend(memT, std::span<const float>(mem_v.data(), mem_v.size()));
        transfer_to_backend(qT, std::span<const float>(q_v.data(), q_v.size()));
        transfer_to_backend(rT, std::span<const float>(r_v.data(), r_v.size()));
        transfer_to_backend(kT, std::span<const float>(mask_v.data(), mask_v.size()));
        compute(g, backend);

        cls_out.push_back(to_vec(si.cls));
        box_out.push_back(to_vec(si.box));
        q_v = to_vec(si.query);
        r_v = to_vec(si.ref);
        if (li + 1 < hc.dec_layers) {
            // 다음 층 마스크. mmdet 규약 두 가지를 그대로 따른다:
            //  ① NMS 는 **지금까지 살아남은 query 끼리만** 돈다(집합이 단조 감소한다).
            //  ② 셀 (i,j) 는 **행이나 열 중 하나라도** 살아있으면 통과, 아니면 가린다.
            std::vector<int> ori;
            for (int64_t i = 0; i < NQ; ++i) if (alive[(size_t)i]) ori.push_back((int)i);
            std::vector<detection> dq(ori.size());
            for (size_t n = 0; n < ori.size(); ++n) {
                const int64_t i = ori[n];
                float best = -1e30f;
                for (int64_t k = 0; k < NC; ++k)
                    best = std::max(best, cls_out.back()[(size_t)i * NC + k]);
                float cx = r_v[(size_t)i * 4 + 0], cy = r_v[(size_t)i * 4 + 1];
                float w = r_v[(size_t)i * 4 + 2], h = r_v[(size_t)i * 4 + 3];
                dq[n] = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f,
                         sigm(best), 0};
            }
            std::vector<char> next((size_t)NQ, 0);
            for (int k : nms(dq, hc.ddq_iou_thr)) next[(size_t)ori[(size_t)k]] = 1;
            alive.swap(next);
            const float NEG = -1e30f;
            for (int64_t i = 0; i < NQ; ++i)
                for (int64_t j = 0; j < NQ; ++j)
                    mask_v[(size_t)i * NQ + j] =
                        (alive[(size_t)i] || alive[(size_t)j]) ? 0.0f : NEG;
        }
    }
    if (const char* pre = std::getenv("MMDET_DUMP_HEAD")) {
        for (size_t l = 0; l < cls_out.size(); ++l)
            dump_bin(std::string(pre) + ".cls." + std::to_string(l) + ".bin", cls_out[l]);
        for (size_t l = 0; l < box_out.size(); ++l)
            dump_bin(std::string(pre) + ".box." + std::to_string(l) + ".bin", box_out[l]);
    }

    // 최종 디코드 — mmdet `DDQDETRHead.predict_by_feat`. 마지막 층 출력에서 살아남은
    // query 만 골라 DETR top-k 를 탄다. 층 사이 NMS 가 이미 중복을 걸렀으므로 최종 NMS 는
    // 없다.
    // ⚠️ 박스는 마지막 층 hidden 에서 낸 sigmoid cxcywh 다. `s.ref` 는 **다음 층 참조점**이라
    //    미묘하게 다르고, 섞으면 값만 조용히 틀린다.
    std::vector<float> fcls, fbox;
    std::vector<float> const& lc = cls_out.back();
    std::vector<float> const& lb = box_out.back();
    int n_alive = 0;
    for (int64_t i = 0; i < NQ; ++i) {
        if (!alive[(size_t)i]) continue;
        fcls.insert(fcls.end(), lc.begin() + i * NC, lc.begin() + (i + 1) * NC);
        fbox.insert(fbox.end(), lb.begin() + i * 4, lb.begin() + (i + 1) * 4);
        ++n_alive;
    }
    detr_params qp;
    qp.num_queries = n_alive;
    qp.num_classes = dp.num_classes;   // 배경 제외 수 (프론트엔드 규약)
    qp.use_sigmoid = true;             // deformable 계열 — 항상 sigmoid+focal
    qp.max_per_img = dp.max_per_img;
    qp.input_w = dp.input_w;
    qp.input_h = dp.input_h;
    return detect_detr(fcls.data(), fbox.data(), qp);
}

}  // namespace

std::vector<detection> mmdet_run(backend_device& backend, model_weights& weights,
                                 model_file const& file, mmdet_backbone_fn backbone,
                                 mmdet_cfg cfg, std::span<const float> in_data,
                                 int input_w, int input_h) {
    anchor_head_cfg& hc = cfg.head;
    det_params& dp = cfg.det;
    dp.input_w = input_w;
    dp.input_h = input_h;
    const int L = (int)dp.strides.size();
    if (L == 0) {
        throw except("이 설정의 head 는 내보내기에서 인식되지 않았다 — 백본만 있다");
    }
    if (in_data.size() != (size_t)3 * input_w * input_h) {
        throw except("입력 크기 {} != {} (전처리·해상도 확인)", in_data.size(),
                     (size_t)3 * input_w * input_h);
    }

    compute_graph graph = compute_graph_init(131072);
    model_ref m(weights, graph);
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, input_w, input_h, 1}, "x");
    ggml_build_forward_expand(graph, input);
    ggml_build_forward_expand(graph, backbone(m, input, file));

    // 백본이 낸 FPN feature 를 이름으로 잡는다.
    std::vector<tensor> feats;
    for (int l = 0; l < L; ++l) {
        tensor f = ggml_graph_get_tensor(graph.graph, ("out_" + std::to_string(l)).c_str());
        if (!f) throw except("백본 출력 out_{} 이 없다 (생성물의 출력 규약 확인)", l);
        feats.push_back(f);
    }
    // deformable 을 쓰는 계열(vfnet·reppoints)은 고정 3×3 격자가 필요하다. mmdet 의
    // dcn_base_offset 과 같은 값 — [y,x] 가 번갈아 놓인 18개다.
    // ⚠️ 안 쓰는 계열에서 만들면 그래프에 안 실려 버퍼가 없고 add_buffer 가 죽는다.
    tensor dcn_base = nullptr;
    if (hc.kind == head_kind::vfnet || hc.kind == head_kind::reppoints) {
        dcn_base = compute_graph_input(m, GGML_TYPE_F32, {18, 1, 1, 1}, "dcn_base");
        tensor_data d = tensor_alloc(dcn_base);
        std::span<float> v = d.as_f32();
        for (int i = 0; i < 9; ++i) {
            v[2 * i] = float(i / 3 - 1);        // y: -1,-1,-1, 0,0,0, 1,1,1
            v[2 * i + 1] = float(i % 3 - 1);    // x: -1,0,1 반복
        }
        m.add_buffer(std::move(d));
    }

    if (hc.kind == head_kind::ddq) {
        return run_ddq(backend, weights, graph, m, input, feats, cfg, in_data);
    }

    std::vector<detection> dets;
    head_outputs ho;
    mmdet_head_forward(m, feats, hc, dcn_base, ho);
    std::vector<tensor>&cls_t = ho.cls, &box_t = ho.box;
    for (tensor t : cls_t) ggml_build_forward_expand(graph, t);
    for (tensor t : box_t) ggml_build_forward_expand(graph, t);
    for (tensor t : ho.ctr) ggml_build_forward_expand(graph, t);
    // ⚠️ extra 갈래도 **그래프에 넣어야** 버퍼가 생긴다. 안 넣고 읽으면
    //    `GGML_ASSERT(buf != NULL && "tensor buffer not set")` 로 죽는다.
    for (auto const& e : ho.extra)
        for (tensor t : e.second) ggml_build_forward_expand(graph, t);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, in_data);
    compute(graph, backend);

    // 검증용 원시 덤프. **디코드 전** 값을 그대로 내보내 torch 의 bbox_head 출력과 직접
    // 대조한다 — NMS 를 거치면 어디가 틀렸는지 짚을 수 없다.
    if (const char* pre = std::getenv("MMDET_DUMP_HEAD")) {
        auto dump = [&](const char* kind, size_t l, tensor t) {
            dump_bin(std::string(pre) + "." + kind + "." + std::to_string(l) + ".bin", to_vec(t));
        };
        // ⚠️ **head 입력(neck 출력)도 같이 낸다.** head 출력만 보면 「조립이 틀렸다」와
        //    「백본·neck 이 이미 틀렸다」를 가를 수 없다.
        for (size_t l = 0; l < feats.size(); ++l) dump("feat", l, feats[l]);
        for (size_t l = 0; l < cls_t.size(); ++l) dump("cls", l, cls_t[l]);
        for (size_t l = 0; l < box_t.size(); ++l) dump("box", l, box_t[l]);
        for (size_t l = 0; l < ho.ctr.size(); ++l) dump("ctr", l, ho.ctr[l]);
        for (auto const& e : ho.extra)
            for (size_t l = 0; l < e.second.size(); ++l) dump(e.first.c_str(), l, e.second[l]);
    }

    // ── CornerNet / CentripetalNet ─────────────────────────────────────────
    // 앵커가 없다 — 코너 heatmap 두 장에서 top-k 를 뽑아 쌍을 만든다. 갈래가 6~8개라
    // out.extra 를 쓰고, **마지막 레벨만** 쓴다(mmdet 도 `tl_heats[-1]` 이다).
    bool corner_done = false;
    if (hc.kind == head_kind::cornernet) {
        if (ho.cls.empty() || ho.box.empty() || ho.ctr.empty() || ho.extra.empty()) {
            throw except("corner head 출력이 비었다");
        }
        auto last = [](std::vector<tensor> const& v) { return v.back(); };
        auto named = [&](const char* tag) -> std::vector<float> {
            for (auto const& e : ho.extra)
                if (e.first == tag && !e.second.empty()) return to_vec(e.second.back());
            return {};
        };
        tensor t_cls = last(ho.cls);
        const int cfh = (int)t_cls->ne[2], cfw = (int)t_cls->ne[1];
        corner_params cp;
        cp.num_classes = dp.num_classes;
        cp.topk = hc.corner_topk;
        cp.local_max_kernel = hc.local_max_kernel;
        cp.distance_threshold = hc.corner_distance_thr;
        cp.score_thr = dp.score_thr;
        cp.nms_thr = dp.nms_thr;
        cp.max_per_img = dp.max_per_img;
        cp.input_w = dp.input_w;
        cp.input_h = dp.input_h;
        cp.centripetal = hc.corner_centripetal;
        dets = detect_corner(to_vec(t_cls), to_vec(last(ho.box)),
                             to_vec(last(ho.ctr)), named("brof"),
                             named("tlemb"), named("bremb"),
                             named("tlcs"), named("brcs"),
                             cfh, cfw, cp);
        corner_done = true;
    }

    // ── CenterNet ──────────────────────────────────────────────────────────
    // 앵커가 없고 레벨도 하나다. heatmap 의 국소 최대점이 중심이고, 같은 자리의 wh/offset 이
    // 상자를 만든다. heat 는 head 안에서 이미 sigmoid 를 거쳤다.
    if (hc.kind == head_kind::centernet) {
        if (ho.cls.empty() || ho.box.empty() || ho.ctr.empty()) {
            throw except("centernet head 출력이 비었다(heatmap·wh·offset 필요)");
        }
        tensor t = ho.cls.back();
        centernet_params cp;
        cp.num_classes = dp.num_classes;
        cp.topk = hc.corner_topk;                 // test_cfg.topk (헤더가 같은 칸을 쓴다)
        cp.local_max_kernel = hc.local_max_kernel;
        cp.input_w = dp.input_w;
        cp.input_h = dp.input_h;
        dets = detect_centernet(to_vec(t), to_vec(ho.box.back()), to_vec(ho.ctr.back()),
                                (int)t->ne[2], (int)t->ne[1], cp);
        corner_done = true;
    }

    // DETR 계열은 출력 인덱스가 **decoder 층**이라 레벨 수와 무관하다.
    const bool is_detr = hc.kind == head_kind::detr ||
                         hc.kind == head_kind::conditional_detr ||
                         hc.kind == head_kind::dab_detr ||
                         hc.kind == head_kind::deformable_detr ||
                         hc.kind == head_kind::dino;

    // YOLOv3 는 레벨당 **한 갈래**다(na×(5+nc) 한 텐서에 box·obj·cls 가 다 들어 있다).
    const bool single_branch = hc.kind == head_kind::yolo;

    if (!corner_done && !is_detr && !single_branch &&
        ((int)cls_t.size() < L || (int)box_t.size() < L)) {
        throw except("head 출력이 {} 레벨에 못 미친다 (cls {}, box {})",
                     L, cls_t.size(), box_t.size());
    }

    const int NL = (is_detr || corner_done) ? 0 : L;
    std::vector<std::vector<float>> cls_v(NL), box_v(NL), ctr_v;
    std::vector<std::pair<int, int>> feat_hw(NL);
    for (int l = 0; l < NL; ++l) {
        feat_hw[l] = { (int)cls_t[l]->ne[2], (int)cls_t[l]->ne[1] };  // (fh, fw)
        cls_v[l] = to_vec(cls_t[l]);
        if (!single_branch) box_v[l] = to_vec(box_t[l]);
    }
    // centerness 갈래가 있으면 score factor 로 넘긴다(ATSS·PAA·DDOD·FCOS…). mmdet 은 이걸
    // top-k 뒤에 곱한다 — 안 넘기면 **박스는 맞고 점수만** 높게 나온다(실측 Δ0.071).
    // ⚠️ YOLACT 의 coeff 갈래는 점수가 아니다 — `ctr_tanh` 로 가른다.
    if (!corner_done && (int)ho.ctr.size() >= L && !hc.ctr_tanh) {
        ctr_v.resize(L);
        for (int l = 0; l < L; ++l) ctr_v[l] = to_vec(ho.ctr[l]);
    }

    // 디코드는 **조립과 다른 축**이다. 조립은 타워 모양(head_kind)으로 갈리지만, 디코드는
    // 박스가 무엇이냐로 갈린다 — 앵커 대비 delta 냐, 격자점 대비 거리냐.
    const bool distance_box = hc.kind == head_kind::fcos || hc.kind == head_kind::gfl ||
                              hc.kind == head_kind::vfnet || hc.kind == head_kind::reppoints ||
                              hc.bbox_mul_stride;

    if (corner_done) {
        // 위에서 채웠다.
    } else if (is_detr) {
        // ⚠️ out.cls/out.box 의 인덱스가 **FPN 레벨이 아니라 decoder 층**이다. mmdet 도
        //    `all_layers_*[-1]` 만 쓴다 — 앞 층을 쓰면 shape 가 맞아 안 죽고 더 거친 박스가
        //    나온다.
        if (ho.cls.empty() || ho.box.empty()) throw except("DETR head 출력이 비었다");
        std::vector<float> cls_last = to_vec(ho.cls.back());
        std::vector<float> box_last = to_vec(ho.box.back());
        detr_params qp;
        qp.num_queries = hc.num_queries;
        qp.num_classes = dp.num_classes;      // 배경을 뺀 수 (프론트엔드가 그렇게 싣는다)
        qp.use_sigmoid = dp.use_sigmoid;      // 고전 DETR=softmax, Deformable/DINO=sigmoid
        qp.max_per_img = dp.max_per_img;
        qp.input_w = dp.input_w;
        qp.input_h = dp.input_h;
        const int q_actual = (int)ho.box.back()->ne[1];
        if (q_actual > 0 && q_actual != qp.num_queries) {
            qp.num_queries = q_actual;   // 설정보다 텐서를 따른다
        }
        dets = detect_detr(cls_last.data(), box_last.data(), qp);
    } else if (hc.kind == head_kind::tood) {
        // 조립기가 이미 `distance2bbox` 까지 그래프로 펴서 **격자 좌표계 xyxy** 를 낸다 —
        // stride 만 곱하면 픽셀이다.
        // ⚠️ cls 는 `sqrt(σ(logits)·σ(align))` 라 **이미 확률**이다. 시그모이드를 또 걸면
        //    박스는 그대로인데 점수만 낮아진다.
        std::vector<detection> cand;
        for (int l = 0; l < L; ++l) {
            const int fh = feat_hw[l].first, fw = feat_hw[l].second, npos = fh * fw;
            const float stride = dp.strides[l];
            float const* cs = cls_v[l].data();
            float const* bx = box_v[l].data();
            for (int pos = 0; pos < npos; ++pos) {
                float const* c1 = cs + (size_t)pos * dp.num_classes;
                int best = 0;
                for (int j = 1; j < dp.num_classes; ++j)
                    if (c1[j] > c1[best]) best = j;
                if (c1[best] <= dp.score_thr) continue;
                float const* b1 = bx + (size_t)pos * 4;
                detection d{b1[0] * stride, b1[1] * stride, b1[2] * stride, b1[3] * stride,
                            c1[best], best};
                if (dp.input_w > 0) {
                    d.x1 = std::min(std::max(d.x1, 0.0f), (float)dp.input_w);
                    d.x2 = std::min(std::max(d.x2, 0.0f), (float)dp.input_w);
                    d.y1 = std::min(std::max(d.y1, 0.0f), (float)dp.input_h);
                    d.y2 = std::min(std::max(d.y2, 0.0f), (float)dp.input_h);
                }
                cand.push_back(d);
            }
        }
        int maxlabel = 0;
        for (auto const& c : cand) maxlabel = std::max(maxlabel, c.label);
        for (int lab = 0; lab <= maxlabel; ++lab) {
            std::vector<detection> per;
            for (auto const& c : cand) if (c.label == lab) per.push_back(c);
            if (per.empty()) continue;
            for (int k : nms(per, dp.nms_thr)) dets.push_back(per[k]);
        }
        std::sort(dets.begin(), dets.end(),
                  [](detection const& a, detection const& b) { return a.score > b.score; });
        if ((int)dets.size() > dp.max_per_img) dets.resize(dp.max_per_img);
    } else if (hc.kind == head_kind::rpn) {
        // 제안 생성. mmdet 은 클래스별이 아니라 **레벨별**로 NMS 를 걸고(level_ids), pre-NMS
        // 점수 임계값이 없다 — `detect_anchor` 로 두면 둘 다 어긋난다.
        rpn_params rp;
        rp.strides = dp.strides;
        rp.octave_base_scale = dp.octave_base_scale;
        rp.octave_scales = dp.octave_scales;
        rp.ratios = dp.ratios;
        for (int k = 0; k < 4; ++k) { rp.means[k] = dp.means[k]; rp.stds[k] = dp.stds[k]; }
        rp.nms_pre = dp.nms_pre;
        rp.nms_thr = dp.nms_thr;
        rp.max_per_img = dp.max_per_img;
        rp.input_w = dp.input_w;
        rp.input_h = dp.input_h;
        dets = detect_rpn(cls_v, box_v, feat_hw, rp);
        // `label` 은 레벨 번호로 돌아온다. 밖에서는 클래스 자리라 0 으로 둔다.
        for (auto& d : dets) d.label = 0;
    } else if (hc.kind == head_kind::sabl) {
        // 변마다 버킷 분류 + 오프셋. 두 번째 box 갈래는 `extra[0]`("bcls") 로 온다.
        sabl_params sp;
        sp.strides = dp.strides;
        sp.anchor_scale = dp.anchor_scale;
        sp.num_buckets = dp.num_buckets;
        sp.bucket_scale = dp.bucket_scale;
        sp.num_classes = dp.num_classes;
        sp.score_thr = dp.score_thr;
        sp.nms_thr = dp.nms_thr;
        sp.nms_pre = dp.nms_pre;
        sp.max_per_img = dp.max_per_img;
        sp.input_w = dp.input_w;
        sp.input_h = dp.input_h;
        std::vector<std::vector<float>> bcls_v(L);
        if (!ho.extra.empty() && (int)ho.extra[0].second.size() >= L)
            for (int l = 0; l < L; ++l) bcls_v[l] = to_vec(ho.extra[0].second[l]);
        dets = detect_sabl(cls_v, bcls_v, box_v, feat_hw, sp);
    } else if (hc.kind == head_kind::yolo) {
        // YOLOv3: 레벨당 **한 갈래**(na×(5+nc))라 조립기가 `out.cls` 에만 담는다. 앵커가
        // (w,h) 쌍이고 objectness 로 먼저 거른다 — 전용 디코더로 보낸다.
        yolov3_params yp;
        yp.strides = dp.strides;
        yp.base_sizes = dp.base_sizes;
        yp.num_classes = dp.num_classes;
        yp.conf_thr = dp.conf_thr;
        yp.score_thr = dp.score_thr;
        yp.nms_thr = dp.nms_thr;
        yp.nms_pre = dp.nms_pre;
        yp.max_per_img = dp.max_per_img;
        yp.input_w = dp.input_w;
        yp.input_h = dp.input_h;
        dets = detect_yolov3(cls_v, feat_hw, yp);
    } else if (hc.kind == head_kind::yolox) {
        // 격자 단위 (dx,dy,log w,log h) + 별도 objectness. 코더가 없어 앵커 파라미터가 안
        // 실리고, 그대로 두면 detect_anchor 가 후보 0개를 낸다.
        // ⚠️ obj 는 `out.ctr` 로 온다. 비면 점수가 cls 만 남아 **박스는 맞고 점수만** 높다.
        yolox_params yp;
        yp.strides = dp.strides;
        yp.prior_offset = hc.center_offset;   // YOLOX 의 MlvlPointGenerator 는 offset=0
        yp.num_classes = dp.num_classes;
        yp.score_thr = dp.score_thr;
        yp.nms_thr = dp.nms_thr;
        yp.max_per_img = dp.max_per_img;
        yp.input_w = dp.input_w;
        yp.input_h = dp.input_h;
        if ((int)ctr_v.size() < L) {
            throw except("YOLOX 인데 objectness 갈래가 {} 레벨뿐이다 (필요 {})",
                         ctr_v.size(), L);
        }
        dets = detect_yolox(cls_v, box_v, ctr_v, feat_hw, yp);
    } else if (distance_box) {
        fcos_params fp;
        fp.strides = dp.strides;
        fp.num_classes = dp.num_classes;
        fp.score_thr = dp.score_thr;
        fp.nms_thr = dp.nms_thr;
        fp.nms_pre = dp.nms_pre;
        fp.max_per_img = dp.max_per_img;
        fp.input_w = dp.input_w;
        fp.input_h = dp.input_h;
        fp.base_edge = dp.base_edge;   // FoveaBox 만 채워져 온다
        // ⚠️ 격자 오프셋을 kind 로 박으면 안 된다. 같은 `fcos` kind 라도 FCOS·NAS-FCOS·
        //    FoveaBox 는 0.5 지만 AutoAssign 은 0 이다. 프론트엔드가 생성기에서 읽어 싣는
        //    값을 그대로 쓴다. 박으면 최대 stride 128 에서 64px 밀린다(실측 64.30px).
        fp.point_offset = hc.center_offset;
        fp.box_xyxy_offset = hc.kind == head_kind::reppoints;
        dets = detect_fcos(cls_v, box_v, ctr_v, feat_hw, fp);
    } else if (dp.score_voting) {
        // PAA·LAD — 점수가 sqrt(cls×iou) 라 iou 갈래(ctr) 없이는 디코드할 수 없다.
        if ((int)ctr_v.size() < L) {
            throw except("PAA 인데 iou 갈래가 {} 레벨뿐이다 (필요 {})", ctr_v.size(), L);
        }
        dets = detect_paa(cls_v, box_v, ctr_v, feat_hw, dp);
    } else if (dp.fast_nms) {
        // YOLACT — softmax + fast NMS. coeff 갈래(ctr_tanh)는 박스 검증에서 안 쓴다.
        dets = detect_yolact(cls_v, box_v, feat_hw, dp);
    } else {
        dets = detect_anchor(cls_v, box_v, feat_hw, dp, ctr_v.empty() ? nullptr : &ctr_v);
    }

    return dets;
}

}  // namespace visp
