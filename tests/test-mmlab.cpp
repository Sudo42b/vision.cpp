#include "visp/ml.h"

#include <array>
#include <cmath>
#include <cstdio>

// Internal builder helper. This test calls the implementation in visioncpp,
// rather than maintaining a second copy of the sampling graph.
namespace visp::mmlab {
tensor sample_tokens(model_ref, tensor, int64_t, int64_t, tensor, tensor);
}

int main() {
    using namespace visp;
    auto graph = compute_graph_init();
    model_ref model(nullptr, &graph);
    auto values = compute_graph_input(model, GGML_TYPE_F32, {2, 12, 2, 1}, "values");
    auto x = compute_graph_input(model, GGML_TYPE_F32, {7, 2, 1, 1}, "x");
    auto y = compute_graph_input(model, GGML_TYPE_F32, {7, 2, 1, 1}, "y");
    auto output = compute_graph_output(model, mmlab::sample_tokens(model, values, 4, 3, x, y));
    auto device = backend_init(backend_type::cpu);
    if (!compute_graph_allocate(graph, device)) return 2;
    std::array<float, 48> data{};
    for (int group = 0; group < 2; ++group)
        for (int pixel = 0; pixel < 12; ++pixel)
            for (int channel = 0; channel < 2; ++channel)
                data[channel + 2 * pixel + 24 * group] = float(1 + channel + 2 * group);
    constexpr std::array<float, 7> xs{-.25f, 0, .75f, 3, 3.25f, -2, 8};
    constexpr std::array<float, 7> ys{0, .5f, 1.25f, 2, 2.25f, 1, 1};
    // Independent expected zero-padding contribution for these coordinates.
    constexpr std::array<float, 7> contribution{.75f, 1, 1, 1, .5625f, 0, 0};
    std::array<float, 14> coordinates_x{}, coordinates_y{};
    for (int group = 0; group < 2; ++group)
        for (int i = 0; i < 7; ++i) {
            coordinates_x[i + 7 * group] = xs[i];
            coordinates_y[i + 7 * group] = ys[i];
        }
    transfer_to_backend(values, span<float const>{data});
    transfer_to_backend(x, span<float const>{coordinates_x});
    transfer_to_backend(y, span<float const>{coordinates_y});
    compute(graph, device);
    std::array<float, 28> result{};
    transfer_from_backend(output, span<float>{result});
    for (int group = 0; group < 2; ++group)
        for (int i = 0; i < 7; ++i)
            for (int channel = 0; channel < 2; ++channel) {
                float expected = float(1 + channel + 2 * group) * contribution[i];
                float actual = result[channel + 2 * i + 14 * group];
                if (std::abs(actual - expected) > 1e-6f) {
                    std::fprintf(stderr, "group=%d point=%d channel=%d: %g != %g\n", group, i, channel, actual, expected);
                    return 1;
                }
            }
    std::puts("MMLab CPU zero-padding sampling: PASS");
}
