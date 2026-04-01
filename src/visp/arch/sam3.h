#include "util/common.h"
#include "util/math.h"
#include "visp/ml.h"
#include "visp/nn.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace visp {
namespace sam3 {

using string32 = fixed_string<32>;

//
// String storage

class string_storage {
    size_t chunk_size = 4096;
    size_t offset = 0;
    std::vector<std::unique_ptr<char[]>> chunks;

  public:
    string_storage() = default;

    std::string_view alloc(std::string_view str) {
        if (chunks.empty() || chunk_size - offset < str.size() + 1) {
            chunks.push_back(std::make_unique<char[]>(chunk_size));
            offset = 0;
        }
        char* chunk = chunks.back().get();
        std::memcpy(chunk + offset, str.data(), str.size());
        chunk[offset + str.size()] = 0;
        offset += str.size() + 1;
        return std::string_view(chunk + offset - str.size() - 1, str.size());
    }
};

class string_list {
    std::vector<std::string_view> strings;
    string_storage storage;

  public:
    string_list() = default;

    size_t size() const { return strings.size(); }
    std::string_view operator[](size_t index) const { return strings[index]; }

    void reserve(size_t n) { strings.reserve(n); }
    void push_back(std::string_view str) { strings.push_back(storage.alloc(str)); }

    using const_iterator = std::vector<std::string_view>::const_iterator;
    const_iterator begin() const { return strings.begin(); }
    const_iterator end() const { return strings.end(); }
};

string_list get_string_list(model_file const& f, char const* key_name) {
    int64_t key_id = f.key(key_name);
    if (gguf_get_arr_type(f.gguf.get(), key_id) != GGUF_TYPE_STRING) {
        throw except(
            "Array type mismatch for key '{}' in model file {}, expected string", key_name, f.path);
    }
    size_t n = gguf_get_arr_n(f.gguf.get(), key_id);
    string_list result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(gguf_get_arr_str(f.gguf.get(), key_id, i));
    }
    return result;
}

//
// CLIP tokenizer

std::string normalize_prompt(std::string_view text) {
    std::string result = std::string(text);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return result;
}

// Encode word to individual character tokens, adding end-of-word marker to the last one
std::vector<string32> split_chars(std::string_view word) {
    auto tokens = std::vector<string32>(word.size());
    for (int i = 0; i < (int)word.size() - 1; ++i) {
        tokens[i] = string32(word[i]);
    }
    if (!tokens.empty()) {
        format(tokens.back(), "{}</w>", word.back());
    }
    return tokens;
}

struct clip_text_tokens {
    std::vector<uint32_t> token_ids;
    std::vector<uint16_t> attention_mask;
};

struct clip_tokenizer {
    string_list vocab_list;
    std::unordered_map<std::string_view, uint32_t> vocab;

    string_list bpe_merges;
    std::unordered_map<std::string_view, int64_t> bpe_rank;

    uint32_t bos_token_id = 49406; // <|startoftext|>
    uint32_t eos_token_id = 49407; // <|endoftext|>
    uint32_t pad_token_id = 49407; // padding
    uint32_t unk_token_id = 49407; // unknown

    std::vector<string32> apply_bpe(std::vector<string32> tokens) const {
        bool changed = true;
        while (changed && tokens.size() > 1) {
            changed = false;

            // Find the merge pair with the lowest rank (highest priority)
            int best_rank = INT_MAX;
            int best_idx = -1;

            for (size_t i = 0; i + 1 < tokens.size(); i++) {
                ASSERT(tokens[i].length + tokens[i + 1].length + 1 < 32);
                auto pair = format<string32>("{} {}", tokens[i].view(), tokens[i + 1].view());
                auto it = bpe_rank.find(pair.view());
                if (it != bpe_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = (int)i;
                }
            }

            if (best_idx >= 0) {
                tokens[best_idx] = format<string32>(
                    "{}{}", tokens[best_idx].view(), tokens[best_idx + 1].view());
                tokens.erase(tokens.begin() + best_idx + 1);
                changed = true;
            }
        }
        return tokens;
    }

    clip_text_tokens tokenize(std::string_view text, int max_tokens) const {
        auto token_ids = std::vector<uint32_t>(max_tokens, pad_token_id);
        auto attn_mask = std::vector<uint16_t>(max_tokens * max_tokens, f16_neg_inf);

        // fill lower diagonal to build causal attention mask for flash attention
        auto attn_fill = [&](int row, int cols = -1) {
            cols = (cols == -1) ? row + 1 : cols;
            std::fill_n(attn_mask.begin() + row * max_tokens, cols, f16_zero);
        };

        int i = 0;
        token_ids[i] = bos_token_id;
        attn_fill(i);
        i++;

        if (!text.empty()) {
            std::string normalized = normalize_prompt(text);

            // Pre-tokenize to letter sequences, single digits, other non-whitespace
            static const std::regex pat("[a-zA-Z]+|[0-9]|[^\\s\\da-zA-Z]+");
            std::vector<std::string> words;
            auto it = std::sregex_iterator(normalized.begin(), normalized.end(), pat);
            for (auto end = std::sregex_iterator(); it != end; ++it) {
                words.push_back(it->str());
            }

            for (std::string const& w : words) {
                auto tokens = apply_bpe(split_chars(w));
                for (string32 const& tok : tokens) {
                    auto it = vocab.find(tok.view());
                    token_ids[i] = (it != vocab.end()) ? it->second : unk_token_id;
                    attn_fill(i);
                    i++;
                    if (i >= max_tokens - 1) {
                        break; // reserve space for eos token
                    }
                }
            }
        }

        token_ids[i] = eos_token_id;
        attn_fill(i);
        for (int row = i + 1; row < max_tokens; ++row) {
            attn_fill(row, i + 1);
        }
        return {token_ids, attn_mask};
    }
};

clip_tokenizer clip_tokenizer_init(model_file const& file) {
    clip_tokenizer tokenizer;
    // Build vocab map: token_string -> id (reverse of the stored id -> token_string list)
    tokenizer.vocab_list = get_string_list(file, "tokenizer.ggml.tokens");
    tokenizer.vocab.reserve(tokenizer.vocab_list.size());
    for (uint32_t i = 0; i < (uint32_t)tokenizer.vocab_list.size(); ++i) {
        tokenizer.vocab[tokenizer.vocab_list[i]] = i;
    }

    // Parse merge rules stored as "token1 token2" strings
    tokenizer.bpe_merges = get_string_list(file, "tokenizer.ggml.merges");
    int64_t rank = 0;
    for (std::string_view const& merge_str : tokenizer.bpe_merges) {
        tokenizer.bpe_rank[merge_str] = rank++;
    }

    tokenizer.bos_token_id = file.get_uint32("tokenizer.ggml.bos_token_id");
    tokenizer.eos_token_id = file.get_uint32("tokenizer.ggml.eos_token_id");
    tokenizer.pad_token_id = file.get_uint32("tokenizer.ggml.padding_token_id");
    tokenizer.unk_token_id = file.get_uint32("tokenizer.ggml.unknown_token_id");
    return tokenizer;
}

//
// CLIP text encoder

tensor clip_text_embed(model_ref m, tensor ids) {
    int64_t seq_len = ids->ne[0];

    tensor token_embeds = m.weights("token_embedding.weight");
    token_embeds = ggml_get_rows(m, token_embeds, ids);

    tensor pos_embeds = m.weights("position_embedding.weight");
    pos_embeds = slice(m, pos_embeds, {}, {0, seq_len}, {}, {});

    return ggml_add(m, token_embeds, pos_embeds);
}

tensor clip_attention(model_ref m, tensor x, tensor attention_mask) {
    const int64_t n_heads = 16;
    const int64_t head_dim = x->ne[0] / n_heads;
    const float scale = 1.f / std::sqrt((float)head_dim);

    tensor q = linear(m["q_proj"], x);
    tensor k = linear(m["k_proj"], x);
    tensor v = linear(m["v_proj"], x);

    q = ggml_reshape_4d(m, q, head_dim, n_heads, q->ne[1], q->ne[2]);
    k = ggml_reshape_4d(m, k, head_dim, n_heads, k->ne[1], k->ne[2]);
    v = ggml_reshape_4d(m, v, head_dim, n_heads, v->ne[1], v->ne[2]);

    return attention(m, q, k, v, attention_mask, scale, m["out_proj"]);
}

tensor clip_mlp(model_ref m, tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu(m, x);
    x = linear(m["fc2"], x);
    return x;
}

tensor clip_encoder_layer(model_ref m, tensor x, tensor attention_mask) {
    tensor residual = x;
    x = layer_norm(m["layer_norm1"], x);
    x = clip_attention(m["self_attn"], x, attention_mask);
    x = ggml_add(m, x, residual);

    residual = x;
    x = layer_norm(m["layer_norm2"], x);
    x = clip_mlp(m["mlp"], x);
    x = ggml_add(m, x, residual);
    return x;
}

tensor clip_text_encode(model_ref m, tensor embeds, tensor attention_mask) {
    model_ref layers = m["layers"];
    for (int i = 0; i < 24; ++i) {
        embeds = clip_encoder_layer(layers[i], embeds, attention_mask);
    }
    return embeds;
}

tensor clip_encode_text(model_ref m, tensor ids, tensor attention_mask) {
    tensor embeds = clip_text_embed(m["embeddings"], ids);
    tensor encoded = clip_text_encode(m["encoder"], embeds, attention_mask);
    encoded = layer_norm(m["final_layer_norm"], encoded);
    return encoded;
}

tensor encode_text(model_ref m, tensor ids, tensor attention_mask) {
    tensor clip_last_hidden = clip_encode_text(m["te.text_model"], ids, attention_mask);
    tensor embeds = linear(m["text_projection"], clip_last_hidden);
    return embeds;
}

//
// Vision encoder

struct rope_2d_positions {
    tensor x; // patch position indices across width
    tensor y; // patch position indices across height
    float scale;
};

struct sam3_vit_params {
    int image_size = 1008;
    int patch_size = 14;
    int window_size = 24;
    int n_layers = 32;
    int n_heads = 16;
    std::array<int, 4> global_attn_indexes = {7, 15, 23, 31};
    std::array<float, 4> scale_factors = {4.0f, 2.0f, 1.0f, 0.5f};
};

struct sam3_params {
    sam3_vit_params vision;
};

tensor vision_embed(model_ref m, tensor image, int patch_size) {
    auto [w, h, c, b] = nelements(image);
    int64_t wp = w / patch_size;
    int64_t hp = h / patch_size;

    // Patch embedding with stride=patch_size
    tensor embed = conv_2d(m["patch_embeddings.projection"], image, patch_size);
    embed = ggml_reshape_3d(m, embed, wp * hp, embed->ne[2], b);
    embed = ggml_permute(m, embed, 1, 0, 2, 3); // -> [hidden_size, n_patches, batch]
    embed = ggml_cont(m, embed);

    tensor pos_embed = m.weights("position_embeddings");
    int64_t hidden_size = pos_embed->ne[0];
    int64_t pretrain_size = (int64_t)(std::sqrt((float)pos_embed->ne[1]) + 0.5f);
    if (wp == pretrain_size && hp == pretrain_size) {
        pos_embed = ggml_reshape_3d(m, pos_embed, hidden_size, wp * hp, 1);
    } else {
        // Tile position embeddings to match image size
        pos_embed = ggml_reshape_4d(m, pos_embed, hidden_size, pretrain_size, pretrain_size, 1);
        pos_embed = ggml_repeat_4d(m, pos_embed, hidden_size, wp, hp, 1);
        pos_embed = ggml_reshape_3d(m, pos_embed, hidden_size, wp * hp, 1);
    }
    return ggml_add(m, embed, pos_embed);
}

tensor mlp(model_ref m, tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu(m, x);
    x = linear(m["fc2"], x);
    return x;
}

tensor window_partition(model_ref m, tensor x, int window) {
    auto [c, w, h, b] = nelements(x);
    // if (m.flags & model_build_flag::window_partition) {
    //     x = ggml_win_part(m, x, window);
    //     x = ggml_reshape_3d(m, x, c, window * window, x->ne[3]);
    //     return x;
    // }
    int64_t px = (window - w % window) % window;
    int64_t py = (window - h % window) % window;
    int64_t npw = (w + px) / window;
    int64_t nph = (h + py) / window;

    if (px > 0 || py > 0) {
        x = ggml_pad(m, x, 0, int(px), int(py), 0);
    }
    x = ggml_reshape_4d(m, x, c * window, npw, window, nph * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, window * window, npw * nph * b);
    return x;
}

tensor window_reverse(model_ref m, tensor x, int w, int h, int window) {
    int64_t c = x->ne[0];
    int64_t b = x->ne[3];
    // if (m.flags & model_build_flag::window_partition) {
    //     x = ggml_reshape_4d(m, x, c, window, window, x->ne[2]);
    //     x = ggml_win_unpart(m, x, w, h, window);
    //     return x;
    // }
    int64_t px = (window - w % window) % window;
    int64_t py = (window - h % window) % window;
    int64_t npw = (w + px) / window;
    int64_t nph = (h + py) / window;

    x = ggml_reshape_4d(m, x, c * window, window, npw, nph * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, c, w + px, h + py, b);
    if (px > 0 || py > 0) {
        x = slice(m, x, {}, {0, w}, {0, h});
        x = ggml_cont(m, x);
    }
    return x;
}

rope_2d_positions build_rope_2d_positions(model_ref m, int n_pos, float scale, char const* name) {
    return {
        compute_graph_input(m, GGML_TYPE_I32, {n_pos, 1, 1, 1}, format<tensor_name>("{}.x", name)),
        compute_graph_input(m, GGML_TYPE_I32, {n_pos, 1, 1, 1}, format<tensor_name>("{}.y", name)),
        scale};
}

rope_2d_positions get_rope_2d_positions(model_ref m, char const* name) {
    tensor_name x_name = format<tensor_name>("{}.x", name);
    tensor_name y_name = format<tensor_name>("{}.y", name);
    tensor x_pos = ggml_get_tensor(m, x_name.c_str());
    tensor y_pos = ggml_get_tensor(m, y_name.c_str());
    ASSERT(x_pos && y_pos);
    return {x_pos, y_pos, 1.0f};
}

void init_rope_2d_positions(rope_2d_positions& pos, int n_pos, int height) {
    std::vector<int32_t> pos_data(n_pos);
    // width dimension
    for (int i = 0; i < n_pos; i++) {
        pos_data[i] = i % height;
    }
    transfer_to_backend(pos.x, as_bytes(pos_data));
    // height dimension
    for (int i = 0; i < n_pos; i++) {
        pos_data[i] = i / height;
    }
    transfer_to_backend(pos.y, as_bytes(pos_data));
}

tensor apply_rope_2d(model_ref m, tensor x, rope_2d_positions const& pos) {
    float const freq_base = 10000.f;
    int64_t const dim = x->ne[0];
    // The head_dim is split into two halves:
    //   first  half (elements 0..dim/2-1)  : apply RoPE using x positions
    //   second half (elements dim/2..dim-1): apply RoPE using y positions
    // Both halves use the same frequency schedule: 1/freq_base^(2i/(dim/2)) for i in [0, dim/4)

    // first half
    tensor first = slice(m, x, {0, dim / 2}, {}, {}, {});
    first = ggml_rope_ext(
        m, first, pos.x,
        nullptr, // freq factors
        dim / 2, // n_dims
        GGML_ROPE_TYPE_NORMAL, 0, freq_base, pos.scale, 0.0f, 1.0f, 0.0f, 0.0f);

    // second half
    tensor second = slice(m, x, {dim / 2, dim}, {}, {}, {});
    second = ggml_rope_ext(
        m, second, pos.y,
        nullptr, // freq factors
        dim / 2, // n_dims
        GGML_ROPE_TYPE_NORMAL, 0, freq_base, pos.scale, 0.0f, 1.0f, 0.0f, 0.0f);

    return ggml_concat(m, first, second, 0);
}

tensor rope_attention(model_ref m, tensor x, int n_heads, rope_2d_positions const& pos) {
    int64_t head_dim = x->ne[0] / n_heads;
    float scale = 1.f / std::sqrt((float)head_dim);

    tensor q = linear(m["q_proj"], x);
    tensor k = linear(m["k_proj"], x);
    tensor v = linear(m["v_proj"], x);

    q = ggml_reshape_4d(m, q, head_dim, n_heads, q->ne[1], q->ne[2]);
    k = ggml_reshape_4d(m, k, head_dim, n_heads, k->ne[1], k->ne[2]);
    v = ggml_reshape_4d(m, v, head_dim, n_heads, v->ne[1], v->ne[2]);

    q = apply_rope_2d(m, q, pos);
    k = apply_rope_2d(m, k, pos);
    return attention(m, q, k, v, nullptr, scale, m["o_proj"]);
}

tensor vision_layer(
    model_ref m, tensor x, int window_size, int n_heads, rope_2d_positions const& rope_pos) {

    tensor residual_attn = x;
    x = layer_norm(m["layer_norm1"], x);

    auto [c, w, h, b] = nelements(x);
    if (window_size > 0) {
        x = window_partition(m, x, window_size);
    } else {
        x = ggml_reshape_3d(m, x, c, w * h, b);
    }
    x = rope_attention(m["attention"], x, n_heads, rope_pos);

    if (window_size > 0) {
        x = window_reverse(m, x, w, h, window_size);
    } else {
        x = ggml_reshape_4d(m, x, c, w, h, b);
    }
    x = ggml_add(m, x, residual_attn);

    tensor residual_mlp = x;
    x = layer_norm(m["layer_norm2"], x);
    x = mlp(m["mlp"], x);
    x = ggml_add(m, x, residual_mlp);
    return named(m, x);
}

tensor vision_transformer(model_ref m, tensor image, sam3_vit_params const& p) {
    tensor x = vision_embed(m["embeddings"], image, p.patch_size);

    int64_t c = x->ne[0];
    int64_t w = image->ne[0] / p.patch_size;
    int64_t h = image->ne[1] / p.patch_size;
    int64_t b = image->ne[3];
    x = ggml_reshape_4d(m, x, c, w, h, b);

    rope_2d_positions pos_window = build_rope_2d_positions(
        m, sqr(p.window_size), 1.0f, "_vit_rope_pos_window");

    int n_pos_global = sqr(p.image_size / p.patch_size);
    float scale_global = float(p.window_size) / float(p.image_size / p.patch_size);
    rope_2d_positions pos_global = build_rope_2d_positions(
        m, n_pos_global, scale_global, "_vit_rope_pos_global");

    x = layer_norm(m["layer_norm"], x);

    model_ref layers = m["layers"];
    for (int i = 0; i < p.n_layers; ++i) {
        bool is_global = contains(span(p.global_attn_indexes), i);
        int window_size = is_global ? 0 : p.window_size;
        auto const& pos = is_global ? pos_global : pos_window;

        x = vision_layer(layers[i], x, window_size, p.n_heads, pos);
    }

    x = ggml_reshape_3d(m, x, c, w * h, b);
    return x;
}

struct vision_output {
    static constexpr int n_layers = 4;

    std::array<tensor, n_layers> fpn_hidden_states;
    std::array<tensor, n_layers> fpn_position_encoding;
};

std::vector<float> generate_sine_position_embedding(
    int width, int height, int n_pos_feats, bool normalize = false) {

    constexpr float temperature = 10000.f;
    constexpr float scale = 2.f * M_PIf;
    constexpr float eps = 1e-6f;

    std::vector<float> dim_t(n_pos_feats);
    for (int k = 0; k < n_pos_feats; ++k) {
        dim_t[k] = std::pow(temperature, 2.f * (k / 2) / n_pos_feats);
    }

    auto out = std::vector<float>(width * height * n_pos_feats * 2);
    for (int h = 0; h < height; ++h) {
        float y = (float)(h + 1);
        if (normalize) {
            y = y / ((float)height + eps) * scale;
        }
        for (int w = 0; w < width; ++w) {
            float x = (float)(w + 1);
            if (normalize) {
                x = x / ((float)width + eps) * scale;
            }

            for (int k = 0; k < n_pos_feats; ++k) {
                float y_val = (k % 2 == 0) ? std::sin(y / dim_t[k]) : std::cos(y / dim_t[k]);
                float x_val = (k % 2 == 0) ? std::sin(x / dim_t[k]) : std::cos(x / dim_t[k]);
                out[k * height * width + h * width + w] = y_val;
                out[(n_pos_feats + k) * height * width + h * width + w] = x_val;
            }
        }
    }
    return out;
}

tensor sine_position_embedding(
    model_ref m, std::array<int64_t, 4> shape, int n_pos_feats, bool normalize) {

    int width = (int)shape[0];
    int height = (int)shape[1];
    tensor pe = ggml_new_tensor_3d(m, GGML_TYPE_F32, width, height, n_pos_feats * 2);
    ggml_set_input(pe);
    ggml_format_name(
        pe, "_sine_position_embedding_%dx%dx%d%s", width, height, n_pos_feats,
        normalize ? "_normalized" : "");
    return pe;
}

tensor fpn_layer(model_ref m, tensor x, int index) {
    switch (index) {
        case 0: // scale factor = 4
            x = conv_transpose_2d(m["scale_layers.0"], x, 2);
            x = ggml_gelu(m, x);
            x = conv_transpose_2d(m["scale_layers.2"], x, 2);
            break;
        case 1: // scale factor = 2
            x = conv_transpose_2d(m["scale_layers.0"], x, 2);
            break;
        case 2: // scale factor = 1
            break;
        case 3: // scale factor = 0.5
            // TODO: what is this last layer used for?
            x = ggml_pool_2d(m, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
            break;
        default: throw except("Invalid FPN layer index {}", index);
    }
    x = conv_2d(m["proj1"], x, 1, 0); // 1x1 pad=0
    x = conv_2d(m["proj2"], x, 1, 1); // 3x3 pad=1
    return x;
}

vision_output vision_neck(model_ref m, tensor x) {
    vision_output out;
    model_ref layers = m["fpn_layers"];

    for (int i = 0; i < vision_output::n_layers; ++i) {
        out.fpn_hidden_states[i] = fpn_layer(layers[i], x, i);
        out.fpn_position_encoding[i] = sine_position_embedding(
            m, nelements(out.fpn_hidden_states[i]), 64, true);
    }
    return out;
}

vision_output encode_vision(model_ref m, tensor image, sam3_vit_params const& p) {
    const auto [w, h, c, b] = nelements(image);
    // Backbone
    tensor x = vision_transformer(m["backbone"], image, p);

    // Neck
    x = ggml_reshape_4d(m, x, x->ne[0], w / p.patch_size, h / p.patch_size, b);
    x = cwhn_to_contiguous_2d(m, x);
    return vision_neck(m["neck"], x);
}

} // namespace sam3

using sam3::clip_text_tokens;

image_data sam3_process_input(image_view img) {
    image_data resized = image_scale(img, i32x2{1008, 1008});
    return image_u8_to_f32(resized, image_format::rgb_f32, {-0.5f}, {2.f});
}

clip_text_tokens clip_tokenize(model_file const& file, std::string_view text) {
    sam3::clip_tokenizer tokenizer = sam3::clip_tokenizer_init(file);
    int max_tokens = file.get_int("sam3.tokenizer.max_length");
    return tokenizer.tokenize(text, max_tokens);
}

} // namespace visp