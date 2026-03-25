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
    x = ggml_gelu_inplace(m, x);
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