// Upstream llama.cpp's logits of every position of one prompt: the model on the CPU, the prompt decoded in one forward
// from an empty context, without flash attention and with an F32 attention cache, written as float32 rows
// [tokens, vocabulary] (`magnitude_conversion/llama.py`).
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: logits MODEL OUT.f32 TOKEN...\n");
        return 2;
    }
    std::vector<llama_token> tokens;
    for (int i = 3; i < argc; ++i) {
        tokens.push_back(atoi(argv[i]));
    }
    const uint32_t n = (uint32_t) tokens.size();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "logits: %s did not load\n", argv[1]);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = (int32_t) std::thread::hardware_concurrency();
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "logits: no context for %u tokens\n", n);
        return 1;
    }

    llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
    for (uint32_t i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = (int32_t) n;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "logits: decode failed\n");
        return 1;
    }
    const int32_t vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "logits: can't write %s\n", argv[2]);
        return 1;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (fwrite(llama_get_logits_ith(ctx, (int32_t) i), sizeof(float), vocabulary, out) != (size_t) vocabulary) {
            fprintf(stderr, "logits: short write\n");
            return 1;
        }
    }
    fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
