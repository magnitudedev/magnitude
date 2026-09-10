// Independent same-GGUF teacher-forced oracle. Build against a recorded llama.cpp
// installation; this is test tooling and is never linked into the engine.
#include <llama.h>
#include <ggml-backend.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: llama_logits MODEL OUTPUT PROMPT cpu|metal\n");
        return 2;
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    const bool cpu = std::strcmp(argv[4], "cpu") == 0;
    mp.n_gpu_layers = cpu ? 0 : -1;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t width = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens(std::strlen(argv[3]) + 16);
    const int32_t count = llama_tokenize(vocab, argv[3], std::strlen(argv[3]),
                                       tokens.data(), tokens.size(), false, true);
    if (count <= 0) throw std::runtime_error("tokenization failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_seq_max = 1;
    cp.n_batch = cp.n_ubatch = count;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = !cpu;
    llama_context *context = llama_init_from_model(model, cp);
    if (!context) return 4;
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    if (llama_decode(context, batch) != 0) return 5;
    FILE *output = std::fopen(argv[2], "wb");
    if (!output) return 6;
    std::fwrite(&count, sizeof(count), 1, output);
    std::fwrite(&width, sizeof(width), 1, output);
    std::fwrite(tokens.data(), sizeof(llama_token), count, output);
    for (int i = 0; i < count; ++i) {
        const float *logits = llama_get_logits_ith(context, i);
        if (!logits || std::fwrite(logits, sizeof(float), width, output) != size_t(width)) return 7;
    }
    if (std::fclose(output)) return 8;
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
