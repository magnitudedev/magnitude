// Independent long-context oracle on exact supplied token IDs. Only the final
// logit row is retained; reference storage does not scale with context*vocabulary.
#include <llama.h>
#include <ggml-backend.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5 || (std::strcmp(argv[4], "cpu") && std::strcmp(argv[4], "metal"))) {
        std::fprintf(stderr, "usage: llama_tail MODEL TOKENS_I32 OUTPUT cpu|metal\n");
        return 2;
    }
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    if (!input) return 3;
    auto bytes = input.tellg();
    if (bytes <= 0 || bytes % sizeof(llama_token) || bytes / sizeof(llama_token) > 262144) return 4;
    std::vector<llama_token> tokens(size_t(bytes) / sizeof(llama_token));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(tokens.data()), bytes)) return 5;
    const int32_t count = tokens.size();
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    const bool cpu = std::strcmp(argv[4], "cpu") == 0;
    mp.n_gpu_layers = cpu ? 0 : -1;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 6;
    const int32_t width = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (auto token : tokens) if (token < 0 || token >= width) return 7;
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(count, 512);
    cp.n_seq_max = 1;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = !cpu;
    llama_context *context = llama_init_from_model(model, cp);
    if (!context) return 8;
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int start = 0; start < count; start += 512) {
        batch.n_tokens = std::min(512, count - start);
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = start + i == count - 1;
        }
        if (llama_decode(context, batch) != 0) return 9;
        std::fprintf(stderr, "oracle processed %d/%d\n", start + batch.n_tokens, count);
    }
    const float *logits = llama_get_logits_ith(context, -1);
    if (!logits) return 10;
    const std::string temporary = std::string(argv[3]) + ".part";
    FILE *output = std::fopen(temporary.c_str(), "wb");
    if (!output) return 11;
    bool written = std::fwrite(&count, sizeof(count), 1, output) == 1
        && std::fwrite(&width, sizeof(width), 1, output) == 1
        && std::fwrite(tokens.data(), sizeof(llama_token), count, output) == size_t(count)
        && std::fwrite(logits, sizeof(float), width, output) == size_t(width);
    if (std::fclose(output) || !written) return 12;
    if (std::rename(temporary.c_str(), argv[3])) return 13;
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
