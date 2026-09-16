// Reference-engine execution only. Magnitude always uses its production TileLang path.
#include <llama.h>
#include <ggml-backend.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    const std::string backend = argv[3];
    const int ordinal = std::stoi(argv[4]);
    const std::string registration = backend == "metal" ? "mtl" : backend;
    ggml_backend_dev_t selected = nullptr;
    int found = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name == registration && found++ == ordinal) selected = device;
    }
    if (!selected) { std::cerr << "requested backend/device unavailable\n"; return 3; }
    ggml_backend_dev_t devices[] = {selected, nullptr};
    mp.devices = devices;
    mp.n_gpu_layers = backend == "cpu" ? 0 : 999;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 4;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::stoul(argv[2]);
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 5; }
    std::ifstream fixture(argv[5]);
    size_t np, nd;
    fixture >> np >> nd;
    std::vector<llama_token> prompt(np), continuation(nd);
    for (auto &token : prompt) fixture >> token;
    for (auto &token : continuation) fixture >> token;
    if (!fixture) return 6;
    auto decode = [&](llama_token *tokens, size_t count) {
        auto batch = llama_batch_get_one(tokens, static_cast<int32_t>(count));
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        llama_synchronize(ctx);
    };
    auto prefill = [&]() {
        for (size_t i = 0; i < np; i += 512) decode(prompt.data()+i, std::min(size_t(512), np-i));
    };
    std::cout << "READY " << ggml_backend_dev_name(selected) << std::endl;
    std::string command;
    try {
        while (std::cin >> command) {
            if (command == "quit") break;
            if (command != "prefill" && command != "decode") throw std::runtime_error("invalid phase");
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_synchronize(ctx);
            if (command == "decode") prefill();
            const auto start = std::chrono::steady_clock::now();
            if (command == "prefill") prefill();
            else for (auto &token : continuation) decode(&token, 1);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::cout << std::setprecision(17) << seconds << std::endl;
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 7;
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
