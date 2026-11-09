// What upstream llama.cpp does with a GGUF alone: its loader reads the file's metadata and vocabulary, its tokenizer
// encodes the fixed strings and its chat path renders the fixed conversations from the file's own template.
//
//   verify-publish <first shard> <fixture.json>
//
// The loader is llama.cpp's own: llama_model_load's metadata and vocabulary steps (src/llama.cpp), without the
// tensors, which this file's companion scales are not in llama.cpp's form. So every key the loader reads for the
// file's architecture, required or optional, is read by the code that defines it, with its own type rules.
// The chat path is llama-server's: OpenAI messages and tools through common_chat_templates_apply, with the template
// arguments as chat_template_kwargs (tools/server/server-common.cpp).
//
// Prints exactly one line per check (metadata, tokenizer, chat), "<check>: pass ..." or "<check>: refused: <what>",
// and exits 1 if any is refused.

#include "chat.h"
#include "json.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static void quiet(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static bool refuse(const char * check, const std::string & what) {
    printf("%s: refused: %s\n", check, what.c_str());
    return false;
}

static std::string ids_text(const std::vector<llama_token> & ids) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); i++) {
        out += (i ? ", " : "") + std::to_string(ids[i]);
    }
    return out + "]";
}

// The tokenizer: the fixture's texts through llama_tokenize, as HF tokenizers encodes them (no implicit special
// tokens, special-token text matched)
static bool check_tokenizer(const llama_vocab * vocab, const common_json & corpus) {
    if (corpus.empty()) {
        return refuse("tokenizer", "the fixture has no texts to encode");
    }
    for (const auto & entry : corpus) {
        const auto text = entry.at("text").get<std::string>();
        const auto want = entry.at("ids").get<std::vector<int>>();
        std::vector<llama_token> got(text.size() + 8);
        const int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), got.data(), (int32_t) got.size(),
                                     /* add_special */ false, /* parse_special */ true);
        if (n < 0) {
            return refuse("tokenizer", "could not encode " + common_json::make(text).dump());
        }
        got.resize(n);
        if (got != want) {
            return refuse("tokenizer", common_json::make(text).dump() + " encodes to " + ids_text(got) +
                                           ", tokenizer.json gives " + ids_text(want));
        }
    }
    printf("tokenizer: pass, %zu texts encode to tokenizer.json's ids\n", corpus.size());
    return true;
}

static bool check_chat(const llama_model * model, const common_json & fixture) {
    if (llama_model_chat_template(model, nullptr) == nullptr) {
        // common_chat_templates_init would render ChatML instead (common/chat.cpp)
        return refuse("chat", "tokenizer.chat_template is absent; llama.cpp would render ChatML");
    }
    auto       tmpls = common_chat_templates_init(model, "");
    const auto now   = std::chrono::system_clock::time_point(std::chrono::seconds(fixture.at("now").get<int64_t>()));
    const auto & renders = fixture.at("renders");
    if (renders.empty()) {
        return refuse("chat", "the fixture has no conversations to render");
    }
    for (const auto & r : renders) {
        const auto name = r.at("name").get<std::string>();
        common_chat_templates_inputs inputs;
        inputs.messages         = common_chat_msgs_parse_oaicompat(r.at("messages"));
        inputs.tools            = common_chat_tools_parse_oaicompat(r.at("tools"));
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.now              = now;
        for (const auto & [key, value] : r.at("template_arguments").items()) {
            inputs.chat_template_kwargs[key] = value.dump();
        }
        if (r.at("template_arguments").contains("enable_thinking")) {
            inputs.enable_thinking = r.at("template_arguments").at("enable_thinking").get<bool>();
        }
        std::string got;
        try {
            got = common_chat_templates_apply(tmpls.get(), inputs).prompt.str();
        } catch (const std::exception & e) {
            return refuse("chat", name + ": " + e.what());
        }
        const auto want = r.at("text").get<std::string>();
        if (got != want) {
            size_t i = 0;
            while (i < got.size() && i < want.size() && got[i] == want[i]) {
                i++;
            }
            return refuse("chat", name + ": differs from the reference at byte " + std::to_string(i) + ": " +
                                      common_json::make(got.substr(i, 80)).dump() + " vs " +
                                      common_json::make(want.substr(i, 80)).dump());
        }
    }
    printf("chat: pass, %zu conversations render byte-equal from the file's tokenizer.chat_template\n",
           renders.size());
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <first shard> <fixture.json>\n", argv[0]);
        return 2;
    }
    llama_log_set(quiet, nullptr);
    std::ifstream     fixture_file(argv[2]);
    std::stringstream fixture_text;
    fixture_text << fixture_file.rdbuf();
    const auto fixture = common_json::parse(fixture_text.str());

    llama_model_params params = llama_model_default_params();
    std::vector<std::string> splits;
    std::unique_ptr<llama_model_loader> ml;
    std::unique_ptr<llama_model> model;
    try {
        ml = std::make_unique<llama_model_loader>(nullptr, nullptr, nullptr, argv[1], splits, nullptr,
                                                  LLAMA_LOAD_MODE_NONE, /* check_tensors */ false,
                                                  /* no_alloc */ true, params.load_mtp, nullptr, nullptr);
        model.reset(llama_model_create(*ml, params));
    } catch (const std::exception & e) {
        refuse("metadata", e.what());
        refuse("tokenizer", "the file did not load");
        refuse("chat", "the file did not load");
        return 1;
    }

    bool ok = true;
    // the full load's metadata step, not the vocab-only one, which returns before the architecture's keys
    model->hparams.vocab_only = false;
    try {
        model->load_hparams(*ml);
        printf("metadata: pass, llama.cpp's loader reads every key it requires for %s\n", ml->get_arch_name().c_str());
    } catch (const std::exception & e) {
        ok = refuse("metadata", e.what());
    }
    bool vocab = true;
    try {
        model->load_vocab(*ml);
    } catch (const std::exception & e) {
        vocab = ok = refuse("tokenizer", e.what());
    }
    if (vocab) {
        ok = check_tokenizer(llama_model_get_vocab(model.get()), fixture.at("corpus")) && ok;
        ok = check_chat(model.get(), fixture) && ok;
    } else {
        ok = refuse("chat", "the vocabulary did not load");
    }
    return ok ? 0 : 1;
}
