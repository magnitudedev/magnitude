#include "chat.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

constexpr int PROTOCOL_VERSION = 1;
constexpr size_t MAX_INPUT_LINE_BYTES = 16 * 1024 * 1024;
bool backend_initialized = false;

using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

const std::vector<std::string> OPERATIONS = {
    "protocol.describe",
    "configuration.inspect",
    "model.metadata",
    "tokenizer.tokenize",
    "tokenizer.token-to-piece",
    "chat-template.render",
    "grammar.json-schema-to-grammar",
    "sampler.apply",
    "decode.execute-plan",
    "state.execute-script",
    "sampler.bench",
    "chat-template.bench",
    "chat-parser.inspect",
    "chat-parser.bench",
    "reasoning-budget.inspect",
    "decode.abort",
    "decode.abort-recovery",
};

struct batch_entry {
    llama_token token;
    llama_pos position;
    std::vector<llama_seq_id> sequence_ids;
    bool request_logits;
};

// Owns every array referenced by the public llama_batch view. Keeping
// construction here makes the exact token/position/sequence/logits boundary
// explicit without relying on a higher-level helper.
class native_batch {
  public:
    explicit native_batch(std::vector<batch_entry> entries) : entries_(std::move(entries)) {
        if (entries_.empty()) {
            throw std::invalid_argument("batch must not be empty");
        }
        if (entries_.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("batch contains too many entries");
        }
        tokens_.reserve(entries_.size());
        positions_.reserve(entries_.size());
        sequence_counts_.reserve(entries_.size());
        sequence_pointers_.reserve(entries_.size());
        logits_.reserve(entries_.size());
        for (auto &entry : entries_) {
            if (entry.sequence_ids.empty()) {
                throw std::invalid_argument("each batch entry must have at least one sequence ID");
            }
            if (entry.sequence_ids.size() >
                static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::invalid_argument("batch entry contains too many sequence IDs");
            }
            tokens_.push_back(entry.token);
            positions_.push_back(entry.position);
            sequence_counts_.push_back(static_cast<int32_t>(entry.sequence_ids.size()));
            sequence_pointers_.push_back(entry.sequence_ids.data());
            logits_.push_back(entry.request_logits ? 1 : 0);
        }
    }

    native_batch(const native_batch &) = delete;
    native_batch &operator=(const native_batch &) = delete;

    llama_batch view() {
        return {
            static_cast<int32_t>(entries_.size()),
            tokens_.data(),
            nullptr,
            positions_.data(),
            sequence_counts_.data(),
            sequence_pointers_.data(),
            logits_.data(),
        };
    }

  private:
    std::vector<batch_entry> entries_;
    std::vector<llama_token> tokens_;
    std::vector<llama_pos> positions_;
    std::vector<int32_t> sequence_counts_;
    std::vector<llama_seq_id *> sequence_pointers_;
    std::vector<int8_t> logits_;
};

template <typename T> T required(const json &value, const char *field) {
    if (!value.contains(field)) {
        throw std::invalid_argument(std::string("missing field: ") + field);
    }
    return value.at(field).get<T>();
}

template <typename T> T optional(const json &value, const char *field, T fallback) {
    return value.contains(field) ? value.at(field).get<T>() : std::move(fallback);
}

void require_only_fields(const json &input, std::initializer_list<const char *> allowed) {
    if (!input.is_object()) {
        throw std::invalid_argument("operation input must be an object");
    }
    for (auto iterator = input.begin(); iterator != input.end(); ++iterator) {
        const bool known = std::any_of(
            allowed.begin(), allowed.end(),
            [&](const char *field) { return iterator.key() == field; });
        if (!known) {
            throw std::invalid_argument("unknown input field: " + iterator.key());
        }
    }
}

json byte_array(const std::string &value) {
    json result = json::array();
    for (const unsigned char byte : value) {
        result.push_back(byte);
    }
    return result;
}

json finite_number(float value) {
    if (std::isfinite(value)) {
        return json{{"value", value}, {"class", "finite"}};
    }
    if (std::isnan(value)) {
        return json{{"value", nullptr}, {"class", "nan"}};
    }
    return json{
        {"value", nullptr},
        {"class", value > 0 ? "positive-infinity" : "negative-infinity"},
    };
}

std::string trigger_type_name(common_grammar_trigger_type type) {
    switch (type) {
    case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
        return "token";
    case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
        return "word";
    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
        return "pattern";
    case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
        return "pattern-full";
    }
    throw std::runtime_error("unknown grammar trigger type");
}

std::string vocab_type_name(enum llama_vocab_type type) {
    switch (type) {
    case LLAMA_VOCAB_TYPE_NONE:
        return "none";
    case LLAMA_VOCAB_TYPE_SPM:
        return "spm";
    case LLAMA_VOCAB_TYPE_BPE:
        return "bpe";
    case LLAMA_VOCAB_TYPE_WPM:
        return "wpm";
    case LLAMA_VOCAB_TYPE_UGM:
        return "ugm";
    case LLAMA_VOCAB_TYPE_RWKV:
        return "rwkv";
    case LLAMA_VOCAB_TYPE_PLAMO2:
        return "plamo2";
    }
    throw std::runtime_error("unknown vocabulary type");
}

std::string rope_type_name(enum llama_rope_type type) {
    switch (type) {
    case LLAMA_ROPE_TYPE_NONE:
        return "none";
    case LLAMA_ROPE_TYPE_NORM:
        return "norm";
    case LLAMA_ROPE_TYPE_NEOX:
        return "neox";
    case LLAMA_ROPE_TYPE_MROPE:
        return "mrope";
    case LLAMA_ROPE_TYPE_VISION:
        return "vision";
    case LLAMA_ROPE_TYPE_IMROPE:
        break;
    }
    throw std::runtime_error("model RoPE type is outside the approved shared projection");
}

void ensure_backend_initialized() {
    if (!backend_initialized) {
        llama_backend_init();
        backend_initialized = true;
    }
}

model_ptr load_model(const json &input, bool vocab_only_default) {
    ensure_backend_initialized();
    const std::string model_path = required<std::string>(input, "modelPath");
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = optional<int32_t>(input, "nGpuLayers", 0);
    params.vocab_only = optional<bool>(input, "vocabOnly", vocab_only_default);
    params.use_mmap = optional<bool>(input, "useMmap", true);
    params.use_mlock = optional<bool>(input, "useMlock", false);
    params.check_tensors = optional<bool>(input, "checkTensors", false);
    model_ptr model(llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
    if (!model) {
        throw std::runtime_error("llama_model_load_from_file failed");
    }
    return model;
}

std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text,
                                  bool add_special, bool parse_special) {
    const size_t initial_size = text.size() + (add_special ? 2 : 0);
    if (initial_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("text is too large to tokenize");
    }
    std::vector<llama_token> tokens(std::max<size_t>(initial_size, 1));
    int32_t count =
        llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                       static_cast<int32_t>(tokens.size()), add_special, parse_special);
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenization result exceeds int32_t");
    }
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), add_special, parse_special);
    }
    if (count < 0) {
        throw std::runtime_error("llama_tokenize failed after resizing output");
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::string token_piece(const llama_vocab *vocab, llama_token token, bool special) {
    std::vector<char> buffer(32);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, special);
    if (count < 0) {
        buffer.resize(static_cast<size_t>(-count));
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, special);
    }
    if (count < 0) {
        throw std::runtime_error("llama_token_to_piece failed after resizing output");
    }
    return std::string(buffer.data(), static_cast<size_t>(count));
}

json pieces_evidence(const llama_vocab *vocab, const std::vector<llama_token> &tokens,
                     bool special) {
    json pieces = json::array();
    for (const llama_token token : tokens) {
        const std::string piece = token_piece(vocab, token, special);
        pieces.push_back({{"token", token}, {"bytes", byte_array(piece)}});
    }
    return pieces;
}

json describe_protocol() {
    json operations = json::array();
    for (const auto &operation : OPERATIONS) {
        operations.push_back(operation);
    }
    return {
        {"protocolVersion", PROTOCOL_VERSION},
        {"transport", "jsonl-stdin-stdout"},
        {"operations", std::move(operations)},
    };
}

context_ptr create_context(llama_model *model, llama_context_params params) {
    context_ptr context(llama_init_from_model(model, params), llama_free);
    if (!context) {
        throw std::runtime_error("llama_init_from_model failed");
    }
    return context;
}

void configure_cpu_resident_context(llama_context_params &params, const json &input) {
    const int32_t threads = required<int32_t>(input, "threads");
    const int32_t batch_threads = required<int32_t>(input, "batchThreads");
    if (threads <= 0) {
        throw std::invalid_argument("threads must be greater than zero");
    }
    if (batch_threads <= 0) {
        throw std::invalid_argument("batchThreads must be greater than zero");
    }
    const bool offload_kqv = required<bool>(input, "offloadKqv");
    const bool operation_offload = required<bool>(input, "operationOffload");
    const std::string flash_attention = required<std::string>(input, "flashAttention");
    if (offload_kqv || operation_offload || flash_attention != "off") {
        throw std::invalid_argument("CPU-resident correctness requires offloadKqv=false, "
                                    "operationOffload=false, and flashAttention=off");
    }
    params.n_threads = threads;
    params.n_threads_batch = batch_threads;
    params.offload_kqv = offload_kqv;
    params.op_offload = operation_offload;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
}

void require_positive(uint32_t value, const char *field) {
    if (value == 0) {
        throw std::invalid_argument(std::string(field) + " must be greater than zero");
    }
}

void validate_token(const llama_vocab *vocab, llama_token token, const char *field) {
    const int32_t token_count = llama_vocab_n_tokens(vocab);
    if (token < 0 || token >= token_count) {
        throw std::invalid_argument(std::string(field) + " must be a valid model token in [0, " +
                                    std::to_string(token_count) + ")");
    }
}

std::vector<int32_t> validated_logit_indices(const json &input, const llama_vocab *vocab,
                                             std::vector<int32_t> fallback) {
    const auto indices =
        optional<std::vector<int32_t>>(input, "logit_indices", std::move(fallback));
    if (indices.empty()) {
        throw std::invalid_argument("logit_indices must not be empty");
    }
    const int32_t token_count = llama_vocab_n_tokens(vocab);
    for (const int32_t index : indices) {
        if (index < 0 || index >= token_count) {
            throw std::invalid_argument("logit index must be a valid model token in [0, " +
                                        std::to_string(token_count) + ")");
        }
    }
    return indices;
}

json selected_logits(const float *logits, const std::vector<int32_t> &indices) {
    if (logits == nullptr) {
        throw std::runtime_error("requested logits row is unavailable");
    }
    json values = json::array();
    for (const int32_t index : indices) {
        const json encoded = finite_number(logits[index]);
        values.push_back({
            {"index", index},
            {"value", encoded.at("value")},
            {"class", encoded.at("class")},
        });
    }
    return values;
}

json decode_status(int32_t code) {
    if (code != 0) {
        throw std::runtime_error("llama_decode returned status " + std::to_string(code));
    }
    return {{"code", code}, {"class", "success"}};
}

json configuration_inspect(const json &input) {
    model_ptr model = load_model(input, false);
    const json context_input = required<json>(input, "context");
    if (!context_input.is_object()) {
        throw std::invalid_argument("context must be an object");
    }

    const uint32_t context_tokens = required<uint32_t>(context_input, "context_tokens");
    const uint32_t batch_tokens = required<uint32_t>(context_input, "batch_tokens");
    const uint32_t micro_batch_tokens = required<uint32_t>(context_input, "micro_batch_tokens");
    const uint32_t sequences = required<uint32_t>(context_input, "sequences");
    const int32_t threads = required<int32_t>(context_input, "threads");
    const int32_t batch_threads = required<int32_t>(context_input, "batch_threads");
    require_positive(context_tokens, "context.context_tokens");
    require_positive(batch_tokens, "context.batch_tokens");
    require_positive(micro_batch_tokens, "context.micro_batch_tokens");
    require_positive(sequences, "context.sequences");
    if (threads <= 0) {
        throw std::invalid_argument("context.threads must be greater than zero");
    }
    if (batch_threads <= 0) {
        throw std::invalid_argument("context.batch_threads must be greater than zero");
    }
    if (micro_batch_tokens > batch_tokens) {
        throw std::invalid_argument(
            "context.micro_batch_tokens must not exceed context.batch_tokens");
    }
    if (sequences > llama_max_parallel_sequences()) {
        throw std::invalid_argument("context.sequences exceeds llama.cpp's maximum");
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = context_tokens;
    params.n_batch = batch_tokens;
    params.n_ubatch = micro_batch_tokens;
    params.n_seq_max = sequences;
    params.n_threads = threads;
    params.n_threads_batch = batch_threads;
    params.embeddings = required<bool>(context_input, "embeddings");
    params.offload_kqv = required<bool>(context_input, "offload_kqv");
    context_ptr context = create_context(model.get(), params);

    return {
        {"effective",
         {
             {"contextTokens", llama_n_ctx(context.get())},
             {"contextTokensPerSequence", llama_n_ctx_seq(context.get())},
             {"batchTokens", llama_n_batch(context.get())},
             {"microBatchTokens", llama_n_ubatch(context.get())},
             {"sequences", llama_n_seq_max(context.get())},
             {"recurrentSequences", llama_n_rs_seq(context.get())},
             {"threads", llama_n_threads(context.get())},
             {"batchThreads", llama_n_threads_batch(context.get())},
         }},
    };
}

json decode_execute_plan(const json &input) {
    model_ptr model = load_model(input, false);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }

    const uint32_t context_tokens = required<uint32_t>(input, "context_tokens");
    require_positive(context_tokens, "context_tokens");
    const std::vector<json> batch_input = required<std::vector<json>>(input, "batch");
    if (batch_input.empty()) {
        throw std::invalid_argument("batch must not be empty");
    }
    const std::vector<int32_t> logit_indices = validated_logit_indices(input, vocab, {});

    std::vector<batch_entry> entries;
    entries.reserve(batch_input.size());
    json plan = json::array();
    llama_seq_id maximum_sequence_id = 0;
    for (size_t batch_index = 0; batch_index < batch_input.size(); ++batch_index) {
        const json &item = batch_input[batch_index];
        if (!item.is_object()) {
            throw std::invalid_argument("each batch entry must be an object");
        }
        const llama_token token = required<llama_token>(item, "token");
        const llama_pos position = required<llama_pos>(item, "position");
        const auto sequence_ids = required<std::vector<llama_seq_id>>(item, "sequence_ids");
        const bool request_logits = required<bool>(item, "request_logits");
        validate_token(vocab, token, "batch token");
        if (position < 0 || static_cast<uint32_t>(position) >= context_tokens) {
            throw std::invalid_argument("batch position must be within context_tokens");
        }
        if (sequence_ids.empty()) {
            throw std::invalid_argument("each batch entry must have at least one sequence ID");
        }
        for (size_t index = 0; index < sequence_ids.size(); ++index) {
            const llama_seq_id sequence_id = sequence_ids[index];
            if (sequence_id < 0) {
                throw std::invalid_argument("sequence IDs must be non-negative");
            }
            if (std::find(sequence_ids.begin(), sequence_ids.begin() + index, sequence_id) !=
                sequence_ids.begin() + index) {
                throw std::invalid_argument("sequence IDs within one batch entry must be unique");
            }
            maximum_sequence_id = std::max(maximum_sequence_id, sequence_id);
        }
        entries.push_back({token, position, sequence_ids, request_logits});
        plan.push_back({
            {"token", token},
            {"position", position},
            {"sequenceIds", sequence_ids},
            {"requestLogits", request_logits},
        });
    }
    const uint32_t sequences = static_cast<uint32_t>(maximum_sequence_id) + 1;
    if (sequences > llama_max_parallel_sequences()) {
        throw std::invalid_argument("batch sequence IDs exceed llama.cpp's maximum");
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = context_tokens;
    params.n_seq_max = sequences;
    configure_cpu_resident_context(params, input);
    context_ptr context = create_context(model.get(), params);
    native_batch batch(std::move(entries));
    const int32_t status_code = llama_decode(context.get(), batch.view());
    llama_synchronize(context.get());
    const json status = decode_status(status_code);

    json logits = json::array();
    for (size_t batch_index = 0; batch_index < batch_input.size(); ++batch_index) {
        if (!required<bool>(batch_input[batch_index], "request_logits")) {
            continue;
        }
        if (batch_index > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("batch logits index exceeds int32_t");
        }
        logits.push_back({
            {"batchIndex", batch_index},
            {"values",
             selected_logits(llama_get_logits_ith(context.get(), static_cast<int32_t>(batch_index)),
                             logit_indices)},
        });
    }
    return {
        {"plan", std::move(plan)},
        {"status", status},
        {"logits", std::move(logits)},
    };
}

json state_execute_script(const json &input) {
    model_ptr model = load_model(input, false);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }

    const std::vector<llama_token> prepare_tokens =
        required<std::vector<llama_token>>(input, "prepare_tokens");
    if (prepare_tokens.empty()) {
        throw std::invalid_argument("prepare_tokens must not be empty");
    }
    if (prepare_tokens.size() > static_cast<size_t>(std::numeric_limits<llama_pos>::max())) {
        throw std::invalid_argument("prepare_tokens is too large");
    }
    for (const llama_token token : prepare_tokens) {
        validate_token(vocab, token, "prepare token");
    }
    const llama_seq_id sequence_id = required<llama_seq_id>(input, "sequence_id");
    if (sequence_id < 0) {
        throw std::invalid_argument("sequence_id must be non-negative");
    }
    const uint32_t sequences = static_cast<uint32_t>(sequence_id) + 1;
    if (sequences > llama_max_parallel_sequences()) {
        throw std::invalid_argument("sequence_id exceeds llama.cpp's maximum");
    }
    const std::vector<json> script = required<std::vector<json>>(input, "operations");
    if (script.empty()) {
        throw std::invalid_argument("operations must not be empty");
    }
    const std::vector<int32_t> logit_indices = validated_logit_indices(input, vocab, {0, 1, 2, 3});

    uint32_t context_tokens = std::max<uint32_t>(32, static_cast<uint32_t>(prepare_tokens.size()));
    for (const json &operation : script) {
        if (!operation.is_object()) {
            throw std::invalid_argument("each state operation must be an object");
        }
        const std::string type = required<std::string>(operation, "type");
        if (type == "remove") {
            const llama_pos start = required<llama_pos>(operation, "position_start");
            const llama_pos end = required<llama_pos>(operation, "position_end");
            if (start < 0 || end <= start) {
                throw std::invalid_argument(
                    "remove requires a non-negative, non-empty position range");
            }
        } else if (type == "decode") {
            const llama_token token = required<llama_token>(operation, "token");
            const llama_pos position = required<llama_pos>(operation, "position");
            validate_token(vocab, token, "decode token");
            if (position < 0) {
                throw std::invalid_argument("decode position must be non-negative");
            }
            if (position == std::numeric_limits<llama_pos>::max()) {
                throw std::invalid_argument("decode position is too large");
            }
            context_tokens = std::max(context_tokens, static_cast<uint32_t>(position) + 1);
            (void)required<bool>(operation, "request_logits");
        } else {
            throw std::invalid_argument("unsupported state operation: " + type);
        }
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = context_tokens;
    params.n_seq_max = sequences;
    configure_cpu_resident_context(params, input);
    context_ptr context = create_context(model.get(), params);

    std::vector<batch_entry> prepare_entries;
    prepare_entries.reserve(prepare_tokens.size());
    for (size_t index = 0; index < prepare_tokens.size(); ++index) {
        prepare_entries.push_back({
            prepare_tokens[index],
            static_cast<llama_pos>(index),
            {sequence_id},
            index + 1 == prepare_tokens.size(),
        });
    }
    native_batch prepare_batch(std::move(prepare_entries));
    decode_status(llama_decode(context.get(), prepare_batch.view()));
    llama_synchronize(context.get());

    llama_memory_t memory = llama_get_memory(context.get());
    if (memory == nullptr) {
        throw std::runtime_error("context has no sequence memory");
    }
    json operation_results = json::array();
    json logits = json::array();
    for (size_t operation_index = 0; operation_index < script.size(); ++operation_index) {
        const json &operation = script[operation_index];
        const std::string type = required<std::string>(operation, "type");
        if (type == "remove") {
            llama_synchronize(context.get());
            const llama_pos start = required<llama_pos>(operation, "position_start");
            const llama_pos end = required<llama_pos>(operation, "position_end");
            const bool removed = llama_memory_seq_rm(memory, sequence_id, start, end);
            operation_results.push_back({
                {"index", operation_index},
                {"type", type},
                {"positionStart", start},
                {"positionEnd", end},
                {"success", removed},
            });
            continue;
        }

        const llama_token token = required<llama_token>(operation, "token");
        const llama_pos position = required<llama_pos>(operation, "position");
        const bool request_logits = required<bool>(operation, "request_logits");
        native_batch decode_batch({{token, position, {sequence_id}, request_logits}});
        const int32_t status_code = llama_decode(context.get(), decode_batch.view());
        llama_synchronize(context.get());
        const json status = decode_status(status_code);
        operation_results.push_back({
            {"index", operation_index},
            {"type", type},
            {"token", token},
            {"position", position},
            {"requestLogits", request_logits},
            {"status", status},
        });
        if (request_logits) {
            logits.push_back({
                {"operationIndex", operation_index},
                {"values", selected_logits(llama_get_logits_ith(context.get(), 0), logit_indices)},
            });
        }
    }

    llama_synchronize(context.get());
    return {
        {"operations", std::move(operation_results)},
        {"state",
         {
             {"sequenceId", sequence_id},
             {"positionMin", llama_memory_seq_pos_min(memory, sequence_id)},
             {"positionMax", llama_memory_seq_pos_max(memory, sequence_id)},
         }},
        {"logits", std::move(logits)},
    };
}

struct abort_probe_context {
    model_ptr model;
    context_ptr context;
    llama_token token;
    std::vector<int32_t> logit_indices;
};

abort_probe_context prepare_abort_probe(const json &input, bool require_logits) {
    const int32_t n_gpu_layers = required<int32_t>(input, "nGpuLayers");
    const bool use_mmap = required<bool>(input, "useMmap");
    const bool use_mlock = required<bool>(input, "useMlock");
    if (n_gpu_layers != 0 || !use_mmap || use_mlock) {
        throw std::invalid_argument(
            "CPU abort parity requires nGpuLayers=0, useMmap=true, and useMlock=false");
    }
    const uint32_t context_tokens = required<uint32_t>(input, "contextTokens");
    require_positive(context_tokens, "contextTokens");
    model_ptr model = load_model(input, false);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }
    const llama_token token = required<llama_token>(input, "token");
    validate_token(vocab, token, "token");
    std::vector<int32_t> logit_indices;
    if (require_logits) {
        logit_indices = required<std::vector<int32_t>>(input, "logitIndices");
        if (logit_indices.empty()) {
            throw std::invalid_argument("logitIndices must not be empty");
        }
        const int32_t token_count = llama_vocab_n_tokens(vocab);
        for (const int32_t index : logit_indices) {
            if (index < 0 || index >= token_count) {
                throw std::invalid_argument(
                    "logit index must be a valid model token in [0, " +
                    std::to_string(token_count) + ")");
            }
        }
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = context_tokens;
    params.n_seq_max = 1;
    configure_cpu_resident_context(params, input);
    context_ptr context = create_context(model.get(), params);
    return {std::move(model), std::move(context), token, std::move(logit_indices)};
}

bool abort_when_signaled(void *data) {
    return *static_cast<const bool *>(data);
}

void observe_presignaled_abort(llama_context *context, llama_token token) {
    bool signaled = true;
    llama_set_abort_callback(context, abort_when_signaled, &signaled);
    native_batch batch({{token, 0, {0}, true}});
    const int32_t status_code = llama_decode(context, batch.view());
    llama_synchronize(context);
    llama_set_abort_callback(context, nullptr, nullptr);
    if (status_code != 2) {
        throw std::runtime_error("pre-signaled llama_decode returned status " +
                                 std::to_string(status_code) + " instead of aborted status 2");
    }
}

json decode_abort(const json &input) {
    require_only_fields(input,
                        {"modelPath", "modelId", "model_id", "nGpuLayers", "useMmap",
                         "useMlock", "contextTokens", "threads", "batchThreads",
                         "offloadKqv", "operationOffload", "flashAttention", "token",
                         "fixturePaths", "engineConfiguration"});
    abort_probe_context probe = prepare_abort_probe(input, false);
    observe_presignaled_abort(probe.context.get(), probe.token);
    return {{"class", "aborted"}};
}

json decode_abort_recovery(const json &input) {
    require_only_fields(input,
                        {"modelPath", "modelId", "model_id", "nGpuLayers", "useMmap",
                         "useMlock", "contextTokens", "threads", "batchThreads",
                         "offloadKqv", "operationOffload", "flashAttention", "token",
                         "logitIndices", "fixturePaths", "engineConfiguration"});
    abort_probe_context probe = prepare_abort_probe(input, true);
    observe_presignaled_abort(probe.context.get(), probe.token);

    llama_memory_t memory = llama_get_memory(probe.context.get());
    if (memory == nullptr) {
        throw std::runtime_error("context has no sequence memory");
    }
    // This is the production recovery boundary: clear all logical memory state
    // without paying to zero-fill the backing storage.
    llama_memory_clear(memory, false);

    native_batch recovery_batch({{probe.token, 0, {0}, true}});
    const int32_t status_code = llama_decode(probe.context.get(), recovery_batch.view());
    llama_synchronize(probe.context.get());
    const json recovery_status = decode_status(status_code);
    return {
        {"abort", {{"class", "aborted"}}},
        {"recovery",
         {
             {"status", recovery_status},
             {"logits",
              selected_logits(llama_get_logits_ith(probe.context.get(), 0),
                              probe.logit_indices)},
         }},
    };
}

json model_metadata(const json &input) {
    model_ptr model = load_model(input, false);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }

    return {
        {"sizeBytes", llama_model_size(model.get())},
        {"parameterCount", llama_model_n_params(model.get())},
        {"dimensions",
         {
             {"contextTrain", llama_model_n_ctx_train(model.get())},
             {"embedding", llama_model_n_embd(model.get())},
             {"layers", llama_model_n_layer(model.get())},
             {"attentionHeads", llama_model_n_head(model.get())},
             {"kvAttentionHeads", llama_model_n_head_kv(model.get())},
             {"slidingWindow", llama_model_n_swa(model.get())},
         }},
        {"architecture",
         {
             {"ropeType", rope_type_name(llama_model_rope_type(model.get()))},
             {"recurrent", llama_model_is_recurrent(model.get())},
             {"hybrid", llama_model_is_hybrid(model.get())},
         }},
        {"vocabulary",
         {
             {"type", vocab_type_name(llama_vocab_type(vocab))},
             {"tokenCount", llama_vocab_n_tokens(vocab)},
             {"addBos", llama_vocab_get_add_bos(vocab)},
             {"specialTokens",
              {
                  {"bos", llama_vocab_bos(vocab)},
                  {"eos", llama_vocab_eos(vocab)},
                  {"newline", llama_vocab_nl(vocab)},
              }},
         }},
        {"metadataCount", llama_model_meta_count(model.get())},
    };
}

json tokenizer_tokenize(const json &input) {
    model_ptr model = load_model(input, true);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }
    const std::string text = required<std::string>(input, "text");
    const bool add_special = optional<bool>(input, "addSpecial", false);
    const bool parse_special = optional<bool>(input, "parseSpecial", false);
    const bool include_pieces = optional<bool>(input, "includePieces", true);
    const bool piece_special = optional<bool>(input, "pieceSpecial", true);
    const std::vector<llama_token> tokens = tokenize(vocab, text, add_special, parse_special);
    json evidence = {
        {"inputBytes", byte_array(text)},
        {"tokens", tokens},
        {"addSpecial", add_special},
        {"parseSpecial", parse_special},
    };
    if (include_pieces) {
        evidence["pieces"] = pieces_evidence(vocab, tokens, piece_special);
        evidence["pieceSpecial"] = piece_special;
    }
    return evidence;
}

json tokenizer_token_to_piece(const json &input) {
    require_only_fields(input, {"modelPath", "modelId", "model_id", "tokens", "special",
                                "lstrip", "fixturePaths", "engineConfiguration"});
    const bool special = required<bool>(input, "special");
    if (!special) {
        throw std::invalid_argument(
            "the production token-to-piece boundary requires special=true");
    }
    const json lstrip = required<json>(input, "lstrip");
    if (!lstrip.is_null()) {
        throw std::invalid_argument(
            "the production token-to-piece boundary requires lstrip=null");
    }
    model_ptr model = load_model(input, true);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }
    const std::vector<llama_token> tokens = required<std::vector<llama_token>>(input, "tokens");
    if (tokens.empty()) {
        throw std::invalid_argument("tokens must not be empty");
    }
    json pieces = json::array();
    json concatenated_bytes = json::array();
    for (const llama_token token : tokens) {
        validate_token(vocab, token, "token");
        const std::string piece = token_piece(vocab, token, true);
        const json bytes = byte_array(piece);
        pieces.push_back(bytes);
        for (const auto &byte : bytes) {
            concatenated_bytes.push_back(byte);
        }
    }
    return {
        {"pieces", std::move(pieces)},
        {"concatenatedBytes", std::move(concatenated_bytes)},
    };
}

llama_sampler *make_sampler(const json &config) {
    const std::string type = required<std::string>(config, "type");
    if (type == "top-k") {
        return llama_sampler_init_top_k(required<int32_t>(config, "k"));
    }
    if (type == "top-p") {
        return llama_sampler_init_top_p(required<float>(config, "p"),
                                        optional<size_t>(config, "minKeep", 1));
    }
    if (type == "min-p") {
        return llama_sampler_init_min_p(required<float>(config, "p"),
                                        optional<size_t>(config, "minKeep", 1));
    }
    if (type == "typical") {
        return llama_sampler_init_typical(required<float>(config, "p"),
                                          optional<size_t>(config, "minKeep", 1));
    }
    if (type == "temperature") {
        return llama_sampler_init_temp(required<float>(config, "temperature"));
    }
    if (type == "temperature-ext") {
        return llama_sampler_init_temp_ext(required<float>(config, "temperature"),
                                           required<float>(config, "delta"),
                                           required<float>(config, "exponent"));
    }
    if (type == "xtc") {
        return llama_sampler_init_xtc(
            required<float>(config, "probability"), required<float>(config, "threshold"),
            optional<size_t>(config, "minKeep", 1), optional<uint32_t>(config, "seed", 0));
    }
    if (type == "top-n-sigma") {
        return llama_sampler_init_top_n_sigma(required<float>(config, "n"));
    }
    if (type == "penalties") {
        return llama_sampler_init_penalties(
            required<int32_t>(config, "lastN"), required<float>(config, "repeat"),
            required<float>(config, "frequency"), required<float>(config, "presence"));
    }
    throw std::invalid_argument("unsupported sampler type: " + type);
}

json sampler_apply(const json &input) {
    const auto candidate_input = required<std::vector<json>>(input, "candidates");
    if (candidate_input.empty()) {
        throw std::invalid_argument("candidates must not be empty");
    }
    std::vector<llama_token_data> candidates;
    candidates.reserve(candidate_input.size());
    for (const auto &candidate : candidate_input) {
        candidates.push_back({
            required<llama_token>(candidate, "id"),
            required<float>(candidate, "logit"),
            optional<float>(candidate, "probability", 0.0F),
        });
    }
    llama_token_data_array data = {candidates.data(), candidates.size(), -1, false};

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = true;
    sampler_ptr chain(llama_sampler_chain_init(chain_params), llama_sampler_free);
    if (!chain) {
        throw std::runtime_error("llama_sampler_chain_init failed");
    }
    for (const auto &config : optional<std::vector<json>>(input, "samplers", {})) {
        llama_sampler *sampler = make_sampler(config);
        if (sampler == nullptr) {
            throw std::runtime_error("sampler constructor returned null");
        }
        llama_sampler_chain_add(chain.get(), sampler);
    }

    const json selection = input.value("selection", json{{"type", "distribution"}, {"seed", 0}});
    const std::string selection_type = required<std::string>(selection, "type");
    if (selection_type == "distribution") {
        llama_sampler_chain_add(chain.get(),
                                llama_sampler_init_dist(optional<uint32_t>(selection, "seed", 0)));
    } else if (selection_type == "greedy") {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    } else if (selection_type != "none") {
        throw std::invalid_argument("unsupported selection type: " + selection_type);
    }

    for (const llama_token token :
         optional<std::vector<llama_token>>(input, "acceptedTokens", {})) {
        llama_sampler_accept(chain.get(), token);
    }
    llama_sampler_apply(chain.get(), &data);

    json transformed = json::array();
    for (size_t index = 0; index < data.size; ++index) {
        const auto logit = finite_number(data.data[index].logit);
        const auto probability = finite_number(data.data[index].p);
        transformed.push_back({
            {"id", data.data[index].id},
            {"logit", logit.at("value")},
            {"logitClass", logit.at("class")},
            {"probability", probability.at("value")},
            {"probabilityClass", probability.at("class")},
        });
    }
    const bool has_selection = data.selected >= 0 && static_cast<size_t>(data.selected) < data.size;
    return {
        {"candidates", std::move(transformed)},
        {"sorted", data.sorted},
        {"selectedIndex", has_selection ? json(data.selected) : json(nullptr)},
        {"selectedToken", has_selection ? json(data.data[data.selected].id) : json(nullptr)},
        {"selection", selection},
    };
}

int64_t positive_elapsed_nanoseconds(const std::chrono::steady_clock::time_point &start,
                                     const std::chrono::steady_clock::time_point &end) {
    return std::max<int64_t>(
        1, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

json timed_probe_evidence(json output, int64_t duration_nanoseconds, json effective_configuration) {
    return {
        {"schemaVersion", PROTOCOL_VERSION},
        {"output", std::move(output)},
        {"measurements", json::array({
                             {
                                 {"name", "duration"},
                                 {"unit", "ns"},
                                 {"samples", json::array({duration_nanoseconds})},
                             },
                         })},
        {"effectiveConfiguration", std::move(effective_configuration)},
    };
}

// Shared measured-loop attestation for primitive microbenchmarks. Unsigned arithmetic is
// deliberately modulo 2^64: `next = (current XOR semantic_result) * 1099511628211`.
// Rust candidates must use `wrapping_mul` with the same seed and one fold per measured result.
constexpr uint64_t LOOP_SEMANTIC_FOLD_SEED = UINT64_C(1469598103934665603);
constexpr uint64_t LOOP_SEMANTIC_FOLD_PRIME = UINT64_C(1099511628211);

uint64_t fold_loop_semantic_result(uint64_t current, uint64_t semantic_result) {
    return (current ^ semantic_result) * LOOP_SEMANTIC_FOLD_PRIME;
}

std::vector<llama_token_data> generate_uniform_candidates(size_t count, uint32_t seed,
                                                          float minimum, float maximum) {
    std::vector<llama_token_data> result;
    result.reserve(count);
    uint32_t state = seed;
    for (size_t index = 0; index < count; ++index) {
        // Language-neutral Numerical Recipes LCG. uint32 overflow is modulo 2^32;
        // the upper 24 bits are scaled by exactly 2^-24 into [0, 1). A Rust
        // implementation is `state =
        // state.wrapping_mul(1664525).wrapping_add(1013904223)` followed by `(state
        // >> 8) as f32 * (1.0_f32 / 16_777_216.0_f32)`.
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        const float unit = static_cast<float>(state >> 8) * (1.0F / 16777216.0F);
        const float logit = minimum + (maximum - minimum) * unit;
        result.push_back({static_cast<llama_token>(index), logit, 0.0F});
    }
    return result;
}

json sampler_bench(const json &input) {
    const size_t candidate_count = required<size_t>(input, "candidate_count");
    if (candidate_count == 0 ||
        candidate_count > static_cast<size_t>(std::numeric_limits<llama_token>::max())) {
        throw std::invalid_argument("candidate_count must fit the positive llama_token range");
    }
    const json generator = required<json>(input, "candidate_generator");
    if (!generator.is_object() ||
        required<std::string>(generator, "kind") != "seeded-uniform-logits") {
        throw std::invalid_argument("candidate_generator.kind must be seeded-uniform-logits");
    }
    const uint32_t seed = required<uint32_t>(generator, "seed");
    const float minimum = required<float>(generator, "minimum");
    const float maximum = required<float>(generator, "maximum");
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum >= maximum) {
        throw std::invalid_argument("candidate generator bounds must be finite and increasing");
    }

    const json sampler_config = required<json>(input, "sampler");
    if (!sampler_config.is_object() || required<std::string>(sampler_config, "type") != "top-k") {
        throw std::invalid_argument("sampler.type must be top-k");
    }
    const int32_t top_k = required<int32_t>(sampler_config, "k");
    if (top_k <= 0 || static_cast<size_t>(top_k) > candidate_count) {
        throw std::invalid_argument(
            "sampler.k must be positive and no larger than candidate_count");
    }
    const uint32_t iterations = required<uint32_t>(input, "iterations");
    const uint32_t warmup_iterations = required<uint32_t>(input, "warmup_iterations");
    require_positive(iterations, "iterations");

    // Candidate generation, storage allocation, and sampler construction are
    // deliberately outside both warmup and measurement intervals. The interval
    // matches llama.cpp's own sampling microbenchmark: copy, apply, and reset
    // only.
    const std::vector<llama_token_data> source =
        generate_uniform_candidates(candidate_count, seed, minimum, maximum);
    std::vector<llama_token_data> working(candidate_count);
    sampler_ptr sampler(llama_sampler_init_top_k(top_k), llama_sampler_free);
    if (!sampler) {
        throw std::runtime_error("llama_sampler_init_top_k failed");
    }
    llama_token_data_array result = {working.data(), working.size(), -1, false};
    const auto apply_once = [&]() {
        std::copy(source.begin(), source.end(), working.begin());
        result = {working.data(), working.size(), -1, false};
        llama_sampler_apply(sampler.get(), &result);
        llama_sampler_reset(sampler.get());
    };
    uint32_t executed_warmup_iterations = 0;
    for (; executed_warmup_iterations < warmup_iterations;
         ++executed_warmup_iterations) {
        apply_once();
    }

    const auto semantic_result_checksum = [&]() {
        if (result.size > working.size()) {
            throw std::runtime_error("sampler returned an invalid candidate count");
        }
        uint64_t value = static_cast<uint64_t>(result.size);
        value = value * LOOP_SEMANTIC_FOLD_PRIME ^ static_cast<uint64_t>(result.sorted);
        const uint64_t selected = result.selected < 0
                                      ? std::numeric_limits<uint64_t>::max()
                                      : static_cast<uint64_t>(result.selected);
        value = value * LOOP_SEMANTIC_FOLD_PRIME ^ selected;
        for (size_t index = 0; index < result.size; ++index) {
            const uint64_t token =
                static_cast<uint64_t>(static_cast<uint32_t>(result.data[index].id));
            value = value * LOOP_SEMANTIC_FOLD_PRIME ^ token;
        }
        return value;
    };
    uint32_t executed_measurement_iterations = 0;
    uint64_t semantic_checksum = LOOP_SEMANTIC_FOLD_SEED;
    const auto start = std::chrono::steady_clock::now();
    for (; executed_measurement_iterations < iterations;
         ++executed_measurement_iterations) {
        apply_once();
        semantic_checksum =
            fold_loop_semantic_result(semantic_checksum, semantic_result_checksum());
    }
    const auto end = std::chrono::steady_clock::now();
    if (result.size > working.size()) {
        throw std::runtime_error("sampler returned an invalid candidate count");
    }

    json result_token_ids = json::array();
    for (size_t index = 0; index < result.size; ++index) {
        result_token_ids.push_back(result.data[index].id);
    }
    const bool has_selection =
        result.selected >= 0 && static_cast<size_t>(result.selected) < result.size;
    json output = {
        {"resultCandidateCount", result.size},
        {"sorted", result.sorted},
        {"resultTokenIds", std::move(result_token_ids)},
        {"executedWarmupIterations", executed_warmup_iterations},
        {"executedMeasurementIterations", executed_measurement_iterations},
        {"semanticChecksum", semantic_checksum},
    };
    if (has_selection) {
        throw std::runtime_error("top-k benchmark unexpectedly selected a candidate");
    }
    return timed_probe_evidence(std::move(output), positive_elapsed_nanoseconds(start, end),
                                json::object());
}

json nullable_token(llama_token token) {
    return token == LLAMA_TOKEN_NULL ? json(nullptr) : json(token);
}

json reasoning_budget_inspect(const json &input) {
    require_only_fields(input,
                        {"modelPath", "modelId", "model_id", "budgetTokens", "startTag",
                         "endTag", "forcedMessage", "controllable", "fixturePaths",
                         "engineConfiguration"});
    const int32_t budget_tokens = required<int32_t>(input, "budgetTokens");
    if (budget_tokens <= 0) {
        throw std::invalid_argument("budgetTokens must be greater than zero");
    }
    const bool controllable = required<bool>(input, "controllable");
    if (!controllable) {
        throw std::invalid_argument(
            "reasoning-budget.inspect requires controllable=true");
    }
    const std::string start_tag = required<std::string>(input, "startTag");
    const std::string end_tag = required<std::string>(input, "endTag");
    const std::string forced_message = required<std::string>(input, "forcedMessage");
    if (start_tag.empty() || end_tag.empty()) {
        throw std::invalid_argument("startTag and endTag must not be empty");
    }

    model_ptr model = load_model(input, true);
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("loaded model has no vocabulary");
    }
    const std::vector<llama_token> start_tokens = tokenize(vocab, start_tag, false, true);
    const std::vector<llama_token> end_tokens = tokenize(vocab, end_tag, false, true);
    const std::vector<llama_token> forced_tokens =
        tokenize(vocab, forced_message + end_tag, false, true);
    if (start_tokens.empty() || end_tokens.empty() || forced_tokens.empty()) {
        throw std::invalid_argument(
            "reasoning-budget tags must tokenize to non-empty model token sequences");
    }

    common_params_sampling params;
    params.reasoning_budget_tokens = budget_tokens;
    params.reasoning_budget_start = start_tokens;
    params.reasoning_budget_end = end_tokens;
    params.reasoning_budget_forced = forced_tokens;
    params.reasoning_budget_message = forced_message;
    params.reasoning_control = controllable;
    common_sampler_ptr sampler(common_sampler_init(model.get(), params));
    if (!sampler) {
        throw std::runtime_error("common_sampler_init failed");
    }

    const bool force_before_start = common_sampler_reasoning_budget_force(sampler.get());
    for (const llama_token token : start_tokens) {
        common_sampler_accept(sampler.get(), token, true);
    }
    const llama_token last_after_start = common_sampler_last(sampler.get());
    const bool force_after_start = common_sampler_reasoning_budget_force(sampler.get());
    const bool force_while_forcing = common_sampler_reasoning_budget_force(sampler.get());
    for (const llama_token token : forced_tokens) {
        common_sampler_accept(sampler.get(), token, true);
    }
    const llama_token last_after_forced_sequence = common_sampler_last(sampler.get());
    const bool force_after_completion = common_sampler_reasoning_budget_force(sampler.get());

    if (force_before_start || !force_after_start || force_while_forcing ||
        force_after_completion) {
        throw std::runtime_error(
            "public common_sampler reasoning-budget transitions violated the expected contract");
    }
    return {
        {"tokenized",
         {
             {"startTokens", start_tokens},
             {"endTokens", end_tokens},
             {"forcedTokens", forced_tokens},
         }},
        {"observations",
         {
             {"forceBeforeStart", force_before_start},
             {"lastTokenAfterStart", nullable_token(last_after_start)},
             {"forceAfterStart", force_after_start},
             {"forceWhileForcing", force_while_forcing},
             {"lastTokenAfterForcedSequence", nullable_token(last_after_forced_sequence)},
             {"forceAfterCompletion", force_after_completion},
         }},
    };
}

json chat_template_render(const json &input) {
    const std::string source = required<std::string>(input, "template");
    const std::string bos = optional<std::string>(input, "bosToken", "");
    const std::string eos = optional<std::string>(input, "eosToken", "");
    auto templates = common_chat_templates_init(nullptr, source, bos, eos);
    if (!templates) {
        throw std::runtime_error("common_chat_templates_init failed");
    }

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(required<json>(input, "messages"));
    inputs.grammar = optional<std::string>(input, "grammar", "");
    inputs.json_schema = input.contains("jsonSchema") ? input.at("jsonSchema").dump() : "";
    inputs.add_generation_prompt = optional<bool>(input, "addGenerationPrompt", true);
    inputs.continue_final_message =
        common_chat_continuation_parse(input.value("continueFinalMessage", json(false)));
    inputs.use_jinja = optional<bool>(input, "useJinja", true);
    inputs.tools = common_chat_tools_parse_oaicompat(input.value("tools", json::array()));
    inputs.tool_choice =
        common_chat_tool_choice_parse_oaicompat(optional<std::string>(input, "toolChoice", "auto"));
    inputs.parallel_tool_calls = optional<bool>(input, "parallelToolCalls", false);
    inputs.reasoning_format =
        common_reasoning_format_from_name(optional<std::string>(input, "reasoningFormat", "none"));
    inputs.enable_thinking = optional<bool>(input, "enableThinking", true);
    inputs.now = std::chrono::system_clock::time_point(
        std::chrono::seconds(optional<int64_t>(input, "nowUnixSeconds", 0)));
    inputs.add_bos = optional<bool>(input, "addBos", false);
    inputs.add_eos = optional<bool>(input, "addEos", false);
    inputs.force_pure_content = optional<bool>(input, "forcePureContent", false);
    if (input.contains("chatTemplateKwargs")) {
        inputs.chat_template_kwargs =
            input.at("chatTemplateKwargs").get<std::map<std::string, std::string>>();
    }

    const common_chat_params params = common_chat_templates_apply(templates.get(), inputs);
    json triggers = json::array();
    for (const auto &trigger : params.grammar_triggers) {
        triggers.push_back({
            {"type", trigger_type_name(trigger.type)},
            {"value", trigger.value},
            {"token", trigger.token},
        });
    }
    json spans = json::array();
    for (const auto &span : params.message_spans) {
        spans.push_back({{"role", span.role}, {"position", span.pos}, {"length", span.len}});
    }
    json caps = json::object();
    for (const auto &[name, enabled] : common_chat_templates_get_caps(templates.get())) {
        caps[name] = enabled;
    }
    return {
        {"source", common_chat_templates_source(templates.get())},
        {"explicitTemplate", common_chat_templates_was_explicit(templates.get())},
        {"capabilities", std::move(caps)},
        {"format", common_chat_format_name(params.format)},
        {"prompt", params.prompt},
        {"grammar", params.grammar},
        {"grammarLazy", params.grammar_lazy},
        {"generationPrompt", params.generation_prompt},
        {"supportsThinking", params.supports_thinking},
        {"thinkingStartTag", params.thinking_start_tag},
        {"thinkingEndTag", params.thinking_end_tag},
        {"grammarTriggers", std::move(triggers)},
        {"preservedTokens", params.preserved_tokens},
        {"additionalStops", params.additional_stops},
        {"parser", params.parser},
        {"messageSpans", std::move(spans)},
    };
}

std::string read_fixture_file(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open fixture: " + path);
    }
    std::string content{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        throw std::runtime_error("failed to read fixture: " + path);
    }
    return content;
}

std::string resolved_fixture_path(const json &input, const char *fixture_id) {
    const json paths = required<json>(input, "fixturePaths");
    if (!paths.is_object()) {
        throw std::invalid_argument("fixturePaths must be an object");
    }
    return required<std::string>(paths, fixture_id);
}

common_chat_templates_inputs chat_bench_inputs(const json &request) {
    if (!request.is_object()) {
        throw std::invalid_argument("chat input fixture must contain an object");
    }
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(required<json>(request, "messages"));
    inputs.grammar = optional<std::string>(request, "grammar", "");
    inputs.json_schema = request.contains("json_schema") && !request.at("json_schema").is_null()
                             ? request.at("json_schema").dump()
                             : "";
    inputs.add_generation_prompt = optional<bool>(request, "add_generation_prompt", true);
    inputs.continue_final_message =
        common_chat_continuation_parse(request.value("continue_final_message", json(false)));
    inputs.use_jinja = optional<bool>(request, "use_jinja", true);
    inputs.tools = common_chat_tools_parse_oaicompat(request.value("tools", json::array()));
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(
        optional<std::string>(request, "tool_choice", "auto"));
    inputs.parallel_tool_calls = optional<bool>(request, "parallel_tool_calls", false);
    inputs.reasoning_format = common_reasoning_format_from_name(
        optional<std::string>(request, "reasoning_format", "none"));
    inputs.enable_thinking = optional<bool>(request, "enable_thinking", true);
    if (request.contains("chat_template_kwargs")) {
        const json &kwargs = request.at("chat_template_kwargs");
        if (!kwargs.is_object()) {
            throw std::invalid_argument("chat_template_kwargs must be an object");
        }
        for (auto item = kwargs.begin(); item != kwargs.end(); ++item) {
            inputs.chat_template_kwargs[item.key()] = item.value().dump();
        }
    }
    inputs.force_pure_content = optional<bool>(request, "force_pure_content", false);
    return inputs;
}

struct chat_preparation_summary {
    common_chat_format format;
    size_t prompt_bytes;
    size_t grammar_bytes;
    size_t generation_prompt_bytes;
    size_t trigger_count;
    size_t preserved_token_count;
    size_t stop_count;
    size_t parser_bytes;
    size_t span_count;
};

chat_preparation_summary summarize_chat_preparation(const common_chat_params &params) {
    return {
        params.format,
        params.prompt.size(),
        params.grammar.size(),
        params.generation_prompt.size(),
        params.grammar_triggers.size(),
        params.preserved_tokens.size(),
        params.additional_stops.size(),
        params.parser.size(),
        params.message_spans.size(),
    };
}

uint64_t chat_summary_checksum(const chat_preparation_summary &summary) {
    uint64_t value = static_cast<uint64_t>(summary.format);
    for (const size_t component : {
             summary.prompt_bytes,
             summary.grammar_bytes,
             summary.generation_prompt_bytes,
             summary.trigger_count,
             summary.preserved_token_count,
             summary.stop_count,
             summary.parser_bytes,
             summary.span_count,
         }) {
        value = value * UINT64_C(1099511628211) ^ static_cast<uint64_t>(component);
    }
    return value;
}

json chat_preparation_semantic_evidence(const common_chat_params &params) {
    const chat_preparation_summary summary = summarize_chat_preparation(params);
    json triggers = json::array();
    for (const auto &trigger : params.grammar_triggers) {
        triggers.push_back({
            {"type", trigger_type_name(trigger.type)},
            {"value", trigger.value},
            {"token", trigger.token},
        });
    }
    json spans = json::array();
    for (const auto &span : params.message_spans) {
        spans.push_back({{"role", span.role}, {"position", span.pos}, {"length", span.len}});
    }
    return {
        {"format", common_chat_format_name(params.format)},
        {"prompt", params.prompt},
        {"grammar", params.grammar},
        {"generationPrompt", params.generation_prompt},
        {"parser", params.parser},
        {"grammarTriggers", std::move(triggers)},
        {"preservedTokens", params.preserved_tokens},
        {"additionalStops", params.additional_stops},
        {"messageSpans", std::move(spans)},
        {"promptBytes", summary.prompt_bytes},
        {"grammarBytes", summary.grammar_bytes},
        {"generationPromptBytes", summary.generation_prompt_bytes},
        {"triggerCount", summary.trigger_count},
        {"preservedTokenCount", summary.preserved_token_count},
        {"stopCount", summary.stop_count},
        {"parserBytes", summary.parser_bytes},
        {"spanCount", summary.span_count},
    };
}

json chat_template_bench(const json &input) {
    // Logical fixture names remain in the runner-derived work contract; only the
    // resolved, digest-verified paths are used for I/O and never escape in
    // operation evidence.
    (void)required<std::string>(input, "template_fixture");
    (void)required<std::string>(input, "input_fixture");
    const uint32_t warmup_iterations = required<uint32_t>(input, "warmup_iterations");
    const uint32_t iterations = required<uint32_t>(input, "iterations");
    require_positive(iterations, "iterations");

    const std::string template_source =
        read_fixture_file(resolved_fixture_path(input, "chatml-basic-template"));
    const json request =
        json::parse(read_fixture_file(resolved_fixture_path(input, "chatml-basic-input")));
    if (!request.is_object()) {
        throw std::invalid_argument("chat input fixture must contain an object");
    }
    const std::string bos = optional<std::string>(request, "bos_token", "");
    const std::string eos = optional<std::string>(request, "eos_token", "");
    auto templates = common_chat_templates_init(nullptr, template_source, bos, eos);
    if (!templates) {
        throw std::runtime_error("common_chat_templates_init failed");
    }

    // Parsing the files and constructing the reusable template handle are
    // outside. Each call converts the neutral request, performs native
    // preparation, and reads a bounded shared semantic summary. JSON value
    // construction and JSONL serialization remain outside.
    const auto prepare_once = [&]() {
        common_chat_templates_inputs inputs = chat_bench_inputs(request);
        const common_chat_params params = common_chat_templates_apply(templates.get(), inputs);
        return summarize_chat_preparation(params);
    };
    volatile uint64_t warmup_semantic_sink = LOOP_SEMANTIC_FOLD_SEED;
    uint32_t executed_warmup_iterations = 0;
    for (; executed_warmup_iterations < warmup_iterations;
         ++executed_warmup_iterations) {
        const chat_preparation_summary summary = prepare_once();
        warmup_semantic_sink = fold_loop_semantic_result(
            warmup_semantic_sink, chat_summary_checksum(summary));
    }

    chat_preparation_summary summary = {};
    uint32_t executed_measurement_iterations = 0;
    uint64_t semantic_checksum = LOOP_SEMANTIC_FOLD_SEED;
    const auto start = std::chrono::steady_clock::now();
    for (; executed_measurement_iterations < iterations;
         ++executed_measurement_iterations) {
        summary = prepare_once();
        semantic_checksum =
            fold_loop_semantic_result(semantic_checksum, chat_summary_checksum(summary));
    }
    const auto end = std::chrono::steady_clock::now();
    (void)warmup_semantic_sink;
    // Build the complete stable semantic projection after timing. This intentionally performs
    // one excluded preparation so full string/vector copying and JSON construction cannot leak
    // into the measured primitive while the evidence proves both sides produced the same result.
    common_chat_templates_inputs evidence_inputs = chat_bench_inputs(request);
    const common_chat_params evidence_params =
        common_chat_templates_apply(templates.get(), evidence_inputs);
    json output = chat_preparation_semantic_evidence(evidence_params);
    output["executedWarmupIterations"] = executed_warmup_iterations;
    output["executedMeasurementIterations"] = executed_measurement_iterations;
    output["semanticChecksum"] = semantic_checksum;
    return timed_probe_evidence(std::move(output),
                                positive_elapsed_nanoseconds(start, end), json::object());
}

json nullable_nonempty_string(const std::string &value) {
    return value.empty() ? json(nullptr) : json(value);
}

json parsed_chat_message_evidence(const common_chat_msg &message) {
    json tool_calls = json::array();
    for (const auto &call : message.tool_calls) {
        tool_calls.push_back({
            {"name", call.name},
            {"arguments", call.arguments},
            {"id", nullable_nonempty_string(call.id)},
        });
    }
    return {
        {"role", message.role},
        {"content", message.content},
        {"toolCalls", std::move(tool_calls)},
        {"reasoningContent", nullable_nonempty_string(message.reasoning_content)},
        {"toolName", nullable_nonempty_string(message.tool_name)},
        {"toolCallId", nullable_nonempty_string(message.tool_call_id)},
    };
}

struct parser_fixture_data {
    std::string template_source;
    json request;
    std::string text;
    std::vector<std::vector<std::string>> chunk_partitions;
};

parser_fixture_data load_parser_fixture_data(const json &input) {
    // Logical fixture names are retained in the runner-derived work contract.
    // Native I/O uses only the digest-verified resolved paths supplied by the
    // runner.
    (void)required<std::string>(input, "template_fixture");
    (void)required<std::string>(input, "input_fixture");
    (void)required<std::string>(input, "content_fixture");
    parser_fixture_data fixtures{
        read_fixture_file(resolved_fixture_path(input, "chatml-basic-template")),
        json::parse(read_fixture_file(resolved_fixture_path(input, "chatml-basic-input"))),
        {},
        {},
    };
    const json content =
        json::parse(read_fixture_file(resolved_fixture_path(input, "content-replay")));
    if (!fixtures.request.is_object() || !content.is_object()) {
        throw std::invalid_argument("chat and parser fixtures must each contain an object");
    }
    fixtures.text = required<std::string>(content, "input");
    fixtures.chunk_partitions =
        required<std::vector<std::vector<std::string>>>(content, "chunkings");
    if (fixtures.chunk_partitions.empty()) {
        throw std::invalid_argument("content fixture must declare at least one chunk partition");
    }
    for (const auto &partition : fixtures.chunk_partitions) {
        if (partition.empty()) {
            throw std::invalid_argument("chunk partitions must not be empty");
        }
        std::string cumulative;
        for (const auto &chunk : partition) {
            cumulative += chunk;
        }
        if (cumulative != fixtures.text) {
            throw std::invalid_argument("each chunk partition must concatenate exactly to text");
        }
    }
    return fixtures;
}

struct prepared_parser_boundary {
    common_chat_params prepared;
    common_reasoning_format reasoning_format;
};

prepared_parser_boundary prepare_content_only_parser(const json &input,
                                                     const parser_fixture_data &fixtures) {
    const bool use_jinja = required<bool>(input, "use_jinja");
    const bool force_pure_content = required<bool>(input, "force_pure_content");
    if (use_jinja || !force_pure_content) {
        throw std::invalid_argument("content-only parser cases require "
                                    "use_jinja=false and force_pure_content=true");
    }
    const std::string bos = optional<std::string>(fixtures.request, "bos_token", "");
    const std::string eos = optional<std::string>(fixtures.request, "eos_token", "");
    auto templates = common_chat_templates_init(nullptr, fixtures.template_source, bos, eos);
    if (!templates) {
        throw std::runtime_error("common_chat_templates_init failed");
    }
    common_chat_templates_inputs prepare_inputs = chat_bench_inputs(fixtures.request);
    prepare_inputs.use_jinja = use_jinja;
    prepare_inputs.force_pure_content = force_pure_content;
    const common_reasoning_format reasoning_format = prepare_inputs.reasoning_format;
    common_chat_params prepared = common_chat_templates_apply(templates.get(), prepare_inputs);
    if (prepared.format != COMMON_CHAT_FORMAT_CONTENT_ONLY || !prepared.parser.empty()) {
        throw std::runtime_error("pinned legacy ChatML preparation did not select "
                                 "the content-only parser boundary");
    }
    return {std::move(prepared), reasoning_format};
}

common_chat_parser_params configured_parser(const prepared_parser_boundary &boundary,
                                            const json &input) {
    const json options = required<json>(input, "parser_options");
    if (!options.is_object()) {
        throw std::invalid_argument("parser_options must be an object");
    }
    common_chat_parser_params parser(boundary.prepared);
    parser.reasoning_format = boundary.reasoning_format;
    parser.reasoning_in_content = required<bool>(options, "reasoning_in_content");
    parser.parse_tool_calls = required<bool>(options, "parse_tool_calls");
    parser.is_continuation = required<bool>(options, "is_continuation");
    parser.echo = required<bool>(options, "echo");
    parser.debug = required<bool>(options, "debug");
    if (!boundary.prepared.parser.empty()) {
        parser.parser.load(boundary.prepared.parser);
    }
    return parser;
}

json semantic_delta_evidence(const common_chat_msg &previous, const common_chat_msg &current) {
    const auto native_diffs = common_chat_msg_diff::compute_diffs(previous, current);
    json result = json::array();
    size_t diff_index = 0;
    if (previous.reasoning_content != current.reasoning_content) {
        if (diff_index >= native_diffs.size() ||
            native_diffs[diff_index].tool_call_index != std::string::npos) {
            throw std::runtime_error("unexpected native reasoning diff projection");
        }
        result.push_back({
            {"kind", "reasoning"},
            {"text", native_diffs[diff_index++].reasoning_content_delta},
        });
    }
    if (previous.content != current.content) {
        if (diff_index >= native_diffs.size() ||
            native_diffs[diff_index].tool_call_index != std::string::npos) {
            throw std::runtime_error("unexpected native content diff projection");
        }
        result.push_back({
            {"kind", "content"},
            {"text", native_diffs[diff_index++].content_delta},
        });
    }
    for (; diff_index < native_diffs.size(); ++diff_index) {
        const auto &native = native_diffs[diff_index];
        if (native.tool_call_index == std::string::npos) {
            throw std::runtime_error("unexpected native chat diff without a semantic field");
        }
        bool has_id = false;
        bool has_name = false;
        std::string id;
        std::string name;
        if (native.tool_call_index < previous.tool_calls.size()) {
            if (native.tool_call_index >= current.tool_calls.size()) {
                throw std::runtime_error("native tool-call diff index is out of range");
            }
            const auto &old_call = previous.tool_calls[native.tool_call_index];
            const auto &new_call = current.tool_calls[native.tool_call_index];
            has_id = old_call.id != new_call.id;
            has_name = old_call.name != new_call.name;
            id = new_call.id;
            name = new_call.name;
        } else {
            has_id = !native.tool_call_delta.id.empty();
            has_name = true;
            id = native.tool_call_delta.id;
            name = native.tool_call_delta.name;
        }
        result.push_back({
            {"kind", "tool-call"},
            {"index", native.tool_call_index},
            {"id", has_id ? json(id) : json(nullptr)},
            {"name", has_name ? json(name) : json(nullptr)},
            {"arguments", native.tool_call_delta.arguments},
        });
    }
    return result;
}

json prepared_content_only_evidence(const json &input) {
    return {
        {"format", "content-only"},
        {"useJinja", required<bool>(input, "use_jinja")},
        {"forcePureContent", required<bool>(input, "force_pure_content")},
    };
}

json chat_parser_inspect(const json &input) {
    const parser_fixture_data fixtures = load_parser_fixture_data(input);
    const prepared_parser_boundary boundary = prepare_content_only_parser(input, fixtures);
    const common_chat_parser_params snapshot_parser = configured_parser(boundary, input);
    const common_chat_msg snapshot = common_chat_parse(fixtures.text, false, snapshot_parser);
    const json snapshot_evidence = parsed_chat_message_evidence(snapshot);

    json partitions = json::array();
    bool all_finals_match = true;
    for (size_t partition_index = 0; partition_index < fixtures.chunk_partitions.size();
         ++partition_index) {
        const auto &partition = fixtures.chunk_partitions[partition_index];
        const common_chat_parser_params stream_parser = configured_parser(boundary, input);
        common_chat_msg previous;
        std::string cumulative;
        json pushes = json::array();
        for (size_t chunk_index = 0; chunk_index < partition.size(); ++chunk_index) {
            const std::string &chunk = partition[chunk_index];
            cumulative += chunk;
            const common_chat_msg parsed = common_chat_parse(cumulative, true, stream_parser);
            json deltas = json::array();
            if (!parsed.empty()) {
                deltas = semantic_delta_evidence(previous, parsed);
                previous = parsed;
            }
            pushes.push_back({
                {"index", chunk_index},
                {"chunkByteLength", chunk.size()},
                {"deltas", std::move(deltas)},
            });
        }
        const common_chat_msg finalized = common_chat_parse(cumulative, false, stream_parser);
        json finish_deltas = json::array();
        if (!finalized.empty()) {
            finish_deltas = semantic_delta_evidence(previous, finalized);
            previous = finalized;
        }
        const json final_evidence = parsed_chat_message_evidence(previous);
        const bool matches_snapshot = final_evidence == snapshot_evidence;
        all_finals_match = all_finals_match && matches_snapshot;
        partitions.push_back({
            {"index", partition_index},
            {"pushes", std::move(pushes)},
            {"finishDeltas", std::move(finish_deltas)},
            {"final", final_evidence},
            {"matchesSnapshot", matches_snapshot},
        });
    }
    return {
        {"prepared", prepared_content_only_evidence(input)},
        {"snapshot", snapshot_evidence},
        {"partitions", std::move(partitions)},
        {"allFinalsMatch", all_finals_match},
    };
}

uint64_t semantic_message_checksum(const common_chat_msg &message) {
    uint64_t result = UINT64_C(1469598103934665603);
    const auto include = [&result](const std::string &value) {
        for (const unsigned char byte : value) {
            result = (result ^ byte) * UINT64_C(1099511628211);
        }
        result = (result ^ UINT64_C(255)) * UINT64_C(1099511628211);
    };
    include(message.role);
    include(message.content);
    include(message.reasoning_content);
    include(message.tool_name);
    include(message.tool_call_id);
    for (const auto &call : message.tool_calls) {
        include(call.id);
        include(call.name);
        include(call.arguments);
    }
    return result ^ static_cast<uint64_t>(message.tool_calls.size());
}

json chat_parser_bench(const json &input) {
    const uint32_t warmup_iterations = required<uint32_t>(input, "warmup_iterations");
    const uint32_t iterations = required<uint32_t>(input, "iterations");
    require_positive(iterations, "iterations");

    // Fixture I/O, template construction, chat preparation, and reusable parser
    // construction are outside both warmup and measurement. The measured
    // operation is final parsing plus a complete semantic-field projection; JSON
    // construction and JSONL serialization are outside.
    const parser_fixture_data fixtures = load_parser_fixture_data(input);
    const prepared_parser_boundary boundary = prepare_content_only_parser(input, fixtures);
    const common_chat_parser_params parser = configured_parser(boundary, input);
    volatile uint64_t semantic_sink = 0;
    for (uint32_t warmup = 0; warmup < warmup_iterations; ++warmup) {
        const common_chat_msg parsed = common_chat_parse(fixtures.text, false, parser);
        semantic_sink = semantic_sink ^ semantic_message_checksum(parsed);
    }

    common_chat_msg parsed;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        parsed = common_chat_parse(fixtures.text, false, parser);
        semantic_sink = semantic_sink ^ semantic_message_checksum(parsed);
    }
    const auto end = std::chrono::steady_clock::now();
    (void)semantic_sink;
    return timed_probe_evidence(
        {
            {"prepared", prepared_content_only_evidence(input)},
            {"inputByteLength", fixtures.text.size()},
            {"message", parsed_chat_message_evidence(parsed)},
        },
        positive_elapsed_nanoseconds(start, end), json::object());
}

json schema_to_grammar(const json &input) {
    const bool force_gbnf = optional<bool>(input, "forceGbnf", false);
    return {
        {"grammar", json_schema_to_grammar(required<json>(input, "schema"), force_gbnf)},
        {"forceGbnf", force_gbnf},
    };
}

json execute(const std::string &operation, const json &input) {
    if (operation == "protocol.describe") {
        return describe_protocol();
    }
    if (operation == "configuration.inspect") {
        return configuration_inspect(input);
    }
    if (operation == "decode.execute-plan") {
        return decode_execute_plan(input);
    }
    if (operation == "model.metadata") {
        return model_metadata(input);
    }
    if (operation == "tokenizer.tokenize") {
        return tokenizer_tokenize(input);
    }
    if (operation == "tokenizer.token-to-piece") {
        return tokenizer_token_to_piece(input);
    }
    if (operation == "sampler.apply") {
        return sampler_apply(input);
    }
    if (operation == "sampler.bench") {
        return sampler_bench(input);
    }
    if (operation == "state.execute-script") {
        return state_execute_script(input);
    }
    if (operation == "reasoning-budget.inspect") {
        return reasoning_budget_inspect(input);
    }
    if (operation == "decode.abort") {
        return decode_abort(input);
    }
    if (operation == "decode.abort-recovery") {
        return decode_abort_recovery(input);
    }
    if (operation == "chat-template.render") {
        return chat_template_render(input);
    }
    if (operation == "chat-template.bench") {
        return chat_template_bench(input);
    }
    if (operation == "chat-parser.inspect") {
        return chat_parser_inspect(input);
    }
    if (operation == "chat-parser.bench") {
        return chat_parser_bench(input);
    }
    if (operation == "grammar.json-schema-to-grammar") {
        return schema_to_grammar(input);
    }
    throw std::invalid_argument("unsupported operation: " + operation);
}

json success(const std::string &case_id, const std::string &operation, json evidence) {
    return {
        {"schemaVersion", PROTOCOL_VERSION},
        {"caseId", case_id},
        {"operation", operation},
        {"status", "ok"},
        {"evidence", std::move(evidence)},
    };
}

json failure(const std::string &case_id, const std::string &operation,
             const std::string &class_name, const std::string &code, const std::string &message) {
    return {
        {"schemaVersion", PROTOCOL_VERSION},
        {"caseId", case_id},
        {"operation", operation},
        {"status", "error"},
        {"error", {{"class", class_name}, {"code", code}, {"message", message}}},
    };
}

} // namespace

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.size() > MAX_INPUT_LINE_BYTES) {
            std::cout << failure("", "", "invalid-input", "input-too-large",
                                 "input line exceeds 16777216 bytes")
                             .dump()
                      << '\n';
            std::cout.flush();
            continue;
        }
        std::string case_id;
        std::string operation;
        try {
            const json request = json::parse(line);
            if (required<int>(request, "schemaVersion") != PROTOCOL_VERSION) {
                throw std::invalid_argument("unsupported schemaVersion");
            }
            case_id = required<std::string>(request, "caseId");
            operation = required<std::string>(request, "operation");
            const json input = request.value("input", json::object());
            std::cout << success(case_id, operation, execute(operation, input)).dump() << '\n';
        } catch (const std::invalid_argument &error) {
            std::cout
                << failure(case_id, operation, "invalid-input", "invalid-case", error.what()).dump()
                << '\n';
        } catch (const json::exception &error) {
            std::cout
                << failure(case_id, operation, "invalid-input", "invalid-case", error.what()).dump()
                << '\n';
        } catch (const std::exception &error) {
            std::cout << failure(case_id, operation, "runtime-error", "native-operation",
                                 error.what())
                             .dump()
                      << '\n';
        }
        std::cout.flush();
    }
    if (backend_initialized) {
        llama_backend_free();
    }
    return 0;
}
