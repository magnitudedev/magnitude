#include "templates.h"
#include "chat.h"
#include "jinja/runtime.h"
#include "json-schema-to-grammar.h"
#include "templates-log.h"
#include "output-stream.h"

#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>

namespace {
using json = common_json;
struct abi_error : std::runtime_error {
    int32_t status;
    abi_error(int32_t status, const char * message) : std::runtime_error(message), status(status) {}
};
struct template_state {
    common_chat_templates_ptr compiled;
    json special_tokens;
    bool reads_preserve_thinking;
};
// A schema keyword the prepared constraint does not enforce, for a tool's
// arguments (named) or for the JSON output (unnamed).
struct subject_relaxation {
    std::optional<std::string>    tool;
    common_chat_schema_relaxation relaxation;

    bool operator<(const subject_relaxation & other) const {
        return std::tie(tool, relaxation) < std::tie(other.tool, other.relaxation);
    }
};
struct prepared_state {
    common_chat_params parameters;
    std::vector<std::string> diagnostics;
    std::set<subject_relaxation> relaxations;
    uint64_t renders;
};
std::mutex mutex;
uint64_t next_handle = 1;
std::unordered_map<uint64_t, std::shared_ptr<const template_state>> templates;
std::unordered_map<uint64_t, std::shared_ptr<const prepared_state>> requests;
std::unordered_map<uint64_t, std::shared_ptr<templates_native::output_stream>> streams;
std::unordered_map<uint64_t, std::unique_ptr<std::string>> buffers;

uint64_t allocate_handle() {
    if (next_handle == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("Native handle space exhausted");
    }
    return next_handle++;
}

templates_buffer owned(std::string value) {
    auto storage = std::make_unique<std::string>(std::move(value));
    templates_buffer result{reinterpret_cast<const uint8_t *>(storage->data()), storage->size(), 0};
    std::lock_guard<std::mutex> lock(mutex);
    auto id = allocate_handle();
    result.owner = id;
    buffers.emplace(id, std::move(storage));
    return result;
}

template<typename Map>
uint64_t insert_handle(Map & table, typename Map::mapped_type value) {
    std::lock_guard<std::mutex> lock(mutex);
    auto id = allocate_handle();
    table.emplace(id, std::move(value));
    return id;
}

template<typename Map>
void erase_handle(Map & table, uint64_t handle, const char * message) {
    typename Map::mapped_type removed;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = table.find(handle);
        if (it == table.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, message); }
        removed = std::move(it->second);
        table.erase(it);
    }
}

template<typename F>
int32_t guarded(templates_buffer * error, F && function) noexcept {
    if (!error) { return TEMPLATES_INVALID_ARGUMENT; }
    *error = {};
    try {
        templates_native::diagnostics.clear();
        templates_native::relaxations.clear();
        templates_native::renders = 0;
        try {
            function();
            return TEMPLATES_OK;
        } catch (const abi_error & e) {
            *error = owned(e.what());
            return e.status;
        } catch (const std::invalid_argument & e) {
            *error = owned(e.what());
            return TEMPLATES_INVALID_ARGUMENT;
        } catch (const std::bad_alloc &) {
            throw;
        } catch (const std::exception & e) {
            *error = owned(e.what());
            return TEMPLATES_NATIVE_ERROR;
        }
    } catch (const std::bad_alloc &) {
        static const std::string_view message = "Native templates allocation failed";
        *error = {reinterpret_cast<const uint8_t *>(message.data()), message.size(), 0};
        return TEMPLATES_OUT_OF_MEMORY;
    } catch (...) {
        static const std::string_view message = "Unknown native templates failure";
        *error = {reinterpret_cast<const uint8_t *>(message.data()), message.size(), 0};
        return TEMPLATES_NATIVE_ERROR;
    }
}

json payload(const uint8_t * input, uint64_t size) {
    constexpr uint64_t maximum_json_bytes = 64 * 1024 * 1024;
    if (!input || size == 0 || size > maximum_json_bytes) {
        throw std::invalid_argument("Invalid JSON input span (limit: 64 MiB)");
    }
    auto data = json::parse(std::string(reinterpret_cast<const char *>(input), static_cast<size_t>(size)));
    if (!data.is_object() || !data.contains("version") || !data.at("version").is_number_integer()
        || data.at("version").get<int64_t>() != 1) {
        throw abi_error(TEMPLATES_VERSION_MISMATCH, "Expected payload version 1");
    }
    return data;
}

std::shared_ptr<const template_state> get_template(uint64_t handle) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = templates.find(handle);
    if (it == templates.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released template handle"); }
    return it->second;
}

void require_output(const void * output) {
    if (!output) { throw std::invalid_argument("Null output pointer"); }
}

std::shared_ptr<const prepared_state> get_request(uint64_t handle) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = requests.find(handle);
    if (it == requests.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released prepared request"); }
    return it->second;
}

std::shared_ptr<templates_native::output_stream> get_stream(uint64_t handle) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = streams.find(handle);
    if (it == streams.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released stream"); }
    return it->second;
}

void append_json_string(std::string & out, std::string_view s) {
    static constexpr char hex[] = "0123456789abcdef";
    out.push_back('"');
    const char * data = s.data();
    const size_t len = s.size();
    size_t start = 0;
    for (size_t i = 0; i < len; ++i) {
        const auto c = static_cast<unsigned char>(data[i]);
        if (c >= 0x20 && c != '"' && c != '\\') {
            continue;
        }
        if (i > start) {
            out.append(data + start, i - start);
        }
        switch (c) {
            case '"':  out.append("\\\"", 2); break;
            case '\\': out.append("\\\\", 2); break;
            case '\b': out.append("\\b", 2);  break;
            case '\f': out.append("\\f", 2);  break;
            case '\n': out.append("\\n", 2);  break;
            case '\r': out.append("\\r", 2);  break;
            case '\t': out.append("\\t", 2);  break;
            default: {
                const char escaped[6] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 0x0f]};
                out.append(escaped, 6);
                break;
            }
        }
        start = i + 1;
    }
    if (len > start) {
        out.append(data + start, len - start);
    }
    out.push_back('"');
}

void append_json_string_array(std::string & out, const std::vector<std::string> & values) {
    out.push_back('[');
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        append_json_string(out, values[i]);
    }
    out.push_back(']');
}

void append_json_hash128(std::string & out, std::string_view s) {
    // Two independent 64-bit FNV-1a/SplitMix64 streams over bytes + length,
    // finalized with SplitMix64 avalanche (128 bits = 32 hex chars).
    uint64_t h1 = 0xcbf29ce484222325ULL;
    uint64_t h2 = 0x84222325cbf29ce4ULL;
    const auto * data = reinterpret_cast<const unsigned char *>(s.data());
    const size_t len = s.size();
    for (size_t i = 0; i < len; ++i) {
        const uint64_t b = data[i];
        h1 = (h1 ^ b) * 0x100000001b3ULL;
        h2 = (h2 ^ (b + 0x9e3779b97f4a7c15ULL)) * 0x9e3779b185ebca87ULL;
    }
    h1 ^= static_cast<uint64_t>(len);
    h2 ^= static_cast<uint64_t>(len) * 0xbf58476d1ce4e5b9ULL;
    auto finalize = [](uint64_t x) -> uint64_t {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    };
    h1 = finalize(h1);
    h2 = finalize(h2);

    static constexpr char hex[] = "0123456789abcdef";
    char buf[34];
    buf[0] = '"';
    for (int i = 0; i < 16; ++i) {
        buf[1 + i] = hex[(h1 >> (60 - i * 4)) & 0x0f];
        buf[17 + i] = hex[(h2 >> (60 - i * 4)) & 0x0f];
    }
    buf[33] = '"';
    out.append(buf, sizeof(buf));
}

prepared_state apply_request_payload(const template_state & state, const uint8_t * input, uint64_t size) {
    auto data = payload(input, size);
    common_chat_templates_inputs request;
    request.messages = common_chat_msgs_parse_oaicompat(data.at("messages"));
    if (request.messages.empty()) { throw std::invalid_argument("Request requires messages"); }
    request.tool_choice = common_chat_tool_choice_parse_oaicompat(data.value("tool_choice", std::string("auto")));
    // Thinking is enabled unless the caller turns it off, as in upstream
    // common chat, so a template's own `enable_thinking` fallback never
    // decides the default.
    request.enable_thinking = true;
    // Schemas are valid JSON Schemas (the host admits only those) and
    // always lower; lowering records what the constraint does not enforce.
    std::set<subject_relaxation> relaxations;
    auto lower = [&](const std::optional<std::string> & tool, const json & schema) {
        for (auto & relaxation : common_chat_schema_from_json(schema).relaxations) {
            relaxations.insert({ tool, std::move(relaxation) });
        }
    };
    if (request.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && data.contains("tools")) {
        for (const auto & tool : data.at("tools")) {
            const auto & function = tool.at("function");
            lower(function.at("name").get<std::string>(), common_chat_tool_parameters(function));
        }
        request.tools = common_chat_tools_parse_oaicompat(data.at("tools"));
    }
    request.parallel_tool_calls = data.value("parallel_tool_calls", true);
    request.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    auto now = data.at("now").get<int64_t>();
    const auto max_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::duration::max()).count();
    const auto min_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::duration::min()).count();
    if (now < min_seconds || now > max_seconds) { throw std::invalid_argument("Timestamp outside native clock range"); }
    request.now = std::chrono::system_clock::from_time_t(static_cast<time_t>(now));
    if (data.contains("template_arguments")) {
        const auto & arguments = data.at("template_arguments");
        if (!arguments.is_object()) { throw std::invalid_argument("template_arguments must be an object"); }
        const std::vector<std::string> reserved = {"messages", "tools", "bos_token", "eos_token",
            "add_generation_prompt", "tokenize", "chat_template", "now", "date_string", "datetime"};
        for (const auto & [key, value] : arguments.items()) {
            if (std::find(reserved.begin(), reserved.end(), key) != reserved.end() || state.special_tokens.contains(key)) {
                throw std::invalid_argument("Reserved template argument: " + key);
            }
            request.chat_template_kwargs[key] = value.dump();
            if (key == "enable_thinking") { request.enable_thinking = value.get<bool>(); }
        }
    }
    // Reasoning stays on every assistant message unless the caller turns it
    // off, so a new user message appends to the previous prompt instead of
    // rewriting earlier replies, and the prefix cache reuses earlier tasks.
    if (state.reads_preserve_thinking && !request.chat_template_kwargs.count("preserve_thinking")) {
        request.chat_template_kwargs["preserve_thinking"] = "true";
    }
    if (data.contains("json_schema")) {
        if (!request.tools.empty()) { throw std::invalid_argument("Combining tools with JSON output is unsupported"); }
        lower(std::nullopt, data.at("json_schema"));
        request.json_schema = data.at("json_schema").dump();
    }
    prepared_state prepared;
    prepared.parameters = common_chat_templates_apply(state.compiled.get(), request);
    // Extracted generators enforce the whole completion. Specialized handlers
    // may still provide activation metadata, which has no meaning in this mode.
    if (prepared.parameters.grammar_lazy) {
        throw std::logic_error("Unexpected lazy grammar from extracted generator");
    }
    prepared.parameters.grammar_triggers.clear();
    prepared.diagnostics = templates_native::diagnostics;
    prepared.renders = templates_native::renders;
    for (auto & recorded : templates_native::relaxations) {
        relaxations.insert({ recorded.tool, std::move(recorded.relaxation) });
    }
    prepared.relaxations = std::move(relaxations);
    return prepared;
}
}

extern "C" {
uint32_t templates_abi_version(void) { return 1; }

int32_t templates_buffer_release(uint64_t owner) {
    try {
        if (!owner) { return TEMPLATES_OK; }
        std::unique_ptr<std::string> removed;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = buffers.find(owner);
            if (it == buffers.end()) { return TEMPLATES_INVALID_HANDLE; }
            removed = std::move(it->second);
            buffers.erase(it);
        }
        return TEMPLATES_OK;
    } catch (...) { return TEMPLATES_NATIVE_ERROR; }
}

int32_t templates_build_info(templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        *output = owned(json({{"version", 1}, {"abi", 1}, {"extraction", 1},
            {"upstream", "930e2fa5995789efbf249a8bf61325bb626e417b"},
            {"build", TEMPLATES_BUILD_ID}}).dump());
    });
}

int32_t templates_template_create(const uint8_t * input, uint64_t size,
    uint64_t * output, templates_buffer * error) {
    if (output) { *output = 0; }
    return guarded(error, [&] {
        require_output(output);
        auto data = payload(input, size);
        auto source = data.at("source").get<std::string>();
        auto tokens = data.contains("special_tokens") ? data.at("special_tokens") : json::object();
        if (!tokens.is_object()) { throw std::invalid_argument("special_tokens must be an object"); }
        for (const auto & [key, value] : tokens.items()) {
            if (!value.is_string()) { throw std::invalid_argument("Special token values must be strings"); }
        }
        auto state = std::make_shared<template_state>();
        state->compiled = common_chat_templates_init(source,
            tokens.value("bos_token", std::string()), tokens.value("eos_token", std::string()));
        state->special_tokens = std::move(tokens);
        state->reads_preserve_thinking = common_chat_templates_reads_preserve_thinking(state->compiled.get());
        *output = insert_handle(templates, std::move(state));
    });
}

int32_t templates_template_release(uint64_t handle, templates_buffer * error) {
    return guarded(error, [&] {
        erase_handle(templates, handle, "Unknown or released template handle");
    });
}

int32_t templates_template_inspect(uint64_t handle, templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto state = get_template(handle);
        *output = owned(json({
            {"version", 1},
            {"capabilities", common_chat_templates_get_caps(state->compiled.get())},
            {"handler_context_variables", common_chat_templates_handler_context_variables(state->compiled.get())},
            {"template_reasoning_identifiers", common_chat_templates_reasoning_identifiers(state->compiled.get())}
        }).dump());
    });
}

int32_t templates_template_render(uint64_t handle, const uint8_t * input, uint64_t size,
    templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto state = get_template(handle);
        auto data = payload(input, size);
        auto context = data.at("context");
        if (!context.is_object()) { throw std::invalid_argument("Render context must be an object"); }
        for (const auto & [key, value] : state->special_tokens.items()) {
            if (context.contains(key)) { throw std::invalid_argument("Context overrides an artifact special token"); }
            context[key] = value;
        }
        auto now = data.at("now").get<int64_t>();
        const auto & compiled = common_chat_templates_get_default(state->compiled.get());
        jinja::context ctx(compiled.source());
        ctx.current_time = static_cast<time_t>(now);
        jinja::global_from_json(ctx, context, false);
        jinja::runtime runtime(ctx);
        auto rendered = runtime.execute(compiled.prog);
        *output = owned(jinja::runtime::gather_string_parts(rendered)->as_string().str());
    });
}

int32_t templates_template_probe(uint64_t handle, const uint8_t * input, uint64_t size,
    templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto state = get_template(handle);
        auto prepared = apply_request_payload(*state, input, size);
        const auto & params = prepared.parameters;
        std::string out;
        out.reserve(params.prompt.size() + params.generation_prompt.size() + params.thinking_start_tag.size() + 256);
        out.append("{\"prompt\":");
        append_json_string(out, params.prompt);
        out.append(",\"generation_prefix\":");
        append_json_string(out, params.generation_prompt);
        out.append(",\"parser_hash\":");
        append_json_hash128(out, params.parser);
        out.append(",\"grammar_hash\":");
        append_json_hash128(out, params.grammar);
        out.append(",\"supports_thinking\":");
        out.append(params.supports_thinking ? "true" : "false");
        out.append(",\"thinking_start\":");
        append_json_string(out, params.thinking_start_tag);
        out.append(",\"thinking_ends\":");
        append_json_string_array(out, params.thinking_end_tags);
        out.push_back('}');
        *output = owned(std::move(out));
    });
}

int32_t templates_request_create(uint64_t handle, const uint8_t * input, uint64_t size,
    uint64_t * output, templates_buffer * error) {
    if (output) { *output = 0; }
    return guarded(error, [&] {
        require_output(output);
        auto state = get_template(handle);
        auto prepared = std::make_shared<prepared_state>(apply_request_payload(*state, input, size));
        *output = insert_handle(requests, std::move(prepared));
    });
}

int32_t templates_request_describe(uint64_t handle, templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto state = get_request(handle);
        const auto & params = state->parameters;
        std::string out;
        out.reserve(params.prompt.size() + params.generation_prompt.size() + params.parser.size() + params.grammar.size() + 1024);
        out.append("{\"version\":1,\"prompt\":");
        append_json_string(out, params.prompt);
        out.append(",\"generation_prefix\":");
        append_json_string(out, params.generation_prompt);
        out.append(",\"parser\":");
        append_json_string(out, params.parser);
        out.append(",\"format\":");
        append_json_string(out, common_chat_format_name(params.format));
        out.append(",\"grammar\":");
        append_json_string(out, params.grammar);
        out.append(",\"grammar_dialect\":\"gbnf\",\"grammar_initial_prefix\":");
        append_json_string(out, params.grammar.empty() ? std::string_view() : std::string_view(params.generation_prompt));
        out.append(",\"grammar_lazy\":");
        out.append(params.grammar_lazy ? "true" : "false");
        out.append(",\"grammar_triggers\":[");
        for (size_t i = 0; i < params.grammar_triggers.size(); ++i) {
            if (i > 0) {
                out.push_back(',');
            }
            const auto & trigger = params.grammar_triggers[i];
            out.append("{\"type\":");
            out.append(std::to_string(static_cast<int>(trigger.type)));
            out.append(",\"value\":");
            append_json_string(out, trigger.value);
            out.append(",\"token\":");
            out.append(std::to_string(trigger.token));
            out.push_back('}');
        }
        out.append("],\"preserved_tokens\":");
        append_json_string_array(out, params.preserved_tokens);
        out.append(",\"additional_stops\":");
        append_json_string_array(out, params.additional_stops);
        out.append(",\"supports_thinking\":");
        out.append(params.supports_thinking ? "true" : "false");
        out.append(",\"thinking_start\":");
        append_json_string(out, params.thinking_start_tag);
        out.append(",\"thinking_ends\":");
        append_json_string_array(out, params.thinking_end_tags);
        out.append(",\"diagnostics\":");
        append_json_string_array(out, state->diagnostics);
        out.append(",\"relaxations\":[");
        bool first_relaxation = true;
        for (const auto & [tool, relaxation] : state->relaxations) {
            if (!first_relaxation) {
                out.push_back(',');
            }
            first_relaxation = false;
            out.append("{\"subject\":");
            if (tool) {
                out.append("{\"kind\":\"tool\",\"name\":");
                append_json_string(out, *tool);
                out.push_back('}');
            } else {
                out.append("{\"kind\":\"output\"}");
            }
            out.append(",\"path\":");
            append_json_string(out, relaxation.path);
            out.append(",\"keyword\":");
            append_json_string(out, relaxation.keyword);
            out.append(",\"reason\":");
            append_json_string(out, common_chat_schema_relaxation::reason_name(relaxation.reason));
            out.push_back('}');
        }
        out.append("],\"renders\":");
        out.append(std::to_string(state->renders));
        out.push_back('}');
        *output = owned(std::move(out));
    });
}

int32_t templates_request_release(uint64_t handle, templates_buffer * error) {
    return guarded(error, [&] {
        erase_handle(requests, handle, "Unknown or released prepared request");
    });
}

int32_t templates_stream_create(uint64_t handle, uint64_t limit, uint64_t * output, templates_buffer * error) {
    if (output) { *output = 0; }
    return guarded(error, [&] {
        require_output(output);
        auto request = get_request(handle);
        auto stream = std::make_shared<templates_native::output_stream>(request->parameters, limit);
        *output = insert_handle(streams, std::move(stream));
    });
}

int32_t templates_stream_feed(uint64_t handle, const uint8_t * bytes, uint64_t size,
    templates_events * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto stream = get_stream(handle);
        *output = stream->feed(bytes, size);
    });
}

int32_t templates_stream_finish(uint64_t handle, uint32_t cause,
    templates_events * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto stream = get_stream(handle);
        *output = stream->finish(cause);
    });
}

int32_t templates_stream_release(uint64_t handle, templates_buffer * error) {
    return guarded(error, [&] {
        erase_handle(streams, handle, "Unknown or released stream");
    });
}
}
