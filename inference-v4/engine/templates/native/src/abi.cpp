#include "templates.h"
#include "chat.h"
#include "jinja/runtime.h"
#include "json-schema-to-grammar.h"
#include "templates-log.h"
#include "output-stream.h"
#include "schema-validation.h"

#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
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
};
struct prepared_state {
    common_chat_params parameters;
    common_chat_parser_params parser;
    std::vector<std::string> diagnostics;
};
std::mutex mutex;
uint64_t next_handle = 1;
std::unordered_map<uint64_t, std::unique_ptr<template_state>> templates;
std::unordered_map<uint64_t, std::shared_ptr<const prepared_state>> requests;
std::unordered_map<uint64_t, std::unique_ptr<templates_native::output_stream>> streams;
std::unordered_map<uint64_t, std::unique_ptr<std::string>> buffers;

uint64_t allocate_handle() {
    if (next_handle == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("Native handle space exhausted");
    }
    return next_handle++;
}

templates_buffer owned(std::string value) {
    auto storage = std::make_unique<std::string>(std::move(value));
    auto id = allocate_handle();
    templates_buffer result{reinterpret_cast<const uint8_t *>(storage->data()), storage->size(), id};
    buffers.emplace(id, std::move(storage));
    return result;
}

template<typename F>
int32_t guarded(templates_buffer * error, F && function) noexcept {
    if (!error) { return TEMPLATES_INVALID_ARGUMENT; }
    *error = {};
    try {
        std::lock_guard<std::mutex> lock(mutex);
        templates_native::diagnostics.clear();
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

template_state & get_template(uint64_t handle) {
    auto it = templates.find(handle);
    if (it == templates.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released template handle"); }
    return *it->second;
}

void require_output(const void * output) {
    if (!output) { throw std::invalid_argument("Null output pointer"); }
}

const prepared_state & get_request(uint64_t handle) {
    auto it = requests.find(handle);
    if (it == requests.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released prepared request"); }
    return *it->second;
}
templates_native::output_stream & get_stream(uint64_t handle) {
    auto it = streams.find(handle);
    if (it == streams.end()) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released stream"); }
    return *it->second;
}
}

extern "C" {
uint32_t templates_abi_version(void) { return 1; }

int32_t templates_buffer_release(uint64_t owner) {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        if (!owner) { return TEMPLATES_OK; }
        return buffers.erase(owner) ? TEMPLATES_OK : TEMPLATES_INVALID_HANDLE;
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
        auto state = std::make_unique<template_state>();
        state->compiled = common_chat_templates_init(source,
            tokens.value("bos_token", std::string()), tokens.value("eos_token", std::string()));
        state->special_tokens = std::move(tokens);
        auto id = allocate_handle();
        templates.emplace(id, std::move(state));
        *output = id;
    });
}

int32_t templates_template_release(uint64_t handle, templates_buffer * error) {
    return guarded(error, [&] {
        if (!templates.erase(handle)) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released template handle"); }
    });
}

int32_t templates_template_inspect(uint64_t handle, templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto & state = get_template(handle);
        *output = owned(json({{"version", 1}, {"capabilities", common_chat_templates_get_caps(state.compiled.get())}}).dump());
    });
}

int32_t templates_template_render(uint64_t handle, const uint8_t * input, uint64_t size,
    templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        auto & state = get_template(handle);
        auto data = payload(input, size);
        auto context = data.at("context");
        if (!context.is_object()) { throw std::invalid_argument("Render context must be an object"); }
        for (const auto & [key, value] : state.special_tokens.items()) {
            if (context.contains(key)) { throw std::invalid_argument("Context overrides an artifact special token"); }
            context[key] = value;
        }
        auto now = data.at("now").get<int64_t>();
        const auto & compiled = common_chat_templates_get_default(state.compiled.get());
        jinja::context ctx(compiled.source());
        ctx.current_time = static_cast<time_t>(now);
        jinja::global_from_json(ctx, context, false);
        jinja::runtime runtime(ctx);
        auto rendered = runtime.execute(compiled.prog);
        *output = owned(jinja::runtime::gather_string_parts(rendered)->as_string().str());
    });
}

int32_t templates_request_create(uint64_t handle, const uint8_t * input, uint64_t size,
    uint64_t * output, templates_buffer * error) {
    if (output) { *output = 0; }
    return guarded(error, [&] {
        require_output(output);
        auto & state = get_template(handle);
        auto data = payload(input, size);
        common_chat_templates_inputs request;
        request.messages = common_chat_msgs_parse_oaicompat(data.at("messages"));
        if (request.messages.empty()) { throw std::invalid_argument("Request requires messages"); }
        request.tool_choice = common_chat_tool_choice_parse_oaicompat(data.value("tool_choice", std::string("auto")));
        if (request.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && data.contains("tools")) {
            for (const auto & tool : data.at("tools")) {
                const auto & function = tool.at("function");
                if (function.contains("parameters")) {
                    templates_native::validate_schema(function.at("parameters"));
                }
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
        if (data.contains("json_schema")) {
            if (!request.tools.empty()) { throw std::invalid_argument("Combining tools with JSON output is unsupported"); }
            templates_native::validate_schema(data.at("json_schema"));
            request.json_schema = data.at("json_schema").dump();
        }
        auto prepared = std::make_shared<prepared_state>();
        prepared->parameters = common_chat_templates_apply(state.compiled.get(), request);
        if (prepared->parameters.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
            // This upstream handler constrains names but parses an arbitrary
            // Gemma dictionary. It cannot advertise strict argument schemas.
            for (const auto & tool : request.tools) {
                const auto schema = json::parse(tool.parameters);
                for (const auto & [key, value] : schema.items()) {
                    if (key == "type" && value == "object") { continue; }
                    if (key == "additionalProperties" && value == true) { continue; }
                    if (key == "description" || key == "title") { continue; }
                    throw std::invalid_argument("Unsupported JSON schema: Gemma tool argument constraints");
                }
            }
        }
        // Extracted generators enforce the whole completion. Specialized handlers
        // may still provide activation metadata, which has no meaning in this mode.
        if (prepared->parameters.grammar_lazy) {
            throw std::logic_error("Unexpected lazy grammar from extracted generator");
        }
        prepared->parameters.grammar_triggers.clear();
        prepared->parser = common_chat_parser_params(prepared->parameters);
        prepared->parser.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        if (!prepared->parameters.parser.empty()) { prepared->parser.parser.load(prepared->parameters.parser); }
        prepared->diagnostics = templates_native::diagnostics;
        auto id = allocate_handle();
        requests.emplace(id, std::move(prepared));
        *output = id;
    });
}

int32_t templates_request_describe(uint64_t handle, templates_buffer * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        const auto & state = get_request(handle);
        const auto & params = state.parameters;
        json triggers = json::array();
        for (const auto & trigger : params.grammar_triggers) {
            triggers.push_back(json({{"type", static_cast<int>(trigger.type)}, {"value", trigger.value}, {"token", trigger.token}}));
        }
        *output = owned(json({{"version", 1}, {"prompt", params.prompt},
            {"generation_prefix", params.generation_prompt}, {"parser", params.parser},
            {"format", common_chat_format_name(params.format)},
            {"grammar", params.grammar}, {"grammar_dialect", "gbnf"},
            {"grammar_initial_prefix", params.grammar.empty() ? std::string() : params.generation_prompt},
            {"grammar_lazy", params.grammar_lazy}, {"grammar_triggers", triggers},
            {"preserved_tokens", params.preserved_tokens}, {"additional_stops", params.additional_stops},
            {"supports_thinking", params.supports_thinking}, {"thinking_start", params.thinking_start_tag},
            {"thinking_ends", params.thinking_end_tags}, {"diagnostics", state.diagnostics}}).dump());
    });
}

int32_t templates_request_release(uint64_t handle, templates_buffer * error) {
    return guarded(error, [&] {
        if (!requests.erase(handle)) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released prepared request"); }
    });
}

int32_t templates_stream_create(uint64_t handle, uint64_t limit, uint64_t * output, templates_buffer * error) {
    if (output) { *output = 0; }
    return guarded(error, [&] {
        require_output(output);
        auto stream = std::make_unique<templates_native::output_stream>(get_request(handle).parameters, limit);
        auto id = allocate_handle();
        streams.emplace(id, std::move(stream));
        *output = id;
    });
}

int32_t templates_stream_feed(uint64_t handle, const uint8_t * bytes, uint64_t size,
    templates_events * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        *output = get_stream(handle).feed(bytes, size);
    });
}

int32_t templates_stream_finish(uint64_t handle, uint32_t cause,
    templates_events * output, templates_buffer * error) {
    if (output) { *output = {}; }
    return guarded(error, [&] {
        require_output(output);
        *output = get_stream(handle).finish(cause);
    });
}

int32_t templates_stream_release(uint64_t handle, templates_buffer * error) {
    return guarded(error, [&] {
        if (!streams.erase(handle)) { throw abi_error(TEMPLATES_INVALID_HANDLE, "Unknown or released stream"); }
    });
}
}
