#include "output-stream.h"
#include "chat-peg-parser.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <stdexcept>

namespace templates_native {
namespace {
struct argument_insertion { size_t offset, start, end; unsigned escaping = 1; };

// A partial Python scalar/container may normalize differently once complete.
// Keep that value unpublished; preceding fields and literal string prefixes are
// still available. Complete mapping remains the upstream implementation.
class streaming_mapper : public common_chat_peg_mapper {
  public:
    std::vector<text_spans> arguments;
    std::vector<std::vector<argument_insertion>> strings;
    text_spans pending_arguments;
    bool named = false;
    explicit streaming_mapper(common_chat_msg & message) : common_chat_peg_mapper(message) {}
    void map(const common_peg_ast_node & node) override {
        if (node.tag == common_chat_peg_builder::TOOL_ARG_VALUE && node.is_partial) { return; }
        if (node.tag == common_chat_peg_builder::TOOL_OPEN) {
            pending_arguments.clear();
            named = false;
        }
        if (node.tag == common_chat_peg_builder::TOOL_ARGS && !node.text.empty() && node.text.front() == '{') {
            size_t length = node.text.size();
            while (length && std::isspace(static_cast<unsigned char>(node.text[length - 1]))) { --length; }
            pending_arguments = {{node.start, node.start + length}};
            if (named) { arguments.back() = pending_arguments; }
            return;
        }
        if (node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE && named) {
            // Let upstream own punctuation and quote state. The source string
            // is escaped only as its new suffix is published, not on every feed.
            auto empty = node;
            empty.text = {};
            common_chat_peg_mapper::map(empty);
            strings.back().push_back({result.tool_calls.back().arguments.size(), node.start, node.end});
            return;
        }
        common_chat_peg_mapper::map(node);
        if (node.tag == common_chat_peg_builder::TOOL_NAME && !result.tool_calls.empty()) {
            named = true;
            arguments.resize(result.tool_calls.size());
            strings.resize(result.tool_calls.size());
            arguments.back() = pending_arguments;
        }
    }
};

size_t utf8_prefix(const std::string & bytes) {
    size_t position = 0;
    while (position < bytes.size()) {
        const auto first = static_cast<uint8_t>(bytes[position]);
        size_t width;
        uint32_t value;
        uint32_t minimum;
        if (first < 0x80) { ++position; continue; }
        if (first >= 0xc2 && first <= 0xdf) { width = 2; value = first & 0x1f; minimum = 0x80; }
        else if (first >= 0xe0 && first <= 0xef) { width = 3; value = first & 0x0f; minimum = 0x800; }
        else if (first >= 0xf0 && first <= 0xf4) { width = 4; value = first & 0x07; minimum = 0x10000; }
        else { throw std::invalid_argument("Invalid UTF-8 output"); }
        const size_t available = std::min(width, bytes.size() - position);
        for (size_t i = 1; i < available; ++i) {
            const auto next = static_cast<uint8_t>(bytes[position + i]);
            if ((next & 0xc0) != 0x80) { throw std::invalid_argument("Invalid UTF-8 continuation"); }
            value = (value << 6) | (next & 0x3f);
        }
        if (available < width) { return position; }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) {
            throw std::invalid_argument("Invalid UTF-8 scalar");
        }
        position += width;
    }
    return position;
}
}

output_stream::output_stream(const common_chat_params & plan, uint64_t limit)
    : context(COMMON_PEG_PARSE_FLAG_LENIENT), format(plan.format), limit(limit) {
    if (!limit || limit > 64 * 1024 * 1024) { throw std::invalid_argument("Invalid output byte limit"); }
    context.append_only = true;
    if (plan.parser.empty()) {
        parser = build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); });
    } else {
        parser.load(plan.parser);
        context.input = plan.generation_prompt;
    }
    for (size_t i = 0; i < parser.size(); ++i) {
        if (auto tag = std::get_if<common_peg_tag_parser>(&parser.get(i))) {
            has_explicit_ids |= tag->tag == common_chat_peg_builder::TOOL_ID;
        }
    }
    // The generation prompt's own reading is the baseline, never published.
    if (auto baseline = read(true)) {
        publish(*baseline);
        values.clear();
        pending_reasoning.clear();
    }
}

// One complete reading of the output so far. A lenient reading treats the end
// of the input as a point where more output may follow; a strict one treats it
// as the end of the output and exists only when the whole output satisfies the
// parser, every call it holds then being complete.
std::optional<output_stream::reading> output_stream::read(bool lenient) {
    context.parse_depth = 0;
    context.flags = lenient ? COMMON_PEG_PARSE_FLAG_LENIENT : COMMON_PEG_PARSE_FLAG_NONE;
    auto result = parser.parse(context);
    if (!lenient && (!result.success() || result.end != context.input.size())) { return std::nullopt; }
    if (result.fail() && result.nodes.empty()) { return std::nullopt; }
    common_chat_msg message;
    std::vector<text_spans> mapped_arguments;
    std::vector<std::vector<argument_insertion>> mapped_strings;
    if (format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        common_chat_peg_gemma4_mapper mapper(message);
        mapper.retain_output = true;
        mapper.from_ast(context.ast, result);
    } else if (format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
        common_chat_peg_minimax_m3_mapper mapper(message);
        mapper.retain_output = true;
        mapper.from_ast(context.ast, result);
    } else {
        streaming_mapper mapper(message);
        mapper.retain_output = true;
        mapper.from_ast(context.ast, result);
        mapped_arguments = std::move(mapper.arguments);
        mapped_strings = std::move(mapper.strings);
    }
    std::vector<bool> closed;
    context.ast.visit_tags(result, [&](const common_peg_ast_node & node) {
        if (node.tag == common_chat_peg_builder::TOOL) { closed.push_back(!node.is_partial); }
    });
    reading next{message.content_output, message.reasoning_output, {}};
    next.calls.resize(message.tool_calls.size());
    for (size_t i = 0; i < next.calls.size(); ++i) {
        const auto & current = message.tool_calls[i];
        auto & call = next.calls[i];
        call.name = current.name;
        call.id = current.id;
        call.closed = !lenient || (i < closed.size() && closed[i]);
        common_peg_text text = current.argument_output;
        if (text.empty() && i < mapped_arguments.size() && !mapped_arguments[i].empty()) {
            for (const auto & span : mapped_arguments[i]) {
                text += common_peg_text::source(span.first, span.second);
            }
        } else if (text.empty()) {
            size_t offset = 0;
            if (i < mapped_strings.size()) {
                for (const auto & insertion : mapped_strings[i]) {
                    text += current.arguments.substr(offset, insertion.offset - offset);
                    auto source = common_peg_text::source(insertion.start, insertion.end);
                    for (unsigned j = 0; j < insertion.escaping; ++j) { source = source.escaped(); }
                    text += source;
                    offset = insertion.offset;
                }
            }
            text += current.arguments.substr(offset);
        }
        call.arguments = std::move(text);
    }
    return next;
}

// Publishes what a reading adds to the published output. A reading that does
// not extend it (retracting or changing published text, a published call or a
// call's identity) publishes nothing and is refused.
bool output_stream::publish(const reading & next) {
    auto new_content = next.content.suffix(content_output, context.input);
    auto new_reasoning = next.reasoning.suffix(reasoning_output, context.input);
    if (!new_content || !new_reasoning) { return false; }
    for (size_t i = next.calls.size(); i < calls.size(); ++i) {
        if (calls[i].started) { return false; }
    }
    std::vector<event_value> emitted;
    auto pending = pending_reasoning;
    bool visible = visible_reasoning;
    if (!visible) {
        pending += *new_reasoning;
        if (new_reasoning->find_first_not_of(" \n\r\t") != std::string::npos) {
            visible = true;
            *new_reasoning = std::move(pending);
            pending.clear();
        } else {
            new_reasoning->clear();
        }
    }
    if (!new_reasoning->empty()) { emitted.push_back({TEMPLATES_REASONING, 0, std::move(*new_reasoning), {}}); }
    if (!new_content->empty()) { emitted.push_back({TEMPLATES_CONTENT, 0, std::move(*new_content), {}}); }

    auto cursors = calls;
    cursors.resize(next.calls.size());
    for (size_t i = 0; i < cursors.size(); ++i) {
        auto & cursor = cursors[i];
        const auto & current = next.calls[i];
        if (current.name.empty()) { continue; }
        if (!cursor.started) {
            if (has_explicit_ids && current.id.empty() && !current.closed) { continue; }
            cursor.name = current.name;
            // Only the model's own ID; the host mints one when the format has none.
            cursor.id = current.id;
            cursor.started = true;
            emitted.push_back({TEMPLATES_TOOL_START, static_cast<uint32_t>(i), cursor.name, cursor.id});
        } else if (cursor.name != current.name || (!current.id.empty() && cursor.id != current.id)) {
            return false;
        }
        auto suffix = current.arguments.suffix(cursor.argument_output, context.input);
        if (!suffix) { return false; }
        if (!suffix->empty()) {
            emitted.push_back({TEMPLATES_TOOL_ARGUMENTS, static_cast<uint32_t>(i), std::move(*suffix), {}});
        }
        cursor.argument_output = current.arguments;
        if (current.closed && !cursor.complete) {
            emitted.push_back({TEMPLATES_TOOL_COMPLETE, static_cast<uint32_t>(i), {}, {}});
            cursor.complete = true;
        }
    }

    content_output = next.content;
    reasoning_output = next.reasoning;
    pending_reasoning = std::move(pending);
    visible_reasoning = visible;
    calls = std::move(cursors);
    values.insert(values.end(), std::make_move_iterator(emitted.begin()), std::make_move_iterator(emitted.end()));
    return true;
}

// The strict reading resolves what the stream withheld at the end of output,
// such as a provisional delimiter. It never reinterprets published output:
// when the model ends inside a construct the stream already published from,
// such as reasoning it never closed, the strict reading backtracks out of that
// construct and reads it again as something else. The end of output then
// closes the construct instead, and the published reading stands, provided it
// leaves no tool call open.
void output_stream::finish_natural() {
    if (auto strict = read(false); strict && publish(*strict)) { return; }
    auto published = read(true);
    const bool open_call = published && std::any_of(published->calls.begin(), published->calls.end(),
        [](const call_reading & call) { return !call.closed; });
    if (!published || open_call || !publish(*published)) {
        throw std::runtime_error("Naturally completed output does not satisfy its parser");
    }
}

templates_events output_stream::batch() {
    events.clear();
    events.reserve(values.size());
    for (const auto & value : values) {
        events.push_back({value.kind, value.index,
            reinterpret_cast<const uint8_t *>(value.text.data()), value.text.size(),
            reinterpret_cast<const uint8_t *>(value.id.data()), value.id.size()});
    }
    return {events.data(), events.size()};
}

templates_events output_stream::feed(const uint8_t * bytes, uint64_t size) {
    if (terminal) { throw std::invalid_argument("Stream is terminal"); }
    values.clear();
    events.clear();
    try {
        if (size > limit - received) { throw std::length_error("Output parser byte limit exceeded"); }
        if (!bytes && size) { throw std::invalid_argument("Null output byte span"); }
        received += size;
        if (size) { utf8_pending.append(reinterpret_cast<const char *>(bytes), static_cast<size_t>(size)); }
        const auto complete = utf8_prefix(utf8_pending);
        if (complete) {
            context.input.append(utf8_pending, 0, complete);
            utf8_pending.erase(0, complete);
            auto next = read(true);
            if (!next) { throw std::runtime_error("Output does not match its prepared parser"); }
            if (!publish(*next)) { throw std::runtime_error("Output parser would change published output"); }
        }
        return batch();
    } catch (...) {
        terminal = true;
        values.clear();
        events.clear();
        throw;
    }
}

templates_events output_stream::finish(uint32_t cause) {
    if (terminal) { throw std::invalid_argument("Stream is terminal"); }
    if (cause > TEMPLATES_FAILED) { throw std::invalid_argument("Unknown terminal cause"); }
    terminal = true;
    values.clear();
    events.clear();
    try {
        if (cause == TEMPLATES_NATURAL) {
            if (!utf8_pending.empty()) { throw std::invalid_argument("Naturally completed output ends inside UTF-8"); }
            finish_natural();
        }
        values.push_back({TEMPLATES_FINISH, cause, {}, {}});
        return batch();
    } catch (...) {
        values.clear();
        events.clear();
        throw;
    }
}
}
