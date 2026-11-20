#include "parsers.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <utility>

#include "templates-log.h"

namespace {

size_t hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct parser_cache_key {
    std::string             tools;
    std::string             json_schema;
    std::string             generation_prompt;
    common_chat_tool_choice tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    common_reasoning_format reasoning_format    = COMMON_REASONING_FORMAT_NONE;
    std::optional<bool>     enable_thinking     = std::nullopt;
    bool                    parallel_tool_calls = false;
    size_t                  hash                = 0;

    bool operator==(const parser_cache_key & other) const {
        return hash == other.hash &&
               tool_choice == other.tool_choice &&
               reasoning_format == other.reasoning_format &&
               enable_thinking == other.enable_thinking &&
               parallel_tool_calls == other.parallel_tool_calls &&
               generation_prompt == other.generation_prompt &&
               json_schema == other.json_schema &&
               tools == other.tools;
    }
};

struct parser_cache_entry {
    parser_cache_key                               key;
    std::string                                    parser;
    std::string                                    grammar;
    bool                                           grammar_lazy = false;
    std::vector<common_grammar_trigger>            grammar_triggers;
    std::vector<templates_native::tool_relaxation> relaxations;
    std::vector<std::string>                       diagnostics;
};

parser_cache_key make_parser_cache_key(
        const autoparser::generation_params & inputs,
        const std::string &                   parser_generation_prompt) {
    parser_cache_key key;
    if (!inputs.tools.is_null()) {
        key.tools = inputs.tools.dump();
    }
    if (!inputs.json_schema.is_null()) {
        key.json_schema = inputs.json_schema.dump();
    }
    key.generation_prompt   = parser_generation_prompt;
    key.tool_choice         = inputs.tool_choice;
    key.reasoning_format    = inputs.reasoning_format;
    key.enable_thinking     = inputs.enable_thinking;
    key.parallel_tool_calls = inputs.parallel_tool_calls;

    size_t h = std::hash<std::string>{}(key.tools);
    h = hash_combine(h, std::hash<std::string>{}(key.json_schema));
    h = hash_combine(h, std::hash<std::string>{}(key.generation_prompt));
    h = hash_combine(h, std::hash<int>{}(static_cast<int>(key.tool_choice)));
    h = hash_combine(h, std::hash<int>{}(static_cast<int>(key.reasoning_format)));
    h = hash_combine(h, std::hash<int>{}(key.enable_thinking.has_value() ? (*key.enable_thinking ? 2 : 1) : 0));
    h = hash_combine(h, std::hash<bool>{}(key.parallel_tool_calls));
    key.hash = h;
    return key;
}

}  // namespace

struct common_chat_parser_cache {
    static constexpr size_t         max_entries = 32;
    std::mutex                      mutex;
    std::vector<parser_cache_entry> entries;
};

std::shared_ptr<common_chat_parser_cache> common_chat_parser_cache_create() {
    return std::make_shared<common_chat_parser_cache>();
}

void common_chat_apply_cached_parser(
        const common_chat_template &                      tmpl,
        const autoparser::generation_params &             inputs,
        const std::string &                               parser_generation_prompt,
        common_chat_params &                              data,
        const std::function<void(common_chat_params &)> & build) {
    const auto & cache = tmpl.cached_parsers;
    if (!cache) {
        build(data);
        return;
    }

    auto key = make_parser_cache_key(inputs, parser_generation_prompt);
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        for (size_t i = 0; i < cache->entries.size(); ++i) {
            if (cache->entries[i].key == key) {
                if (i > 0) {
                    std::rotate(cache->entries.begin(),
                                cache->entries.begin() + static_cast<std::ptrdiff_t>(i),
                                cache->entries.begin() + static_cast<std::ptrdiff_t>(i + 1));
                }
                const auto & hit = cache->entries.front();
                data.parser           = hit.parser;
                data.grammar          = hit.grammar;
                data.grammar_lazy     = hit.grammar_lazy;
                data.grammar_triggers = hit.grammar_triggers;
                if (!hit.relaxations.empty()) {
                    templates_native::relaxations.insert(
                        templates_native::relaxations.end(),
                        hit.relaxations.begin(),
                        hit.relaxations.end());
                }
                if (!hit.diagnostics.empty()) {
                    templates_native::diagnostics.insert(
                        templates_native::diagnostics.end(),
                        hit.diagnostics.begin(),
                        hit.diagnostics.end());
                }
                return;
            }
        }
    }

    const size_t relaxations_before = templates_native::relaxations.size();
    const size_t diagnostics_before = templates_native::diagnostics.size();

    build(data);

    parser_cache_entry entry;
    entry.key              = std::move(key);
    entry.parser           = data.parser;
    entry.grammar          = data.grammar;
    entry.grammar_lazy     = data.grammar_lazy;
    entry.grammar_triggers = data.grammar_triggers;
    if (templates_native::relaxations.size() > relaxations_before) {
        entry.relaxations.assign(
            templates_native::relaxations.begin() + static_cast<std::ptrdiff_t>(relaxations_before),
            templates_native::relaxations.end());
    }
    if (templates_native::diagnostics.size() > diagnostics_before) {
        entry.diagnostics.assign(
            templates_native::diagnostics.begin() + static_cast<std::ptrdiff_t>(diagnostics_before),
            templates_native::diagnostics.end());
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    for (size_t i = 0; i < cache->entries.size(); ++i) {
        if (cache->entries[i].key == entry.key) {
            if (i > 0) {
                std::rotate(cache->entries.begin(),
                            cache->entries.begin() + static_cast<std::ptrdiff_t>(i),
                            cache->entries.begin() + static_cast<std::ptrdiff_t>(i + 1));
            }
            return;
        }
    }
    if (cache->entries.size() >= common_chat_parser_cache::max_entries) {
        cache->entries.pop_back();
    }
    cache->entries.insert(cache->entries.begin(), std::move(entry));
}

void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

void foreach_parameter(const json & function, const std::function<void(const common_chat_schema_property &, const common_chat_schema_document_ptr &)> & fn) {
    auto                       params = common_chat_tool_parameters(function);
    auto                       doc    = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));
    const common_chat_schema * root   = doc->root.get();
    for (size_t hops = 0; root->kind() == common_chat_schema::KIND_REF && hops <= doc->refs.size(); hops++) {
        root = static_cast<const common_chat_schema_ref *>(root)->target;
    }
    const auto * object = dynamic_cast<const common_chat_schema_object *>(root);
    if (!object) {
        // Named arguments express one object's properties; under any other
        // argument schema they are written as none, which it may reject.
        templates_native::relax(function.at("name"), *doc->root, "type", common_chat_schema_relaxation::REASON_UNENFORCED);
        return;
    }
    for (const auto & prop : object->properties) {
        fn(prop, doc);
    }
}

// The rules are the product of a prefix check for `open` and a scanner for
// `close`.
void unopened_text_grammar(const common_grammar_builder & builder,
                           const std::string &            name,
                           const std::string &            open,
                           const std::string &            close,
                           bool                           including) {
    std::set<char> letters(close.begin(), close.end());
    letters.insert(open.begin(), open.end());
    // Progress through `open` (-1 once it can no longer match), and through `close`.
    using state = std::pair<int, size_t>;
    const auto advance_close = [&](size_t matched, char c) {
        const std::string text = close.substr(0, matched) + c;
        for (size_t length = std::min(text.size(), close.size()); length > 0; length--) {
            if (text.compare(text.size() - length, length, close, 0, length) == 0) {
                return length;
            }
        }
        return size_t{0};
    };
    std::map<state, std::string> names;
    std::vector<state> pending;
    const auto name_of = [&](const state & s) {
        auto found = names.find(s);
        if (found != names.end()) {
            return found->second;
        }
        auto rule = names.empty() ? name : name + "-" + std::to_string(names.size());
        names.emplace(s, rule);
        pending.push_back(s);
        return rule;
    };
    const auto char_class = [](const std::set<char> & chars, bool negated) {
        std::string out = negated ? "[^" : "[";
        for (char c : chars) {
            char escaped[5];
            std::snprintf(escaped, sizeof(escaped), "\\x%02X", (unsigned char) c);
            out += escaped;
        }
        return out + "]";
    };
    name_of({0, 0});
    for (size_t index = 0; index < pending.size(); index++) {
        const auto [opened, closed] = pending[index];
        std::vector<std::string> alternatives;
        if (!including) {
            alternatives.push_back("");
        }
        for (char c : letters) {
            int next_open = -1;
            if (opened >= 0 && c == open[opened]) {
                if ((size_t) opened + 1 == open.size()) {
                    continue;  // the text begins with the opener
                }
                next_open = opened + 1;
            }
            const auto next_close = advance_close(closed, c);
            if (next_close < close.size()) {
                alternatives.push_back(char_class({ c }, false) + " " + name_of({ next_open, next_close }));
            } else if (including) {
                alternatives.push_back(char_class({ c }, false));
            }
        }
        alternatives.push_back(char_class(letters, true) + " " + name_of({ -1, 0 }));
        builder.add_rule(names.at({ opened, closed }), string_join(alternatives, " | "));
    }
}
