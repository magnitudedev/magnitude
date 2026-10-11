#include "parsers.h"

#include <cstdio>
#include <map>
#include <utility>

#include "templates-log.h"

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
