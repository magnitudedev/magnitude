#include "parsers.h"

namespace workaround {

// Gemma4 uses a custom tool_responses field instead of role:tool messages.
//
// This will transform a sequence of messages:
//   assistant(tool_call+) -> tool+ -> assistant(content)
//
// Into a single assistant message containing a tool_responses field:
//   assistant(content + tool_call + tool_responses)
//
// This is necessary for the Gemma4 chat template to properly format the prompt.
// See https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
struct gemma4_model_turn_builder {
    json & messages;
    size_t pos;
    json tool_calls = json::array();
    json tool_responses = json::array();
    json content;
    json reasoning_content;

    gemma4_model_turn_builder(json & msgs, size_t pos) : messages(msgs), pos(pos) {}

    void collect() {
        // Collect the first assistant message
        auto & msg = messages[pos];
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            // According to the prompt formatting guide, we need to preserve reasoning_content
            // between function calls. The current chat templates do not support this, but we will do it anyway.
            reasoning_content = msg.at("reasoning_content");
        }
        for (auto & tc : msg.at("tool_calls")) {
            tool_calls.push_back(tc);
        }
        pos++;

        // Collect tool call results
        while (pos < messages.size() && messages[pos].value("role", "") == "tool") {
            collect_result(messages[pos]);
            pos++;
        }

        // Check if the next assistant message is the final message
        if (pos < messages.size() && messages[pos].value("role", "") == "assistant") {
            auto & next = messages[pos];
            if (!has_tool_calls(next) && has_content(next)) {
                content = next.at("content");
                pos++;
            }
        }
    }

    void collect_result(const json & curr) {
        json response;
        if (curr.contains("content")) {
            const auto & content = curr.at("content");
            if (content.is_string()) {
                // Try to parse the content as JSON; fall back to raw string
                try {
                    response = json::parse(content.get<std::string>());
                } catch (...) {
                    response = content;
                }
            } else {
                response = content;
            }
        }

        std::string name;

        // Match name with corresponding tool call
        size_t idx = tool_responses.size();
        if (idx < tool_calls.size()) {
            auto & tc = tool_calls[idx];
            if (tc.contains("function")) {
                name = tc.at("function").value("name", "");
            }
        }

        // Fallback to the tool call id
        if (name.empty()) {
            name = curr.value("tool_call_id", "");
        }

        tool_responses.push_back({{"name", name}, {"response", response}});
    }

    json build() {
        collect();

        json msg = {
            {"role", "assistant"},
            {"tool_calls", tool_calls},
        };
        if (!tool_responses.empty()) {
            msg["tool_responses"] = tool_responses;
        }
        if (!content.is_null()) {
            msg["content"] = content;
        }
        if (!reasoning_content.is_null()) {
            msg["reasoning_content"] = reasoning_content;
        }
        return msg;
    }

    static bool has_content(const json & msg) {
        if (!msg.contains("content") || msg.at("content").is_null()) {
            return false;
        }
        const auto & content = msg.at("content");
        if (content.is_string() && !content.get<std::string>().empty()) {
            return true;
        }
        if (content.is_array() && !content.empty()) {
            return true;
        }
        return false;
    }

    static bool has_tool_calls(const json & msg) {
        return msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty();
    }
};

void convert_tool_responses_gemma4(json & messages) {
    json result = json::array();
    size_t i = 0;

    while (i < messages.size()) {
        auto & msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            result.push_back(msg);
            i++;
            continue;
        }

        gemma4_model_turn_builder builder(messages, i);
        result.push_back(builder.build());
        i = builder.pos;
    }

    messages = result;
}

}

namespace {

// Tool arguments constrained by the function's JSON schema, in the dictionary
// syntax the template renders them in (`format_argument`): unquoted keys in
// `dictsort` order, strings between <|"|> delimiters. Every rule is named
// `<kind>--<path>` after the node kind common_chat_peg_gemma4_mapper converts
// to JSON, so parsing and the generation grammar come from the same rules.
class gemma4_argument_rules {
  public:
    gemma4_argument_rules(common_chat_peg_builder & p, std::string prefix, std::string tool) :
        p(p), prefix(std::move(prefix)), tool(std::move(tool)) {}

    common_peg_parser visit(const common_chat_schema & schema) {
        switch (schema.kind()) {
            case common_chat_schema::KIND_ANY:
                return p.ref("gemma4-value");
            case common_chat_schema::KIND_NULL:
                return p.ref("gemma4-null");
            case common_chat_schema::KIND_BOOLEAN:
                return p.ref("gemma4-bool");
            case common_chat_schema::KIND_NUMBER:
                bounds(schema, static_cast<const common_chat_schema_number &>(schema));
                return p.ref("gemma4-number");
            case common_chat_schema::KIND_INTEGER: {
                bounds(schema, static_cast<const common_chat_schema_integer &>(schema));
                return p.rule("gemma4-number--" + path(), p.sequence({
                    p.optional(p.literal("-")),
                    p.choice({p.literal("0"), p.chars("[1-9]", 1, 1) + p.chars("[0-9]", 0, -1)}),
                    p.negate(p.chars("[0-9.eE+-]", 1, 1)),
                }));
            }
            case common_chat_schema::KIND_STRING: {
                const auto & string = static_cast<const common_chat_schema_string &>(schema);
                if (!string.pattern.empty()) {
                    unenforced(schema, "pattern");
                }
                if (string.min_length > 0) {
                    unenforced(schema, "minLength");
                }
                if (string.max_length >= 0) {
                    unenforced(schema, "maxLength");
                }
                return p.ref("gemma4-string");
            }
            case common_chat_schema::KIND_CONST: {
                auto rules = literals(schema, { static_cast<const common_chat_schema_const &>(schema).value }, "const");
                return rules ? rules->front() : p.ref("gemma4-value");
            }
            case common_chat_schema::KIND_ENUM: {
                auto rules = literals(schema, static_cast<const common_chat_schema_enum &>(schema).values, "enum");
                return rules ? alternatives(*rules) : p.ref("gemma4-value");
            }
            case common_chat_schema::KIND_ANY_OF: {
                std::vector<common_peg_parser> values;
                for (const auto & child : static_cast<const common_chat_schema_any_of &>(schema).children) {
                    values.push_back(visit(*child));
                }
                return alternatives(values);
            }
            case common_chat_schema::KIND_REF: {
                const auto * target = static_cast<const common_chat_schema_ref &>(schema).target;
                // The placeholder rule lets a recursive schema refer to itself.
                auto index = references.emplace(target, references.size()).first->second;
                return p.rule("gemma4-value--" + prefix + "-ref-" + std::to_string(index),
                              [&]() { return visit(*target); });
            }
            case common_chat_schema::KIND_ARRAY:
                return array(static_cast<const common_chat_schema_array &>(schema));
            case common_chat_schema::KIND_OBJECT:
                return object(static_cast<const common_chat_schema_object &>(schema));
            case common_chat_schema::KIND_TUPLE:
                unenforced(schema, "prefixItems");
                return p.ref("gemma4-array");
            case common_chat_schema::KIND_NEVER:
                throw std::logic_error("unsettled schema node");
        }
        throw std::logic_error("unhandled schema node");
    }

  private:
    common_chat_peg_builder &                     p;
    std::string                                   prefix;
    std::string                                   tool;
    size_t                                        next = 0;
    std::map<const common_chat_schema *, size_t>  references;

    // The dictionary syntax does not enforce this keyword.
    void unenforced(const common_chat_schema & schema, const std::string & keyword) {
        templates_native::relax(tool, schema, keyword, common_chat_schema_relaxation::REASON_UNENFORCED);
    }

    // Numbers are written without bounds.
    void bounds(const common_chat_schema & schema, const common_chat_schema_numeric & numeric) {
        if (numeric.minimum) {
            unenforced(schema, numeric.minimum->exclusive ? "exclusiveMinimum" : "minimum");
        }
        if (numeric.maximum) {
            unenforced(schema, numeric.maximum->exclusive ? "exclusiveMaximum" : "maximum");
        }
    }

    std::string path() { return prefix + "-" + std::to_string(next++); }

    // The listed values (const or enum) rendered literally, or none when the
    // list admits any value instead: a structured value is listed (those are
    // not rendered), or every value contains the string delimiter. Strings
    // containing the delimiter cannot be written and are left out.
    std::optional<std::vector<common_peg_parser>> literals(const common_chat_schema & schema, const std::vector<json> & values,
                                                           const std::string & keyword) {
        if (std::any_of(values.begin(), values.end(), [](const json & value) { return value.is_object() || value.is_array(); })) {
            unenforced(schema, keyword);
            return std::nullopt;
        }
        std::vector<common_peg_parser> rules;
        for (const auto & value : values) {
            if (value.is_string() && value.get<std::string>().find("<|\"|>") != std::string::npos) {
                templates_native::relax(tool, schema, keyword, common_chat_schema_relaxation::REASON_UNREPRESENTABLE);
                continue;
            }
            rules.push_back(constant(value));
        }
        if (rules.empty()) {
            unenforced(schema, keyword);
            return std::nullopt;
        }
        return rules;
    }

    // A scalar rendered literally.
    common_peg_parser constant(const json & value) {
        auto at = path();
        if (value.is_string()) {
            const auto text = value.get<std::string>();
            return p.rule("gemma4-string--" + at,
                          p.literal("<|\"|>") + p.rule("gemma4-string-content--" + at, p.literal(text)) + p.literal("<|\"|>"));
        }
        if (value.is_boolean()) {
            return p.rule("gemma4-bool--" + at, p.literal(value.get<bool>() ? "true" : "false"));
        }
        if (value.is_null()) {
            return p.rule("gemma4-null--" + at, p.literal("null"));
        }
        return p.rule("gemma4-number--" + at, p.literal(value.dump()));
    }

    // One of several values. A PEG choice commits to the first alternative
    // that matches, so each must reach the end of the value.
    common_peg_parser alternatives(const std::vector<common_peg_parser> & values) {
        auto end = p.peek(p.choice({p.literal(","), p.literal("}"), p.literal("]"), p.literal(" "), p.literal("\n"), p.literal("\t")}));
        auto choice = p.choice();
        for (const auto & value : values) {
            choice |= value + end;
        }
        return p.rule("gemma4-value--" + path(), choice);
    }

    common_peg_parser array(const common_chat_schema_array & array) {
        if (array.items->kind() == common_chat_schema::KIND_ANY && array.min_items == 0 && array.max_items < 0) {
            return p.ref("gemma4-array");
        }
        auto item     = visit(*array.items);
        auto elements = p.eps();
        if (array.max_items != 0) {
            auto rest = p.repeat(p.literal(",") + p.space() + item, std::max(array.min_items - 1, 0),
                                 array.max_items < 0 ? -1 : array.max_items - 1);
            elements = array.min_items > 0 ? item + rest : p.optional(item + rest);
        }
        return p.rule("gemma4-array--" + path(), p.literal("[") + p.space() + elements + p.space() + p.literal("]"));
    }

    common_peg_parser object(const common_chat_schema_object & object) {
        const auto * additional = object.additional();
        if (object.properties.empty() && additional && additional->kind() == common_chat_schema::KIND_ANY) {
            return p.ref("gemma4-dict");
        }
        if (additional) {
            // Declared and undeclared members in one sorted dictionary.
            unenforced(object, "additionalProperties");
            return p.ref("gemma4-dict");
        }
        std::vector<const common_chat_schema_property *> properties;
        for (const auto & property : object.properties) {
            if (property.name.empty() || property.name.find_first_of(":}") != std::string::npos) {
                // A dictionary key is bare text without ':' or '}'.
                templates_native::relax(tool, *property.schema, "properties",
                                        common_chat_schema_relaxation::REASON_UNREPRESENTABLE);
                continue;
            }
            properties.push_back(&property);
        }
        // `dictsort` orders keys case-insensitively.
        auto folded = [](const std::string & text) {
            std::string result = text;
            std::transform(result.begin(), result.end(), result.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return result;
        };
        std::stable_sort(properties.begin(), properties.end(), [&](const auto * a, const auto * b) {
            return folded(a->name) < folded(b->name);
        });
        // Built from the last key: `first` holds the members from here on when
        // none precedes them, `later` the members after one that did.
        auto first = p.eps();
        auto later = p.eps();
        for (auto it = properties.rbegin(); it != properties.rend(); ++it) {
            const auto & property = **it;
            auto at     = path();
            auto key    = p.rule("gemma4-dict-key--" + at,
                                 p.rule("gemma4-dict-key-name--" + at, p.literal(property.name)) + p.literal(":"));
            auto member = p.rule("gemma4-dict-kv--" + at, key + p.space() + visit(*property.schema));
            auto after  = p.literal(",") + p.space() + member;
            if (property.required) {
                first = member + later;
                later = after + later;
            } else {
                first = p.choice({member + later, first});
                later = p.optional(after) + later;
            }
        }
        return p.rule("gemma4-dict--" + path(), p.literal("{") + p.space() + first + p.space() + p.literal("}"));
    }
};

}

common_chat_params common_chat_params_init_gemma4(const common_chat_template &    tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);

    if (inputs.add_generation_prompt && string_ends_with(data.prompt, "<turn|>\n")) {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.generation_prompt = "<|turn>model\n";
        data.prompt += data.generation_prompt;
    }

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_USER,      "<|turn>user"  },
        { COMMON_CHAT_ROLE_ASSISTANT, "<|turn>model" },
    };

    data.format            = COMMON_CHAT_FORMAT_PEG_GEMMA4;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<|channel>thought";
    data.thinking_end_tags  = {"<channel|>"};

    data.preserved_tokens = {
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|turn>",
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = string_ends_with(data.prompt, "<turn|>\n") ? "<|turn>model\n" : "";
        data.generation_prompt += "<|channel>thought\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "<channel|>" + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.optional(p.literal("<|turn>model\n")));

        if (extract_reasoning) {
            p.rule("thought", p.literal("<|channel>thought") + p.space() + p.reasoning(p.until("<channel|>")) + p.literal("<channel|>"));
        } else {
            p.rule("thought", p.content(p.literal("<|channel>thought") + p.space() + p.until("<channel|>") + p.literal("<channel|>")));
        }

        auto consume_empty_channels = p.gbnf(p.zero_or_more(p.literal("<|channel>") + p.negate(p.literal("thought"))), "");
        auto thought = (p.peek(p.literal("<|channel>")) + consume_empty_channels + p.ref("thought")) | p.negate(p.literal("<|channel>"));

        if (has_response_format) {
            auto response_format = p.literal("```json") <<
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) <<
                p.literal("```");
            return start + p.optional(thought) + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Gemma4 tool calling syntax
            // Rules should match traversal logic in gemma4_to_json()
            p.rule("gemma4-string-content", p.until("<|\"|>"));
            p.rule("gemma4-string", p.literal("<|\"|>") + p.ref("gemma4-string-content") + p.literal("<|\"|>"));
            p.rule("gemma4-bool", p.json_bool());
            p.rule("gemma4-null", p.json_null());
            p.rule("gemma4-number", p.json_number());
            p.rule("gemma4-dict-key", p.rule("gemma4-dict-key-name", p.chars("[^:}]", 1, -1)) + p.literal(":"));
            p.rule("gemma4-dict-kv", p.ref("gemma4-dict-key") + p.space() + p.ref("gemma4-value"));
            p.rule("gemma4-dict", [&]() {
                auto ws = p.space();
                auto member = p.ref("gemma4-dict-kv");
                auto members = p.sequence({member, p.zero_or_more(p.sequence({p.literal(","), ws, member}))});
                return p.sequence({
                    p.literal("{"), ws,
                    p.choice({p.literal("}"), p.sequence({members, ws, p.literal("}")})})
                });
            });
            p.rule("gemma4-array", [&]() {
                auto ws = p.space();
                auto value = p.ref("gemma4-value");
                auto elements = p.sequence({value, p.zero_or_more(p.sequence({p.literal(","), ws, value}))});
                return p.sequence({
                    p.literal("["), ws,
                    p.choice({p.literal("]"), p.sequence({elements, ws, p.literal("]")})})
                });
            });
            p.rule("gemma4-value", [&]() {
                return p.choice({
                    p.ref("gemma4-string"), p.ref("gemma4-dict"), p.ref("gemma4-array"),
                    p.ref("gemma4-number"), p.ref("gemma4-bool"), p.ref("gemma4-null")
                });
            });

            auto tool_choice = p.choice();

            size_t index = 0;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto   schema   = common_chat_schema_from_json(common_chat_tool_parameters(function));
                gemma4_argument_rules arguments(p, "tool-" + std::to_string(index++), name);

                tool_choice |= p.rule("tool-" + name, p.tool(p.sequence({
                    p.tool_open(p.tool_name(p.literal(name)) + p.peek(p.literal("{"))),
                    p.tool_args(arguments.visit(*schema.root)),
                })));
            });

            auto tool_call = p.trigger_rule("tool-call", p.repeat(
                "<|tool_call>call:" + tool_choice + "<tool_call|>",
                /* min = */ inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0,
                /* max = */ inputs.parallel_tool_calls ? -1 : 1
            ));

            auto scan_to_toolcall = p.rule("scan-to-toolcall", p.until("<|tool_call>"));
            auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>", "<|tool_call>"})));
            auto message = p.rule("message", thought + content);
            return start + p.zero_or_more(message) + scan_to_toolcall + tool_call;
        }

        // Gemma 4 may emit an extra <|channel>thought\n<channel|> at the end of the content. It may
        // also emit a single trailing <channel|> token. Consume all complete reasoning blocks and
        // then stop at the first unmatched <channel|> token.
        auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>"})));
        auto message = p.rule("message", thought + content);
        return start + p.one_or_more(message);
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = false;  // Enforce the whole completion, including trigger bytes.
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call>" },
        };
    }

    return data;
}
