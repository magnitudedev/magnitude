#include "parsers.h"

// Inkling 2 (Thinking Machines) format:
//
// Each assistant turn is a sequence of role-prefixed messages terminated by
// <|content_model_end_sampling|> (the model's EOS token):
//   - reasoning: <|message_model|><|content_thinking|>{reasoning}<|end_message|>
//   - content:   <|message_model|><|content_text|>{content}<|end_message|>
//   - tool call: <|message_model|>{name}<|content_invoke_tool_json|>{"name":"{name}","args":{...}}<|end_message|>
//
// The official chat_template.jinja does not append a generation prompt; the
// model emits <|message_model|> itself unless reasoning is disabled (in which
// case we set reasoning_effort="none" and prefill <|message_model|><|content_text|>).
common_chat_params common_chat_params_init_inkling2(const common_chat_template &          tmpl,
                                                    const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string MODEL_ROLE   = "<|message_model|>";
    const std::string USER_ROLE    = "<|message_user|>";
    const std::string SYSTEM_ROLE  = "<|message_system|>";
    const std::string TOOL_ROLE    = "<|message_tool|>";
    const std::string THINKING     = "<|content_thinking|>";
    const std::string TEXT         = "<|content_text|>";
    const std::string INVOKE_TOOL  = "<|content_invoke_tool_json|>";
    const std::string CONTENT_IMG  = "<|content_image|>";
    const std::string CONTENT_XML  = "<|content_xml|>";
    const std::string END_MSG      = "<|end_message|>";
    const std::string END_SAMPLING = "<|content_model_end_sampling|>";

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    // Convert internal `media_marker` image parts (whether preserved as typed
    // parts or flattened into a string by `messages_inp_normalizer`) into typed
    // `{"type": "image"}` + `{"type": "text"}` parts so the template renders
    // `<|content_image|><|unused_200054|><|end_message|>` as its own image
    // message rather than inside `<|content_text|>`, and ensure historical
    // tool call arguments are parsed JSON objects (the template rejects strings).
    const std::string IMG_PLACEHOLDER = CONTENT_IMG + "<|unused_200054|>" + END_MSG;
    auto adjusted_messages = json::array();
    for (auto msg : inputs.messages) {
        const std::string role = msg.value("role", "");
        if ((role == "user" || role == "tool") && msg.contains("content")) {
            if (msg.at("content").is_array()) {
                auto parts = json::array();
                for (const auto & part : msg.at("content")) {
                    if (part.is_object() && part.value("type", "") == "media_marker") {
                        parts.push_back(json{ { "type", "image" } });
                    } else {
                        parts.push_back(part);
                    }
                }
                msg["content"] = std::move(parts);
            } else if (msg.at("content").is_string()) {
                const std::string text = msg.at("content").get<std::string>();
                if (text.find(IMG_PLACEHOLDER) != std::string::npos) {
                    auto parts = json::array();
                    size_t pos = 0;
                    while (pos < text.size()) {
                        size_t found = text.find(IMG_PLACEHOLDER, pos);
                        if (found == std::string::npos) {
                            parts.push_back(json{ { "type", "text" }, { "text", text.substr(pos) } });
                            break;
                        }
                        if (found > pos) {
                            parts.push_back(json{ { "type", "text" }, { "text", text.substr(pos, found - pos) } });
                        }
                        parts.push_back(json{ { "type", "image" } });
                        pos = found + IMG_PLACEHOLDER.size();
                    }
                    msg["content"] = std::move(parts);
                }
            }
        }
        if (msg.contains("tool_calls") && msg.at("tool_calls").is_array()) {
            for (auto & tc : msg["tool_calls"]) {
                if (tc.contains("function") && tc["function"].contains("arguments")) {
                    auto & args = tc["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            args = json::parse(args.get<std::string>());
                        } catch (const std::exception & e) {
                            throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                        }
                    }
                }
            }
        }
        adjusted_messages.push_back(std::move(msg));
    }

    auto reasoning_disabled = inputs.enable_thinking.has_value() && !*inputs.enable_thinking;
    auto names_effort       = inputs.extra_context.is_object() &&
                              inputs.extra_context.contains("reasoning_effort");
    std::optional<json> additional_context;
    if (reasoning_disabled && !names_effort) {
        additional_context = json{ { "reasoning_effort", "none" } };
    }

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs, adjusted_messages, std::nullopt, additional_context);
    data.generation_prompt  = "";
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = MODEL_ROLE + THINKING;
    data.thinking_end_tags  = { END_MSG };
    data.additional_stops   = { END_SAMPLING };

    data.preserved_tokens = {
        MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, TOOL_ROLE,
        THINKING, TEXT, INVOKE_TOOL, CONTENT_IMG, CONTENT_XML,
        END_MSG, END_SAMPLING,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, MODEL_ROLE  },
        { COMMON_CHAT_ROLE_USER,      USER_ROLE   },
        { COMMON_CHAT_ROLE_SYSTEM,    SYSTEM_ROLE },
        { COMMON_CHAT_ROLE_TOOL,      TOOL_ROLE   },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = MODEL_ROLE + THINKING + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += END_MSG + MODEL_ROLE + TEXT + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    } else if (reasoning_disabled && !include_grammar) {
        data.generation_prompt = MODEL_ROLE + TEXT;
        data.prompt           += data.generation_prompt;
    }

    common_chat_apply_cached_parser(tmpl, inputs, "", data, [&](common_chat_params & data) {
        auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
            auto end_turn = p.optional(p.literal(END_SAMPLING)) + p.end();

            auto reasoning_text = extract_reasoning
                ? p.reasoning(p.until(END_MSG))
                : p.content(p.until(END_MSG));
            auto thinking = extract_reasoning
                ? p.rule("thinking", p.literal(MODEL_ROLE + THINKING) + reasoning_text + p.literal(END_MSG))
                : p.rule("thinking", p.content(p.literal(MODEL_ROLE + THINKING) + p.until(END_MSG) + p.literal(END_MSG)));

            auto unclosed_thinking = p.literal(MODEL_ROLE + THINKING) + reasoning_text + p.end();

            if (has_response_format) {
                auto response_text = p.rule("response-format",
                    p.literal(MODEL_ROLE + TEXT) +
                    p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                    p.literal(END_MSG));
                return p.zero_or_more(thinking) + response_text + end_turn;
            }

            auto final_msg = p.rule("final",
                p.literal(MODEL_ROLE + TEXT) + p.content(p.until(END_MSG)));

            if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
                auto tool_choice = p.choice();
                foreach_function(inputs.tools, [&](const json & tool) {
                    const auto &      function  = tool.at("function");
                    const std::string name      = function.at("name");
                    const auto        params    = common_chat_tool_parameters(function);
                    const std::string name_json = json(name).dump();

                    auto tool_parser = p.tool(
                        p.tool_open(
                            p.literal(MODEL_ROLE) + p.tool_name(p.literal(name)) + p.literal(INVOKE_TOOL) +
                            p.literal("{") + p.space() +
                            p.literal("\"name\"") + p.space() + p.literal(":") + p.space() +
                            p.literal(name_json) + p.space() + p.literal(",") + p.space() +
                            p.literal("\"args\"") + p.space() + p.literal(":") + p.space()) +
                        p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", params)) +
                        p.tool_close(p.space() + p.literal("}") + p.literal(END_MSG)));

                    tool_choice |= p.rule("tool-" + name, tool_parser);
                });

                auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
                auto tool_calls = p.trigger_rule("tool-call", p.repeat(tool_choice, 1, max_calls));

                if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                    return p.zero_or_more(thinking) + tool_calls + end_turn;
                }

                auto text_and_opt_calls = final_msg + p.optional(p.literal(END_MSG) + p.optional(tool_calls)) + end_turn;
                if (extract_reasoning) {
                    return p.zero_or_more(thinking) + ((tool_calls + end_turn) | text_and_opt_calls | unclosed_thinking);
                }
                return p.zero_or_more(thinking) + ((tool_calls + end_turn) | text_and_opt_calls);
            }

            auto closed_final = p.one_or_more(final_msg + p.optional(p.literal(END_MSG))) + end_turn;
            if (extract_reasoning) {
                return p.zero_or_more(thinking) + (closed_final | unclosed_thinking);
            }
            return p.zero_or_more(thinking) + closed_final;
        });

        data.parser = parser.save();

        if (include_grammar) {
            data.grammar_lazy = false;
            data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
                parser.build_grammar(builder, data.grammar_lazy);
            });
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, INVOKE_TOOL },
            };
        }
    });

    return data;
}
