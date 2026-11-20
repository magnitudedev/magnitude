#include "value.h"
#include "runtime.h"
#include "caps.h"

// note: the json dependency is only for defining input in a convenient way
// we can remove it in the future when we figure out a better way to define inputs using jinja::value
#include "json.h"

#include <functional>
#include <sstream>

#define FILENAME "jinja-caps"

using json = common_json;

namespace jinja {

using caps_ctx_fn = std::function<void(context &)>;
using caps_analyze_fn = std::function<void(context &, bool, value &, value &, const std::string &)>;

void caps_apply_preserve_reasoning(jinja::context & ctx, bool enabled) {
    ctx.set_val("preserve_thinking",         mk_val<value_bool>(enabled));
    ctx.set_val("clear_thinking",            mk_val<value_bool>(!enabled));
    ctx.set_val("truncate_history_thinking", mk_val<value_bool>(!enabled));
    ctx.set_val("drop_thinking",             mk_val<value_bool>(!enabled));
}

void caps_apply_reasoning_effort(jinja::context & ctx, const std::string & effort) {
    value var = mk_val<value_string>(effort); // bind to the same value for stats
    ctx.set_val("reasoning_effort",   var);
    ctx.set_val("reasoning_strength", var);
}

static void caps_try_execute(jinja::program & prog,
                             const json & input,
                             const caps_ctx_fn & ctx_fn,
                             const caps_analyze_fn & analyze_fn,
                             bool track_stats = true) {
    context ctx;
    ctx.current_time = 946684800; // Frozen capability inspection context.
    ctx.is_get_stats = track_stats;
    jinja::global_from_json(ctx, input, track_stats);

    if (ctx_fn) {
        ctx_fn(ctx);
    }

    auto messages = ctx.get_val("messages");
    auto tools = ctx.get_val("tools");

    bool success = false;
    std::string result;
    try {
        jinja::runtime runtime(ctx);
        auto results = runtime.execute(prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        result = parts->as_string().str();
        success = true;
    } catch (const std::exception & e) {
        JJ_DEBUG("Exception during execution: %s", e.what());
        result = "";
        // ignore exceptions during capability analysis
    }

    analyze_fn(ctx, success, messages, tools, result);
}

// for debugging only
static void caps_print_stats(value & v, const std::string & path) {
    if (!g_jinja_debug) {
        return;
    }
    std::string ops;
    if (v->stats.has_op(value_t::stats_t::OP_TEST_IS_STRING)) ops += "test_is_string ";
    if (v->stats.has_op(value_t::stats_t::OP_SELECTATTR))     ops += "selectattr ";
    if (v->stats.has_op(value_t::stats_t::OP_ARRAY_ACCESS))   ops += "array_access ";
    JJ_DEBUG("Value %s, type: %s %s, ops: %s",
                path.c_str(),
                v->type().c_str(),
                v->stats.used ? "(used)" : "",
                ops.c_str());
}

std::map<std::string, bool> caps::to_map() const {
    return {
        {"supports_string_content", supports_string_content},
        {"supports_typed_content", supports_typed_content},
        {"supports_tools", supports_tools},
        {"supports_tool_calls", supports_tool_calls},
        {"supports_parallel_tool_calls", supports_parallel_tool_calls},
        {"supports_system_role", supports_system_role},
        {"supports_preserve_reasoning", supports_preserve_reasoning},
        {"supports_reasoning_effort", supports_reasoning_effort},
        {"supports_object_arguments", supports_object_arguments},
    };
}

std::string caps::to_string() const {
    std::ostringstream ss;
    ss << "Caps(\n";
    for (const auto & [key, value] : to_map()) {
        ss << "  " << key << "=" << (value ? "true" : "false") << "\n";
    }
    ss << ")";
    return ss.str();
}

static json make_caps_input(json messages, json tools = json::array()) {
    return json{
        {"messages", std::move(messages)},
        {"tools", std::move(tools)},
        {"bos_token", ""},
        {"eos_token", ""},
        {"add_generation_prompt", true}
    };
}

static const json & sample_tools() {
    static const json tools = json::array({
        {
            {"name", "tool"},
            {"type", "function"},
            {"function", {
                {"name", "tool1"},
                {"description", "Tool description"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"arg", {
                            {"type", "string"},
                            {"description", "Arg description"},
                        }},
                    }},
                    {"required", json::array({ "arg" })},
                }},
            }},
        },
    });
    return tools;
}

caps caps_get(jinja::program & prog, const std::string & src) {
    caps result;

    const bool has_src = !src.empty();
    const auto src_contains = [&](const char * needle) {
        return !has_src || src.find(needle) != std::string::npos;
    };

    JJ_DEBUG("%s\n", ">>> Running capability check: typed content");

    bool checks_for_string = false;
    static const std::string content_marker = "STRING_MARKER";

    static const json input_string_content = make_caps_input(json::array({
        {
            {"role", "user"},
            {"content", content_marker}
        }
    }));

    // case: typed content support
    caps_try_execute(
        prog,
        input_string_content,
        nullptr, // ctx_fn
        [&](context &, bool success, value & messages, value &, const std::string & rendered) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (content->stats.has_op(value_t::stats_t::OP_TEST_IS_STRING)) {
                // checked if content is string
                checks_for_string = true;
            }
            bool used_as_array = content->stats.has_op(value_t::stats_t::OP_SELECTATTR) ||
                                 content->stats.has_op(value_t::stats_t::OP_ARRAY_ACCESS);
            if (used_as_array) {
                // accessed as an array
                result.supports_typed_content = true;
            }
            if (!success) {
                // failed to execute with content as string
                result.supports_string_content = false;
            } else if (used_as_array && rendered.find(content_marker) == std::string::npos) {
                // edge case: string may be accessed for checking, but does not appear in the output
                result.supports_string_content = false;
            }
        }
    );

    if (checks_for_string && !result.supports_typed_content) {
        static const json input_typed_content = make_caps_input(json::array({
            {
                {"role", "user"},
                {"content", json::array({})}
            }
        }));
        caps_try_execute(
            prog,
            input_typed_content,
            nullptr, // ctx_fn
            [&](context &, bool success, value & messages, value &, const std::string &) {
                auto & content = messages->at(0)->at("content");
                caps_print_stats(content, "messages[0].content");
                bool used_as_array = content->stats.has_op(value_t::stats_t::OP_SELECTATTR) ||
                                     content->stats.has_op(value_t::stats_t::OP_ARRAY_ACCESS);
                if (used_as_array && success) {
                    // accessed as an array
                    result.supports_typed_content = true;
                }
            }
        );
    }

    JJ_DEBUG("%s\n", ">>> Running capability check: system prompt");

    static const json input_system = make_caps_input(json::array({
        {
            {"role", "system"},
            {"content", "System message"}
        },
        {
            {"role", "user"},
            {"content", "User message"}
        },
    }));

    // case: system prompt support
    caps_try_execute(
        prog,
        input_system,
        nullptr, // ctx_fn
        [&](context &, bool, value & messages, value &, const std::string &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (!content->stats.used) {
                result.supports_system_role = false;
            }
        }
    );

    const bool may_use_tools = src_contains("tools") || src_contains("tool_calls") || src_contains("tojson");
    if (!may_use_tools) {
        result.supports_tools = false;
        result.supports_tool_calls = false;
        result.supports_parallel_tool_calls = false;
        result.supports_object_arguments = false;
    } else {
        JJ_DEBUG("%s\n", ">>> Running capability check: single tool with object arguments support");

        bool tool_probe_succeeded = false;

        static const json input_tool_obj = make_caps_input(json::array({
            {
                {"role", "user"},
                {"content", "User message"},
            },
            {
                {"role", "assistant"},
                {"content", ""}, // Some templates expect content to be empty with tool calls
                {"tool_calls", json::array({
                    {
                        {"id", "call00001"},
                        {"type", "function"},
                        {"function", {
                            {"name", "tool1"},
                            {"arguments", {
                                {"arg", "value"}
                            }}
                        }}
                    }
                })}
            },
            {
                {"role", "tool"},
                {"content", "Tool response"},
                {"tool_call_id", "call00001"}
            },
            {
                {"role", "assistant"},
                {"content", "The tool response was 'tool response'"}
            },
            {
                {"role", "user"},
                {"content", "User message"},
            },
        }), sample_tools());

        // case: tools support: single call with object arguments
        caps_try_execute(
            prog,
            input_tool_obj,
            nullptr, // ctx_fn
            [&](context &, bool success, value & messages, value & tools, const std::string &) {
                if (!success) {
                    return; // Nothing can be inferred
                }
                tool_probe_succeeded = true;

                auto & tool_name = tools->at(0)->at("function")->at("name");
                caps_print_stats(tool_name, "tools[0].function.name");
                caps_print_stats(tools, "tools");
                if (!tool_name->stats.used) {
                    result.supports_tools = false;
                }

                auto & tool_calls = messages->at(1)->at("tool_calls");;
                caps_print_stats(tool_calls, "messages[1].tool_calls");
                if (!tool_calls->stats.used) {
                    result.supports_tool_calls = false;
                    return;
                }

                auto & tool_arg = tool_calls->at(0)->at("function")->at("arguments")->at("arg");
                caps_print_stats(tool_arg, "messages[1].tool_calls[0].function.arguments.arg");
                if (tool_arg->stats.used) {
                    result.supports_object_arguments = true;
                }
            }
        );

        if (!result.supports_object_arguments && !(tool_probe_succeeded && !result.supports_tool_calls)) {
            JJ_DEBUG("%s\n", ">>> Running capability check: single tool with string arguments support");

            static const json input_tool_str = make_caps_input(json::array({
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
                {
                    {"role", "assistant"},
                    {"content", ""}, // Some templates expect content to be empty with tool calls
                    {"tool_calls", json::array({
                        {
                            {"id", "call00001"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool1"},
                                {"arguments", R"({"arg": "value"})"}
                            }}
                        }
                    })}
                },
                {
                    {"role", "tool"},
                    {"content", "Tool response"},
                    {"tool_call_id", "call00001"}
                },
                {
                    {"role", "assistant"},
                    {"content", "The tool response was 'tool response'"}
                },
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
            }), sample_tools());

            // case: tools support: single call with string arguments
            caps_try_execute(
                prog,
                input_tool_str,
                nullptr, // ctx_fn
                [&](context &, bool success, value & messages, value & tools, const std::string &) {
                    if (!success) {
                        result.supports_tool_calls = false;
                        result.supports_tools = false;
                        return;
                    }

                    auto & tool_name = tools->at(0)->at("function")->at("name");
                    caps_print_stats(tool_name, "tools[0].function.name");
                    caps_print_stats(tools, "tools");
                    if (!tool_name->stats.used) {
                        result.supports_tools = false;
                    }

                    auto & tool_calls = messages->at(1)->at("tool_calls");
                    caps_print_stats(tool_calls, "messages[1].tool_calls");
                    if (!tool_calls->stats.used) {
                        result.supports_tool_calls = false;
                        return;
                    }
                }
            );
        }

        if (!result.supports_tool_calls) {
            result.supports_parallel_tool_calls = false;
        } else {
            JJ_DEBUG("%s\n", ">>> Running capability check: parallel tool support");

            auto make_parallel_input = [](const json & args) {
                return make_caps_input(json::array({
                    {
                        {"role", "user"},
                        {"content", "User message"},
                    },
                    {
                        {"role", "assistant"},
                        {"content", ""}, // Some templates expect content to be empty with tool calls
                        {"tool_calls", json::array({
                            {
                                {"id", "call00001"},
                                {"type", "function"},
                                {"function", {
                                    {"name", "tool1"},
                                    {"arguments", args}
                                }}
                            },
                            {
                                {"id", "call00002"},
                                {"type", "function"},
                                {"function", {
                                    {"name", "tool1"},
                                    {"arguments", args}
                                }}
                            }
                        })}
                    },
                    {
                        {"role", "tool"},
                        {"content", "Tool response"},
                        {"tool_call_id", "call00001"}
                    },
                    {
                        {"role", "assistant"},
                        {"content", "The tool response was 'tool response'"}
                    },
                    {
                        {"role", "user"},
                        {"content", "User message"},
                    },
                }), sample_tools());
            };
            static const json input_parallel_obj = make_parallel_input(json{{"arg", "value"}});
            static const json input_parallel_str = make_parallel_input(json(R"({"arg": "value"})"));

            // case: tools support: parallel calls
            caps_try_execute(
                prog,
                result.supports_object_arguments ? input_parallel_obj : input_parallel_str,
                nullptr, // ctx_fn
                [&](context &, bool success, value & messages, value &, const std::string &) {
                    if (!success) {
                        result.supports_parallel_tool_calls = false;
                        return;
                    }

                    auto & tool_calls = messages->at(1)->at("tool_calls");
                    caps_print_stats(tool_calls, "messages[1].tool_calls");

                    // check for second tool call usage
                    auto & tool_call_1 = tool_calls->at(1)->at("function");
                    caps_print_stats(tool_call_1, "messages[1].tool_calls[1].function");
                    if (!tool_call_1->stats.used) {
                        result.supports_parallel_tool_calls = false;
                    }
                }
            );
        }
    }

    if (src_contains("reasoning_content")) {
        JJ_DEBUG("%s\n", ">>> Running capability check: preserve reasoning");

        // case: preserve reasoning content in chat history
        static const std::string reasoning_placeholder = "<REASONING_CONTENT_PLACEHOLDER>";
        static const json input_preserve_reasoning = make_caps_input(json::array({
            {
                {"role", "user"},
                {"content", "User message"}
            },
            {
                {"role", "assistant"},
                {"content", "Assistant message"},
                // check of reasoning_content deeper in the history, not just the last assistant message
                {"reasoning_content", reasoning_placeholder}
            },
            {
                {"role", "user"},
                {"content", "User message"}
            },
            {
                {"role", "assistant"},
                {"content", "Assistant message"},
                {"reasoning_content", "Reasoning content"}
            },
            {
                {"role", "user"},
                {"content", "User message"}
            },
        }));
        caps_try_execute(
            prog,
            input_preserve_reasoning,
            [&](context & ctx) {
                ctx.set_val("enable_thinking", mk_val<value_bool>(true));
                caps_apply_preserve_reasoning(ctx, true);
            },
            [&](context &, bool, value &, value &, const std::string & output) {
                // note: we cannot use stats here because the reasoning_content may be used for "if" condition test, but not actually outputted in the final result
                if (output.find(reasoning_placeholder) != std::string::npos) {
                    result.supports_preserve_reasoning = true;
                }
            },
            /* track_stats = */ false
        );
    }

    if (src_contains("reasoning_effort") || src_contains("reasoning_strength")) {
        JJ_DEBUG("%s\n", ">>> Running capability check: reasoning effort");

        static const json input_reasoning_effort = make_caps_input(json::array({
            {
                {"role", "user"},
                {"content", "User message"}
            },
        }));

        // case: reasoning effort level
        caps_try_execute(
            prog,
            input_reasoning_effort,
            [&](context & ctx) {
                ctx.set_val("enable_thinking", mk_val<value_bool>(true));
                caps_apply_reasoning_effort(ctx, "low");
            },
            [&](context & ctx, bool, value &, value &, const std::string &) {
                value effort = ctx.get_val("reasoning_effort");
                caps_print_stats(effort, "reasoning_effort");
                result.supports_reasoning_effort = effort->stats.used;
            }
        );
    }

    JJ_DEBUG("%s\n", result.to_string().c_str());

    return result;
}

} // namespace jinja
