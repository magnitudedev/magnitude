#include "peg-parser.h"
#include <cassert>
#include <iostream>
#include <tuple>

using capture = std::tuple<std::string, size_t, size_t, bool>;
static std::vector<capture> captures(const common_peg_parse_context & ctx, const common_peg_parse_result & result) {
    std::vector<capture> found;
    ctx.ast.visit(result, [&](const common_peg_ast_node & node) {
        if (!node.tag.empty()) { found.emplace_back(node.tag, node.start, node.end, node.is_partial); }
    });
    return found;
}

static void compare(const common_peg_arena & parser, common_peg_parse_context & cached) {
    common_peg_parse_context fresh(cached.input, cached.flags);
    const auto expected = parser.parse(fresh);
    const auto actual = parser.parse(cached);
    assert(actual.type == expected.type);
    assert(actual.start == expected.start && actual.end == expected.end);
    assert(captures(cached, actual) == captures(fresh, expected));
    if (!cached.tags_only) {
        using full_node = std::tuple<std::string, std::string, size_t, size_t, std::string, bool, size_t>;
        auto tree = [](const auto & ctx, const auto & result) {
            std::vector<full_node> nodes;
            ctx.ast.visit(result, [&](const auto & node) {
                nodes.emplace_back(node.rule, node.tag, node.start, node.end,
                                   std::string(node.text), node.is_partial, node.children.size());
            });
            return nodes;
        };
        assert(tree(cached, actual) == tree(fresh, expected));
    }
    assert(cached.scans.size() <= 65536);
    assert(cached.completed.size() <= 65536);
    assert(cached.retained_nodes <= cached.max_retained_nodes);
}

static void check(const common_peg_arena & parser, const std::string & input, bool tags_only) {
    common_peg_parse_context cached(COMMON_PEG_PARSE_FLAG_LENIENT);
    cached.append_only = true;
    cached.tags_only = tags_only;
    for (size_t i = 0; i <= input.size(); ++i) {
        if (i) { cached.input.push_back(input[i - 1]); }
        compare(parser, cached);
        // Natural termination may follow any prefix. It never precedes append
        // on the same stream, so give finalization its own context copy.
        auto final = cached;
        final.flags = COMMON_PEG_PARSE_FLAG_NONE;
        compare(parser, final);
    }
}

int main() {
    std::vector<common_peg_arena> parsers;
    parsers.push_back(build_peg_parser([](auto & p) {
        return p.tag("body", p.zero_or_more(p.literal("ab") | p.literal("a") | p.literal("b"))) + p.end();
    }));
    parsers.push_back(build_peg_parser([](auto & p) {
        return p.tag("body", p.zero_or_more(p.negate(p.literal("ab")) + p.any())) + p.end();
    }));
    parsers.push_back(build_peg_parser([](auto & p) {
        return p.zero_or_more(p.tag("item", p.literal("a") | p.literal("b"))) + p.end();
    }));
    parsers.push_back(build_peg_parser([](auto & p) {
        return p.tag("body", p.zero_or_more(p.until("ab") + p.literal("ab"))) + p.end();
    }));
    parsers.push_back(build_peg_parser([](auto & p) {
        return p.tag("body", p.zero_or_more((p.peek(p.end()) | p.literal("a")) + p.any())) + p.end();
    }));
    for (const auto & parser : parsers) {
        for (unsigned bits = 0; bits < 256; ++bits) {
            std::string input;
            for (unsigned i = 0; i < 8; ++i) { input += (bits & (1u << i)) ? 'a' : 'b'; }
            check(parser, input, false);
            check(parser, input, true);
        }
    }
    const auto json = build_peg_parser([](auto & p) { return p.tag("json", p.json()) + p.end(); });
    for (const std::string input : {
        R"({"x":[1,12,-1.5e+2,true,false,null,{"y":"a\u1234\"\\b"}],"z":"😀"})",
        R"([{"a":[1,2,3]}, {"b":[4,5,6]}])", R"([1e+])", R"({"x":"\u12x4"})",
        "{\"x\": \t\r\n  1   , \"y\":  2  }"}) {
        check(json, input, false);
        check(json, input, true);
    }
    // Saturation must retain the outer repetition's stable prefix, and pointers
    // held across child parsing must stay valid. Compare the completed AST too.
    common_peg_parse_context large(COMMON_PEG_PARSE_FLAG_LENIENT);
    large.append_only = true;
    large.tags_only = true;
    large.input = "[0";
    for (size_t i = 0; i < 18000; ++i) {
        large.input += ", {\"a\":[1,2,3]}";
        json.parse(large);
    }
    large.input += "]";
    assert(large.scans.size() == 65536);
    large.flags = COMMON_PEG_PARSE_FLAG_NONE;
    compare(json, large);
    // Copying a context and reallocating input must rebase all retained views.
    auto copied = large;
    copied.input.reserve(copied.input.capacity() * 2);
    compare(json, copied);
    copied.ast.clear();
    compare(json, copied);
    std::cout << "Append-only PEG matches fresh parsing, including cache saturation\n";
}
