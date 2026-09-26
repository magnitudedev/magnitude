#pragma once

#include "json-schema.h"
#include "json.h"
#include "output-text.h"
#include <tuple>

#include <memory>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <variant>

struct common_grammar_builder;

class common_peg_parser_builder;

using common_peg_parser_id = size_t;
constexpr common_peg_parser_id COMMON_PEG_INVALID_PARSER_ID = static_cast<common_peg_parser_id>(-1);

using common_peg_ast_id = size_t;
constexpr common_peg_ast_id COMMON_PEG_INVALID_AST_ID = static_cast<common_peg_ast_id>(-1);

// Lightweight wrapper around common_peg_parser_id for convenience
class common_peg_parser {
    common_peg_parser_id id_;
    common_peg_parser_builder & builder_;

  public:
    common_peg_parser(const common_peg_parser & other) : id_(other.id_), builder_(other.builder_) {}
    common_peg_parser(common_peg_parser_id id, common_peg_parser_builder & builder) : id_(id), builder_(builder) {}

    common_peg_parser & operator=(const common_peg_parser & other);
    common_peg_parser & operator+=(const common_peg_parser & other);
    common_peg_parser & operator|=(const common_peg_parser & other);

    operator common_peg_parser_id() const { return id_; }
    common_peg_parser_id id() const { return id_; }

    common_peg_parser_builder & builder() const { return builder_; }

    // Creates a sequence
    common_peg_parser operator+(const common_peg_parser & other) const;

    // Creates a sequence separated by spaces.
    common_peg_parser operator<<(const common_peg_parser & other) const;

    // Creates a choice
    common_peg_parser operator|(const common_peg_parser & other) const;

    common_peg_parser operator+(const char * str) const;
    common_peg_parser operator+(const std::string & str) const;
    common_peg_parser operator<<(const char * str) const;
    common_peg_parser operator<<(const std::string & str) const;
    common_peg_parser operator|(const char * str) const;
    common_peg_parser operator|(const std::string & str) const;
};

common_peg_parser operator+(const char * str, const common_peg_parser & p);
common_peg_parser operator+(const std::string & str, const common_peg_parser & p);
common_peg_parser operator<<(const char * str, const common_peg_parser & p);
common_peg_parser operator<<(const std::string & str, const common_peg_parser & p);
common_peg_parser operator|(const char * str, const common_peg_parser & p);
common_peg_parser operator|(const std::string & str, const common_peg_parser & p);

enum common_peg_parse_result_type {
    COMMON_PEG_PARSE_RESULT_FAIL            = 0,
    COMMON_PEG_PARSE_RESULT_SUCCESS         = 1,
    COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT = 2,
};

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type);

struct common_peg_ast_node {
    common_peg_ast_id id;
    std::string rule;
    std::string tag;
    size_t start;
    size_t end;
    std::string_view text;
    std::vector<common_peg_ast_id> children;

    bool is_partial = false;
    bool contains_tags = false;
};

struct common_peg_parse_result;

using common_peg_ast_visitor = std::function<void(const common_peg_ast_node & node)>;

class common_peg_ast_arena {
    std::vector<common_peg_ast_node> nodes_;
    mutable std::map<std::pair<size_t, size_t>, common_peg_text> output_cache;
    struct output_prefix {
        std::vector<common_peg_ast_id> children;
        common_peg_text text;
    };
    mutable std::map<std::tuple<size_t, std::string, std::string, size_t>, output_prefix> output_prefixes;
    size_t stable_nodes = 0;
    std::string_view source_input;

  public:
    common_peg_ast_id add_node(
        const std::string & rule,
        const std::string & tag,
        size_t start,
        size_t end,
        std::string_view text,
        std::vector<common_peg_ast_id> children,
        bool is_partial = false
    ) {
        common_peg_ast_id id = nodes_.size();
        nodes_.push_back({id, rule, tag, start, end, text, std::move(children), is_partial});
        auto & node = nodes_.back();
        node.contains_tags = !node.tag.empty();
        for (auto child : node.children) { node.contains_tags |= nodes_[child].contains_tags; }
        return id;
    }

    const common_peg_ast_node & get(common_peg_ast_id id) const { return nodes_.at(id); }

    common_peg_ast_id find_by_tag(const common_peg_ast_node & parent, const std::string & tag, int max_depth = 3) const;
    common_peg_ast_id find_by_rule(const common_peg_ast_node & parent, const std::string & tag, int max_depth = 3) const;

    size_t size() const { return nodes_.size(); }

    void clear() {
        nodes_.clear(); output_cache.clear(); output_prefixes.clear(); stable_nodes = 0;
    }
    void seal(size_t count, std::string_view input) { stable_nodes = count; source_input = input; }
    std::string materialize(const common_peg_text & text) const { return text.str(source_input); }
    template<class Transform>
    common_peg_text output(common_peg_ast_id id, size_t operation, Transform transform) const {
        const auto key = std::make_pair(operation, id);
        if (auto found = output_cache.find(key); found != output_cache.end()) { return found->second; }
        auto text = transform();
        if (id < stable_nodes && output_cache.size() < 262144) { output_cache.emplace(key, text); }
        return text;
    }
    template<class Transform>
    common_peg_text output_children(const common_peg_ast_node & node, size_t operation,
                                   const std::vector<common_peg_ast_id> & children,
                                   const char * separator, Transform transform, bool skip_empty = false) const {
        const auto key = std::make_tuple(operation, node.rule, node.tag, node.start);
        auto found = output_prefixes.find(key);
        output_prefix * prefix = found == output_prefixes.end() ? nullptr : &found->second;
        if (!prefix && output_prefixes.size() < 65536) { prefix = &output_prefixes[key]; }
        if (prefix && (prefix->children.size() > children.size() ||
            !std::equal(prefix->children.begin(), prefix->children.end(), children.begin()))) {
            *prefix = {};
        }
        size_t i = prefix ? prefix->children.size() : 0;
        common_peg_text text = prefix ? prefix->text : common_peg_text{};
        bool stable = true;
        for (; i < children.size(); ++i) {
            auto child = children[i];
            auto part = transform(child);
            if (!skip_empty || !part.empty()) {
                if (skip_empty ? !text.empty() : i > 0) { text += separator; }
                text += part;
            }
            stable &= child < stable_nodes;
            if (prefix && stable) {
                prefix->children.push_back(child);
                prefix->text = text;
            }
        }
        return text;
    }
    void truncate(size_t count) { nodes_.resize(count); }
    void rebase(std::string_view input) {
        for (auto & node : nodes_) {
            node.text = input.substr(node.start, node.end - node.start);
        }
    }

    void visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const;
    void visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const;

    void visit_tags(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const {
        const auto & node = get(id);
        if (!node.contains_tags) { return; }
        if (!node.tag.empty()) { visitor(node); }
        for (auto child : node.children) { visit_tags(child, visitor); }
    }
    void visit_tags(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const;
    std::string dump();
};

struct common_peg_parse_result {
    common_peg_parse_result_type type = COMMON_PEG_PARSE_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    std::vector<common_peg_ast_id> nodes;

    common_peg_parse_result() = default;

    common_peg_parse_result(common_peg_parse_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end, std::vector<common_peg_ast_id> nodes)
        : type(type), start(start), end(end), nodes(std::move(nodes)) {}

    bool fail() const { return type == COMMON_PEG_PARSE_RESULT_FAIL; }
    bool need_more_input() const { return type == COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT; }
    bool success() const { return type == COMMON_PEG_PARSE_RESULT_SUCCESS; }
};

enum common_peg_parse_flags {
    COMMON_PEG_PARSE_FLAG_NONE    = 0,
    COMMON_PEG_PARSE_FLAG_LENIENT = 1 << 0,
    COMMON_PEG_PARSE_FLAG_DEBUG   = 1 << 1,
};

inline common_peg_parse_flags operator|(common_peg_parse_flags a, common_peg_parse_flags b) {
    return static_cast<common_peg_parse_flags>(int(a) | int(b));
}

inline common_peg_parse_flags & operator|=(common_peg_parse_flags & a, common_peg_parse_flags b) {
    return a = a | b;
}

inline common_peg_parse_flags operator&(common_peg_parse_flags a, common_peg_parse_flags b) {
    return static_cast<common_peg_parse_flags>(int(a) & int(b));
}

inline common_peg_parse_flags operator~(common_peg_parse_flags a) {
    return static_cast<common_peg_parse_flags>(~int(a));
}

struct common_peg_parse_context {
    std::string input;
    common_peg_parse_flags flags;
    common_peg_ast_arena ast;

    int parse_depth;

    // Opt-in only for an immutable arena and an append-only input. Cache complete
    // scanner prefixes, never PEG choices/results or provisional delimiters.
    bool append_only = false;
    // Generic chat mapping consumes tags, whereas specialized mappers also use
    // named rule nodes. Never suppress those nodes without an explicit opt-in.
    bool tags_only = false;
    size_t boundary_reads = 0;
    struct scan_progress {
        size_t position;
        int count;
        std::vector<common_peg_ast_id> nodes;
    };
    // Stable nodes are immutable. The tail above this watermark is provisional
    // and discarded before the next append. Bounds limit retained cache storage;
    // exhausting a cache changes cost, never accepted language or output.
    static constexpr size_t max_retained_nodes = 262144;
    size_t retained_nodes = 0;
    const char * input_storage = nullptr;
    std::map<std::pair<common_peg_parser_id, size_t>, common_peg_parse_result> completed;

    bool retain_nodes() {
        if (ast.size() > max_retained_nodes) { return false; }
        retained_nodes = ast.size();
        return true;
    }
    void begin_parse() {
        if (!append_only) { return; }
        if (ast.size() < retained_nodes) {
            // Explicitly clearing the caller-owned AST resets its dependent caches.
            retained_nodes = 0;
            completed.clear();
            scans.clear();
        }
        ast.truncate(retained_nodes);
        if (input_storage != input.data()) { ast.rebase(input); }
        input_storage = input.data();
        boundary_reads = 0;
        parse_depth = 0;
    }
    std::map<std::pair<common_peg_parser_id, size_t>, scan_progress> scans;
    scan_progress * scan(common_peg_parser_id parser, size_t start) {
        if (!append_only) { return nullptr; }
        const auto key = std::make_pair(parser, start);
        if (auto found = scans.find(key); found != scans.end()) { return &found->second; }
        // Keep early (outer) prefixes when full. Clearing would repeatedly lose
        // the enclosing repetition's progress on large structured arguments.
        if (scans.size() >= 65536) { return nullptr; }
        return &scans.emplace(key, scan_progress{start, 0}).first->second;
    }

    common_peg_parse_context(common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : flags(flags), parse_depth(0) {}

    common_peg_parse_context(const std::string & input, common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : input(input), flags(flags), parse_depth(0) {}

    bool is_lenient() const { return flags & COMMON_PEG_PARSE_FLAG_LENIENT; }
    bool is_debug() const { return flags & COMMON_PEG_PARSE_FLAG_DEBUG; }
};

class common_peg_arena;

// Parser variants
struct common_peg_epsilon_parser {};

struct common_peg_start_parser {};

struct common_peg_end_parser {};

struct common_peg_literal_parser {
    std::string literal;
};

struct common_peg_sequence_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_choice_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_repetition_parser {
    common_peg_parser_id child;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_and_parser {
    common_peg_parser_id child;
};

struct common_peg_not_parser {
    common_peg_parser_id child;
};

struct common_peg_any_parser {};

struct common_peg_space_parser {};

struct common_peg_chars_parser {
    struct char_range {
        uint32_t start;
        uint32_t end;
        bool contains(uint32_t codepoint) const { return codepoint >= start && codepoint <= end; }
    };

    std::string pattern;
    std::vector<char_range> ranges;
    bool negated;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_string_parser {
    char delimiter;
};

struct common_peg_until_parser {
    std::vector<std::string> delimiters;
};

struct common_peg_schema_parser {
    common_peg_parser_id child;
    std::string name;
    common_chat_schema_document_ptr doc;  // owns node
    const common_chat_schema * node = nullptr;

    // Indicates if the GBNF should accept a raw string that matches the schema.
    bool raw;
};

struct common_peg_rule_parser {
    std::string name;
    common_peg_parser_id child;
    bool trigger;
};

struct common_peg_ref_parser {
    std::string name;
};

struct common_peg_atomic_parser {
    common_peg_parser_id child;
};

struct common_peg_tag_parser {
    common_peg_parser_id child;
    std::string tag;
};

struct common_peg_gbnf_parser {
    common_peg_parser_id child;
    std::string grammar;
};

struct common_peg_ac_parser {
    common_peg_parser_id child;
    std::vector<std::string> delimiters;
};

// Variant holding all parser types
using common_peg_parser_variant = std::variant<
    common_peg_epsilon_parser,
    common_peg_start_parser,
    common_peg_end_parser,
    common_peg_literal_parser,
    common_peg_sequence_parser,
    common_peg_choice_parser,
    common_peg_repetition_parser,
    common_peg_and_parser,
    common_peg_not_parser,
    common_peg_any_parser,
    common_peg_space_parser,
    common_peg_chars_parser,
    common_peg_string_parser,
    common_peg_until_parser,
    common_peg_schema_parser,
    common_peg_rule_parser,
    common_peg_ref_parser,
    common_peg_atomic_parser,
    common_peg_tag_parser,
    common_peg_gbnf_parser,
    common_peg_ac_parser
>;

class common_peg_arena {
    std::vector<common_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_peg_parser_id> rules_;
    common_peg_parser_id root_ = COMMON_PEG_INVALID_PARSER_ID;

  public:
    const common_peg_parser_variant & get(common_peg_parser_id id) const { return parsers_.at(id); }
    common_peg_parser_variant & get(common_peg_parser_id id) { return parsers_.at(id); }

    size_t size() const { return parsers_.size(); }
    bool empty() const { return parsers_.empty(); }

    common_peg_parser_id get_rule(const std::string & name) const;
    bool has_rule(const std::string & name) const { return rules_.find(name) != rules_.end(); }

    common_peg_parser_id root() const { return root_; }
    void set_root(common_peg_parser_id id) { root_ = id; }

    common_peg_parse_result parse(common_peg_parse_context & ctx, size_t start = 0) const;
    common_peg_parse_result parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const;

    void resolve_refs();

    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;

    std::string dump(common_peg_parser_id id) const;

    common_json to_json() const;
    static common_peg_arena from_json(const common_json & j);

    std::string save() const;
    void load(const std::string & data);

    friend class common_peg_parser_builder;

  private:
    std::string dump_impl(common_peg_parser_id id, std::set<common_peg_parser_id> & visited) const;

    common_peg_parser_id add_parser(common_peg_parser_variant parser);
    void add_rule(const std::string & name, common_peg_parser_id id);

    common_peg_parser_id resolve_ref(common_peg_parser_id id);
};

class common_peg_parser_builder {
    common_peg_arena arena_;

    common_peg_parser wrap(common_peg_parser_id id) { return common_peg_parser(id, *this); }
    common_peg_parser add(const common_peg_parser_variant & p) { return wrap(arena_.add_parser(p)); }

  public:
    common_peg_parser_builder();

    // Match nothing, always succeed.
    //   S -> ε
    common_peg_parser eps() { return add(common_peg_epsilon_parser{}); }

    // Matches the start of the input.
    //   S -> ^
    common_peg_parser start() { return add(common_peg_start_parser{}); }

    // Matches the end of the input.
    //   S -> $
    common_peg_parser end() { return add(common_peg_end_parser{}); }

    // Matches an exact literal string.
    //   S -> "hello"
    common_peg_parser literal(const std::string & literal) { return add(common_peg_literal_parser{literal}); }

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_peg_parser sequence() { return add(common_peg_sequence_parser{}); }
    common_peg_parser sequence(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser sequence(const std::vector<common_peg_parser> & parsers);
    common_peg_parser sequence(std::initializer_list<common_peg_parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_peg_parser choice() { return add(common_peg_choice_parser{}); }
    common_peg_parser choice(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser choice(const std::vector<common_peg_parser> & parsers);
    common_peg_parser choice(std::initializer_list<common_peg_parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_peg_parser one_or_more(const common_peg_parser & p) { return repeat(p, 1, -1); }

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_peg_parser zero_or_more(const common_peg_parser & p) { return repeat(p, 0, -1); }

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_peg_parser optional(const common_peg_parser & p) { return repeat(p, 0, 1); }

    // Positive lookahead: succeeds if child parser succeeds, consumes no input.
    //   S -> &A
    common_peg_parser peek(const common_peg_parser & p) { return add(common_peg_and_parser{p}); }

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_peg_parser negate(const common_peg_parser & p) { return add(common_peg_not_parser{p}); }

    // Matches any single character.
    //   S -> .
    common_peg_parser any() { return add(common_peg_any_parser{}); }

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser chars(const std::string & classes, int min = 1, int max = -1);

    // Creates a lightweight reference to a named rule (resolved during build()).
    // Use this for forward references in recursive grammars.
    //   expr_ref -> expr
    common_peg_parser ref(const std::string & name) { return add(common_peg_ref_parser{name}); }

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    common_peg_parser space() { return add(common_peg_space_parser{}); }

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until(const std::string & delimiter) { return add(common_peg_until_parser{{delimiter}}); }

    // Matches all characters until one of the delimiters in the list is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until_one_of(const std::vector<std::string> & delimiters) { return add(common_peg_until_parser{delimiters}); }

    // Matches everything
    //   S -> .*
    common_peg_parser rest() { return until_one_of({}); }

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser repeat(const common_peg_parser & p, int min, int max) { return add(common_peg_repetition_parser{p, min,max}); }

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_peg_parser repeat(const common_peg_parser & p, int n) { return repeat(p, n, n); }

    // Matches a double-quoted string: '"' content '"' space
    common_peg_parser double_quoted_string();

    // Matches a single-quoted string: "'" content "'" space
    common_peg_parser single_quoted_string();

    // Matches a string that accepts both double-quoted and single-quoted styles.
    common_peg_parser quoted_string();

    // Matches string content without the surrounding delimiter.
    common_peg_parser string_content(char delimiter);

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    common_peg_parser json();
    common_peg_parser json_object();
    common_peg_parser json_string();
    common_peg_parser json_array();
    common_peg_parser json_number();
    common_peg_parser json_bool();
    common_peg_parser json_null();

    // Matches a JSON object member with a key and associated parser as the
    // value.
    common_peg_parser json_member(const std::string & key, const common_peg_parser & p);

    // Creates a complete Python format parser supporting dicts, arrays, strings, numbers, booleans, and None.
    // Differs from JSON: uses True/False/None, accepts both single and double-quoted strings.
    //   value -> dict | array | string | number | True | False | None
    common_peg_parser python_value();
    common_peg_parser python_dict();
    common_peg_parser python_string();
    common_peg_parser python_array();
    common_peg_parser python_number();
    common_peg_parser python_bool();
    common_peg_parser python_null();

    // A marker, i.e. text delimited by a pair of <> or []
    common_peg_parser marker();

    // Wraps a parser with the schema its GBNF is generated from, a node of the document that owns it
    common_peg_parser schema(const common_peg_parser & p, const std::string & name, common_chat_schema_document_ptr doc, const common_chat_schema & node, bool raw = false);

    // Parses the JSON schema into a document of its own
    common_peg_parser schema(const common_peg_parser & p, const std::string & name, const common_json & schema, bool raw = false);

    // Creates a named rule, stores it in the grammar, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", json_obj | json_arr | ...)
    common_peg_parser rule(const std::string & name, const common_peg_parser & p, bool trigger = false);

    // Creates a named rule using a builder function, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", [&]() { return json_object() | json_array() | ... })
    common_peg_parser rule(const std::string & name, const std::function<common_peg_parser()> & builder, bool trigger = false);

    // Creates a trigger rule. When generating a lazy grammar from the parser,
    // only trigger rules and descendents are emitted.
    common_peg_parser trigger_rule(const std::string & name, const common_peg_parser & p) { return rule(name, p, true); }
    common_peg_parser trigger_rule(const std::string & name, const std::function<common_peg_parser()> & builder) { return rule(name, builder, true); }

    // Creates an atomic parser. Atomic parsers do not create an AST node if
    // the child results in a partial parse, i.e. NEEDS_MORE_INPUT. This is
    // intended for situations where partial output is undesirable.
    common_peg_parser atomic(const common_peg_parser & p) { return add(common_peg_atomic_parser{p}); }

    // Tags create nodes in the generated AST for semantic purposes.
    // Unlike rules, you can tag multiple nodes with the same tag.
    common_peg_parser tag(const std::string & tag, const common_peg_parser & p) { return add(common_peg_tag_parser{p.id(), tag}); }

    // Wraps a child parser but emits a custom GBNF grammar string instead of
    // the child's grammar. Parsing delegates entirely to the child.
    common_peg_parser gbnf(const common_peg_parser & p, const std::string & grammar) { return add(common_peg_gbnf_parser{p, grammar}); }

    // Wraps a child parser but emits a GBNF grammar built from the Aho-Corasick
    // automaton of `delimiters`, matching everything up to and including the
    // first delimiter. Parsing delegates entirely to the child, which is
    // responsible for consuming the delimiter (e.g. until(D) + literal(D)).
    common_peg_parser ac(const common_peg_parser & p, const std::vector<std::string> & delimiters);
    common_peg_parser ac(const common_peg_parser & p, const std::string & delimiter) { return ac(p, std::vector<std::string>{delimiter}); }

    void set_root(const common_peg_parser & p);

    common_peg_arena build();
};

// Helper function for building parsers
common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);
