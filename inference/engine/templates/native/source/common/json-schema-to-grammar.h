#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <optional>
#include <string>

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_chat_schema_document & schema);

// Whether a JSON Schema pattern translates to a grammar over JSON string text.
bool json_schema_pattern_supported(const std::string & pattern);

// The GBNF of plain decimal JSON numbers within the bounds (integers only when
// integral); null when no value is within them.
std::optional<std::string> json_schema_numeric_range(const common_chat_schema_numeric & bounds, bool integral);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)>    add_rule;
    std::function<std::string(const std::string &, const common_chat_schema &)> add_schema;
};

std::string gbnf_format_literal(const std::string & literal);

// A JSON string's \u escape after its backslash: one character outside the
// surrogates, or a complete surrogate pair. JSON parsers reject lone surrogates.
extern const char * const GBNF_JSON_UNICODE_ESCAPE;

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb);
