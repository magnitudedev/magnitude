#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Symbolic support only. No inference headers or numerical runtime dependencies.
#define TEMPLATES_ASSERT(condition) do { if (!(condition)) throw std::logic_error("Native templates invariant: " #condition); } while (false)
#define TEMPLATES_ABORT(message) throw std::logic_error(message)

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};
struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    int32_t token = -1;
};
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    COMMON_REASONING_FORMAT_DEEPSEEK,
};

std::string string_join(const std::vector<std::string> &, const std::string &);
std::vector<std::string> string_split(const std::string &, const std::string &);
template <typename T>
std::vector<T> string_split(const std::string & text, char delimiter) {
    static_assert(std::is_same_v<T, std::string>);
    return string_split(text, std::string(1, delimiter));
}
std::string string_repeat(const std::string &, size_t);
void string_replace_all(std::string &, const std::string &, const std::string &);
inline bool string_starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}
inline bool string_starts_with(std::string_view s, char prefix) { return !s.empty() && s.front() == prefix; }
inline bool string_ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

struct common_chat_schema;
void templates_require_unconstrained_raw_string(const common_chat_schema & schema);
