#pragma once

#include "json-schema.h"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Symbolic support only. No inference headers or numerical runtime dependencies.
#define TEMPLATES_ASSERT(condition) do { if (!(condition)) throw std::logic_error("Native templates invariant: " #condition); } while (false)
#define TEMPLATES_ABORT(message) throw std::logic_error(message)

// The UTC calendar fields of `time`; false outside the representable range.
// Windows provides the reentrant conversion as gmtime_s, POSIX as gmtime_r.
inline bool templates_utc_calendar(std::time_t time, std::tm & calendar) {
#if defined(_WIN32)
    return gmtime_s(&calendar, &time) == 0;
#else
    return gmtime_r(&time, &calendar) != nullptr;
#endif
}

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

namespace templates_native {
// A schema keyword a tool's argument syntax cannot enforce, recorded by the
// family that builds the grammar. Drained by the ABI at operation boundaries.
struct tool_relaxation {
    std::string                   tool;
    common_chat_schema_relaxation relaxation;
};
extern thread_local std::vector<tool_relaxation> relaxations;
void relax(const std::string & tool, const common_chat_schema & node, const std::string & keyword,
           common_chat_schema_relaxation::reason_kind reason);
}

// Raw argument text is any string without the closing delimiter. Records the
// string constraints of the schema that raw text leaves unenforced.
void templates_raw_string_argument(const std::string & tool, const common_chat_schema & schema);
// The strings a tagged raw argument value may be: nullopt when any text is
// allowed, otherwise exactly these values (none when the schema admits no
// string). Values containing `delimiter` cannot be written and are left out.
// Records every constraint raw text leaves unenforced.
std::optional<std::vector<std::string>> templates_raw_string_values(const std::string & tool, const common_chat_schema & schema,
                                                                    const std::string & delimiter);
