// String helpers extracted from llama.cpp common/common.cpp; see upstream/LICENSE.
#include "templates-support.h"
#include "templates-log.h"
#include <cstdarg>
#include <cstdio>
thread_local std::vector<std::string> templates_native::diagnostics;
void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

void templates_native::diagnostic(const char * format, ...) {
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    int size = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (size < 0) { va_end(args); throw std::runtime_error("Native diagnostic formatting failed"); }
    std::vector<char> buffer(static_cast<size_t>(size) + 1);
    std::vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);
    diagnostics.emplace_back(buffer.data(), static_cast<size_t>(size));
}

thread_local std::vector<templates_native::tool_relaxation> templates_native::relaxations;

void templates_native::relax(const std::string & tool, const common_chat_schema & node, const std::string & keyword,
                             common_chat_schema_relaxation::reason_kind reason) {
    relaxations.push_back({ tool, { node.path, keyword, reason } });
}

namespace {
using relaxation = common_chat_schema_relaxation;

// Whether the schema admits every string.
bool admits_every_string(const common_chat_schema & schema, int depth = 0) {
    switch (schema.kind()) {
        case common_chat_schema::KIND_ANY:
            return true;
        case common_chat_schema::KIND_STRING: {
            const auto & string = static_cast<const common_chat_schema_string &>(schema);
            return string.pattern.empty() && string.min_length == 0 && string.max_length < 0;
        }
        case common_chat_schema::KIND_REF: {
            const auto * target = static_cast<const common_chat_schema_ref &>(schema).target;
            return depth < 32 && admits_every_string(*target, depth + 1);
        }
        case common_chat_schema::KIND_ANY_OF:
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(schema).children) {
                if (admits_every_string(*child, depth + 1)) {
                    return true;
                }
            }
            return false;
        default:
            return false;
    }
}

// The string constraints a schema places, which raw text does not enforce.
void record_string_constraints(const std::string & tool, const common_chat_schema & schema, int depth = 0) {
    switch (schema.kind()) {
        case common_chat_schema::KIND_STRING: {
            const auto & string = static_cast<const common_chat_schema_string &>(schema);
            if (!string.pattern.empty()) {
                templates_native::relax(tool, schema, "pattern", relaxation::REASON_UNENFORCED);
            }
            if (string.min_length > 0) {
                templates_native::relax(tool, schema, "minLength", relaxation::REASON_UNENFORCED);
            }
            if (string.max_length >= 0) {
                templates_native::relax(tool, schema, "maxLength", relaxation::REASON_UNENFORCED);
            }
            break;
        }
        case common_chat_schema::KIND_CONST:
            if (static_cast<const common_chat_schema_const &>(schema).value.is_string()) {
                templates_native::relax(tool, schema, "const", relaxation::REASON_UNENFORCED);
            }
            break;
        case common_chat_schema::KIND_ENUM:
            if (schema.may_be_string()) {
                templates_native::relax(tool, schema, "enum", relaxation::REASON_UNENFORCED);
            }
            break;
        case common_chat_schema::KIND_REF:
            if (depth < 32) {
                record_string_constraints(tool, *static_cast<const common_chat_schema_ref &>(schema).target, depth + 1);
            }
            break;
        case common_chat_schema::KIND_ANY_OF:
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(schema).children) {
                record_string_constraints(tool, *child, depth + 1);
            }
            break;
        default:
            break;
    }
}

std::optional<std::vector<std::string>> raw_string_values(const std::string & tool, const common_chat_schema & schema,
                                                          const std::string & delimiter, int depth) {
    auto writable = [&](const std::string & value, const char * keyword) {
        if (value.find(delimiter) == std::string::npos) {
            return true;
        }
        templates_native::relax(tool, schema, keyword, relaxation::REASON_UNREPRESENTABLE);
        return false;
    };
    switch (schema.kind()) {
        case common_chat_schema::KIND_ANY:
            return std::nullopt;
        case common_chat_schema::KIND_STRING:
            record_string_constraints(tool, schema);
            return std::nullopt;
        case common_chat_schema::KIND_CONST: {
            const auto & value = static_cast<const common_chat_schema_const &>(schema).value;
            if (value.is_string() && writable(value.get<std::string>(), "const")) {
                return std::vector<std::string>{ value.get<std::string>() };
            }
            return std::vector<std::string>{};
        }
        case common_chat_schema::KIND_ENUM: {
            std::vector<std::string> values;
            for (const auto & value : static_cast<const common_chat_schema_enum &>(schema).values) {
                if (value.is_string() && writable(value.get<std::string>(), "enum") &&
                    std::find(values.begin(), values.end(), value.get<std::string>()) == values.end()) {
                    values.push_back(value.get<std::string>());
                }
            }
            return values;
        }
        case common_chat_schema::KIND_REF:
            if (depth >= 32) {
                return std::nullopt;
            }
            return raw_string_values(tool, *static_cast<const common_chat_schema_ref &>(schema).target, delimiter, depth + 1);
        case common_chat_schema::KIND_ANY_OF: {
            std::vector<std::string> values;
            bool                     any_text = false;
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(schema).children) {
                auto child_values = raw_string_values(tool, *child, delimiter, depth + 1);
                if (!child_values) {
                    any_text = true;
                    continue;
                }
                for (auto & value : *child_values) {
                    if (std::find(values.begin(), values.end(), value) == values.end()) {
                        values.push_back(std::move(value));
                    }
                }
            }
            if (any_text) {
                return std::nullopt;
            }
            return values;
        }
        default:
            return std::vector<std::string>{};
    }
}
}  // namespace

void templates_raw_string_argument(const std::string & tool, const common_chat_schema & schema) {
    if (!admits_every_string(schema)) {
        record_string_constraints(tool, schema);
    }
}

std::optional<std::vector<std::string>> templates_raw_string_values(const std::string & tool, const common_chat_schema & schema,
                                                                    const std::string & delimiter) {
    if (admits_every_string(schema)) {
        return std::nullopt;
    }
    return raw_string_values(tool, schema, delimiter, 0);
}
