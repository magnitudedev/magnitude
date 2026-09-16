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

#include "json-schema.h"
void templates_require_unconstrained_raw_string(const common_chat_schema & schema) {
    if (!schema.may_be_string()) { return; }
    if (schema.kind() == common_chat_schema::KIND_STRING) {
        const auto & string = static_cast<const common_chat_schema_string &>(schema);
        if (string.pattern.empty() && string.format == common_chat_schema::FORMAT_NONE &&
            string.min_length == 0 && string.max_length == -1) { return; }
    }
    throw std::invalid_argument("Unsupported JSON schema: tagged raw strings cannot enforce this constraint");
}
