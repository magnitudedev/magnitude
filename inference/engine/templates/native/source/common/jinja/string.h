#pragma once

#include <optional>
#include <string>
#include <vector>

#include "utils.h"

namespace jinja {

// allow differentiate between user input strings and template strings
// every byte keeps the marking of the byte it came from:
// - transformations that keep bytes (case, strip, slicing, splitting, replacement, indentation,
//   formatting, joining, concatenation) keep each byte's part
// - text that cannot be traced to particular bytes (a JSON or repr serialization) is input if any
//   of its sources is
struct string_part {
    bool is_input = false; // may skip parsing special tokens if true
    std::string val;

    bool is_uppercase() const;
    bool is_lowercase() const;
};

struct string {
    std::vector<string_part> parts;
    string() = default;
    string(const std::string & v, bool user_input = false) {
        parts.push_back({user_input, v});
    }
    string(int v) {
        parts.push_back({false, std::to_string(v)});
    }
    string(double v) {
        parts.push_back({false, std::to_string(v)});
    }

    // mark all parts as user input
    void mark_input();

    std::string str() const;
    size_t length() const;
    // whether both concatenate to the same text, without materializing either
    bool equals(const string & other) const noexcept;
    void hash_update(hasher & hash) const noexcept;
    bool any_part_is_input() const;
    bool is_uppercase() const;
    bool is_lowercase() const;

    string & append(const string & other);

    // the bytes [pos, pos + len), each in a part marked as the byte's own part
    string substr(size_t pos, size_t len) const;

    // in-place transformations

    string uppercase();
    string lowercase();
    string capitalize();
    string titlecase();
    string strip(bool left, bool right, std::optional<const std::string_view> chars = std::nullopt);
};

// Template text a host joins into a request string. Request JSON is valid UTF-8, which never uses
// the bytes 0xFE and 0xFF, so text a host brackets with them can never be the request's own:
// `template_text` brackets text (itself UTF-8) as template text, and `json_text` reads a JSON
// string back, its bracketed bytes template text and the rest marked `input`.
std::string template_text(const std::string & text);
string json_text(const std::string & text, bool input);

} // namespace jinja
