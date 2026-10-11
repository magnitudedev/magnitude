#include "json-schema-to-grammar.h"
#include "templates-support.h"
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = common_json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

// ---------------------------------------------------------------------------
// Numeric ranges
//
// A bounded number is written in plain decimal notation (no exponent), which
// every finite decimal value has, so the grammar admits exactly the values
// within the bounds. Digit counts are capped as for unbounded numbers, and
// never below what the bounds themselves need.

static std::string digit_class(char from, char to) {
    return from == to ? std::string("[") + from + "]" : std::string("[") + from + "-" + to + "]";
}

static std::string digits(size_t min_count, size_t max_count) {
    if (min_count == max_count) {
        return min_count == 1 ? "[0-9]" : "[0-9]{" + std::to_string(min_count) + "}";
    }
    return "[0-9]{" + std::to_string(min_count) + "," + std::to_string(max_count) + "}";
}

// Digit strings of one length from `low` to `high`, inclusive.
static std::string same_length_range(const std::string & low, const std::string & high) {
    size_t i = 0;
    while (i < low.size() && low[i] == high[i]) {
        i++;
    }
    std::string prefix = i > 0 ? "\"" + low.substr(0, i) + "\"" : "";
    if (i == low.size()) {
        return prefix;
    }
    const size_t      rest    = low.size() - i - 1;
    const std::string zeros   = std::string(rest, '0');
    const std::string nines   = std::string(rest, '9');
    const std::string low_rest  = low.substr(i + 1);
    const std::string high_rest = high.substr(i + 1);
    std::vector<std::string> alternatives;
    if (rest == 0) {
        alternatives.push_back(digit_class(low[i], high[i]));
    } else {
        const char first = low_rest == zeros ? low[i] : (char) (low[i] + 1);
        const char last  = high_rest == nines ? high[i] : (char) (high[i] - 1);
        if (low_rest != zeros) {
            alternatives.push_back(digit_class(low[i], low[i]) + " (" + same_length_range(low_rest, nines) + ")");
        }
        if (first <= last) {
            alternatives.push_back(digit_class(first, last) + " " + digits(rest, rest));
        }
        if (high_rest != nines) {
            alternatives.push_back(digit_class(high[i], high[i]) + " (" + same_length_range(zeros, high_rest) + ")");
        }
    }
    auto body = alternatives.size() == 1 ? alternatives[0] : "(" + string_join(alternatives, " | ") + ")";
    return prefix.empty() ? body : prefix + " " + body;
}

static int compare_digit_strings(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) {
        return a.size() < b.size() ? -1 : 1;
    }
    int c = a.compare(b);
    return c < 0 ? -1 : c > 0 ? 1 : 0;
}

static std::string increment_digits(std::string value) {
    for (size_t i = value.size(); i-- > 0;) {
        if (value[i] != '9') {
            value[i]++;
            return value;
        }
        value[i] = '0';
    }
    return "1" + value;
}

// The canonical integer one less; `value` is positive.
static std::string decrement_digits(std::string value) {
    for (size_t i = value.size(); i-- > 0;) {
        if (value[i] != '0') {
            value[i]--;
            break;
        }
        value[i] = '9';
    }
    auto first = value.find_first_not_of('0');
    return first == std::string::npos ? "0" : value.substr(first);
}

// Canonical nonnegative integers from `low` to `high` (inclusive) of at most
// `longest` digits, by digit count.
static std::vector<std::pair<size_t, std::string>> integer_ranges(const std::string &                low,
                                                                  const std::optional<std::string> & high,
                                                                  size_t                             longest) {
    std::vector<std::pair<size_t, std::string>> ranges;
    for (size_t length = low.size(); length <= longest && (!high || length <= high->size()); length++) {
        const std::string from = length == low.size() ? low : "1" + std::string(length - 1, '0');
        const std::string to   = high && length == high->size() ? *high : std::string(length, '9');
        if (compare_digit_strings(from, to) <= 0) {
            ranges.emplace_back(length, same_length_range(from, to));
        }
    }
    return ranges;
}

// A bound on a magnitude (a nonnegative decimal).
struct magnitude_bound {
    std::string integer;
    std::string fraction;
    bool        exclusive = false;
};

// The fraction digits after an integer part, from position `i`, while the
// digits so far equal the lower and/or upper bound's fraction digits.
class fraction_grammar {
    enum lower_state { LOWER_FREE, LOWER_TIGHT, LOWER_PENDING };  // pending: equal to an exclusive bound
    enum upper_state { UPPER_FREE, UPPER_TIGHT, UPPER_ZEROS, UPPER_DEAD };

    const magnitude_bound * low_;
    const magnitude_bound * high_;
    size_t                  cap_;

  public:
    fraction_grammar(const magnitude_bound * low, const magnitude_bound * high, size_t cap) : low_(low), high_(high), cap_(cap) {}

    // The optional fraction part, or null when no fraction (not even none) satisfies the bounds.
    std::optional<std::string> part(bool integral) const {
        lower_state lower = low_ ? LOWER_TIGHT : LOWER_FREE;
        upper_state upper = high_ ? UPPER_TIGHT : UPPER_FREE;
        normalize(0, lower, upper);
        if (upper == UPPER_DEAD) {
            return std::nullopt;
        }
        const bool none = can_end(0, lower, upper);
        auto       more = integral ? std::nullopt : from(0, lower, upper, /* may_end = */ false);
        if (none && more) {
            return "(\".\" " + *more + ")?";
        }
        if (more) {
            return "\".\" " + *more;
        }
        if (none) {
            return std::string();
        }
        return std::nullopt;
    }

  private:
    void normalize(size_t i, lower_state & lower, upper_state & upper) const {
        if (lower == LOWER_TIGHT && i >= low_->fraction.size()) {
            lower = low_->exclusive ? LOWER_PENDING : LOWER_FREE;
        }
        if (upper == UPPER_TIGHT && i >= high_->fraction.size()) {
            upper = high_->exclusive ? UPPER_DEAD : UPPER_ZEROS;
        }
    }

    bool can_end(size_t i, lower_state lower, upper_state upper) const {
        // Ending pads with zeros: below a lower bound with digits left, and
        // equal to a bound the digits have reached.
        return lower == LOWER_FREE && upper != UPPER_DEAD && (upper != UPPER_TIGHT || i < high_->fraction.size());
    }

    // Digits from position `i`; null when no continuation satisfies the bounds.
    std::optional<std::string> from(size_t i, lower_state lower, upper_state upper, bool may_end) const {
        normalize(i, lower, upper);
        if (upper == UPPER_DEAD) {
            return std::nullopt;
        }
        const bool   end       = may_end && can_end(i, lower, upper);
        const size_t remaining = i < cap_ ? cap_ - i : 0;
        if (remaining == 0) {
            return end ? std::optional<std::string>("") : std::nullopt;
        }
        if (lower == LOWER_FREE && upper == UPPER_FREE) {
            return digits(may_end ? 0 : 1, remaining);
        }
        if (lower == LOWER_FREE && upper == UPPER_ZEROS) {
            return "[0]{" + std::to_string(may_end ? 0 : 1) + "," + std::to_string(remaining) + "}";
        }
        // Group the next digit by the state it leads to.
        std::vector<std::string> alternatives;
        char                     start = 0;
        std::optional<std::pair<lower_state, upper_state>> group;
        auto flush = [&](char last) {
            if (!group) {
                return;
            }
            auto next = from(i + 1, group->first, group->second, true);
            if (next) {
                alternatives.push_back(digit_class(start, last) + (next->empty() ? "" : " " + *next));
            }
        };
        for (char d = '0'; d <= '9'; d++) {
            std::optional<std::pair<lower_state, upper_state>> next;
            lower_state                                        l = lower;
            upper_state                                        u = upper;
            bool                                               allowed = true;
            if (lower == LOWER_TIGHT) {
                const char bound = low_->fraction[i];
                l                = d > bound ? LOWER_FREE : LOWER_TIGHT;
                allowed          = d >= bound;
            } else if (lower == LOWER_PENDING) {
                l = d > '0' ? LOWER_FREE : LOWER_PENDING;
            }
            if (upper == UPPER_TIGHT) {
                const char bound = high_->fraction[i];
                u                = d < bound ? UPPER_FREE : UPPER_TIGHT;
                allowed          = allowed && d <= bound;
            } else if (upper == UPPER_ZEROS) {
                allowed = allowed && d == '0';
            }
            if (allowed) {
                next = std::make_pair(l, u);
            }
            if (next != group) {
                flush((char) (d - 1));
                group = next;
                start = d;
            }
        }
        flush('9');
        if (alternatives.empty()) {
            return end ? std::optional<std::string>("") : std::nullopt;
        }
        auto body = alternatives.size() == 1 ? alternatives[0] : "(" + string_join(alternatives, " | ") + ")";
        if (end) {
            return "(" + body + ")?";
        }
        return body;
    }
};

// Magnitudes (nonnegative decimals, no sign) within the bounds, written with
// at most `budget` digits, or null when none.
static std::optional<std::string> magnitude_range(const std::optional<magnitude_bound> & low,
                                                  const std::optional<magnitude_bound> & high,
                                                  bool integral, size_t budget) {
    if (low && high) {
        int c = compare_digit_strings(low->integer, high->integer);
        if (c == 0) {
            const size_t width = std::max(low->fraction.size(), high->fraction.size());
            c = (low->fraction + std::string(width - low->fraction.size(), '0'))
                    .compare(high->fraction + std::string(width - high->fraction.size(), '0'));
        }
        if (c > 0 || (c == 0 && (low->exclusive || high->exclusive))) {
            return std::nullopt;
        }
    }
    // Fraction digits left after an integer part of `length` digits.
    auto fraction_cap = [&](size_t length) { return length < budget ? budget - length : 0; };
    auto with_fraction = [&](const std::string & integer, const std::string & fraction) {
        return fraction.empty() ? integer : integer + " " + fraction;
    };
    auto literal = [&](const magnitude_bound * lower, const magnitude_bound * upper, const std::string & integer) {
        auto fraction = fraction_grammar(lower, upper, fraction_cap(integer.size())).part(integral);
        return fraction ? std::optional<std::string>(with_fraction("\"" + integer + "\"", *fraction)) : std::nullopt;
    };
    std::vector<std::string> alternatives;
    if (low && high && low->integer == high->integer) {
        if (auto value = literal(&*low, &*high, low->integer)) {
            alternatives.push_back(*value);
        }
    } else {
        // Integer parts strictly between the bounds' integer parts.
        const std::string between_low = low ? increment_digits(low->integer) : "0";
        std::optional<std::string> between_high;
        bool                       between = true;
        if (high) {
            if (high->integer == "0") {
                between = false;
            } else {
                between_high = decrement_digits(high->integer);
            }
        }
        if (between) {
            for (const auto & [length, integers] : integer_ranges(between_low, between_high, budget)) {
                const size_t cap = fraction_cap(length);
                alternatives.push_back(with_fraction(integers, integral || cap == 0 ? "" : "(\".\" " + digits(1, cap) + ")?"));
            }
        }
        if (low) {
            if (auto value = literal(&*low, nullptr, low->integer)) {
                alternatives.push_back(*value);
            }
        }
        if (high) {
            if (auto value = literal(nullptr, &*high, high->integer)) {
                alternatives.push_back(*value);
            }
        }
    }
    if (alternatives.empty()) {
        return std::nullopt;
    }
    return alternatives.size() == 1 ? alternatives[0] : "(" + string_join(alternatives, " | ") + ")";
}

std::optional<std::string> json_schema_numeric_range(const common_chat_schema_numeric & bounds, bool integral) {
    // Validators compare decimals as doubles. Decimals of at most 15 digits
    // read back as distinct doubles, so one within the bounds stays within
    // them as a double; integers of at most 18 digits are read exactly.
    const size_t budget = integral ? 18 : 15;
    auto magnitude = [](const common_chat_numeric_bound & bound) {
        return magnitude_bound{ bound.value.integer, bound.value.fraction, bound.exclusive };
    };
    const auto & minimum = bounds.minimum;
    const auto & maximum = bounds.maximum;
    std::vector<std::string> alternatives;

    // Zero and positive values, written without a sign.
    const bool nonnegative = !maximum || (!maximum->value.negative && !(maximum->value.is_zero() && maximum->exclusive));
    if (nonnegative) {
        std::optional<magnitude_bound> low;
        std::optional<magnitude_bound> high;
        if (minimum && !minimum->value.negative && !(minimum->value.is_zero() && !minimum->exclusive)) {
            low = magnitude(*minimum);
        }
        if (maximum) {
            high = magnitude(*maximum);
        }
        if (auto range = magnitude_range(low, high, integral, budget)) {
            alternatives.push_back(*range);
        }
    }
    // Negative values: a sign and a positive magnitude.
    const bool negative = !minimum || minimum->value.negative;
    if (negative) {
        magnitude_bound low{ "0", "", true };
        if (maximum && maximum->value.negative) {
            low = magnitude(*maximum);
        }
        std::optional<magnitude_bound> high;
        if (minimum) {
            high = magnitude(*minimum);
        }
        if (auto range = magnitude_range(low, high, integral, budget)) {
            alternatives.push_back("\"-\" " + *range);
        }
    }
    if (alternatives.empty()) {
        return std::nullopt;
    }
    return alternatives.size() == 1 ? alternatives[0] : "(" + string_join(alternatives, " | ") + ")";
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

const char * const GBNF_JSON_UNICODE_ESCAPE =
    "\"u\" ([0-9a-cA-CeEfF] [0-9a-fA-F]{3} | [dD] [0-7] [0-9a-fA-F]{2}"
    " | [dD] [89abAB] [0-9a-fA-F]{2} \"\\\\u\" [dD] [c-fC-F] [0-9a-fA-F]{2})";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\")", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
    {"char",   {std::string("[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | ") + GBNF_JSON_UNICODE_ESCAPE + ")", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
    {"null", {"\"null\"", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static const int MAX_PATTERN_DEPTH = 100;

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$'};

// A pattern constrains a string's value; the grammar constrains its JSON text.
// Characters JSON must escape (quotation mark, reverse solidus, controls)
// match as their escape sequences, never raw.

using codepoint_ranges = std::vector<std::pair<uint32_t, uint32_t>>;

static codepoint_ranges normalize_ranges(codepoint_ranges ranges) {
    std::sort(ranges.begin(), ranges.end());
    codepoint_ranges merged;
    for (const auto & range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second + 1) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}

static codepoint_ranges subtract_ranges(const codepoint_ranges & ranges, const codepoint_ranges & removed) {
    codepoint_ranges result;
    for (auto range : normalize_ranges(ranges)) {
        for (const auto & cut : normalize_ranges(removed)) {
            if (cut.second < range.first || cut.first > range.second) {
                continue;
            }
            if (cut.first > range.first) {
                result.push_back({ range.first, cut.first - 1 });
            }
            if (cut.second >= range.second) {
                range.first = range.second + 1;
                break;
            }
            range.first = cut.second + 1;
        }
        if (range.first <= range.second) {
            result.push_back(range);
        }
    }
    return result;
}

static bool ranges_contain(const codepoint_ranges & ranges, uint32_t cp) {
    return std::any_of(ranges.begin(), ranges.end(), [&](const auto & range) { return cp >= range.first && cp <= range.second; });
}

// Characters a JSON string writes only as escape sequences.
static const codepoint_ranges JSON_ESCAPED = { { 0x00, 0x1F }, { 0x22, 0x22 }, { 0x5C, 0x5C } };

// The GBNF literal body matching how a JSON string writes one character.
static std::string json_text_literal(uint32_t cp) {
    switch (cp) {
        case 0x22: return "\\\\\\\"";
        case 0x5C: return "\\\\\\\\";
        case 0x08: return "\\\\b";
        case 0x0C: return "\\\\f";
        case 0x0A: return "\\\\n";
        case 0x0D: return "\\\\r";
        case 0x09: return "\\\\t";
        default: break;
    }
    if (cp < 0x20) {
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "\\\\u%04x", cp);
        return buffer;
    }
    std::string text;
    if (cp < 0x80) {
        text += (char) cp;
    } else if (cp < 0x800) {
        text += (char) (0xC0 | (cp >> 6));
        text += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        text += (char) (0xE0 | (cp >> 12));
        text += (char) (0x80 | ((cp >> 6) & 0x3F));
        text += (char) (0x80 | (cp & 0x3F));
    } else {
        text += (char) (0xF0 | (cp >> 18));
        text += (char) (0x80 | ((cp >> 12) & 0x3F));
        text += (char) (0x80 | ((cp >> 6) & 0x3F));
        text += (char) (0x80 | (cp & 0x3F));
    }
    return text;
}

static std::string class_endpoint(uint32_t cp) {
    char buffer[16];
    if (cp <= 0xFFFF) {
        snprintf(buffer, sizeof(buffer), "\\u%04X", cp);
    } else {
        snprintf(buffer, sizeof(buffer), "\\U%08X", cp);
    }
    return buffer;
}

static std::string class_items(const codepoint_ranges & ranges) {
    std::string items;
    // Surrogates are not characters of UTF-8 text.
    for (const auto & range : subtract_ranges(ranges, { { 0xD800, 0xDFFF } })) {
        items += class_endpoint(range.first);
        if (range.second != range.first) {
            items += "-" + class_endpoint(range.second);
        }
    }
    return items;
}

// A GBNF expression matching the JSON text of one character in the set
// (or outside it, when negated). Empty when no character matches.
static std::string json_text_class(const codepoint_ranges & ranges, bool negated) {
    std::vector<std::string> alternatives;
    if (negated) {
        codepoint_ranges excluded = ranges;
        excluded.insert(excluded.end(), JSON_ESCAPED.begin(), JSON_ESCAPED.end());
        alternatives.push_back("[^" + class_items(excluded) + "]");
        for (uint32_t cp : { 0x22u, 0x5Cu, 0x08u, 0x0Cu, 0x0Au, 0x0Du, 0x09u }) {
            if (!ranges_contain(ranges, cp)) {
                alternatives.push_back("\"" + json_text_literal(cp) + "\"");
            }
        }
    } else {
        auto raw = subtract_ranges(ranges, JSON_ESCAPED);
        if (!raw.empty()) {
            alternatives.push_back("[" + class_items(raw) + "]");
        }
        for (const auto & range : normalize_ranges(ranges)) {
            for (uint32_t cp = range.first; cp <= std::min<uint32_t>(range.second, 0x5C); cp++) {
                if (ranges_contain(JSON_ESCAPED, cp)) {
                    alternatives.push_back("\"" + json_text_literal(cp) + "\"");
                }
            }
        }
    }
    if (alternatives.empty()) {
        return "";
    }
    if (alternatives.size() == 1) {
        return alternatives[0];
    }
    return "(" + string_join(alternatives, " | ") + ")";
}

static const codepoint_ranges DIGIT_RANGES = { { '0', '9' } };
static const codepoint_ranges WORD_RANGES  = { { '0', '9' }, { 'A', 'Z' }, { '_', '_' }, { 'a', 'z' } };
static const codepoint_ranges SPACE_RANGES = {
    { 0x09, 0x0D }, { 0x20, 0x20 }, { 0xA0, 0xA0 }, { 0x1680, 0x1680 }, { 0x2000, 0x200A },
    { 0x2028, 0x2029 }, { 0x202F, 0x202F }, { 0x205F, 0x205F }, { 0x3000, 0x3000 }, { 0xFEFF, 0xFEFF },
};
static const codepoint_ranges LINE_TERMINATORS = { { 0x0A, 0x0A }, { 0x0D, 0x0D }, { 0x2028, 0x2029 } };

// The ranges of a character class escape (\d, \w, \s and their negations).
static bool class_escape(char c, codepoint_ranges & ranges, bool & negated) {
    switch (c) {
        case 'd': ranges = DIGIT_RANGES; negated = false; return true;
        case 'D': ranges = DIGIT_RANGES; negated = true;  return true;
        case 'w': ranges = WORD_RANGES;  negated = false; return true;
        case 'W': ranges = WORD_RANGES;  negated = true;  return true;
        case 's': ranges = SPACE_RANGES; negated = false; return true;
        case 'S': ranges = SPACE_RANGES; negated = true;  return true;
        default:  return false;
    }
}

// Decodes the UTF-8 character at `i`, advancing past it.
static uint32_t decode_utf8(const std::string & text, size_t & i) {
    const unsigned char lead = text[i];
    size_t              length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : 4;
    uint32_t            cp     = length == 1 ? lead : length == 2 ? lead & 0x1F : length == 3 ? lead & 0x0F : lead & 0x07;
    for (size_t k = 1; k < length && i + k < text.size(); k++) {
        cp = (cp << 6) | ((unsigned char) text[i + k] & 0x3F);
    }
    i += length;
    return cp;
}

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

// Decodes the escape at `i` (a backslash) when it names one character,
// advancing past it. False for escapes naming no single character: class
// escapes, backreferences, assertions and property escapes.
static bool decode_escape(const std::string & pattern, size_t & i, uint32_t & cp) {
    if (i + 1 >= pattern.size()) {
        return false;
    }
    auto hex = [&](size_t start, size_t count, uint32_t & value) {
        if (start + count > pattern.size()) {
            return false;
        }
        value = 0;
        for (size_t k = start; k < start + count; k++) {
            const char h = pattern[k];
            if (!std::isxdigit((unsigned char) h)) {
                return false;
            }
            value = value * 16 + (uint32_t) (h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10);
        }
        return true;
    };
    const char c = pattern[i + 1];
    switch (c) {
        case 't': cp = 0x09; i += 2; return true;
        case 'n': cp = 0x0A; i += 2; return true;
        case 'v': cp = 0x0B; i += 2; return true;
        case 'f': cp = 0x0C; i += 2; return true;
        case 'r': cp = 0x0D; i += 2; return true;
        case '0':
            if (i + 2 < pattern.size() && std::isdigit((unsigned char) pattern[i + 2])) {
                return false;
            }
            cp = 0;
            i += 2;
            return true;
        case 'x':
            if (!hex(i + 2, 2, cp)) {
                return false;
            }
            i += 4;
            return true;
        case 'u': {
            if (i + 2 < pattern.size() && pattern[i + 2] == '{') {
                auto close = pattern.find('}', i + 3);
                if (close == std::string::npos || close - (i + 3) < 1 || close - (i + 3) > 6 ||
                    !hex(i + 3, close - (i + 3), cp) || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
                    return false;
                }
                i = close + 1;
                return true;
            }
            if (!hex(i + 2, 4, cp)) {
                return false;
            }
            size_t next = i + 6;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                uint32_t low = 0;
                if (next + 1 < pattern.size() && pattern[next] == '\\' && pattern[next + 1] == 'u' &&
                    hex(next + 2, 4, low) && low >= 0xDC00 && low <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    i  = next + 6;
                    return true;
                }
                return false;
            }
            if (cp >= 0xDC00 && cp <= 0xDFFF) {
                return false;
            }
            i = next;
            return true;
        }
        default:
            break;
    }
    // Identity escapes of punctuation.
    if ((unsigned char) c < 0x80 && std::ispunct((unsigned char) c)) {
        cp = (uint32_t) c;
        i += 2;
        return true;
    }
    return false;
}

// Parses the class at `i` ('['), advancing past its ']'.
template <typename Unsupported, typename Invalid>
static void parse_class(const std::string & pattern, size_t & i, codepoint_ranges & ranges, bool & negated) {
    i++;
    negated = i < pattern.size() && pattern[i] == '^';
    if (negated) {
        i++;
    }
    auto atom = [&](uint32_t & cp) {
        if (pattern[i] != '\\') {
            cp = decode_utf8(pattern, i);
            return;
        }
        if (i + 1 < pattern.size() && pattern[i + 1] == 'b') {
            cp = 0x08;  // backspace, within a class
            i += 2;
            return;
        }
        if (!decode_escape(pattern, i, cp)) {
            throw Unsupported("unsupported escape in character class: " + pattern.substr(i, 2));
        }
    };
    while (i < pattern.size() && pattern[i] != ']') {
        codepoint_ranges escaped;
        bool             escaped_negated = false;
        if (pattern[i] == '\\' && i + 1 < pattern.size() && class_escape(pattern[i + 1], escaped, escaped_negated)) {
            if (escaped_negated) {
                throw Unsupported("negated class escape inside a character class");
            }
            ranges.insert(ranges.end(), escaped.begin(), escaped.end());
            i += 2;
            continue;
        }
        uint32_t first = 0;
        atom(first);
        if (i + 1 < pattern.size() && pattern[i] == '-' && pattern[i + 1] != ']') {
            i++;
            if (pattern[i] == '\\' && i + 1 < pattern.size() && class_escape(pattern[i + 1], escaped, escaped_negated)) {
                throw Unsupported("class escape as a range endpoint");
            }
            uint32_t last = 0;
            atom(last);
            if (last < first) {
                throw Invalid("character class range out of order");
            }
            ranges.push_back({ first, last });
        } else {
            ranges.push_back({ first, first });
        }
    }
    if (i >= pattern.size()) {
        throw Invalid("unterminated character class");
    }
    i++;
}

class common_chat_schema_converter {
private:
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb);
    std::map<std::string, std::string> _rules;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;

    template <typename T>
    static const T & as(const common_chat_schema & node) {
        return static_cast<const T &>(node);
    }

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<common_chat_schema_ptr> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(*alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    // thrown when the pattern is a valid regex with no grammar equivalent
    struct unsupported_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // thrown when the pattern is not a valid regex
    struct invalid_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // Lowering keeps only patterns this translates; see json_schema_pattern_supported.
    std::string _visit_pattern(const std::string & pattern, const std::string & name) {
        try {
            return _pattern_to_rule(pattern, name);
        } catch (const unsupported_pattern & err) {
            throw std::logic_error("lowered pattern " + pattern + " is not translatable: " + err.what());
        } catch (const invalid_pattern & err) {
            throw std::logic_error("lowered pattern " + pattern + " is invalid: " + err.what());
        }
    }

    // The JSON text of one character a pattern's `.` matches: any but a line terminator.
    std::string _dot_rule() {
        return _add_rule("dot", json_text_class(LINE_TERMINATORS, true));
    }

    std::string _pattern_to_rule(const std::string & pattern, const std::string & name) {
        // A pattern matches anywhere in the string unless anchored.
        std::string sub_pattern    = pattern;
        const bool  anchored_start = !sub_pattern.empty() && sub_pattern.front() == '^';
        if (anchored_start) {
            sub_pattern.erase(0, 1);
        }
        bool anchored_end = false;
        if (!sub_pattern.empty() && sub_pattern.back() == '$') {
            size_t backslashes = 0;
            for (size_t k = sub_pattern.size() - 1; k > 0 && sub_pattern[k - 1] == '\\'; k--) {
                backslashes++;
            }
            if (backslashes % 2 == 0) {
                anchored_end = true;
                sub_pattern.pop_back();
            }
        }
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();
        int paren_depth = 0;

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            std::vector<literal_or_rule> seq;

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(_dot_rule(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else if (i + 1 < length && sub_pattern[i + 1] == '<' && i + 2 < length &&
                                   sub_pattern[i + 2] != '=' && sub_pattern[i + 2] != '!') {
                            // a named group matches as a group
                            auto close = sub_pattern.find('>', i + 2);
                            if (close == std::string::npos) {
                                throw invalid_pattern("unterminated group name");
                            }
                            i = close + 1;
                        } else {
                            // lookaround, inline flags, ...
                            throw unsupported_pattern("unsupported group syntax");
                        }
                    }
                    paren_depth++;
                    if (paren_depth > MAX_PATTERN_DEPTH) {
                        throw unsupported_pattern("pattern nesting too deep");
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (paren_depth == 0) {
                        throw invalid_pattern("unbalanced parentheses");
                    }
                    paren_depth--;
                    return join_seq();
                } else if (c == '^' || c == '$') {
                    throw unsupported_pattern("anchor inside the pattern");
                } else if (c == '[') {
                    codepoint_ranges ranges;
                    bool             negated = false;
                    parse_class<unsupported_pattern, invalid_pattern>(sub_pattern, i, ranges, negated);
                    auto expression = json_text_class(ranges, negated);
                    if (expression.empty()) {
                        throw unsupported_pattern("character class matches nothing");
                    }
                    seq.emplace_back(expression, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        throw unsupported_pattern("unterminated curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    if (nums.size() != 1 && nums.size() != 2) {
                        throw unsupported_pattern("wrong number of values in curly brackets");
                    }
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::logic_error &) {
                        throw unsupported_pattern("invalid number in curly brackets");
                    }
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else if (c == '\\') {
                    if (i == length - 1) {
                        throw invalid_pattern("trailing backslash");
                    }
                    codepoint_ranges ranges;
                    bool             negated = false;
                    if (class_escape(sub_pattern[i + 1], ranges, negated)) {
                        seq.emplace_back(json_text_class(ranges, negated), false);
                        i += 2;
                        continue;
                    }
                    uint32_t cp = 0;
                    if (!decode_escape(sub_pattern, i, cp)) {
                        throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                    }
                    seq.emplace_back(json_text_literal(cp), true);
                } else if (NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end()) {
                    // a stray ']' or '}'
                    throw unsupported_pattern(std::string("unsupported character: ") + c);
                } else {
                    // One character, as its JSON text; a following quantifier applies to it alone.
                    seq.emplace_back(json_text_literal(decode_utf8(sub_pattern, i)), true);
                }
            }
            return join_seq();
        };

        auto rule = to_rule(transform());
        if (paren_depth != 0) {
            throw invalid_pattern("unbalanced parentheses");
        }

        std::vector<std::string> parts = { "\"\\\"\"" };
        if (!anchored_start) {
            parts.push_back(_dot_rule() + "*");
        }
        if (!rule.empty()) {
            parts.push_back("(" + rule + ")");
        }
        if (!anchored_end && (anchored_start || !rule.empty())) {
            parts.push_back(_dot_rule() + "*");
        }
        parts.push_back("\"\\\"\"");
        return _add_rule(name, string_join(parts, " "));
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["]
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
    */
    // A string other than each of `strings`: branches spell a string's JSON
    // text character by character, and leave it at the first character none
    // continues with. Every branch is valid JSON string text.
    std::string _not_strings(const std::vector<std::string> & strings) {
        common_trie trie(strings);

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(size_t)> visit = [&](size_t idx) {
            const auto & node = trie.nodes[idx];
            codepoint_ranges rejects;
            auto first = true;
            for (const auto & [cpt, child] : node.children) {
                rejects.push_back({ cpt, cpt });
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "\"" << json_text_literal(cpt) << "\"";
                if (!trie.nodes[child].children.empty()) {
                    out << " (";
                    visit(child);
                    out << ")";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                out << " | " << json_text_class(rejects, true) << " " << char_rule << "*";
            }
        };
        visit(0);

        out << " )";
        if (trie.nodes[0].pattern < 0) {
            out << "?";
        }
        out << " [\"]";
        return out.str();
    }

    std::string _resolve_ref(const common_chat_schema_ref & schema) {
        auto it = schema.ref.find('#');
        std::string ref_fragment = it != std::string::npos ? schema.ref.substr(it + 1) : schema.ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(schema.ref) == _refs_being_resolved.end()) {
            if (!schema.target) {
                _errors.push_back("Unresolved $ref " + schema.ref);
                return "";
            }
            _refs_being_resolved.insert(schema.ref);
            ref_name = visit(*schema.target, ref_name);
            _refs_being_resolved.erase(schema.ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, const common_chat_schema *>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const common_chat_schema * additional_properties)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            std::string prop_rule_name = visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if (additional_properties) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties->kind() != common_chat_schema::KIND_ANY ? visit(*additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        if (required_props.empty() && optional_props.empty()) {
            return "\"{\" space \"}\"";
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " space \"}\"";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    common_chat_schema_converter() {
        _rules["space"] = SPACE_RULE;
    }

    std::string add_schema(const std::string & name, const common_chat_schema & schema) {
        return visit(schema, name);
    }

    static std::string _generate_constant_rule(const json & value) {
        return format_literal(value.dump());
    }

    std::string _visit_primitive(const std::string & rule_name, const std::string & type) {
        return _add_primitive(rule_name == "root" ? "root" : type, PRIMITIVE_RULES.at(type));
    }

    std::string _visit_numeric(const std::string & rule_name, const common_chat_schema_numeric & bounds, bool integral) {
        if (!bounds.bounded()) {
            return _visit_primitive(rule_name, integral ? "integer" : "number");
        }
        auto range = json_schema_numeric_range(bounds, integral);
        if (!range) {
            throw std::logic_error("lowered numeric range admits no value");
        }
        return _add_rule(rule_name, *range);
    }

    std::string visit(const common_chat_schema & schema, const std::string & name) {
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub_name  = name + (name.empty() ? "" : "-");

        switch (schema.kind()) {
            case common_chat_schema::KIND_REF:
                return _add_rule(rule_name, _resolve_ref(as<common_chat_schema_ref>(schema)));
            case common_chat_schema::KIND_ANY_OF:
                return _add_rule(rule_name, _generate_union_rule(name, as<common_chat_schema_any_of>(schema).children));
            case common_chat_schema::KIND_NEVER:
                throw std::logic_error("unsettled schema node");
            case common_chat_schema::KIND_CONST:
                return _add_rule(rule_name, _generate_constant_rule(as<common_chat_schema_const>(schema).value));
            case common_chat_schema::KIND_ENUM: {
                std::vector<std::string> enum_values;
                for (const auto & v : as<common_chat_schema_enum>(schema).values) {
                    enum_values.push_back(_generate_constant_rule(v));
                }
                return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ")");
            }
            case common_chat_schema::KIND_OBJECT: {
                const auto & obj = as<common_chat_schema_object>(schema);
                if (obj.properties.empty() && obj.additional() && obj.additional()->kind() == common_chat_schema::KIND_ANY) {
                    return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
                }
                std::vector<std::pair<std::string, const common_chat_schema *>> properties;
                std::unordered_set<std::string> required;
                for (const auto & prop : obj.properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (prop.required) {
                        required.insert(prop.name);
                    }
                }
                return _add_rule(rule_name, _build_object_rule(properties, required, name, obj.additional()));
            }
            case common_chat_schema::KIND_TUPLE: {
                const auto & items = as<common_chat_schema_tuple>(schema).items;
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(*items[i], sub_name + "tuple-" + std::to_string(i));
                }
                rule += " space \"]\"";
                return _add_rule(rule_name, rule);
            }
            case common_chat_schema::KIND_ARRAY: {
                const auto & arr = as<common_chat_schema_array>(schema);
                if (arr.items->kind() == common_chat_schema::KIND_ANY && arr.min_items == 0 && arr.max_items < 0) {
                    return _visit_primitive(rule_name, "array");
                }
                std::string item_rule_name = visit(*arr.items, sub_name + "item");
                int max_items = arr.max_items < 0 ? std::numeric_limits<int>::max() : arr.max_items;
                return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, arr.min_items, max_items, "\",\" space") + " space \"]\"");
            }
            case common_chat_schema::KIND_STRING: {
                const auto & str = as<common_chat_schema_string>(schema);
                if (!str.pattern.empty()) {
                    return _visit_pattern(str.pattern, rule_name);
                }
                if (str.format == common_chat_schema::FORMAT_UUID) {
                    return _visit_primitive(rule_name, "uuid");
                }
                if (str.format != common_chat_schema::FORMAT_NONE) {
                    std::string prim_name = std::string(str.format == common_chat_schema::FORMAT_DATE ? "date" : str.format == common_chat_schema::FORMAT_TIME ? "time" : "date-time") + "-string";
                    return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
                }
                if (str.min_length > 0 || str.max_length >= 0) {
                    std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
                    int max_len = str.max_length < 0 ? std::numeric_limits<int>::max() : str.max_length;
                    return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, str.min_length, max_len) + " \"\\\"\"");
                }
                return _visit_primitive(rule_name, "string");
            }
            case common_chat_schema::KIND_INTEGER:
                return _visit_numeric(rule_name, as<common_chat_schema_integer>(schema), true);
            case common_chat_schema::KIND_NUMBER:
                return _visit_numeric(rule_name, as<common_chat_schema_number>(schema), false);
            case common_chat_schema::KIND_BOOLEAN:
                return _visit_primitive(rule_name, "boolean");
            case common_chat_schema::KIND_NULL:
                return _visit_primitive(rule_name, "null");
            case common_chat_schema::KIND_ANY:
                return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        return "";
    }

    // Lowered schemas convert exactly; any conversion error is a defect.
    void check_errors() {
        if (!_errors.empty()) {
            throw std::logic_error("lowered JSON schema conversion failed: " + string_join(_errors, "; "));
        }
    }

    // Whether a pattern translates to a grammar.
    bool pattern_supported(const std::string & pattern) {
        try {
            _pattern_to_rule(pattern, "pattern");
            return true;
        } catch (const unsupported_pattern &) {
            return false;
        } catch (const invalid_pattern &) {
            return false;
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    return json_schema_to_grammar(common_chat_schema_from_json(schema));
}

bool json_schema_pattern_supported(const std::string & pattern) {
    return common_chat_schema_converter().pattern_supported(pattern);
}

std::string json_schema_to_grammar(const common_chat_schema_document & schema) {
    common_chat_schema_converter converter;
    converter.visit(*schema.root, "");
    converter.check_errors();
    return converter.format_grammar();
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb) {
    common_chat_schema_converter converter;
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const common_chat_schema & schema) {
            return converter.add_schema(name == "root" ? "" : name, schema);
        },
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}
