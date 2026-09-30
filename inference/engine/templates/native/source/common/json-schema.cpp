#include "json-schema.h"
#include "json-schema-to-grammar.h"
#include "templates-support.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

using json = common_json;

// ---------------------------------------------------------------------------
// Exact decimals

static std::string strip_leading_zeros(const std::string & digits) {
    auto first = digits.find_first_not_of('0');
    return first == std::string::npos ? "0" : digits.substr(first);
}

static std::string strip_trailing_zeros(const std::string & digits) {
    auto last = digits.find_last_not_of('0');
    return last == std::string::npos ? "" : digits.substr(0, last + 1);
}

common_chat_decimal common_chat_decimal::from_json(const common_json & value) {
    if (!value.is_number() || (value.is_number_float() && !std::isfinite(value.get<double>()))) {
        throw std::logic_error("not a finite JSON number: " + value.dump());
    }
    // Doubles dump as the shortest text that reads back as the same double.
    const std::string text = value.dump();
    common_chat_decimal result;
    size_t              i = 0;
    if (i < text.size() && text[i] == '-') {
        result.negative = true;
        i++;
    }
    std::string digits;
    long long   point = -1;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; i++) {
        if (text[i] == '.') {
            point = (long long) digits.size();
        } else {
            digits += text[i];
        }
    }
    if (point < 0) {
        point = (long long) digits.size();
    }
    if (i < text.size()) {
        point += std::stoll(text.substr(i + 1));
    }
    std::string integer;
    std::string fraction;
    if (point <= 0) {
        integer  = "0";
        fraction = std::string((size_t) -point, '0') + digits;
    } else if (point >= (long long) digits.size()) {
        integer = digits + std::string((size_t) point - digits.size(), '0');
    } else {
        integer  = digits.substr(0, (size_t) point);
        fraction = digits.substr((size_t) point);
    }
    result.integer  = strip_leading_zeros(integer);
    result.fraction = strip_trailing_zeros(fraction);
    if (result.is_zero()) {
        result.negative = false;
    }
    return result;
}

std::string common_chat_decimal::to_string() const {
    return (negative ? "-" : "") + integer + (fraction.empty() ? "" : "." + fraction);
}

static int compare_magnitudes(const common_chat_decimal & a, const common_chat_decimal & b) {
    if (a.integer.size() != b.integer.size()) {
        return a.integer.size() < b.integer.size() ? -1 : 1;
    }
    if (int c = a.integer.compare(b.integer)) {
        return c < 0 ? -1 : 1;
    }
    const size_t width = std::max(a.fraction.size(), b.fraction.size());
    const auto   fa    = a.fraction + std::string(width - a.fraction.size(), '0');
    const auto   fb    = b.fraction + std::string(width - b.fraction.size(), '0');
    int          c     = fa.compare(fb);
    return c < 0 ? -1 : c > 0 ? 1 : 0;
}

int common_chat_decimal_compare(const common_chat_decimal & a, const common_chat_decimal & b) {
    if (a.negative != b.negative) {
        return a.negative ? -1 : 1;
    }
    int magnitude = compare_magnitudes(a, b);
    return a.negative ? -magnitude : magnitude;
}

// ---------------------------------------------------------------------------
// Nodes

template <typename T>
common_chat_schema_ptr common_chat_schema_node<T>::clone() const {
    return std::make_unique<T>(static_cast<const T &>(*this));
}

template struct common_chat_schema_node<common_chat_schema_any>;
template struct common_chat_schema_node<common_chat_schema_never>;
template struct common_chat_schema_node<common_chat_schema_ref>;
template struct common_chat_schema_node<common_chat_schema_any_of>;
template struct common_chat_schema_node<common_chat_schema_const>;
template struct common_chat_schema_node<common_chat_schema_enum>;
template struct common_chat_schema_node<common_chat_schema_null>;
template struct common_chat_schema_node<common_chat_schema_boolean>;
template struct common_chat_schema_node<common_chat_schema_number>;
template struct common_chat_schema_node<common_chat_schema_integer>;
template struct common_chat_schema_node<common_chat_schema_string>;
template struct common_chat_schema_node<common_chat_schema_array>;
template struct common_chat_schema_node<common_chat_schema_tuple>;
template struct common_chat_schema_node<common_chat_schema_object>;

common_chat_schema_any_of::common_chat_schema_any_of(const common_chat_schema_any_of & other) :
    common_chat_schema_node(other) {
    for (const auto & child : other.children) {
        children.push_back(child->clone());
    }
}

common_chat_schema_array::common_chat_schema_array(const common_chat_schema_array & other) :
    common_chat_schema_node(other),
    items(other.items->clone()),
    min_items(other.min_items),
    max_items(other.max_items) {}

common_chat_schema_tuple::common_chat_schema_tuple(const common_chat_schema_tuple & other) :
    common_chat_schema_node(other) {
    for (const auto & item : other.items) {
        items.push_back(item->clone());
    }
}

common_chat_schema_object::common_chat_schema_object(const common_chat_schema_object & other) :
    common_chat_schema_node(other),
    undeclared(other.undeclared),
    additional_properties(other.additional_properties ? other.additional_properties->clone() : nullptr) {
    for (const auto & property : other.properties) {
        properties.push_back({ property.name, property.schema->clone(), property.required });
    }
}

const common_chat_schema * common_chat_schema_object::additional() const {
    static const common_chat_schema_any any;
    switch (undeclared) {
        case UNDECLARED_ANY:    return &any;
        case UNDECLARED_NONE:   return nullptr;
        case UNDECLARED_STATED: return additional_properties.get();
    }
    return nullptr;
}

const char * common_chat_schema_relaxation::reason_name(reason_kind reason) {
    switch (reason) {
        case REASON_UNENFORCED:       return "unenforced";
        case REASON_UNREPRESENTABLE:  return "unrepresentable";
        case REASON_UNSATISFIABLE:    return "unsatisfiable";
    }
    return "?";
}

bool common_chat_schema_relaxation::operator<(const common_chat_schema_relaxation & other) const {
    return std::tie(path, keyword, reason) < std::tie(other.path, other.keyword, other.reason);
}

bool common_chat_schema_relaxation::operator==(const common_chat_schema_relaxation & other) const {
    return path == other.path && keyword == other.keyword && reason == other.reason;
}

// ---------------------------------------------------------------------------
// Lowering

namespace {

using relaxation = common_chat_schema_relaxation;

// Every keyword with validation meaning, in every draft, is lowered below:
// enforced, or recorded as a relaxation. Type-specific keywords constrain only
// values of their type. Anything else is an annotation (including unknown
// keywords) and is ignored exactly.

const std::vector<std::string> OBJECT_KEYWORDS = {
    "properties", "required", "additionalProperties", "patternProperties", "propertyNames", "minProperties",
    "maxProperties", "dependentRequired", "dependentSchemas", "dependencies", "unevaluatedProperties",
};
const std::vector<std::string> ARRAY_KEYWORDS = {
    "items", "prefixItems", "additionalItems", "minItems", "maxItems", "uniqueItems", "contains", "minContains",
    "maxContains", "unevaluatedItems",
};
const std::vector<std::string> STRING_KEYWORDS  = { "minLength", "maxLength", "pattern" };
const std::vector<std::string> NUMERIC_KEYWORDS = { "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf" };

// Integers enumerated to enforce multipleOf exactly.
constexpr int64_t MAX_MULTIPLES = 256;

bool has_any(const json & schema, const std::vector<std::string> & keys) {
    return std::any_of(keys.begin(), keys.end(), [&](const std::string & key) { return schema.contains(key); });
}

std::string pointer_segment(const std::string & name) {
    std::string result;
    for (char c : name) {
        if (c == '~') {
            result += "~0";
        } else if (c == '/') {
            result += "~1";
        } else {
            result += c;
        }
    }
    return result;
}

[[noreturn]] void malformed(const std::string & path, const std::string & message) {
    throw std::logic_error("JSON schema was not admitted: " + path + ": " + message);
}

template <typename T, typename... Args>
std::unique_ptr<T> make(const std::string & path, Args &&... args) {
    auto node  = std::make_unique<T>(std::forward<Args>(args)...);
    node->path = path;
    return node;
}

common_chat_schema_ptr never(const std::string & path, const std::string & cause) {
    return make<common_chat_schema_never>(path, cause);
}

size_t utf8_length(const std::string & text) {
    size_t count = 0;
    for (unsigned char c : text) {
        if ((c & 0xC0) != 0x80) {
            count++;
        }
    }
    return count;
}

enum class admission { yes, no, unknown };

class common_chat_schema_builder {
    const json &                                    root_;
    common_chat_schema_document &                   doc_;
    // Targets by reference; null while the target is being built.
    std::map<std::string, common_chat_schema_ptr>   refs_;
    std::set<relaxation>                            relaxations_;

    void relax(const std::string & path, const std::string & keyword, relaxation::reason_kind reason) {
        relaxations_.insert({ path, keyword, reason });
    }

    // ----- references

    // The JSON value a local JSON pointer reference names, or null when the
    // reference is not a local JSON pointer.
    const json * resolve_pointer(const std::string & ref) {
        if (ref.empty() || ref[0] != '#') {
            return nullptr;
        }
        const json * target = &root_;
        if (ref.size() == 1) {
            return target;
        }
        if (ref[1] != '/') {
            return nullptr;  // an anchor
        }
        std::string decoded;
        for (size_t i = 1; i < ref.size(); i++) {
            if (ref[i] == '%' && i + 2 < ref.size() && std::isxdigit((unsigned char) ref[i + 1]) &&
                std::isxdigit((unsigned char) ref[i + 2])) {
                decoded += (char) std::stoi(ref.substr(i + 1, 2), nullptr, 16);
                i += 2;
            } else {
                decoded += ref[i];
            }
        }
        size_t start = 1;
        while (start <= decoded.size()) {
            size_t      end     = decoded.find('/', start);
            std::string segment = decoded.substr(start, end == std::string::npos ? std::string::npos : end - start);
            std::string name;
            for (size_t i = 0; i < segment.size(); i++) {
                if (segment[i] == '~' && i + 1 < segment.size() && (segment[i + 1] == '0' || segment[i + 1] == '1')) {
                    name += segment[i + 1] == '0' ? '~' : '/';
                    i++;
                } else {
                    name += segment[i];
                }
            }
            if (target->is_object() && target->contains(name)) {
                target = &target->at(name);
            } else if (target->is_array() && !name.empty() &&
                       std::all_of(name.begin(), name.end(), [](char c) { return std::isdigit((unsigned char) c); }) &&
                       std::stoull(name) < target->size()) {
                target = &target->at(std::stoull(name));
            } else {
                return nullptr;
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
        return target;
    }

    common_chat_schema_ptr build_ref(const json & value, const std::string & path) {
        if (!value.is_string()) {
            malformed(path, "$ref must be a string");
        }
        const std::string ref    = value.get<std::string>();
        const json *      target = resolve_pointer(ref);
        if (!target) {
            // Anchors and identifier-relative references resolve against
            // schema resources this lowering does not track.
            relax(path, "$ref", relaxation::REASON_UNENFORCED);
            return make<common_chat_schema_any>(path);
        }
        if (refs_.find(ref) == refs_.end()) {
            // Reserve the key first, so that a cycle back to this $ref stops here.
            refs_[ref] = nullptr;
            refs_[ref] = build(*target, ref);
        }
        return make<common_chat_schema_ref>(path, ref);
    }

    // The node a reference names, following reference chains; null while it is being built.
    const common_chat_schema * resolve(const common_chat_schema & node) {
        const common_chat_schema * current = &node;
        for (size_t hops = 0; current && current->kind() == common_chat_schema::KIND_REF; hops++) {
            if (hops > refs_.size()) {
                return nullptr;  // a cycle of aliases
            }
            auto it = refs_.find(static_cast<const common_chat_schema_ref *>(current)->ref);
            current = it == refs_.end() ? nullptr : it->second.get();
        }
        return current;
    }

    // ----- typed schemas

    common_chat_schema_ptr build_numeric(const json & schema, const std::string & path, bool integer) {
        common_chat_schema_numeric bounds;
        auto tighten = [&](std::optional<common_chat_numeric_bound> & slot, common_chat_numeric_bound bound, bool lower) {
            if (!slot) {
                slot = bound;
                return;
            }
            int c = common_chat_decimal_compare(bound.value, slot->value);
            if ((lower && c > 0) || (!lower && c < 0) || (c == 0 && bound.exclusive)) {
                slot = bound;
            }
        };
        // Draft 4 spells exclusive bounds as booleans modifying minimum and maximum.
        const bool exclusive_minimum = schema.contains("exclusiveMinimum") && schema.at("exclusiveMinimum").is_boolean() &&
                                       schema.at("exclusiveMinimum").get<bool>();
        const bool exclusive_maximum = schema.contains("exclusiveMaximum") && schema.at("exclusiveMaximum").is_boolean() &&
                                       schema.at("exclusiveMaximum").get<bool>();
        if (schema.contains("minimum")) {
            tighten(bounds.minimum, { common_chat_decimal::from_json(schema.at("minimum")), exclusive_minimum }, true);
        }
        if (schema.contains("exclusiveMinimum") && schema.at("exclusiveMinimum").is_number()) {
            tighten(bounds.minimum, { common_chat_decimal::from_json(schema.at("exclusiveMinimum")), true }, true);
        }
        if (schema.contains("maximum")) {
            tighten(bounds.maximum, { common_chat_decimal::from_json(schema.at("maximum")), exclusive_maximum }, false);
        }
        if (schema.contains("exclusiveMaximum") && schema.at("exclusiveMaximum").is_number()) {
            tighten(bounds.maximum, { common_chat_decimal::from_json(schema.at("exclusiveMaximum")), true }, false);
        }
        if (bounds.bounded() && !json_schema_numeric_range(bounds, integer)) {
            return never(path, "minimum");
        }
        if (schema.contains("multipleOf")) {
            auto enumerated = integer ? enumerate_multiples(schema.at("multipleOf"), bounds, path) : nullptr;
            if (enumerated) {
                return enumerated;
            }
            relax(path, "multipleOf", relaxation::REASON_UNENFORCED);
        }
        if (integer) {
            auto node = make<common_chat_schema_integer>(path);
            static_cast<common_chat_schema_numeric &>(*node) = bounds;
            return node;
        }
        auto node = make<common_chat_schema_number>(path);
        static_cast<common_chat_schema_numeric &>(*node) = bounds;
        return node;
    }

    // The integers within the bounds that are multiples of an integral divisor,
    // when there are few enough to list; otherwise null.
    common_chat_schema_ptr enumerate_multiples(const json & divisor, const common_chat_schema_numeric & bounds, const std::string & path) {
        auto k = common_chat_decimal::from_json(divisor);
        if (!k.is_integer() || k.negative || k.is_zero() || !bounds.minimum || !bounds.maximum ||
            k.integer.size() > 15 || bounds.minimum->value.integer.size() > 15 || bounds.maximum->value.integer.size() > 15) {
            return nullptr;
        }
        auto to_double = [](const common_chat_decimal & d) {
            return std::stod(d.to_string());
        };
        const int64_t step  = std::stoll(k.integer);
        const double  low   = to_double(bounds.minimum->value);
        const double  high  = to_double(bounds.maximum->value);
        int64_t       first = (int64_t) std::ceil(low / (double) step) * step;
        if ((double) first == low && bounds.minimum->exclusive) {
            first += step;
        }
        int64_t last = (int64_t) std::floor(high / (double) step) * step;
        if ((double) last == high && bounds.maximum->exclusive) {
            last -= step;
        }
        if (first > last) {
            return never(path, "multipleOf");
        }
        if ((last - first) / step + 1 > MAX_MULTIPLES) {
            return nullptr;
        }
        auto node = make<common_chat_schema_enum>(path);
        for (int64_t value = first; value <= last; value += step) {
            node->values.push_back(value);
        }
        return node;
    }

    common_chat_schema_ptr build_string(const json & schema, const std::string & path) {
        auto node        = make<common_chat_schema_string>(path);
        node->min_length = count(schema, "minLength", 0, path);
        node->max_length = count(schema, "maxLength", -1, path);
        if (node->max_length >= 0 && node->min_length > node->max_length) {
            return never(path, "minLength");
        }
        if (schema.contains("pattern")) {
            const auto & pattern = schema.at("pattern");
            if (!pattern.is_string()) {
                malformed(path, "pattern must be a string");
            }
            if (json_schema_pattern_supported(pattern.get<std::string>())) {
                node->pattern = pattern.get<std::string>();
            } else {
                relax(path, "pattern", relaxation::REASON_UNENFORCED);
            }
        }
        if (!node->pattern.empty()) {
            // A pattern's grammar does not count characters.
            if (schema.contains("minLength") && node->min_length > 0) {
                relax(path, "minLength", relaxation::REASON_UNENFORCED);
            }
            if (schema.contains("maxLength")) {
                relax(path, "maxLength", relaxation::REASON_UNENFORCED);
            }
            node->min_length = 0;
            node->max_length = -1;
        } else if (node->min_length == 0 && node->max_length < 0 && schema.contains("format") &&
                   schema.at("format").is_string()) {
            // Formats are annotations; recognized ones shape the string.
            const auto format = schema.at("format").get<std::string>();
            if (format == "date") {
                node->format = common_chat_schema::FORMAT_DATE;
            } else if (format == "time") {
                node->format = common_chat_schema::FORMAT_TIME;
            } else if (format == "date-time") {
                node->format = common_chat_schema::FORMAT_DATE_TIME;
            } else if (format == "uuid" ||
                       (format.size() == 5 && format.compare(0, 4, "uuid") == 0 && format[4] >= '1' && format[4] <= '5')) {
                node->format = common_chat_schema::FORMAT_UUID;
            }
        }
        return node;
    }

    // A non-negative count keyword, saturated to int.
    static int count(const json & schema, const char * key, int fallback, const std::string & path) {
        if (!schema.contains(key)) {
            return fallback;
        }
        const auto & value = schema.at(key);
        if (!value.is_number() || value.get<double>() < 0) {
            malformed(path, std::string(key) + " must be a non-negative integer");
        }
        return (int) std::min(value.get<double>(), (double) std::numeric_limits<int>::max());
    }

    common_chat_schema_ptr build_array(const json & schema, const std::string & path) {
        const int min_items = count(schema, "minItems", 0, path);
        const int max_items = count(schema, "maxItems", -1, path);
        if (max_items >= 0 && min_items > max_items) {
            return never(path, "minItems");
        }
        if (schema.contains("uniqueItems") && schema.at("uniqueItems") == true) {
            relax(path, "uniqueItems", relaxation::REASON_UNENFORCED);
        }
        if (schema.contains("contains")) {
            relax(path, "contains", relaxation::REASON_UNENFORCED);
        }
        if (schema.contains("unevaluatedItems") && schema.at("unevaluatedItems") != true) {
            relax(path, "unevaluatedItems", relaxation::REASON_UNENFORCED);
        }

        // Draft 2020-12 lists leading items in prefixItems and the rest in
        // items; earlier drafts list them in items and the rest in additionalItems.
        const json * prefix = nullptr;
        const json * rest   = nullptr;
        std::string  prefix_key;
        std::string  rest_key;
        if (schema.contains("prefixItems")) {
            prefix     = &schema.at("prefixItems");
            prefix_key = "prefixItems";
            if (schema.contains("items")) {
                rest     = &schema.at("items");
                rest_key = "items";
            }
        } else if (schema.contains("items") && schema.at("items").is_array()) {
            prefix     = &schema.at("items");
            prefix_key = "items";
            if (schema.contains("additionalItems")) {
                rest     = &schema.at("additionalItems");
                rest_key = "additionalItems";
            }
        } else if (schema.contains("items")) {
            rest     = &schema.at("items");
            rest_key = "items";
        }

        auto rest_schema = rest ? build(*rest, path + "/" + rest_key) : make<common_chat_schema_any>(path);
        if (prefix && prefix->is_array() && prefix->empty()) {
            prefix = nullptr;
        }
        if (!prefix) {
            auto node       = make<common_chat_schema_array>(path);
            node->items     = std::move(rest_schema);
            node->min_items = min_items;
            node->max_items = max_items;
            return node;
        }
        if (!prefix->is_array()) {
            malformed(path, prefix_key + " must be an array");
        }
        std::vector<common_chat_schema_ptr> items;
        for (size_t i = 0; i < prefix->size(); i++) {
            items.push_back(build(prefix->at(i), path + "/" + prefix_key + "/" + std::to_string(i)));
        }
        // A tuple admits exactly its items, a prefix every valid array can take
        // when that many items are allowed.
        size_t length = items.size();
        if (max_items >= 0 && (size_t) max_items < length) {
            length = (size_t) max_items;
        }
        if ((size_t) min_items <= length) {
            auto node = make<common_chat_schema_tuple>(path);
            items.resize(length);
            node->items = std::move(items);
            return node;
        }
        if (rest_schema->kind() == common_chat_schema::KIND_NEVER) {
            return never(path, "minItems");
        }
        // More items are required than the prefix lists: any listed schema may
        // appear at any position.
        relax(path, prefix_key, relaxation::REASON_UNENFORCED);
        auto alternatives  = make<common_chat_schema_any_of>(path);
        alternatives->children = std::move(items);
        alternatives->children.push_back(std::move(rest_schema));
        auto node       = make<common_chat_schema_array>(path);
        node->items     = std::move(alternatives);
        node->min_items = min_items;
        node->max_items = max_items;
        return node;
    }

    common_chat_schema_ptr build_object(const json & schema, const std::string & path) {
        auto node = make<common_chat_schema_object>(path);
        if (schema.contains("properties")) {
            const auto & properties = schema.at("properties");
            if (!properties.is_object()) {
                malformed(path, "properties must be an object");
            }
            for (const auto & [name, property] : properties.items()) {
                node->properties.push_back({ name, build(property, path + "/properties/" + pointer_segment(name)), false });
            }
        }
        // An object that lists properties is closed unless its schema
        // explicitly allows more.
        node->undeclared = schema.contains("properties") ? common_chat_schema_object::UNDECLARED_NONE
                                                         : common_chat_schema_object::UNDECLARED_ANY;
        if (schema.contains("additionalProperties")) {
            const auto & additional = schema.at("additionalProperties");
            node->undeclared        = common_chat_schema_object::UNDECLARED_STATED;
            if (additional.is_boolean()) {
                if (additional.get<bool>()) {
                    node->additional_properties = make<common_chat_schema_any>(path + "/additionalProperties");
                }
            } else {
                node->additional_properties = build(additional, path + "/additionalProperties");
            }
        }
        const bool pattern_properties = schema.contains("patternProperties") && schema.at("patternProperties").is_object() &&
                                        !schema.at("patternProperties").empty();
        if (schema.contains("required")) {
            const auto & required = schema.at("required");
            if (!required.is_array()) {
                malformed(path, "required must be an array");
            }
            for (const auto & entry : required) {
                if (!entry.is_string()) {
                    malformed(path, "required names must be strings");
                }
                const auto name = entry.get<std::string>();
                auto it = std::find_if(node->properties.begin(), node->properties.end(),
                                       [&](const common_chat_schema_property & p) { return p.name == name; });
                if (it != node->properties.end()) {
                    it->required = true;
                    continue;
                }
                // A required name without a declared schema takes the schema of
                // undeclared properties.
                const auto property_path = path + "/properties/" + pointer_segment(name);
                common_chat_schema_ptr property_schema;
                if (node->undeclared != common_chat_schema_object::UNDECLARED_STATED) {
                    property_schema = make<common_chat_schema_any>(property_path);
                } else if (node->additional_properties) {
                    property_schema = node->additional_properties->clone();
                } else if (pattern_properties) {
                    relax(path, "patternProperties", relaxation::REASON_UNENFORCED);
                    property_schema = make<common_chat_schema_any>(property_path);
                } else {
                    return never(path, "required");
                }
                node->properties.push_back({ name, std::move(property_schema), true });
            }
        }
        const bool open = node->additional() != nullptr;
        if (pattern_properties && open) {
            relax(path, "patternProperties", relaxation::REASON_UNENFORCED);
        }
        if (schema.contains("propertyNames") && schema.at("propertyNames") != true &&
            schema.at("propertyNames") != json::object()) {
            relax(path, "propertyNames", relaxation::REASON_UNENFORCED);
        }
        if (schema.contains("minProperties")) {
            const auto required = std::count_if(node->properties.begin(), node->properties.end(),
                                                [](const common_chat_schema_property & p) { return p.required; });
            if (count(schema, "minProperties", 0, path) > required) {
                relax(path, "minProperties", relaxation::REASON_UNENFORCED);
            }
        }
        if (schema.contains("maxProperties") &&
            (open || (size_t) count(schema, "maxProperties", 0, path) < node->properties.size())) {
            relax(path, "maxProperties", relaxation::REASON_UNENFORCED);
        }
        for (const auto * key : { "dependentRequired", "dependentSchemas", "dependencies" }) {
            if (schema.contains(key)) {
                relax(path, key, relaxation::REASON_UNENFORCED);
            }
        }
        if (schema.contains("unevaluatedProperties") && schema.at("unevaluatedProperties") != true) {
            relax(path, "unevaluatedProperties", relaxation::REASON_UNENFORCED);
        }
        return node;
    }

    common_chat_schema_ptr build_type(const std::string & type, const json & schema, const std::string & path) {
        if (type == "null") {
            return make<common_chat_schema_null>(path);
        }
        if (type == "boolean") {
            return make<common_chat_schema_boolean>(path);
        }
        if (type == "integer" || type == "number") {
            return build_numeric(schema, path, type == "integer");
        }
        if (type == "string") {
            return build_string(schema, path);
        }
        if (type == "array") {
            return build_array(schema, path);
        }
        if (type == "object") {
            return build_object(schema, path);
        }
        malformed(path, "unknown type " + type);
    }

    // The schema's type and type-specific keywords. Without a type, the types
    // whose keywords appear; without either, any value.
    common_chat_schema_ptr build_typed(const json & schema, const std::string & path) {
        std::vector<std::string> types;
        if (schema.contains("type")) {
            const auto & type = schema.at("type");
            if (type.is_string()) {
                types.push_back(type.get<std::string>());
            } else if (type.is_array() && !type.empty()) {
                for (const auto & entry : type) {
                    if (!entry.is_string()) {
                        malformed(path, "type entries must be strings");
                    }
                    types.push_back(entry.get<std::string>());
                }
            } else {
                malformed(path, "type must be a string or a nonempty array");
            }
        } else {
            if (has_any(schema, OBJECT_KEYWORDS)) {
                types.push_back("object");
            }
            if (has_any(schema, ARRAY_KEYWORDS)) {
                types.push_back("array");
            }
            if (has_any(schema, STRING_KEYWORDS)) {
                types.push_back("string");
            }
            if (has_any(schema, NUMERIC_KEYWORDS)) {
                types.push_back("number");
            }
        }
        if (types.empty()) {
            return make<common_chat_schema_any>(path);
        }
        if (types.size() == 1) {
            return build_type(types[0], schema, path);
        }
        auto node = make<common_chat_schema_any_of>(path);
        for (const auto & type : types) {
            node->children.push_back(build_type(type, schema, path));
        }
        return node;
    }

    common_chat_schema_ptr build_alternatives(const json & alternatives, const std::string & path) {
        if (!alternatives.is_array() || alternatives.empty()) {
            malformed(path, "alternatives must be a nonempty array");
        }
        auto node = make<common_chat_schema_any_of>(path);
        for (size_t i = 0; i < alternatives.size(); i++) {
            node->children.push_back(build(alternatives.at(i), path + "/" + std::to_string(i)));
        }
        return node;
    }

    // ----- composition

    // Whether a value can satisfy the node: decided for scalars, unknown for
    // structures the node constrains.
    admission admits(const common_chat_schema & node, const json & value, int depth = 0) {
        if (depth > 32) {
            return admission::unknown;
        }
        switch (node.kind()) {
            case common_chat_schema::KIND_ANY:
                return admission::yes;
            case common_chat_schema::KIND_NEVER:
                return admission::no;
            case common_chat_schema::KIND_REF: {
                const auto * target = resolve(node);
                return target ? admits(*target, value, depth + 1) : admission::unknown;
            }
            case common_chat_schema::KIND_ANY_OF: {
                bool unknown = false;
                for (const auto & child : static_cast<const common_chat_schema_any_of &>(node).children) {
                    auto result = admits(*child, value, depth + 1);
                    if (result == admission::yes) {
                        return admission::yes;
                    }
                    unknown = unknown || result == admission::unknown;
                }
                return unknown ? admission::unknown : admission::no;
            }
            case common_chat_schema::KIND_CONST:
                return static_cast<const common_chat_schema_const &>(node).value == value ? admission::yes : admission::no;
            case common_chat_schema::KIND_ENUM: {
                const auto & values = static_cast<const common_chat_schema_enum &>(node).values;
                return std::find(values.begin(), values.end(), value) != values.end() ? admission::yes : admission::no;
            }
            case common_chat_schema::KIND_NULL:
                return value.is_null() ? admission::yes : admission::no;
            case common_chat_schema::KIND_BOOLEAN:
                return value.is_boolean() ? admission::yes : admission::no;
            case common_chat_schema::KIND_NUMBER:
            case common_chat_schema::KIND_INTEGER: {
                if (!value.is_number() || (value.is_number_float() && !std::isfinite(value.get<double>()))) {
                    return admission::no;
                }
                const auto   decimal = common_chat_decimal::from_json(value);
                const auto & bounds  = node.kind() == common_chat_schema::KIND_NUMBER ?
                    static_cast<const common_chat_schema_numeric &>(static_cast<const common_chat_schema_number &>(node)) :
                    static_cast<const common_chat_schema_numeric &>(static_cast<const common_chat_schema_integer &>(node));
                if (node.kind() == common_chat_schema::KIND_INTEGER && !decimal.is_integer()) {
                    return admission::no;
                }
                if (bounds.minimum) {
                    int c = common_chat_decimal_compare(decimal, bounds.minimum->value);
                    if (c < 0 || (c == 0 && bounds.minimum->exclusive)) {
                        return admission::no;
                    }
                }
                if (bounds.maximum) {
                    int c = common_chat_decimal_compare(decimal, bounds.maximum->value);
                    if (c > 0 || (c == 0 && bounds.maximum->exclusive)) {
                        return admission::no;
                    }
                }
                return admission::yes;
            }
            case common_chat_schema::KIND_STRING: {
                if (!value.is_string()) {
                    return admission::no;
                }
                const auto & string = static_cast<const common_chat_schema_string &>(node);
                const auto   text   = value.get<std::string>();
                const auto   length = utf8_length(text);
                if (length < (size_t) string.min_length || (string.max_length >= 0 && length > (size_t) string.max_length)) {
                    return admission::no;
                }
                if (!string.pattern.empty()) {
                    // std::regex matches bytes, which are characters only in ASCII.
                    auto ascii = [](const std::string & s) {
                        return std::all_of(s.begin(), s.end(), [](unsigned char c) { return c < 0x80; });
                    };
                    if (!ascii(text) || !ascii(string.pattern)) {
                        return admission::unknown;
                    }
                    try {
                        return std::regex_search(text, std::regex(string.pattern, std::regex::ECMAScript)) ?
                                   admission::yes : admission::no;
                    } catch (const std::regex_error &) {
                        return admission::unknown;
                    }
                }
                return admission::yes;
            }
            case common_chat_schema::KIND_ARRAY:
            case common_chat_schema::KIND_TUPLE:
                return value.is_array() ? admission::unknown : admission::no;
            case common_chat_schema::KIND_OBJECT:
                return value.is_object() ? admission::unknown : admission::no;
        }
        return admission::unknown;
    }

    common_chat_schema_ptr intersect_values(std::vector<json> values, const common_chat_schema & other, const std::string & path) {
        std::vector<json> kept;
        for (auto & value : values) {
            switch (admits(other, value)) {
                case admission::yes:
                    kept.push_back(std::move(value));
                    break;
                case admission::no:
                    break;
                case admission::unknown:
                    relax(other.path, "enum", relaxation::REASON_UNENFORCED);
                    kept.push_back(std::move(value));
                    break;
            }
        }
        if (kept.empty()) {
            return never(path, "enum");
        }
        auto node    = make<common_chat_schema_enum>(path);
        node->values = std::move(kept);
        return node;
    }

    static std::vector<json> listed_values(const common_chat_schema & node) {
        if (node.kind() == common_chat_schema::KIND_CONST) {
            return { static_cast<const common_chat_schema_const &>(node).value };
        }
        return static_cast<const common_chat_schema_enum &>(node).values;
    }

    static const common_chat_schema_numeric & numeric(const common_chat_schema & node) {
        if (node.kind() == common_chat_schema::KIND_NUMBER) {
            return static_cast<const common_chat_schema_number &>(node);
        }
        return static_cast<const common_chat_schema_integer &>(node);
    }

    // Values that satisfy both nodes. Keywords that cannot be combined are
    // recorded; the result then admits a superset of the intersection.
    common_chat_schema_ptr intersect(common_chat_schema_ptr a, common_chat_schema_ptr b, const std::string & path) {
        using K = common_chat_schema;
        if (a->kind() == K::KIND_ANY) {
            return b;
        }
        if (b->kind() == K::KIND_ANY) {
            return a;
        }
        if (a->kind() == K::KIND_NEVER) {
            return a;
        }
        if (b->kind() == K::KIND_NEVER) {
            return b;
        }
        for (auto * side : { &a, &b }) {
            if ((*side)->kind() == K::KIND_REF) {
                auto &       other  = side == &a ? b : a;
                const auto * target = resolve(**side);
                if (!target) {
                    // A reference into the schema being built cannot be combined.
                    relax(path, "$ref", relaxation::REASON_UNENFORCED);
                    return std::move(other);
                }
                return intersect(target->clone(), std::move(other), path);
            }
        }
        for (auto * side : { &a, &b }) {
            if ((*side)->kind() == K::KIND_ANY_OF) {
                auto & other  = side == &a ? b : a;
                auto   result = make<common_chat_schema_any_of>((*side)->path);
                for (auto & child : static_cast<common_chat_schema_any_of &>(**side).children) {
                    result->children.push_back(intersect(std::move(child), other->clone(), path));
                }
                return result;
            }
        }
        const bool a_listed = a->kind() == K::KIND_ENUM || a->kind() == K::KIND_CONST;
        const bool b_listed = b->kind() == K::KIND_ENUM || b->kind() == K::KIND_CONST;
        if (a_listed) {
            return intersect_values(listed_values(*a), *b, a->path);
        }
        if (b_listed) {
            return intersect_values(listed_values(*b), *a, b->path);
        }
        const bool a_numeric = a->kind() == K::KIND_NUMBER || a->kind() == K::KIND_INTEGER;
        const bool b_numeric = b->kind() == K::KIND_NUMBER || b->kind() == K::KIND_INTEGER;
        if (a_numeric && b_numeric) {
            common_chat_schema_numeric bounds = numeric(*a);
            const auto &               other  = numeric(*b);
            if (other.minimum) {
                if (!bounds.minimum) {
                    bounds.minimum = other.minimum;
                } else {
                    int c = common_chat_decimal_compare(other.minimum->value, bounds.minimum->value);
                    if (c > 0 || (c == 0 && other.minimum->exclusive)) {
                        bounds.minimum = other.minimum;
                    }
                }
            }
            if (other.maximum) {
                if (!bounds.maximum) {
                    bounds.maximum = other.maximum;
                } else {
                    int c = common_chat_decimal_compare(other.maximum->value, bounds.maximum->value);
                    if (c < 0 || (c == 0 && other.maximum->exclusive)) {
                        bounds.maximum = other.maximum;
                    }
                }
            }
            const bool integer = a->kind() == K::KIND_INTEGER || b->kind() == K::KIND_INTEGER;
            if (bounds.bounded() && !json_schema_numeric_range(bounds, integer)) {
                return never(path, "minimum");
            }
            if (integer) {
                auto node = make<common_chat_schema_integer>(a->path);
                static_cast<common_chat_schema_numeric &>(*node) = bounds;
                return node;
            }
            auto node = make<common_chat_schema_number>(a->path);
            static_cast<common_chat_schema_numeric &>(*node) = bounds;
            return node;
        }
        if (a->kind() != b->kind()) {
            const bool arrays = (a->kind() == K::KIND_ARRAY || a->kind() == K::KIND_TUPLE) &&
                                (b->kind() == K::KIND_ARRAY || b->kind() == K::KIND_TUPLE);
            if (!arrays) {
                return never(path, "type");
            }
        }
        switch (a->kind()) {
            case K::KIND_NULL:
            case K::KIND_BOOLEAN:
                return a;
            case K::KIND_STRING: {
                auto &       left  = static_cast<common_chat_schema_string &>(*a);
                const auto & right = static_cast<const common_chat_schema_string &>(*b);
                left.min_length    = std::max(left.min_length, right.min_length);
                if (right.max_length >= 0) {
                    left.max_length = left.max_length < 0 ? right.max_length : std::min(left.max_length, right.max_length);
                }
                if (left.max_length >= 0 && left.min_length > left.max_length) {
                    return never(path, "minLength");
                }
                if (!right.pattern.empty()) {
                    if (left.pattern.empty()) {
                        left.pattern = right.pattern;
                    } else if (left.pattern != right.pattern) {
                        relax(right.path, "pattern", relaxation::REASON_UNENFORCED);
                    }
                }
                if (!left.pattern.empty() && (left.min_length > 0 || left.max_length >= 0)) {
                    relax(path, left.min_length > 0 ? "minLength" : "maxLength", relaxation::REASON_UNENFORCED);
                    left.min_length = 0;
                    left.max_length = -1;
                }
                if (left.format == K::FORMAT_NONE && left.pattern.empty() && left.min_length == 0 && left.max_length < 0) {
                    left.format = right.format;
                }
                if (!left.pattern.empty() || left.min_length > 0 || left.max_length >= 0) {
                    left.format = K::FORMAT_NONE;
                }
                return a;
            }
            case K::KIND_ARRAY:
            case K::KIND_TUPLE:
                return intersect_arrays(std::move(a), std::move(b), path);
            case K::KIND_OBJECT:
                return intersect_objects(std::move(a), std::move(b), path);
            default:
                break;
        }
        throw std::logic_error("unhandled schema intersection");
    }

    common_chat_schema_ptr intersect_arrays(common_chat_schema_ptr a, common_chat_schema_ptr b, const std::string & path) {
        using K = common_chat_schema;
        if (a->kind() == K::KIND_ARRAY && b->kind() == K::KIND_TUPLE) {
            std::swap(a, b);
        }
        if (a->kind() == K::KIND_ARRAY) {
            auto &       left  = static_cast<common_chat_schema_array &>(*a);
            auto &       right = static_cast<common_chat_schema_array &>(*b);
            left.items         = intersect(std::move(left.items), std::move(right.items), path);
            left.min_items     = std::max(left.min_items, right.min_items);
            if (right.max_items >= 0) {
                left.max_items = left.max_items < 0 ? right.max_items : std::min(left.max_items, right.max_items);
            }
            if (left.max_items >= 0 && left.min_items > left.max_items) {
                return never(path, "minItems");
            }
            return a;
        }
        auto & tuple = static_cast<common_chat_schema_tuple &>(*a);
        if (b->kind() == K::KIND_ARRAY) {
            auto & array = static_cast<common_chat_schema_array &>(*b);
            if (tuple.items.size() < (size_t) array.min_items ||
                (array.max_items >= 0 && tuple.items.size() > (size_t) array.max_items)) {
                return never(path, "minItems");
            }
            for (auto & item : tuple.items) {
                item = intersect(std::move(item), array.items->clone(), path);
            }
            return a;
        }
        // Both are prefixes; the longer one's length is kept.
        auto & other = static_cast<common_chat_schema_tuple &>(*b);
        if (other.items.size() > tuple.items.size()) {
            std::swap(tuple.items, other.items);
        }
        for (size_t i = 0; i < other.items.size(); i++) {
            tuple.items[i] = intersect(std::move(tuple.items[i]), std::move(other.items[i]), path);
        }
        return a;
    }

    common_chat_schema_ptr intersect_objects(common_chat_schema_ptr a, common_chat_schema_ptr b, const std::string & path) {
        using object = common_chat_schema_object;
        auto & left  = static_cast<object &>(*a);
        auto & right = static_cast<object &>(*b);
        // The schema a property undeclared on one side must satisfy there, or
        // null when that side forbids it. Closing unstated objects is this
        // lowering's policy, not the schema's, so it applies to the result only.
        auto undeclared = [](const object & side) -> const common_chat_schema * {
            static const common_chat_schema_any any;
            return side.undeclared == object::UNDECLARED_STATED ? side.additional_properties.get() : &any;
        };
        std::vector<common_chat_schema_property> properties;
        auto merge_side = [&](object & side, object & other) -> bool {
            for (auto & property : side.properties) {
                auto it = std::find_if(properties.begin(), properties.end(),
                                       [&](const common_chat_schema_property & p) { return p.name == property.name; });
                if (it != properties.end()) {
                    continue;  // merged from the other side
                }
                auto match = std::find_if(other.properties.begin(), other.properties.end(),
                                          [&](const common_chat_schema_property & p) { return p.name == property.name; });
                if (match != other.properties.end()) {
                    properties.push_back({ property.name,
                                           intersect(std::move(property.schema), std::move(match->schema), path),
                                           property.required || match->required });
                    continue;
                }
                const auto * constraint = undeclared(other);
                if (!constraint) {
                    if (property.required) {
                        return false;
                    }
                    continue;
                }
                properties.push_back({ property.name, intersect(std::move(property.schema), constraint->clone(), path),
                                       property.required });
            }
            return true;
        };
        if (!merge_side(left, right) || !merge_side(right, left)) {
            return never(path, "additionalProperties");
        }
        auto node        = make<object>(left.path);
        node->properties = std::move(properties);
        const bool left_stated  = left.undeclared == object::UNDECLARED_STATED;
        const bool right_stated = right.undeclared == object::UNDECLARED_STATED;
        if (left_stated || right_stated) {
            node->undeclared = object::UNDECLARED_STATED;
            if (left_stated && right_stated) {
                if (left.additional_properties && right.additional_properties) {
                    node->additional_properties =
                        intersect(std::move(left.additional_properties), std::move(right.additional_properties), path);
                }
            } else {
                node->additional_properties = std::move(left_stated ? left.additional_properties : right.additional_properties);
            }
        } else if (left.undeclared == object::UNDECLARED_NONE || right.undeclared == object::UNDECLARED_NONE) {
            node->undeclared = object::UNDECLARED_NONE;
        }
        return node;
    }

    // The types a node admits, through references built so far; null when a
    // reference is still being built.
    std::optional<common_chat_schema::type_set> types_of(const common_chat_schema & node, int depth = 0) {
        if (depth > 32) {
            return std::nullopt;
        }
        switch (node.kind()) {
            case common_chat_schema::KIND_REF: {
                const auto * target = resolve(node);
                if (!target) {
                    return std::nullopt;
                }
                return types_of(*target, depth + 1);
            }
            case common_chat_schema::KIND_ANY_OF: {
                common_chat_schema::type_set types;
                for (const auto & child : static_cast<const common_chat_schema_any_of &>(node).children) {
                    auto child_types = types_of(*child, depth + 1);
                    if (!child_types) {
                        return std::nullopt;
                    }
                    types |= *child_types;
                }
                return types;
            }
            default:
                return node.value_types();
        }
    }

    // Whether no value satisfies two alternatives: different types, different
    // listed values, or objects that require one property with different values.
    bool disjoint(const common_chat_schema & a, const common_chat_schema & b) {
        const auto * left  = resolve(a);
        const auto * right = resolve(b);
        if (!left || !right) {
            return false;
        }
        auto left_types  = types_of(*left);
        auto right_types = types_of(*right);
        if (!left_types || !right_types) {
            return false;
        }
        auto types = *left_types;
        types &= *right_types;
        if (types.empty()) {
            return true;
        }
        auto listed = [](const common_chat_schema & node) {
            return node.kind() == common_chat_schema::KIND_ENUM || node.kind() == common_chat_schema::KIND_CONST;
        };
        if (listed(*left) && listed(*right)) {
            for (const auto & value : listed_values(*left)) {
                const auto others = listed_values(*right);
                if (std::find(others.begin(), others.end(), value) != others.end()) {
                    return false;
                }
            }
            return true;
        }
        if (left->kind() == common_chat_schema::KIND_OBJECT && right->kind() == common_chat_schema::KIND_OBJECT) {
            const auto & lo = static_cast<const common_chat_schema_object &>(*left);
            const auto & ro = static_cast<const common_chat_schema_object &>(*right);
            for (const auto & lp : lo.properties) {
                for (const auto & rp : ro.properties) {
                    if (lp.name == rp.name && lp.required && rp.required && disjoint(*lp.schema, *rp.schema)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    common_chat_schema_ptr build(const json & schema, const std::string & path) {
        if (schema.is_boolean()) {
            if (schema.get<bool>()) {
                return make<common_chat_schema_any>(path);
            }
            return never(path, "false");
        }
        if (!schema.is_object()) {
            malformed(path, "a schema must be an object or a boolean");
        }
        std::vector<common_chat_schema_ptr> parts;
        parts.push_back(build_typed(schema, path));
        if (schema.contains("enum")) {
            const auto & values = schema.at("enum");
            if (!values.is_array() || values.empty()) {
                malformed(path, "enum must be a nonempty array");
            }
            auto node = make<common_chat_schema_enum>(path);
            for (const auto & value : values) {
                node->values.push_back(value);
            }
            parts.push_back(std::move(node));
        }
        if (schema.contains("const")) {
            parts.push_back(make<common_chat_schema_const>(path, schema.at("const")));
        }
        if (schema.contains("$ref")) {
            parts.push_back(build_ref(schema.at("$ref"), path));
        }
        if (schema.contains("allOf")) {
            const auto & children = schema.at("allOf");
            if (!children.is_array() || children.empty()) {
                malformed(path, "allOf must be a nonempty array");
            }
            for (size_t i = 0; i < children.size(); i++) {
                parts.push_back(build(children.at(i), path + "/allOf/" + std::to_string(i)));
            }
        }
        if (schema.contains("anyOf")) {
            parts.push_back(build_alternatives(schema.at("anyOf"), path + "/anyOf"));
        }
        if (schema.contains("oneOf")) {
            auto alternatives = build_alternatives(schema.at("oneOf"), path + "/oneOf");
            const auto & children = static_cast<const common_chat_schema_any_of &>(*alternatives).children;
            bool exclusive = true;
            for (size_t i = 0; i < children.size() && exclusive; i++) {
                for (size_t j = i + 1; j < children.size() && exclusive; j++) {
                    exclusive = disjoint(*children[i], *children[j]);
                }
            }
            if (!exclusive) {
                // Values matching several alternatives are admitted.
                relax(path, "oneOf", relaxation::REASON_UNENFORCED);
            }
            parts.push_back(std::move(alternatives));
        }
        for (const auto * key : { "not", "$dynamicRef", "$recursiveRef" }) {
            if (schema.contains(key)) {
                relax(path, key, relaxation::REASON_UNENFORCED);
            }
        }
        if (schema.contains("if") && (schema.contains("then") || schema.contains("else"))) {
            relax(path, "if", relaxation::REASON_UNENFORCED);
        }
        auto node = std::move(parts[0]);
        for (size_t i = 1; i < parts.size(); i++) {
            node = intersect(std::move(node), std::move(parts[i]), path);
        }
        return node;
    }

    // ----- settling

    // Removes every value-less node. A value-less alternative is dropped; a
    // value-less optional property may not appear; a value-less value that must
    // appear admits any value instead, and is recorded.
    common_chat_schema_ptr settle(common_chat_schema_ptr node) {
        using K = common_chat_schema;
        switch (node->kind()) {
            case K::KIND_ANY_OF: {
                auto & children = static_cast<common_chat_schema_any_of &>(*node).children;
                std::vector<common_chat_schema_ptr> kept;
                std::vector<common_chat_schema_ptr> nevers;
                for (auto & child : children) {
                    auto settled = settle(std::move(child));
                    (settled->kind() == K::KIND_NEVER ? nevers : kept).push_back(std::move(settled));
                }
                if (kept.empty()) {
                    return std::move(nevers.front());
                }
                if (kept.size() == 1) {
                    return std::move(kept.front());
                }
                children = std::move(kept);
                return node;
            }
            case K::KIND_OBJECT: {
                auto & object = static_cast<common_chat_schema_object &>(*node);
                if (object.additional_properties) {
                    object.additional_properties = settle(std::move(object.additional_properties));
                    if (object.additional_properties->kind() == K::KIND_NEVER) {
                        object.additional_properties.reset();
                    }
                }
                std::vector<common_chat_schema_property> kept;
                for (auto & property : object.properties) {
                    property.schema = settle(std::move(property.schema));
                    if (property.schema->kind() != K::KIND_NEVER) {
                        kept.push_back(std::move(property));
                    } else if (property.required) {
                        property.schema = value_position(std::move(property.schema));
                        kept.push_back(std::move(property));
                    } else if (object.additional()) {
                        // Absent from the declared properties, the name may
                        // still appear as an undeclared property.
                        relax(property.schema->path, static_cast<const common_chat_schema_never &>(*property.schema).cause,
                              relaxation::REASON_UNENFORCED);
                    }
                }
                object.properties = std::move(kept);
                return node;
            }
            case K::KIND_ARRAY: {
                auto & array = static_cast<common_chat_schema_array &>(*node);
                array.items  = settle(std::move(array.items));
                if (array.items->kind() == K::KIND_NEVER) {
                    if (array.min_items == 0) {
                        array.max_items = 0;
                        array.items     = make<common_chat_schema_any>(array.items->path);
                    } else {
                        array.items = value_position(std::move(array.items));
                    }
                }
                return node;
            }
            case K::KIND_TUPLE: {
                for (auto & item : static_cast<common_chat_schema_tuple &>(*node).items) {
                    item = value_position(settle(std::move(item)));
                }
                return node;
            }
            default:
                return node;
        }
    }

    // A settled node where some value must appear: never admits any value instead.
    common_chat_schema_ptr value_position(common_chat_schema_ptr node) {
        if (node->kind() != common_chat_schema::KIND_NEVER) {
            return node;
        }
        const auto & never = static_cast<const common_chat_schema_never &>(*node);
        relax(never.path, never.cause, relaxation::REASON_UNSATISFIABLE);
        return make<common_chat_schema_any>(never.path);
    }

    void link(common_chat_schema & node) {
        using K = common_chat_schema;
        switch (node.kind()) {
            case K::KIND_REF: {
                auto & ref = static_cast<common_chat_schema_ref &>(node);
                ref.target = doc_.refs.at(ref.ref).get();
                break;
            }
            case K::KIND_ANY_OF:
                for (auto & child : static_cast<common_chat_schema_any_of &>(node).children) {
                    link(*child);
                }
                break;
            case K::KIND_ARRAY:
                link(*static_cast<common_chat_schema_array &>(node).items);
                break;
            case K::KIND_TUPLE:
                for (auto & item : static_cast<common_chat_schema_tuple &>(node).items) {
                    link(*item);
                }
                break;
            case K::KIND_OBJECT: {
                auto & object = static_cast<common_chat_schema_object &>(node);
                for (auto & property : object.properties) {
                    link(*property.schema);
                }
                if (object.additional_properties) {
                    link(*object.additional_properties);
                }
                break;
            }
            case K::KIND_NEVER:
                throw std::logic_error("unsettled schema node");
            default:
                break;
        }
    }

  public:
    common_chat_schema_builder(const json & root, common_chat_schema_document & doc) : root_(root), doc_(doc) {}

    void build() {
        auto root = build(root_, "#");
        doc_.root = value_position(settle(std::move(root)));
        for (auto & [ref, target] : refs_) {
            doc_.refs[ref] = value_position(settle(std::move(target)));
        }
        link(*doc_.root);
        for (auto & [ref, target] : doc_.refs) {
            link(*target);
        }
        doc_.relaxations.assign(relaxations_.begin(), relaxations_.end());
    }
};

}  // namespace

common_chat_schema_document common_chat_schema_from_json(const common_json & schema) {
    common_chat_schema_document doc;
    common_chat_schema_builder(schema, doc).build();
    return doc;
}

static common_chat_schema::value_type json_type(const common_json & value) {
    if (value.is_null()) {
        return common_chat_schema::TYPE_NULL;
    }
    if (value.is_boolean()) {
        return common_chat_schema::TYPE_BOOLEAN;
    }
    // A number with no fractional part is an integer, however it is written.
    if (value.is_number_integer() || (value.is_number_float() && std::trunc(value.get<double>()) == value.get<double>())) {
        return common_chat_schema::TYPE_INTEGER;
    }
    if (value.is_number()) {
        return common_chat_schema::TYPE_NUMBER;
    }
    if (value.is_string()) {
        return common_chat_schema::TYPE_STRING;
    }
    if (value.is_array()) {
        return common_chat_schema::TYPE_ARRAY;
    }
    return common_chat_schema::TYPE_OBJECT;
}

static common_chat_schema::type_set value_types_impl(const common_chat_schema & s, std::unordered_set<const common_chat_schema *> & visited) {
    switch (s.kind()) {
        case common_chat_schema::KIND_ANY:
            return common_chat_schema::type_set::all();
        case common_chat_schema::KIND_NEVER:
            return {};
        case common_chat_schema::KIND_NULL:
            return { common_chat_schema::TYPE_NULL };
        case common_chat_schema::KIND_BOOLEAN:
            return { common_chat_schema::TYPE_BOOLEAN };
        case common_chat_schema::KIND_NUMBER:
            return { common_chat_schema::TYPE_NUMBER, common_chat_schema::TYPE_INTEGER };
        case common_chat_schema::KIND_INTEGER:
            return { common_chat_schema::TYPE_INTEGER };
        case common_chat_schema::KIND_STRING:
            return { common_chat_schema::TYPE_STRING };
        case common_chat_schema::KIND_ARRAY:
        case common_chat_schema::KIND_TUPLE:
            return { common_chat_schema::TYPE_ARRAY };
        case common_chat_schema::KIND_OBJECT:
            return { common_chat_schema::TYPE_OBJECT };
        case common_chat_schema::KIND_CONST:
            return { json_type(static_cast<const common_chat_schema_const &>(s).value) };
        case common_chat_schema::KIND_ENUM: {
            common_chat_schema::type_set types;
            for (const auto & value : static_cast<const common_chat_schema_enum &>(s).values) {
                types.add(json_type(value));
            }
            return types;
        }
        case common_chat_schema::KIND_REF: {
            const auto * target = static_cast<const common_chat_schema_ref &>(s).target;
            if (!target || !visited.insert(target).second) {
                // a cycle contributes no type, to be safe
                return {};
            }
            auto types = value_types_impl(*target, visited);
            visited.erase(target);
            return types;
        }
        case common_chat_schema::KIND_ANY_OF: {
            common_chat_schema::type_set types;
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(s).children) {
                types |= value_types_impl(*child, visited);
            }
            return types;
        }
    }
    return {};
}

common_chat_schema::type_set common_chat_schema::value_types() const {
    std::unordered_set<const common_chat_schema *> visited;
    return value_types_impl(*this, visited);
}

static bool may_be_string_impl(const common_chat_schema & s, std::unordered_set<const common_chat_schema *> & visited) {
    switch (s.kind()) {
        case common_chat_schema::KIND_STRING:
            return true;
        case common_chat_schema::KIND_CONST:
            return static_cast<const common_chat_schema_const &>(s).value.is_string();
        case common_chat_schema::KIND_ENUM:
            for (const auto & v : static_cast<const common_chat_schema_enum &>(s).values) {
                if (v.is_string()) {
                    return true;
                }
            }
            return false;
        case common_chat_schema::KIND_REF: {
            // a cycle is taken as not a string, to be safe
            const auto * target = static_cast<const common_chat_schema_ref &>(s).target;
            if (!target || !visited.insert(target).second) {
                return false;
            }
            bool result = may_be_string_impl(*target, visited);
            visited.erase(target);
            return result;
        }
        case common_chat_schema::KIND_ANY_OF:
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(s).children) {
                if (may_be_string_impl(*child, visited)) {
                    return true;
                }
            }
            return false;
        default:
            return false;
    }
}

bool common_chat_schema::may_be_string() const {
    std::unordered_set<const common_chat_schema *> visited;
    return may_be_string_impl(*this, visited);
}

const char * common_chat_schema::kind_name(node_kind kind) {
    switch (kind) {
        case KIND_ANY:     return "any";
        case KIND_REF:     return "ref";
        case KIND_ANY_OF:  return "anyOf";
        case KIND_CONST:   return "const";
        case KIND_ENUM:    return "enum";
        case KIND_NULL:    return "null";
        case KIND_BOOLEAN: return "boolean";
        case KIND_NUMBER:  return "number";
        case KIND_INTEGER: return "integer";
        case KIND_STRING:  return "string";
        case KIND_ARRAY:   return "array";
        case KIND_TUPLE:   return "tuple";
        case KIND_OBJECT:  return "object";
        case KIND_NEVER:   return "never";
    }
    return "?";
}

const char * common_chat_schema::type_name(value_type type) {
    switch (type) {
        case TYPE_NULL:    return "null";
        case TYPE_BOOLEAN: return "boolean";
        case TYPE_NUMBER:  return "number";
        case TYPE_INTEGER: return "integer";
        case TYPE_STRING:  return "string";
        case TYPE_ARRAY:   return "array";
        case TYPE_OBJECT:  return "object";
    }
    return "?";
}
