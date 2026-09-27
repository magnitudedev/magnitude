#include "schema-validation.h"

#include <set>
#include <stdexcept>
#include <string>

namespace templates_native {
namespace {
using json = common_json;
const std::set<std::string> annotations = {
    "title", "description", "default", "examples", "$comment", "$schema", "$id",
    "deprecated", "readOnly", "writeOnly", "$defs", "definitions"
};
const std::set<std::string> supported = {
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "const", "anyOf", "$ref", "pattern"
};
[[noreturn]] void fail(const std::string & path, const std::string & message) {
    throw std::invalid_argument("Unsupported JSON schema at " + path + ": " + message);
}
void walk(const json & schema, const std::string & path, unsigned depth, size_t & count) {
    if (depth > 64 || ++count > 10000) { fail(path, "schema resource limit exceeded"); }
    if (!schema.is_object()) { fail(path, "expected an object schema"); }
    for (const auto & [key, value] : schema.items()) {
        if (!annotations.count(key) && !supported.count(key)) { fail(path, "keyword " + key); }
    }
    for (const auto * definitions : {"$defs", "definitions"}) {
        if (schema.contains(definitions)) {
            if (!schema.at(definitions).is_object()) { fail(path, "definitions must be an object"); }
            for (const auto & [name, child] : schema.at(definitions).items()) {
                walk(child, path + "/" + definitions + "/" + name, depth + 1, count);
            }
        }
    }
    // Upstream chooses these branches before structural/type restrictions.
    // Reject intersections it would silently discard.
    for (const auto * exclusive : {"$ref", "anyOf", "const", "enum"}) {
        if (!schema.contains(exclusive)) { continue; }
        for (const auto & [key, value] : schema.items()) {
            if (key != exclusive && !annotations.count(key) && key != "type") {
                fail(path, std::string(exclusive) + " with sibling constraint " + key);
            }
        }
        if (schema.contains("type") && (std::string(exclusive) == "$ref" || std::string(exclusive) == "anyOf")) {
            fail(path, std::string(exclusive) + " with sibling type");
        }
    }
    if (schema.contains("$ref")) {
        const auto & ref = schema.at("$ref");
        if (!ref.is_string()) { fail(path, "$ref must be a string"); }
        auto value = ref.get<std::string>();
        if ((value.rfind("#/$defs/", 0) != 0 && value.rfind("#/definitions/", 0) != 0) ||
            value.find('~') != std::string::npos) {
            fail(path, "only local definitions references without escaped segments are qualified");
        }
    }
    if (schema.contains("anyOf")) {
        const auto & alts = schema.at("anyOf");
        if (!alts.is_array() || alts.empty()) { fail(path, "anyOf must be nonempty"); }
        for (size_t i = 0; i < alts.size(); ++i) {
            walk(alts[i], path + "/anyOf/" + std::to_string(i), depth + 1, count);
        }
    }
    std::set<std::string> types;
    if (schema.contains("type")) {
        auto add_type = [&](const json & type) {
            if (!type.is_string()) { fail(path, "type must contain strings"); }
            const auto name = type.get<std::string>();
            if (name != "object" && name != "array" && name != "string" && name != "number" &&
                name != "integer" && name != "boolean" && name != "null") { fail(path, "unknown type " + name); }
            types.insert(name);
        };
        const auto & type = schema.at("type");
        if (type.is_array()) {
            if (type.empty()) { fail(path, "type must be nonempty"); }
            for (const auto & entry : type) { add_type(entry); }
        } else { add_type(type); }
    }
    auto matches_type = [&](const json & value) {
        if (types.empty()) { return true; }
        return (value.is_null() && types.count("null")) ||
            (value.is_boolean() && types.count("boolean")) ||
            (value.is_string() && types.count("string")) ||
            (value.is_object() && types.count("object")) ||
            (value.is_array() && types.count("array")) ||
            (value.is_number() && types.count("number")) ||
            (value.is_number_integer() && types.count("integer"));
    };
    if (schema.contains("const") && !matches_type(schema.at("const"))) {
        fail(path, "const conflicts with type");
    }
    if (schema.contains("enum")) {
        const auto & values = schema.at("enum");
        if (!values.is_array() || values.empty()) { fail(path, "enum must be nonempty"); }
        for (const auto & value : values) {
            if (!matches_type(value)) { fail(path, "enum conflicts with type"); }
        }
    }
    const std::set<std::string> object_keys = {"properties", "required", "additionalProperties"};
    for (const auto & [key, value] : schema.items()) {
        const auto expected = object_keys.count(key) ? "object" : key == "items" ? "array" : key == "pattern" ? "string" : "";
        if (*expected && (types.size() != 1 || !types.count(expected))) {
            fail(path, key + " requires an explicit matching type");
        }
    }
    if (schema.contains("properties")) {
        const auto & properties = schema.at("properties");
        if (!properties.is_object()) { fail(path, "properties must be an object"); }
        for (const auto & [name, child] : properties.items()) {
            walk(child, path + "/properties/" + name, depth + 1, count);
        }
    }
    if (schema.contains("required")) {
        const auto & required = schema.at("required");
        if (!required.is_array()) { fail(path, "required must be an array"); }
        for (const auto & name : required) {
            if (!name.is_string() || !schema.contains("properties") ||
                !schema.at("properties").contains(name.get<std::string>())) {
                fail(path, "required names must have declared property schemas");
            }
        }
    }
    if (schema.contains("additionalProperties") && !schema.at("additionalProperties").is_boolean()) {
        walk(schema.at("additionalProperties"), path + "/additionalProperties", depth + 1, count);
    }
    if (schema.contains("items")) { walk(schema.at("items"), path + "/items", depth + 1, count); }
    if (schema.contains("pattern") && !schema.at("pattern").is_string()) { fail(path, "pattern must be a string"); }
}
}
void validate_schema(const common_json & schema) {
    size_t count = 0;
    walk(schema, "#", 0, count);
}
}
