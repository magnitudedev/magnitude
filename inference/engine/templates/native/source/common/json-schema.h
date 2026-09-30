#pragma once

#include "json.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// A JSON Schema lowered to the constraint that grammars enforce.
//
// Lowering is total over valid JSON Schemas. Every keyword is either enforced
// exactly, loosened to a superset of its values and recorded as a relaxation,
// or an annotation with no validation meaning. A lowered node never admits a
// value its source schema rejects unless a relaxation says so.

struct common_chat_schema {
    enum node_kind {
        KIND_ANY,
        KIND_REF,
        KIND_ANY_OF,
        KIND_CONST,
        KIND_ENUM,
        KIND_NULL,
        KIND_BOOLEAN,
        KIND_NUMBER,
        KIND_INTEGER,
        KIND_STRING,
        KIND_ARRAY,
        KIND_TUPLE,
        KIND_OBJECT,
        // Admits no value. Lowering settles it before returning a document.
        KIND_NEVER,
    };

    enum value_type {
        TYPE_NULL,
        TYPE_BOOLEAN,
        TYPE_NUMBER,
        TYPE_INTEGER,
        TYPE_STRING,
        TYPE_ARRAY,
        TYPE_OBJECT,
    };

    // Annotation formats the grammar shapes strings to. Formats never assert.
    enum string_format {
        FORMAT_NONE,
        FORMAT_UUID,  // uuid, uuid1 .. uuid5
        FORMAT_DATE,
        FORMAT_TIME,
        FORMAT_DATE_TIME,
    };

    class type_set {
        uint32_t mask_ = 0;

      public:
        type_set() = default;
        type_set(std::initializer_list<value_type> types) {
            for (auto type : types) {
                add(type);
            }
        }

        static type_set all() {
            return { TYPE_NULL, TYPE_BOOLEAN, TYPE_NUMBER, TYPE_INTEGER, TYPE_STRING, TYPE_ARRAY, TYPE_OBJECT };
        }

        void add(value_type type) { mask_ |= 1u << type; }

        bool has(value_type type) const { return (mask_ & (1u << type)) != 0; }
        bool is_only(value_type type) const { return mask_ == (1u << type); }
        bool empty() const { return mask_ == 0; }

        type_set & operator|=(const type_set & other) { mask_ |= other.mask_; return *this; }
        type_set & operator&=(const type_set & other) { mask_ &= other.mask_; return *this; }

        bool operator==(const type_set & other) const { return mask_ == other.mask_; }
        bool operator!=(const type_set & other) const { return mask_ != other.mask_; }
    };

    // The JSON pointer of the schema this node was lowered from.
    std::string path;

    virtual ~common_chat_schema() = default;
    virtual node_kind kind() const = 0;
    virtual std::unique_ptr<common_chat_schema> clone() const = 0;

    type_set value_types() const;

    // Whether a value matching the schema may be a string, through any branch of it.
    bool may_be_string() const;

    static const char * kind_name(node_kind kind);
    static const char * type_name(value_type type);
};

using common_chat_schema_ptr = std::unique_ptr<common_chat_schema>;

template <typename T>
struct common_chat_schema_node : common_chat_schema {
    common_chat_schema_ptr clone() const override;
};

struct common_chat_schema_any : common_chat_schema_node<common_chat_schema_any> {
    node_kind kind() const override { return KIND_ANY; }
};

struct common_chat_schema_never : common_chat_schema_node<common_chat_schema_never> {
    std::string cause;  // the keyword that admits no value

    explicit common_chat_schema_never(std::string cause) : cause(std::move(cause)) {}

    node_kind kind() const override { return KIND_NEVER; }
};

// {"$ref": "#/..."}: a JSON pointer into the same document
struct common_chat_schema_ref : common_chat_schema_node<common_chat_schema_ref> {
    std::string                ref;
    const common_chat_schema * target = nullptr;  // owned by common_chat_schema_document::refs

    explicit common_chat_schema_ref(std::string ref) : ref(std::move(ref)) {}

    node_kind kind() const override { return KIND_REF; }
};

// anyOf; oneOf (exactly, when its branches are disjoint); or a "type" array
// expanded to one alternative per type
struct common_chat_schema_any_of : common_chat_schema_node<common_chat_schema_any_of> {
    std::vector<common_chat_schema_ptr> children;

    common_chat_schema_any_of() = default;
    common_chat_schema_any_of(const common_chat_schema_any_of & other);

    node_kind kind() const override { return KIND_ANY_OF; }
};

struct common_chat_schema_const : common_chat_schema_node<common_chat_schema_const> {
    common_json value;

    explicit common_chat_schema_const(common_json value) : value(std::move(value)) {}

    node_kind kind() const override { return KIND_CONST; }
};

struct common_chat_schema_enum : common_chat_schema_node<common_chat_schema_enum> {
    std::vector<common_json> values;

    node_kind kind() const override { return KIND_ENUM; }
};

struct common_chat_schema_null : common_chat_schema_node<common_chat_schema_null> {
    node_kind kind() const override { return KIND_NULL; }
};

struct common_chat_schema_boolean : common_chat_schema_node<common_chat_schema_boolean> {
    node_kind kind() const override { return KIND_BOOLEAN; }
};

// An exact decimal: optional sign, integer digits without leading zeros
// ("0" for zero) and fraction digits without trailing zeros. Zero is never negative.
struct common_chat_decimal {
    bool        negative = false;
    std::string integer  = "0";
    std::string fraction;

    // Parses a JSON number, in any notation, into its exact decimal value.
    static common_chat_decimal from_json(const common_json & value);

    bool        is_zero() const { return integer == "0" && fraction.empty(); }
    bool        is_integer() const { return fraction.empty(); }
    std::string to_string() const;
};

// <0, 0 or >0 as a is less than, equal to or greater than b.
int common_chat_decimal_compare(const common_chat_decimal & a, const common_chat_decimal & b);

struct common_chat_numeric_bound {
    common_chat_decimal value;
    bool                exclusive = false;
};

// Bounds apply to the decimal value, exclusive bounds included.
struct common_chat_schema_numeric {
    std::optional<common_chat_numeric_bound> minimum;
    std::optional<common_chat_numeric_bound> maximum;

    bool bounded() const { return minimum || maximum; }
};

struct common_chat_schema_number : common_chat_schema_node<common_chat_schema_number>, common_chat_schema_numeric {
    node_kind kind() const override { return KIND_NUMBER; }
};

struct common_chat_schema_integer : common_chat_schema_node<common_chat_schema_integer>, common_chat_schema_numeric {
    node_kind kind() const override { return KIND_INTEGER; }
};

struct common_chat_schema_string : common_chat_schema_node<common_chat_schema_string> {
    std::string   pattern;  // empty when absent; always a pattern the grammar translates
    string_format format     = FORMAT_NONE;
    int           min_length = 0;
    int           max_length = -1;  // -1 for unbounded

    node_kind kind() const override { return KIND_STRING; }
};

struct common_chat_schema_array : common_chat_schema_node<common_chat_schema_array> {
    common_chat_schema_ptr items;  // a common_chat_schema_any when items are unconstrained
    int                    min_items = 0;
    int                    max_items = -1;  // -1 for unbounded

    common_chat_schema_array() = default;
    common_chat_schema_array(const common_chat_schema_array & other);

    node_kind kind() const override { return KIND_ARRAY; }
};

// Exactly these items, in order.
struct common_chat_schema_tuple : common_chat_schema_node<common_chat_schema_tuple> {
    std::vector<common_chat_schema_ptr> items;

    common_chat_schema_tuple() = default;
    common_chat_schema_tuple(const common_chat_schema_tuple & other);

    node_kind kind() const override { return KIND_TUPLE; }
};

struct common_chat_schema_property {
    std::string            name;
    common_chat_schema_ptr schema;
    bool                   required = false;
};

struct common_chat_schema_object : common_chat_schema_node<common_chat_schema_object> {
    // Which undeclared properties may appear.
    enum undeclared_rule {
        // The schema neither lists properties nor says: any may appear.
        UNDECLARED_ANY,
        // The schema lists properties without allowing more: none may appear.
        UNDECLARED_NONE,
        // The schema says: those matching `additional_properties`, or none when null.
        UNDECLARED_STATED,
    };

    std::vector<common_chat_schema_property> properties;  // in schema order
    undeclared_rule                          undeclared = UNDECLARED_ANY;
    common_chat_schema_ptr                   additional_properties;

    common_chat_schema_object() = default;
    common_chat_schema_object(const common_chat_schema_object & other);

    // The schema undeclared properties must match, or null when none may appear.
    const common_chat_schema * additional() const;

    node_kind kind() const override { return KIND_OBJECT; }
};

// A keyword of the source schema the lowered constraint does not enforce.
struct common_chat_schema_relaxation {
    enum reason_kind {
        // The constraint admits values the keyword rejects.
        REASON_UNENFORCED,
        // Values the output format cannot write are left out.
        REASON_UNREPRESENTABLE,
        // The keyword admits no value; the constraint admits values instead.
        REASON_UNSATISFIABLE,
    };

    std::string path;
    std::string keyword;
    reason_kind reason = REASON_UNENFORCED;

    static const char * reason_name(reason_kind reason);

    bool operator<(const common_chat_schema_relaxation & other) const;
    bool operator==(const common_chat_schema_relaxation & other) const;
};

struct common_chat_schema_document {
    common_chat_schema_ptr                        root;
    std::map<std::string, common_chat_schema_ptr> refs;
    std::vector<common_chat_schema_relaxation>    relaxations;
};

// A document shared by the PEG parsers built from its nodes, which it keeps alive
using common_chat_schema_document_ptr = std::shared_ptr<const common_chat_schema_document>;

// Lowers a valid JSON Schema. Never fails for a schema that is valid under its
// draft; a malformed schema is a caller defect (std::logic_error).
common_chat_schema_document common_chat_schema_from_json(const common_json & schema);
