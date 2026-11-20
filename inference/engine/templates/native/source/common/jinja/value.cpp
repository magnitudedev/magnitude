#include "runtime.h"
#include "unicode.h"
#include "value.h"

// for converting from JSON to jinja values
#include "json.h"
#include "templates-support.h"

#include <sstream>
#include <string>
#include <cctype>
#include <vector>
#include <optional>
#include <algorithm>
#include <unordered_set>

#define FILENAME "jinja-value"

namespace jinja {

// func_args method implementations

value func_args::get_kwarg(const std::string & key, value default_val) const {
    for (const auto & arg : args) {
        if (is_val<value_kwarg>(arg)) {
            auto * kwarg = cast_val<value_kwarg>(arg);
            if (kwarg->key == key) {
                return kwarg->val;
            }
        }
    }
    return default_val;
}

value func_args::get_kwarg_or_pos(const std::string & key, size_t pos) const {
    value val = get_kwarg(key, mk_val<value_undefined>());

    if (val->is_undefined() && pos < count() && !is_val<value_kwarg>(args[pos])) {
        return args[pos];
    }

    return val;
}

value func_args::get_pos(size_t pos) const {
    if (count() > pos) {
        return args[pos];
    }
    throw raised_exception("Function '" + func_name + "' expected at least " + std::to_string(pos + 1) + " arguments, got " + std::to_string(count()));
}

value func_args::get_pos(size_t pos, value default_val) const {
    if (count() > pos) {
        return args[pos];
    }
    return default_val;
}

void func_args::push_back(const value & val) {
    args.push_back(val);
}

void func_args::push_front(const value & val) {
    args.insert(args.begin(), val);
}

const std::vector<value> & func_args::get_args() const {
    return args;
}

/**
 * Function that mimics Python's array slicing.
 */
template<typename T>
static T slice(const T & array, int64_t start, int64_t stop, int64_t step = 1) {
    int64_t len = static_cast<int64_t>(array.size());
    int64_t direction = (step > 0) ? 1 : ((step < 0) ? -1 : 0);
    int64_t start_val = 0;
    int64_t stop_val = 0;
    if (direction >= 0) {
        start_val = start;
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)0);
        } else {
            start_val = std::min(start_val, len);
        }

        stop_val = stop;
        if (stop_val < 0) {
            stop_val = std::max(len + stop_val, (int64_t)0);
        } else {
            stop_val = std::min(stop_val, len);
        }
    } else {
        start_val = start;
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)0);
        } else {
            start_val = std::min(start_val, len - 1);
        }

        stop_val = stop;
        if (stop_val < -1) {
            stop_val = std::max(len + stop_val, (int64_t)-1);
        } else {
            stop_val = std::min(stop_val, len - 1);
        }
    }
    T result;
    if (direction == 0) {
        return result;
    }
    for (int64_t i = start_val; direction * i < direction * stop_val; i += step) {
        if (i >= 0 && i < len) {
            result.push_back(array[static_cast<size_t>(i)]);
        }
    }
    return result;
}

template<typename T>
static value empty_value_fn(const func_args &) {
    if constexpr (std::is_same_v<T, value_int>) {
        return mk_val<T>(0);
    } else if constexpr (std::is_same_v<T, value_float>) {
        return mk_val<T>(0.0);
    } else if constexpr (std::is_same_v<T, value_bool>) {
        return mk_val<T>(false);
    } else {
        return mk_val<T>();
    }
}
template<typename... Ts>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    const value & val = args.get_pos(0);
    bool is_type = (is_val<Ts>(val) || ...);
    return mk_val<value_bool>(is_type);
}
template<value_compare_op op>
static value test_compare_fn(const func_args & args) {
    args.ensure_count(2, 2);
    return mk_val<value_bool>(value_compare(args.get_pos(0), args.get_pos(1), op));
}

static value from_json(const common_json & j, bool mark_input);

// The value an item is compared by in `sort`, `unique`, `min`, `max` and `dictsort`, as Jinja's
// `make_attrgetter(..., postprocess=ignore_case)`: the item's `attribute` (an index of an array or a
// key of an object) when one is named, and a string in lower case unless `case_sensitive`.
static value comparison_key(const value & item, const value & attribute, bool case_sensitive) {
    value key = item;
    if (!attribute->is_undefined()) {
        if (is_val<value_int>(attribute) && is_val<value_array>(item)) {
            key = item->at(attribute->as_int());
        } else if (!is_val<value_int>(attribute) && is_val<value_object>(item)) {
            key = item->at(attribute);
        } else {
            throw raised_exception("unsupported attribute " + attribute->type() + " of " + item->type());
        }
    }
    if (case_sensitive || !is_val<value_string>(key)) {
        return key;
    }
    return mk_val<value_string>(key->as_string().lowercase());
}

// The positions of `keys` in Python's stable sorted order, ascending or descending, each key computed once.
static std::vector<size_t> stable_order(const std::vector<value> & keys, bool reverse) {
    std::vector<size_t> order(keys.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return value_compare(keys[a], keys[b], reverse ? value_compare_op::gt : value_compare_op::lt);
    });
    return order;
}

static std::vector<value> to_iterable_items(const value & v, const char * fn_name) {
    if (is_val<value_array>(v)) {
        return v->as_array();
    }
    if (is_val<value_string>(v)) {
        const auto & jstr = v->as_string();
        const std::string s = jstr.str();
        std::vector<value> out;
        out.reserve(s.size());
        for (char ch : s) {
            auto c = mk_val<value_string>(std::string(1, ch));
            c->val_str.mark_input_based_on(jstr);
            out.push_back(std::move(c));
        }
        return out;
    }
    if (is_val<value_object>(v)) {
        const auto & obj = v->as_ordered_object();
        std::vector<value> out;
        out.reserve(obj.size());
        for (const auto & pair : obj) {
            out.push_back(pair.first);
        }
        return out;
    }
    if (v->is_undefined()) {
        return {};
    }
    throw raised_exception(std::string(fn_name) + ": expected iterable, got " + v->type());
}

static void dict_update_from_args(value_object_t & obj, const func_args & args, size_t start_pos, const char * fn_name) {
    const auto & all_args = args.get_args();
    value pos_arg;
    for (size_t i = start_pos; i < all_args.size(); ++i) {
        if (!is_val<value_kwarg>(all_args[i])) {
            if (pos_arg) {
                throw raised_exception(std::string(fn_name) + ": expected at most 1 positional argument");
            }
            pos_arg = all_args[i];
        }
    }
    if (pos_arg) {
        if (is_val<value_object>(pos_arg)) {
            for (const auto & pair : pos_arg->as_ordered_object()) {
                obj.insert(pair.first, pair.second);
            }
        } else {
            for (const auto & item : to_iterable_items(pos_arg, fn_name)) {
                auto pair = to_iterable_items(item, fn_name);
                if (pair.size() != 2) {
                    throw raised_exception(std::string(fn_name) + ": dictionary update sequence element has length " + std::to_string(pair.size()) + "; 2 is required");
                }
                obj.insert(pair[0], pair[1]);
            }
        }
    }
    for (size_t i = start_pos; i < all_args.size(); ++i) {
        if (is_val<value_kwarg>(all_args[i])) {
            const auto * kwarg = cast_val<value_kwarg>(all_args[i]);
            obj.insert(kwarg->key, kwarg->val);
        }
    }
}

static value sum_values(const std::vector<value> & items, const value & start_val, const value & attribute) {
    if (!start_val->is_undefined() && is_val<value_array>(start_val)) {
        auto out = mk_val<value_array>(start_val->as_array());
        for (const auto & item : items) {
            value v = comparison_key(item, attribute, true);
            if (!is_val<value_array>(v)) {
                throw raised_exception("sum: expected array item, got " + v->type());
            }
            for (const auto & elem : v->as_array()) {
                out->push_back(elem);
            }
        }
        return out;
    }
    bool is_float = false;
    int64_t int_acc = 0;
    double flt_acc = 0.0;
    if (!start_val->is_undefined()) {
        if (is_val<value_float>(start_val)) {
            is_float = true;
            flt_acc = start_val->as_float();
        } else if (is_val<value_int>(start_val) || is_val<value_bool>(start_val)) {
            int_acc = start_val->as_int();
        } else {
            throw raised_exception("sum: unsupported start type " + start_val->type());
        }
    }
    for (const auto & item : items) {
        value v = comparison_key(item, attribute, true);
        if (is_val<value_float>(v)) {
            if (!is_float) {
                flt_acc = static_cast<double>(int_acc);
                is_float = true;
            }
            flt_acc += v->as_float();
        } else if (is_val<value_int>(v) || is_val<value_bool>(v)) {
            if (is_float) {
                flt_acc += static_cast<double>(v->as_int());
            } else {
                int_acc += v->as_int();
            }
        } else {
            throw raised_exception("sum: unsupported operand type " + v->type());
        }
    }
    return is_float ? value(mk_val<value_float>(flt_acc)) : value(mk_val<value_int>(int_acc));
}

static std::pair<size_t, size_t> clamp_slice_bounds(size_t len, const value & start_val, const value & end_val) {
    int64_t n = static_cast<int64_t>(len);
    int64_t start = start_val->is_undefined() || start_val->is_none() ? 0 : start_val->as_int();
    int64_t end = end_val->is_undefined() || end_val->is_none() ? n : end_val->as_int();
    if (start < 0) start += n;
    if (end < 0) end += n;
    start = std::clamp<int64_t>(start, 0, n);
    end = std::clamp<int64_t>(end, 0, n);
    return {static_cast<size_t>(start), static_cast<size_t>(end)};
}

template<bool is_max>
static value global_min_max(const func_args & args) {
    const char * fn_name = is_max ? "max" : "min";
    std::vector<value> pos;
    for (const auto & arg : args.get_args()) {
        if (!is_val<value_kwarg>(arg)) {
            pos.push_back(arg);
        }
    }
    if (pos.empty()) {
        throw raised_exception(std::string(fn_name) + "() expected at least 1 argument");
    }
    std::vector<value> items = pos.size() == 1 ? to_iterable_items(pos[0], fn_name) : pos;
    if (items.empty()) {
        value def = args.get_kwarg("default", mk_val<value_undefined>());
        if (!def->is_undefined()) {
            return def;
        }
        throw raised_exception(std::string(fn_name) + "() arg is an empty sequence");
    }
    value best = items[0];
    for (size_t i = 1; i < items.size(); ++i) {
        if (value_compare(items[i], best, is_max ? value_compare_op::gt : value_compare_op::lt)) {
            best = items[i];
        }
    }
    return best;
}

static value round_value(const func_args & args) {
    args.ensure_count(1, 2);
    value v = args.get_pos(0);
    if (!v->is_numeric()) {
        throw raised_exception("round: expected numeric value, got " + v->type());
    }
    value ndigits = args.get_kwarg_or_pos("precision", 1);
    if (ndigits->is_undefined() || ndigits->is_none()) {
        if (is_val<value_int>(v) || is_val<value_bool>(v)) {
            return mk_val<value_int>(v->as_int());
        }
        return mk_val<value_int>(static_cast<int64_t>(std::llrint(v->as_float())));
    }
    int64_t n = ndigits->as_int();
    double scale = std::pow(10.0, static_cast<double>(n));
    double r = std::rint(v->as_float() * scale) / scale;
    if (is_val<value_int>(v) || is_val<value_bool>(v)) {
        return n >= 0 ? mk_val<value_int>(v->as_int()) : mk_val<value_int>(static_cast<int64_t>(r));
    }
    return mk_val<value_float>(r);
}

static void append_codepoint_as_ascii_json_escape(std::string & out, uint32_t codepoint) {
    auto append_u16 = [&out](uint32_t value) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(value));
        out += buf;
    };

    if (codepoint <= 0xFFFF) {
        append_u16(codepoint);
        return;
    }

    codepoint -= 0x10000;
    append_u16(0xD800 + ((codepoint >> 10) & 0x3FF));
    append_u16(0xDC00 + (codepoint & 0x3FF));
}

static std::string json_ensure_ascii_preserving_format(const std::string & json_str) {
    std::string output;
    output.reserve(json_str.size());

    bool in_string = false;
    bool escaped = false;

    for (size_t pos = 0; pos < json_str.size();) {
        const char ch = json_str[pos];
        if (!in_string) {
            output.push_back(ch);
            if (ch == '"') {
                in_string = true;
            }
            ++pos;
            continue;
        }

        if (escaped) {
            output.push_back(ch);
            escaped = false;
            ++pos;
            continue;
        }

        if (ch == '\\') {
            output.push_back(ch);
            escaped = true;
            ++pos;
            continue;
        }

        if (ch == '"') {
            output.push_back(ch);
            in_string = false;
            ++pos;
            continue;
        }

        const unsigned char uch = static_cast<unsigned char>(ch);
        if (uch < 0x80) {
            output.push_back(ch);
            ++pos;
            continue;
        }

        auto parsed = common_parse_utf8_codepoint(json_str, pos);
        if (parsed.status != utf8_parse_result::SUCCESS) {
            output += "\\ufffd";
            ++pos;
            continue;
        }

        append_codepoint_as_ascii_json_escape(output, parsed.codepoint);
        pos += parsed.bytes_consumed;
    }

    return output;
}

static value tojson(const func_args & args) {
    args.ensure_count(1, 5);
    value val_ascii      = args.get_kwarg_or_pos("ensure_ascii", 1);
    value val_indent     = args.get_kwarg_or_pos("indent",       2);
    value val_separators = args.get_kwarg_or_pos("separators",   3);
    value val_sort       = args.get_kwarg_or_pos("sort_keys",    4);
    int indent = -1;
    if (args.ctx.is_get_stats) {
        // mark as used (recursively) for stats
        auto val_input = args.get_pos(0);
        value_t::stats_t::mark_used(const_cast<value&>(val_input), true);
    }
    if (is_val<value_int>(val_indent)) {
        indent = static_cast<int>(val_indent->as_int());
    }
    const bool sort_keys = val_sort->as_bool(); // undefined == false
    const bool ensure_ascii = val_ascii->as_bool(); // undefined == false
    auto separators = (is_val<value_array>(val_separators) ? val_separators : mk_val<value_array>())->as_array();
    std::string item_sep = separators.size() > 0 ? separators[0]->as_string().str() : (indent < 0 ? ", " : ",");
    std::string key_sep = separators.size() > 1 ? separators[1]->as_string().str() : ": ";
    std::string json_str = value_to_json(args.get_pos(0), indent, item_sep, key_sep, sort_keys);
    if (ensure_ascii) {
        json_str = json_ensure_ascii_preserving_format(json_str);
    }
    return mk_val<value_string>(json_str);
}

static bool eval_named_test(const context & ctx, const std::string & test_name, const value & target, const func_args & args, size_t extra_start, const char * fn_name) {
    const auto & builtins = global_builtins();
    auto it = builtins.find("test_is_" + test_name);
    if (it == builtins.end()) {
        throw raised_exception(std::string(fn_name) + ": unknown test '" + test_name + "'");
    }
    func_args test_args(const_cast<context &>(ctx));
    test_args.push_back(target);
    for (size_t i = extra_start; i < args.count(); ++i) {
        test_args.push_back(args.get_pos(i));
    }
    return it->second(test_args)->as_bool();
}

template<bool is_reject>
static value select_filter(const func_args & args) {
    args.ensure_count(1);
    const auto items = to_iterable_items(args.get_pos(0), is_reject ? "reject" : "select");
    auto out = mk_val<value_array>();
    const bool has_test = args.count() >= 2;
    std::string test_name;
    if (has_test) {
        if (!is_val<value_string>(args.get_pos(1))) {
            throw raised_exception(std::string(is_reject ? "reject" : "select") + ": test name must be a string");
        }
        test_name = args.get_pos(1)->as_string().str();
    }
    for (const auto & item : items) {
        bool is_selected = has_test
            ? eval_named_test(args.ctx, test_name, item, args, 2, is_reject ? "reject" : "select")
            : item->as_bool();
        if constexpr (is_reject) is_selected = !is_selected;
        if (is_selected) out->push_back(item);
    }
    return out;
}

template<bool is_reject>
static value selectattr(const func_args & args) {
    args.ensure_count(2);
    const auto items = to_iterable_items(args.get_pos(0), is_reject ? "rejectattr" : "selectattr");
    value attribute = args.get_pos(1);
    if (!is_val<value_string>(attribute)) {
        throw raised_exception(std::string(is_reject ? "rejectattr" : "selectattr") + ": attribute name must be a string");
    }
    const bool has_test = args.count() >= 3;
    std::string test_name;
    if (has_test) {
        if (!is_val<value_string>(args.get_pos(2))) {
            throw raised_exception(std::string(is_reject ? "rejectattr" : "selectattr") + ": test name must be a string");
        }
        test_name = args.get_pos(2)->as_string().str();
    }
    auto out = mk_val<value_array>();
    value val_default = mk_val<value_undefined>();
    for (const auto & item : items) {
        if (!is_val<value_object>(item)) {
            throw raised_exception(std::string(is_reject ? "rejectattr" : "selectattr") + ": item is not an object");
        }
        value attr_val = item->at(attribute, val_default);
        bool is_selected = has_test
            ? eval_named_test(args.ctx, test_name, attr_val, args, 3, is_reject ? "rejectattr" : "selectattr")
            : attr_val->as_bool();
        if constexpr (is_reject) is_selected = !is_selected;
        if (is_selected) out->push_back(item);
    }
    return out;
}

static value default_value(const func_args & args) {
    args.ensure_count(1, 3);
    value def_val = args.get_kwarg_or_pos("default_value", 1);
    if (args.count() == 1 && def_val->is_undefined()) {
        def_val = mk_val<value_string>("");
    }
    value val_check = args.get_kwarg_or_pos("boolean", 2);
    bool check_bool = val_check->as_bool(); // undefined == false
    bool no_value = check_bool
        ? (!args.get_pos(0)->as_bool())
        : (args.get_pos(0)->is_undefined() || args.get_pos(0)->is_none());
    return no_value ? def_val : args.get_pos(0);
}

const func_builtins & global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string msg = args.get_pos(0)->as_string().str();
            throw raised_exception("Jinja Exception: " + msg);
        }},
        {"namespace", [](const func_args & args) -> value {
            auto out = mk_val<value_object>();
            for (const auto & arg : args.get_args()) {
                if (!is_val<value_kwarg>(arg)) {
                    throw raised_exception("namespace() arguments must be kwargs");
                }
                auto kwarg = cast_val<value_kwarg>(arg);
                JJ_DEBUG("namespace: adding key '%s'", kwarg->key.c_str());
                out->insert(kwarg->key, kwarg->val);
            }
            return out;
        }},
        {"strftime_now", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string format = args.get_pos(0)->as_string().str();
            // get current time
            // TODO: make sure this is the same behavior as Python's strftime
            char buf[100];
            std::tm utc_time{};
            if (!templates_utc_calendar(args.ctx.current_time, utc_time)) { throw raised_exception("Timestamp outside UTC calendar range"); }
            if (std::strftime(buf, sizeof(buf), format.c_str(), &utc_time)) {
                return mk_val<value_string>(std::string(buf));
            } else {
                throw raised_exception("strftime_now: failed to format time");
            }
        }},
        {"range", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            args.ensure_vals<value_int, value_int, value_int>(true, false, false);

            auto arg0 = args.get_pos(0);
            auto arg1 = args.get_pos(1, mk_val<value_undefined>());
            auto arg2 = args.get_pos(2, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }

            auto out = mk_val<value_array>();
            if (step == 0) {
                throw raised_exception("range() step argument must not be zero");
            }
            if (step > 0) {
                for (int64_t i = start; i < stop; i += step) {
                    out->push_back(mk_val<value_int>(i));
                }
            } else {
                for (int64_t i = start; i > stop; i += step) {
                    out->push_back(mk_val<value_int>(i));
                }
            }
            return out;
        }},
        {"tojson", tojson},
        {"dict", [](const func_args & args) -> value {
            auto out = mk_val<value_object>();
            dict_update_from_args(*out, args, 0, "dict");
            return out;
        }},
        {"zip", [](const func_args & args) -> value {
            auto out = mk_val<value_array>();
            if (args.count() == 0) {
                return out;
            }
            std::vector<std::vector<value>> seqs;
            seqs.reserve(args.count());
            size_t min_len = SIZE_MAX;
            for (const auto & arg : args.get_args()) {
                seqs.push_back(to_iterable_items(arg, "zip"));
                min_len = std::min(min_len, seqs.back().size());
            }
            for (size_t i = 0; i < min_len; ++i) {
                std::vector<value> tup;
                tup.reserve(seqs.size());
                for (const auto & seq : seqs) {
                    tup.push_back(seq[i]);
                }
                out->push_back(mk_val<value_tuple>(std::move(tup)));
            }
            return out;
        }},
        {"enumerate", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            const auto items = to_iterable_items(args.get_pos(0), "enumerate");
            value val_start = args.get_kwarg_or_pos("start", 1);
            int64_t start = val_start->is_undefined() ? 0 : val_start->as_int();
            auto out = mk_val<value_array>();
            for (size_t i = 0; i < items.size(); ++i) {
                out->push_back(mk_val<value_tuple>(std::vector<value>{
                    mk_val<value_int>(start + static_cast<int64_t>(i)),
                    items[i],
                }));
            }
            return out;
        }},
        {"len", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            const value & v = args.get_pos(0);
            if (is_val<value_array>(v)) {
                return mk_val<value_int>(static_cast<int64_t>(v->as_array().size()));
            }
            if (is_val<value_string>(v)) {
                return mk_val<value_int>(static_cast<int64_t>(v->as_string().length()));
            }
            if (is_val<value_object>(v)) {
                return mk_val<value_int>(static_cast<int64_t>(v->as_ordered_object().size()));
            }
            if (v->is_undefined()) {
                return mk_val<value_int>(0);
            }
            throw raised_exception("len(): object of type " + v->type() + " has no len()");
        }},
        {"str", [](const func_args & args) -> value {
            args.ensure_count(0, 1);
            if (args.count() == 0 || args.get_pos(0)->is_undefined()) {
                return mk_val<value_string>("");
            }
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_count(0, 2);
            if (args.count() == 0) {
                return mk_val<value_int>(0);
            }
            value v = args.get_pos(0);
            value val_base = args.get_kwarg_or_pos("base", 1);
            if (is_val<value_string>(v)) {
                const int base = val_base->is_undefined() ? 10 : static_cast<int>(val_base->as_int());
                if (base != 0 && (base < 2 || base > 36)) {
                    throw raised_exception("int() base must be 0 or between 2 and 36");
                }
                return mk_val<value_int>(std::stoll(v->as_string().str(), nullptr, base));
            }
            if (!val_base->is_undefined()) {
                throw raised_exception("int() can't convert non-string with explicit base");
            }
            if (is_val<value_int>(v) || is_val<value_bool>(v)) {
                return mk_val<value_int>(v->as_int());
            }
            if (is_val<value_float>(v)) {
                return mk_val<value_int>(static_cast<int64_t>(v->as_float()));
            }
            throw raised_exception("int(): cannot convert " + v->type());
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_count(0, 1);
            if (args.count() == 0) {
                return mk_val<value_float>(0.0);
            }
            value v = args.get_pos(0);
            if (is_val<value_string>(v)) {
                return mk_val<value_float>(std::stod(v->as_string().str()));
            }
            if (is_val<value_int>(v) || is_val<value_float>(v)) {
                return mk_val<value_float>(v->as_float());
            }
            if (is_val<value_bool>(v)) {
                return mk_val<value_float>(v->as_bool() ? 1.0 : 0.0);
            }
            throw raised_exception("float(): cannot convert " + v->type());
        }},
        {"bool", [](const func_args & args) -> value {
            args.ensure_count(0, 1);
            return mk_val<value_bool>(args.count() == 0 ? false : args.get_pos(0)->as_bool());
        }},
        {"list", [](const func_args & args) -> value {
            args.ensure_count(0, 1);
            if (args.count() == 0) {
                return mk_val<value_array>();
            }
            return mk_val<value_array>(to_iterable_items(args.get_pos(0), "list"));
        }},
        {"min", global_min_max<false>},
        {"max", global_min_max<true>},
        {"sum", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            const auto items = to_iterable_items(args.get_pos(0), "sum");
            value start_val = args.get_kwarg_or_pos("start", 1);
            return sum_values(items, start_val, mk_val<value_undefined>());
        }},
        {"abs", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            value v = args.get_pos(0);
            if (is_val<value_int>(v) || is_val<value_bool>(v)) {
                int64_t n = v->as_int();
                return mk_val<value_int>(n < 0 ? -n : n);
            }
            if (is_val<value_float>(v)) {
                return mk_val<value_float>(std::fabs(v->as_float()));
            }
            throw raised_exception("abs(): bad operand type " + v->type());
        }},
        {"round", round_value},
        {"sorted", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            const auto items = to_iterable_items(args.get_pos(0), "sorted");
            const bool reverse = args.get_kwarg_or_pos("reverse", 1)->as_bool();
            auto out = mk_val<value_array>();
            for (size_t i : stable_order(items, reverse)) {
                out->push_back(items[i]);
            }
            return out;
        }},
        {"reversed", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            auto items = to_iterable_items(args.get_pos(0), "reversed");
            std::reverse(items.begin(), items.end());
            return mk_val<value_array>(std::move(items));
        }},
        {"any", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            const auto items = to_iterable_items(args.get_pos(0), "any");
            return mk_val<value_bool>(std::any_of(items.begin(), items.end(), [](const value & v) {
                return v->as_bool();
            }));
        }},
        {"all", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            const auto items = to_iterable_items(args.get_pos(0), "all");
            return mk_val<value_bool>(std::all_of(items.begin(), items.end(), [](const value & v) {
                return v->as_bool();
            }));
        }},

        // tests
        {"test_is_boolean", test_type_fn<value_bool>},
        {"test_is_callable", test_type_fn<value_func>},
        {"test_is_odd", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_bool>(val % 2 != 0);
        }},
        {"test_is_even", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_bool>(val % 2 == 0);
        }},
        {"test_is_false", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.get_pos(0)) && !args.get_pos(0)->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_true", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.get_pos(0)) && args.get_pos(0)->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_divisibleby", [](const func_args & args) -> value {
            args.ensure_vals<value_int, value_int>();
            bool res = args.get_pos(0)->val_int % args.get_pos(1)->val_int == 0;
            return mk_val<value_bool>(res);
        }},
        {"test_is_string", test_type_fn<value_string>},
        {"test_is_integer", test_type_fn<value_int>},
        {"test_is_float", test_type_fn<value_float>},
        {"test_is_number", test_type_fn<value_int, value_float>},
        {"test_is_iterable", test_type_fn<value_array, value_string, value_object, value_undefined>},
        {"test_is_sequence", test_type_fn<value_array, value_string, value_undefined>},
        {"test_is_mapping", test_type_fn<value_object>},
        {"test_is_lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.get_pos(0)->val_str.is_lowercase());
        }},
        {"test_is_upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.get_pos(0)->val_str.is_uppercase());
        }},
        {"test_is_none", test_type_fn<value_none>},
        {"test_is_defined", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool res = !args.get_pos(0)->is_undefined();
            JJ_DEBUG("test_is_defined: result=%d", res ? 1 : 0);
            return mk_val<value_bool>(res);
        }},
        {"test_is_undefined", test_type_fn<value_undefined>},
        {"test_is_eq", test_compare_fn<value_compare_op::eq>},
        {"test_is_equalto", test_compare_fn<value_compare_op::eq>},
        {"test_is_ge", test_compare_fn<value_compare_op::ge>},
        {"test_is_gt", test_compare_fn<value_compare_op::gt>},
        {"test_is_greaterthan", test_compare_fn<value_compare_op::gt>},
        {"test_is_le", test_compare_fn<value_compare_op::le>},
        {"test_is_lt", test_compare_fn<value_compare_op::lt>},
        {"test_is_lessthan", test_compare_fn<value_compare_op::lt>},
        {"test_is_ne", test_compare_fn<value_compare_op::ne>},
        {"test_is_in", [](const func_args & args) -> value {
            args.ensure_count(2);
            auto needle   = args.get_pos(0);
            auto haystack = args.get_pos(1);
            if (is_val<value_undefined>(haystack)) {
                return mk_val<value_bool>(false);
            }
            if (is_val<value_array>(haystack)) {
                for (const auto & item : haystack->as_array()) {
                    if (*needle == *item) {
                        return mk_val<value_bool>(true);
                    }
                }
                return mk_val<value_bool>(false);
            }
            if (is_val<value_string>(haystack)) {
                if (!is_val<value_string>(needle)) {
                    throw raised_exception("'in' test expects args[1] as string when args[0] is string, got args[1] as " + needle->type());
                }
                return mk_val<value_bool>(
                    haystack->as_string().str().find(needle->as_string().str()) != std::string::npos);
            }
            if (is_val<value_object>(haystack)) {
                return mk_val<value_bool>(haystack->has_key(needle));
            }
            throw raised_exception("'in' test expects iterable as first argument, got " + haystack->type());
        }},
        {"test_is_test", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            auto & builtins = global_builtins();
            std::string test_name = args.get_pos(0)->val_str.str();
            auto it = builtins.find("test_is_" + test_name);
            bool res = it != builtins.end();
            return mk_val<value_bool>(res);
        }},
        {"test_is_sameas", [](const func_args & args) -> value {
            // Check if an object points to the same memory address as another object
            (void)args;
            throw not_implemented_exception("sameas test not implemented");
        }},
        {"test_is_escaped", [](const func_args & args) -> value {
            (void)args;
            throw not_implemented_exception("escaped test not implemented");
        }},
        {"test_is_filter", [](const func_args & args) -> value {
            (void)args;
            throw not_implemented_exception("filter test not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_int>(val < 0 ? -val : val);
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            return mk_val<value_int>(args.get_pos(0)->as_int());
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            double val = static_cast<double>(args.get_pos(0)->as_int());
            return mk_val<value_float>(val);
        }},
        {"round", round_value},
        {"safe", tojson},
        {"string", tojson},
        {"tojson", tojson},
    };
    return builtins;
}


const func_builtins & value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            double val = args.get_pos(0)->as_float();
            return mk_val<value_float>(val < 0.0 ? -val : val);
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            int64_t val = static_cast<int64_t>(args.get_pos(0)->as_float());
            return mk_val<value_int>(val);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            return mk_val<value_float>(args.get_pos(0)->as_float());
        }},
        {"round", round_value},
        {"safe", tojson},
        {"string", tojson},
        {"tojson", tojson},
    };
    return builtins;
}

static bool string_startswith(const std::string & str, const std::string & prefix) {
    if (str.length() < prefix.length()) return false;
    return str.compare(0, prefix.length(), prefix) == 0;
}

static bool string_endswith(const std::string & str, const std::string & suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

static bool string_matches_affix(const std::string & str, const value & affix, bool (*matcher)(const std::string &, const std::string &), const char * fn_name) {
    if (is_val<value_string>(affix)) {
        return matcher(str, affix->as_string().str());
    }
    if (is_val<value_array>(affix)) {
        for (const auto & item : affix->as_array()) {
            if (!is_val<value_string>(item)) {
                throw raised_exception(std::string(fn_name) + ": tuple elements must be strings");
            }
            if (matcher(str, item->as_string().str())) {
                return true;
            }
        }
        return false;
    }
    throw raised_exception(std::string(fn_name) + ": expected string or tuple of strings, got " + affix->type());
}

const func_builtins & value_string_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().uppercase();
            return mk_val<value_string>(str);
        }},
        {"lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().lowercase();
            return mk_val<value_string>(str);
        }},
        {"strip", [](const func_args & args) -> value {
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("strip() first argument must be a string");
            }
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true, val_chars->as_string().str()));
            }
        }},
        {"rstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true, val_chars->as_string().str()));
            }
        }},
        {"lstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false, val_chars->as_string().str()));
            }
        }},
        {"title", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().titlecase();
            return mk_val<value_string>(str);
        }},
        {"capitalize", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().capitalize();
            return mk_val<value_string>(str);
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string();
            return mk_val<value_int>(str.length());
        }},
        {"startswith", [](const func_args & args) -> value {
            args.ensure_count(2, 2);
            args.ensure_vals<value_string>();
            std::string str = args.get_pos(0)->as_string().str();
            return mk_val<value_bool>(string_matches_affix(str, args.get_pos(1), string_startswith, "startswith"));
        }},
        {"endswith", [](const func_args & args) -> value {
            args.ensure_count(2, 2);
            args.ensure_vals<value_string>();
            std::string str = args.get_pos(0)->as_string().str();
            return mk_val<value_bool>(string_matches_affix(str, args.get_pos(1), string_endswith, "endswith"));
        }},
        {"split", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("split() first argument must be a string");
            }
            const auto & jstr = val_input->as_string();
            std::string str = jstr.str();
            value val_sep = args.get_kwarg_or_pos("sep", 1);
            value val_max = args.get_kwarg_or_pos("maxsplit", 2);
            int64_t maxsplit = val_max->is_undefined() ? -1 : val_max->as_int();
            auto result = mk_val<value_array>();
            auto push_part = [&](std::string part) {
                auto res = mk_val<value_string>(std::move(part));
                res->val_str.mark_input_based_on(jstr);
                result->push_back(std::move(res));
            };
            if (val_sep->is_undefined() || val_sep->is_none()) {
                size_t i = 0;
                const size_t n = str.size();
                while (i < n && std::isspace(static_cast<unsigned char>(str[i]))) ++i;
                while (i < n) {
                    if (maxsplit == 0) {
                        push_part(str.substr(i));
                        break;
                    }
                    size_t start = i;
                    while (i < n && !std::isspace(static_cast<unsigned char>(str[i]))) ++i;
                    push_part(str.substr(start, i - start));
                    if (maxsplit > 0) --maxsplit;
                    while (i < n && std::isspace(static_cast<unsigned char>(str[i]))) ++i;
                }
                return result;
            }
            std::string delim = val_sep->as_string().str();
            if (delim.empty()) {
                throw raised_exception("empty separator");
            }
            size_t pos = 0;
            while (maxsplit != 0 && (pos = str.find(delim)) != std::string::npos) {
                push_part(str.substr(0, pos));
                str.erase(0, pos + delim.length());
                if (maxsplit > 0) --maxsplit;
            }
            push_part(std::move(str));
            return result;
        }},
        {"rsplit", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("rsplit() first argument must be a string");
            }
            const auto & jstr = val_input->as_string();
            std::string str = jstr.str();
            value val_sep = args.get_kwarg_or_pos("sep", 1);
            value val_max = args.get_kwarg_or_pos("maxsplit", 2);
            int64_t maxsplit = val_max->is_undefined() ? -1 : val_max->as_int();
            auto result = mk_val<value_array>();
            auto push_part = [&](std::string part) {
                auto res = mk_val<value_string>(std::move(part));
                res->val_str.mark_input_based_on(jstr);
                result->push_back(std::move(res));
            };
            if (val_sep->is_undefined() || val_sep->is_none()) {
                size_t end = str.size();
                while (end > 0 && std::isspace(static_cast<unsigned char>(str[end - 1]))) --end;
                while (end > 0) {
                    if (maxsplit == 0) {
                        push_part(str.substr(0, end));
                        break;
                    }
                    size_t stop = end;
                    while (end > 0 && !std::isspace(static_cast<unsigned char>(str[end - 1]))) --end;
                    push_part(str.substr(end, stop - end));
                    if (maxsplit > 0) --maxsplit;
                    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end - 1]))) --end;
                }
                result->reverse();
                return result;
            }
            std::string delim = val_sep->as_string().str();
            if (delim.empty()) {
                throw raised_exception("empty separator");
            }
            size_t pos = 0;
            while (maxsplit != 0 && (pos = str.rfind(delim)) != std::string::npos) {
                push_part(str.substr(pos + delim.length()));
                str.erase(pos);
                if (maxsplit > 0) --maxsplit;
            }
            push_part(std::move(str));
            result->reverse();
            return result;
        }},
        {"replace", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string, value_string, value_int>(true, true, true, false);
            std::string str = args.get_pos(0)->as_string().str();
            std::string old_str = args.get_pos(1)->as_string().str();
            std::string new_str = args.get_pos(2)->as_string().str();
            int64_t count = args.count() > 3 ? args.get_pos(3)->as_int() : -1;
            if (count != 0 && old_str != new_str) {
                int64_t remaining = count;
                if (old_str.empty()) {
                    std::string new_res;
                    for (size_t i = 0; i <= str.length(); ++i) {
                        if (remaining != 0) {
                            new_res += new_str;
                            if (remaining > 0) --remaining;
                        }
                        if (i < str.length()) {
                            new_res.push_back(str[i]);
                        }
                    }
                    str = std::move(new_res);
                } else {
                    size_t pos = 0;
                    while (remaining != 0 && (pos = str.find(old_str, pos)) != std::string::npos) {
                        str.replace(pos, old_str.length(), new_str);
                        pos += new_str.length();
                        if (remaining > 0) --remaining;
                    }
                }
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            return res;
        }},
        {"format", [](const func_args & args) -> value {
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("format() first argument must be a string");
            }
            const jinja::string & fmt = val_input->as_string();
            const bool fmt_is_input = fmt.all_parts_are_input();

            const std::string str = fmt.str();
            jinja::string result;
            std::string literal;
            auto flush_literal = [&]() {
                if (!literal.empty()) {
                    result.parts.push_back({fmt_is_input, literal});
                    literal.clear();
                }
            };

            size_t arg_idx = 1; // positional args follow the format string
            for (size_t i = 0; i < str.size(); ++i) {
                if (str[i] != '{') {
                    literal += str[i];
                    continue;
                }
                if (i + 1 >= str.size() || str[i + 1] != '}') {
                    throw not_implemented_exception("format() only supports simple '{}' placeholders");
                }
                ++i;
                flush_literal();
                const jinja::string arg_str = args.get_pos(arg_idx++)->as_string();
                result.parts.insert(result.parts.end(), arg_str.parts.begin(), arg_str.parts.end());
            }
            flush_literal();
            return mk_val<value_string>(result);
        }},
        {"int", [](const func_args & args) -> value {
            value val_input   = args.get_pos(0);
            value val_default = args.get_kwarg_or_pos("default", 1);
            value val_base    = args.get_kwarg_or_pos("base",    2);
            const int base = val_base->is_undefined() ? 10 : val_base->as_int();
            if (base != 0 && (base < 2 || base > 36)) {
                // an out-of-range base makes std::stoi fail fast on the MSVC CRT instead of throwing
                throw raised_exception("int() base must be 0 or between 2 and 36");
            }
            if (is_val<value_string>(val_input) == false) {
                throw raised_exception("int() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            try {
                return mk_val<value_int>(std::stoi(str, nullptr, base));
            } catch (...) {
                return mk_val<value_int>(val_default->is_undefined() ? 0 : val_default->as_int());
            }
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_default = args.get_kwarg_or_pos("default", 1);
            std::string str = args.get_pos(0)->as_string().str();
            try {
                return mk_val<value_float>(std::stod(str));
            } catch (...) {
                return mk_val<value_float>(val_default->is_undefined() ? 0.0 : val_default->as_float());
            }
        }},
        {"string", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_string>();
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_string, value_int, value_int, value_int>(true, true, false, false);

            auto arg0 = args.get_pos(1);
            auto arg1 = args.get_pos(2, mk_val<value_undefined>());
            auto arg2 = args.get_pos(3, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto input = args.get_pos(0);
            auto sliced = slice(input->as_string().str(), start, stop, step);
            auto res = mk_val<value_string>(sliced);
            res->val_str.mark_input_based_on(input->as_string());
            return res;
        }},
        {"safe", [](const func_args & args) -> value {
            // no-op for now
            args.ensure_vals<value_string>();
            return args.get_pos(0);
        }},
        {"tojson", tojson},
        {"indent", [](const func_args &args) -> value {
            args.ensure_count(1, 4);
            value val_input  = args.get_pos(0);
            value val_width  = args.get_kwarg_or_pos("width", 1);
            const bool first = args.get_kwarg_or_pos("first", 2)->as_bool(); // undefined == false
            const bool blank = args.get_kwarg_or_pos("blank", 3)->as_bool(); // undefined == false
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("indent() first argument must be a string");
            }
            std::string indent;
            if (is_val<value_int>(val_width)) {
                indent.assign(val_width->as_int(), ' ');
            } else if (is_val<value_string>(val_width)) {
                indent = val_width->as_string().str();
            } else {
                indent = "    ";
            }
            std::string indented;
            std::string input = val_input->as_string().str();
            std::istringstream iss = std::istringstream(input);
            std::string line;
            while (std::getline(iss, line)) {
                if (!indented.empty()) {
                    indented.push_back('\n');
                }
                if ((indented.empty() ? first : (!line.empty() || blank))) {
                    indented += indent;
                }
                indented += line;
            }
            if (!input.empty() && input.back() == '\n') {
                indented.push_back('\n');
                if (blank) {
                    indented += indent;
                }
            }

            auto res = mk_val<value_string>(indented);
            res->val_str.mark_input_based_on(val_input->as_string());
            return res;
        }},
        {"join", [](const func_args & args) -> value {
            args.ensure_count(2, 2);
            args.ensure_vals<value_string>();
            const std::string delim = args.get_pos(0)->as_string().str();
            const auto items = to_iterable_items(args.get_pos(1), "join");
            std::string result;
            for (size_t i = 0; i < items.size(); ++i) {
                if (i > 0) {
                    result += delim;
                }
                result += items[i]->as_string().str();
            }
            return mk_val<value_string>(result);
        }},
        {"from_json", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            const jinja::string & str = args.get_pos(0)->as_string();
            return from_json(common_json::parse(str.str()), str.all_parts_are_input());
        }},
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_array>(to_iterable_items(args.get_pos(0), "list"));
        }},
        {"find", [](const func_args & args) -> value {
            args.ensure_count(2, 4);
            args.ensure_vals<value_string, value_string, value_int, value_int>(true, true, false, false);
            const std::string str = args.get_pos(0)->as_string().str();
            const std::string sub = args.get_pos(1)->as_string().str();
            auto [start, end] = clamp_slice_bounds(str.size(), args.get_pos(2, mk_val<value_undefined>()), args.get_pos(3, mk_val<value_undefined>()));
            if (start > end) {
                return mk_val<value_int>(-1);
            }
            size_t pos = str.find(sub, start);
            if (pos == std::string::npos || pos + sub.size() > end) {
                return mk_val<value_int>(-1);
            }
            return mk_val<value_int>(static_cast<int64_t>(pos));
        }},
        {"rfind", [](const func_args & args) -> value {
            args.ensure_count(2, 4);
            args.ensure_vals<value_string, value_string, value_int, value_int>(true, true, false, false);
            const std::string str = args.get_pos(0)->as_string().str();
            const std::string sub = args.get_pos(1)->as_string().str();
            auto [start, end] = clamp_slice_bounds(str.size(), args.get_pos(2, mk_val<value_undefined>()), args.get_pos(3, mk_val<value_undefined>()));
            if (start > end || sub.size() > end - start) {
                return mk_val<value_int>(-1);
            }
            size_t pos = str.rfind(sub, end - sub.size());
            if (pos == std::string::npos || pos < start) {
                return mk_val<value_int>(-1);
            }
            return mk_val<value_int>(static_cast<int64_t>(pos));
        }},
        {"splitlines", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            args.ensure_vals<value_string>();
            const auto & jstr = args.get_pos(0)->as_string();
            const std::string str = jstr.str();
            const bool keepends = args.get_kwarg_or_pos("keepends", 1)->as_bool();
            auto out = mk_val<value_array>();
            size_t i = 0;
            const size_t n = str.size();
            while (i < n) {
                size_t start = i;
                while (i < n && str[i] != '\n' && str[i] != '\r') ++i;
                size_t eol = i;
                if (i < n) {
                    if (str[i] == '\r' && i + 1 < n && str[i + 1] == '\n') {
                        i += 2;
                    } else {
                        ++i;
                    }
                }
                auto line = mk_val<value_string>(str.substr(start, (keepends ? i : eol) - start));
                line->val_str.mark_input_based_on(jstr);
                out->push_back(std::move(line));
            }
            return out;
        }},
        {"removeprefix", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            const auto & jstr = args.get_pos(0)->as_string();
            const std::string str = jstr.str();
            const std::string prefix = args.get_pos(1)->as_string().str();
            if (!string_startswith(str, prefix)) {
                return args.get_pos(0);
            }
            auto res = mk_val<value_string>(str.substr(prefix.size()));
            res->val_str.mark_input_based_on(jstr);
            return res;
        }},
        {"removesuffix", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            const auto & jstr = args.get_pos(0)->as_string();
            const std::string str = jstr.str();
            const std::string suffix = args.get_pos(1)->as_string().str();
            if (suffix.empty() || !string_endswith(str, suffix)) {
                return args.get_pos(0);
            }
            auto res = mk_val<value_string>(str.substr(0, str.size() - suffix.size()));
            res->val_str.mark_input_based_on(jstr);
            return res;
        }},
        {"count", [](const func_args & args) -> value {
            args.ensure_count(2, 4);
            args.ensure_vals<value_string, value_string, value_int, value_int>(true, true, false, false);
            const std::string str = args.get_pos(0)->as_string().str();
            const std::string sub = args.get_pos(1)->as_string().str();
            auto [start, end] = clamp_slice_bounds(str.size(), args.get_pos(2, mk_val<value_undefined>()), args.get_pos(3, mk_val<value_undefined>()));
            if (start > end) {
                return mk_val<value_int>(0);
            }
            if (sub.empty()) {
                return mk_val<value_int>(static_cast<int64_t>(end - start + 1));
            }
            int64_t cnt = 0;
            size_t pos = start;
            while (pos + sub.size() <= end && (pos = str.find(sub, pos)) != std::string::npos && pos + sub.size() <= end) {
                ++cnt;
                pos += sub.size();
            }
            return mk_val<value_int>(cnt);
        }},
    };
    return builtins;
}


const func_builtins & value_bool_t::get_builtins() const {
    static const func_handler tostring = [](const func_args & args) -> value {
        args.ensure_vals<value_bool>();
        bool val = args.get_pos(0)->as_bool();
        return mk_val<value_string>(val ? "True" : "False");
    };
    static const func_builtins builtins = {
        {"default", default_value},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_int>(val ? 1 : 0);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_float>(val ? 1.0 : 0.0);
        }},
        {"safe", tostring},
        {"string", tostring},
        {"tojson", tojson},
    };
    return builtins;
}

const func_builtins & value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            auto result = mk_val<value_array>();
            for (const auto& v : arr) {
                result->push_back(v);
            }
            return result;
        }},
        {"first", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[0];
        }},
        {"last", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[arr.size() - 1];
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            return mk_val<value_int>(static_cast<int64_t>(arr.size()));
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_array, value_int, value_int, value_int>(true, true, false, false);

            auto val  = args.get_pos(0);
            auto arg0 = args.get_pos(1);
            auto arg1 = args.get_pos(2, mk_val<value_undefined>());
            auto arg2 = args.get_pos(3, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto arr = slice(val->as_array(), start, stop, step);
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr)) : mk_val<value_array>(std::move(arr));
        }},
        {"selectattr", selectattr<false>},
        {"select", select_filter<false>},
        {"rejectattr", selectattr<true>},
        {"reject", select_filter<true>},
        {"join", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("join() first argument must be an array");
            }
            value val_delim = args.get_kwarg_or_pos("d",         1);
            value attribute = args.get_kwarg_or_pos("attribute", 2);
            const auto & arr = args.get_pos(0)->as_array();
            const bool attr_is_int = is_val<value_int>(attribute);
            if (!attribute->is_undefined() && !is_val<value_string>(attribute) && !attr_is_int) {
                throw raised_exception("join() attribute must be string or integer");
            }
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            const std::string delim = val_delim->is_undefined() ? "" : val_delim->as_string().str();
            std::string result;
            for (size_t i = 0; i < arr.size(); ++i) {
                value val_arr = arr[i];
                if (!attribute->is_undefined()) {
                    if (attr_is_int && is_val<value_array>(val_arr)) {
                        val_arr = val_arr->at(attr_int);
                    } else if (!attr_is_int && is_val<value_object>(val_arr)) {
                        val_arr = val_arr->at(attribute);
                    }
                }
                if (!is_val<value_string>(val_arr) && !is_val<value_int>(val_arr) && !is_val<value_float>(val_arr)) {
                    throw raised_exception("join() can only join arrays of strings or numerics");
                }
                result += val_arr->as_string().str();
                if (i < arr.size() - 1) {
                    result += delim;
                }
            }
            return mk_val<value_string>(result);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            if (args.ctx.is_get_stats) {
                // mark as used (recursively) for stats
                auto val_input = args.get_pos(0);
                value_t::stats_t::mark_used(const_cast<value&>(val_input), true);
            }
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"tojson", tojson},
        {"map", [](const func_args & args) -> value {
            args.ensure_count(2);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("map: first argument must be an array");
            }
            if (!is_val<value_kwarg>(args.get_args().at(1))) {
                throw not_implemented_exception("map: filter-mapping not implemented");
            }
            value val       = args.get_pos(0);
            value attribute = args.get_kwarg_or_pos("attribute", 1);
            const bool attr_is_int = is_val<value_int>(attribute);
            if (!is_val<value_string>(attribute) && !attr_is_int) {
                throw raised_exception("map: attribute must be string or integer");
            }
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            value default_val = args.get_kwarg("default", mk_val<value_undefined>());
            auto out = mk_val<value_array>();
            auto arr = val->as_array();
            for (const auto & item : arr) {
                value attr_val;
                if (attr_is_int) {
                    attr_val = is_val<value_array>(item) ? item->at(attr_int, default_val) : default_val;
                } else {
                    attr_val = is_val<value_object>(item) ? item->at(attribute, default_val) : default_val;
                }
                out->push_back(attr_val);
            }
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(out->as_array())) : out;
        }},
        {"append", [](const func_args & args) -> value {
            args.ensure_count(2);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("append: first argument must be an array");
            }
            const value_array_t * arr = cast_val<value_array>(args.get_pos(0));
            // need to use const_cast here to modify the array
            value_array_t * arr_editable = const_cast<value_array_t *>(arr);
            arr_editable->push_back(args.get_pos(1));
            return args.get_pos(0);
        }},
        {"pop", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            args.ensure_vals<value_array, value_int>(true, false);
            int64_t index = args.count() == 2 ? args.get_pos(1)->as_int() : -1;
            const value_array_t * arr = cast_val<value_array>(args.get_pos(0));
            // need to use const_cast here to modify the array
            value_array_t * arr_editable = const_cast<value_array_t *>(arr);
            return arr_editable->pop_at(index);
        }},
        {"sort", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("sort: first argument must be an array");
            }
            value val         = args.get_pos(0);
            value val_reverse = args.get_kwarg_or_pos("reverse",        1);
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 2);
            value attribute   = args.get_kwarg_or_pos("attribute",      3);
            const bool case_sensitive = val_case->as_bool(); // undefined == false
            const bool reverse = val_reverse->as_bool(); // undefined == false
            const auto & items = val->as_array();
            std::vector<value> keys;
            for (const auto & item : items) {
                keys.push_back(comparison_key(item, attribute, case_sensitive));
            }
            std::vector<value> arr;
            for (size_t i : stable_order(keys, reverse)) {
                arr.push_back(items[i]);
            }
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr)) : mk_val<value_array>(std::move(arr));
        }},
        {"reverse", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            value val = args.get_pos(0);
            std::vector<value> arr = val->as_array(); // copy
            std::reverse(arr.begin(), arr.end());
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr)) : mk_val<value_array>(std::move(arr));
        }},
        {"min", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_array>();
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 1);
            value attribute   = args.get_kwarg_or_pos("attribute",      2);
            const bool case_sensitive = val_case->as_bool(); // undefined == false
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            value result = arr[0];
            for (size_t i = 1; i < arr.size(); ++i) {
                if (value_compare(comparison_key(arr[i], attribute, case_sensitive),
                                  comparison_key(result, attribute, case_sensitive), value_compare_op::lt)) {
                    result = arr[i];
                }
            }
            return result;
        }},
        {"max", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_array>();
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 1);
            value attribute   = args.get_kwarg_or_pos("attribute",      2);
            const bool case_sensitive = val_case->as_bool(); // undefined == false
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            value result = arr[0];
            for (size_t i = 1; i < arr.size(); ++i) {
                if (value_compare(comparison_key(arr[i], attribute, case_sensitive),
                                  comparison_key(result, attribute, case_sensitive), value_compare_op::gt)) {
                    result = arr[i];
                }
            }
            return result;
        }},
        {"unique", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            args.ensure_vals<value_array>();
            value val       = args.get_pos(0);
            value val_case  = args.get_kwarg_or_pos("case_sensitive", 1);
            value attribute = args.get_kwarg_or_pos("attribute",      2);
            const bool case_sensitive = val_case->as_bool(); // undefined == false
            // Each item whose key is new, in order, as Jinja's set of seen keys.
            std::unordered_set<value, value_hasher, value_equivalence> seen;
            auto out = mk_val<value_array>();
            for (const auto & item : val->as_array()) {
                value key = comparison_key(item, attribute, case_sensitive);
                if (!key->is_hashable()) {
                    throw raised_exception("unique: unhashable " + key->type());
                }
                if (seen.insert(key).second) {
                    out->push_back(item);
                }
            }
            return out;
        }},
        {"sum", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            args.ensure_vals<value_array>();
            value attribute = args.get_kwarg_or_pos("attribute", 1);
            value start_val = args.get_kwarg_or_pos("start",     2);
            return sum_values(args.get_pos(0)->as_array(), start_val, attribute);
        }},
        {"extend", [](const func_args & args) -> value {
            args.ensure_count(2, 2);
            args.ensure_vals<value_array>();
            auto * arr = const_cast<value_array_t *>(cast_val<value_array>(args.get_pos(0)));
            for (const auto & item : to_iterable_items(args.get_pos(1), "extend")) {
                arr->push_back(item);
            }
            return mk_val<value_none>();
        }},
        {"insert", [](const func_args & args) -> value {
            args.ensure_count(3, 3);
            args.ensure_vals<value_array, value_int>();
            auto * arr = const_cast<value_array_t *>(cast_val<value_array>(args.get_pos(0)));
            if (arr->is_immutable()) {
                throw std::runtime_error("Attempting to modify immutable type");
            }
            int64_t n = static_cast<int64_t>(arr->val_arr.size());
            int64_t idx = args.get_pos(1)->as_int();
            if (idx < 0) idx += n;
            idx = std::clamp<int64_t>(idx, 0, n);
            arr->val_arr.insert(arr->val_arr.begin() + idx, args.get_pos(2));
            return mk_val<value_none>();
        }},
        {"remove", [](const func_args & args) -> value {
            args.ensure_count(2, 2);
            args.ensure_vals<value_array>();
            auto * arr = const_cast<value_array_t *>(cast_val<value_array>(args.get_pos(0)));
            if (arr->is_immutable()) {
                throw std::runtime_error("Attempting to modify immutable type");
            }
            const value & target = args.get_pos(1);
            for (auto it = arr->val_arr.begin(); it != arr->val_arr.end(); ++it) {
                if (*(*it) == *target) {
                    arr->val_arr.erase(it);
                    return mk_val<value_none>();
                }
            }
            throw raised_exception("remove: value not in list");
        }},
        {"index", [](const func_args & args) -> value {
            args.ensure_count(2, 4);
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            const value & target = args.get_pos(1);
            auto [start, end] = clamp_slice_bounds(arr.size(), args.get_pos(2, mk_val<value_undefined>()), args.get_pos(3, mk_val<value_undefined>()));
            for (size_t i = start; i < end; ++i) {
                if (*arr[i] == *target) {
                    return mk_val<value_int>(static_cast<int64_t>(i));
                }
            }
            throw raised_exception("index: value not in list");
        }},
        {"count", [](const func_args & args) -> value {
            args.ensure_count(2, 2);
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            const value & target = args.get_pos(1);
            int64_t cnt = 0;
            for (const auto & item : arr) {
                if (*item == *target) {
                    ++cnt;
                }
            }
            return mk_val<value_int>(cnt);
        }},
        {"copy", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            args.ensure_vals<value_array>();
            return mk_val<value_array>(args.get_pos(0)->as_array());
        }},
        {"clear", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            args.ensure_vals<value_array>();
            auto * arr = const_cast<value_array_t *>(cast_val<value_array>(args.get_pos(0)));
            if (arr->is_immutable()) {
                throw std::runtime_error("Attempting to modify immutable type");
            }
            arr->val_arr.clear();
            return mk_val<value_none>();
        }},
    };
    return builtins;
}

[[noreturn]] static value object_join_not_implemented(const func_args &) {
    throw not_implemented_exception("object join not implemented");
}

static value object_keys(const func_args & args) {
    args.ensure_vals<value_object>();
    const auto & obj = args.get_pos(0)->as_ordered_object();
    auto result = mk_val<value_array>();
    for (const auto & pair : obj) {
        result->push_back(pair.first);
    }
    return result;
}

static value object_values(const func_args & args) {
    args.ensure_vals<value_object>();
    const auto & obj = args.get_pos(0)->as_ordered_object();
    auto result = mk_val<value_array>();
    for (const auto & pair : obj) {
        result->push_back(pair.second);
    }
    return result;
}

static value object_items(const func_args & args) {
    args.ensure_vals<value_object>();
    const auto & obj = args.get_pos(0)->as_ordered_object();
    auto result = mk_val<value_array>();
    for (const auto & pair : obj) {
        auto item = mk_val<value_tuple>(pair);
        result->push_back(std::move(item));
    }
    return result;
}

const func_builtins & value_object_t::get_builtins() const {
    if (!has_builtins) {
        static const func_builtins no_builtins = {};
        return no_builtins;
    }

    static const func_builtins builtins = {
        {"get", [](const func_args & args) -> value {
            args.ensure_count(2, 3);
            if (!is_val<value_object>(args.get_pos(0))) {
                throw raised_exception("get: first argument must be an object");
            }
            if (!is_val<value_string>(args.get_pos(1))) {
                throw raised_exception("get: second argument must be a string (key)");
            }
            value default_val = mk_val<value_none>();
            if (args.count() == 3) {
                default_val = args.get_pos(2);
            }
            const value obj = args.get_pos(0);
            const value key = args.get_pos(1);
            return obj->at(key, default_val);
        }},
        {"keys", object_keys},
        {"values", object_values},
        {"items", object_items},
        {"update", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            auto * obj = const_cast<value_object_t *>(cast_val<value_object>(args.get_pos(0)));
            dict_update_from_args(*obj, args, 1, "update");
            return mk_val<value_none>();
        }},
        {"pop", [](const func_args & args) -> value {
            args.ensure_count(2, 3);
            args.ensure_vals<value_object>();
            auto * obj = const_cast<value_object_t *>(cast_val<value_object>(args.get_pos(0)));
            const value & key = args.get_pos(1);
            if (value * found = obj->find_key(key)) {
                value saved = *found;
                obj->erase(key);
                return saved;
            }
            if (args.count() == 3) {
                return args.get_pos(2);
            }
            throw raised_exception("pop: key '" + key->as_string().str() + "' not found");
        }},
        {"setdefault", [](const func_args & args) -> value {
            args.ensure_count(2, 3);
            args.ensure_vals<value_object>();
            auto * obj = const_cast<value_object_t *>(cast_val<value_object>(args.get_pos(0)));
            const value & key = args.get_pos(1);
            if (value * found = obj->find_key(key)) {
                return *found;
            }
            value def = args.count() == 3 ? args.get_pos(2) : mk_val<value_none>();
            obj->insert(key, def);
            return def;
        }},
        {"copy", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            args.ensure_vals<value_object>();
            value val = args.get_pos(0);
            return mk_val<value_object>(val);
        }},
        {"clear", [](const func_args & args) -> value {
            args.ensure_count(1, 1);
            args.ensure_vals<value_object>();
            auto * obj = const_cast<value_object_t *>(cast_val<value_object>(args.get_pos(0)));
            if (obj->is_immutable()) {
                throw std::runtime_error("Attempting to modify immutable type");
            }
            obj->val_obj.clear();
            obj->unordered.clear();
            return mk_val<value_none>();
        }},
    };
    return builtins;
}

const func_builtins & value_object_t::get_filters() const {
    static const func_builtins filters = {
        {"default", default_value},
        {"tojson", tojson},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            if (args.ctx.is_get_stats) {
                auto val_input = args.get_pos(0);
                value_t::stats_t::mark_used(const_cast<value&>(val_input), true);
            }
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"safe", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            return mk_val<value_int>(static_cast<int64_t>(obj.size()));
        }},
        {"list", object_keys},
        {"keys", object_keys},
        {"values", object_values},
        {"items", object_items},
        {"dictsort", [](const func_args & args) -> value {
            value val_input   = args.get_pos(0);
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 1);
            value val_by      = args.get_kwarg_or_pos("by",             2);
            value val_reverse = args.get_kwarg_or_pos("reverse",        3);
            const bool case_sensitive = val_case->as_bool(); // undefined == false
            const bool reverse = val_reverse->as_bool(); // undefined == false
            const bool by_value = is_val<value_string>(val_by) && val_by->as_string().str() == "value" ? true : false;
            const value no_attribute = mk_val<value_undefined>();
            auto result = mk_val<value_object>(val_input); // copy
            const auto pairs = result->val_obj;
            std::vector<value> keys;
            for (const auto & pair : pairs) {
                keys.push_back(comparison_key(by_value ? pair.second : pair.first, no_attribute, case_sensitive));
            }
            result->val_obj.clear();
            for (size_t i : stable_order(keys, reverse)) {
                result->val_obj.push_back(pairs[i]);
            }
            return result;
        }},
        {"join", object_join_not_implemented},
    };
    return filters;
}

const func_builtins & value_none_t::get_builtins() const {
    static const func_handler tostring = [](const func_args &) -> value {
        return mk_val<value_string>("None");
    };
    static const func_builtins builtins = {
        {"default", default_value},
        {"tojson", tojson},
        {"string", tostring},
        {"safe", tostring},
        {"items", empty_value_fn<value_array>},
        {"map", empty_value_fn<value_array>},
        {"reject", empty_value_fn<value_array>},
        {"rejectattr", empty_value_fn<value_array>},
        {"select", empty_value_fn<value_array>},
        {"selectattr", empty_value_fn<value_array>},
        {"unique", empty_value_fn<value_array>},
    };
    return builtins;
}


const func_builtins & value_undefined_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"capitalize", empty_value_fn<value_string>},
        {"first", empty_value_fn<value_undefined>},
        {"items", empty_value_fn<value_array>},
        {"join", empty_value_fn<value_string>},
        {"last", empty_value_fn<value_undefined>},
        {"length", empty_value_fn<value_int>},
        {"list", empty_value_fn<value_array>},
        {"lower", empty_value_fn<value_string>},
        {"map", empty_value_fn<value_array>},
        {"max", empty_value_fn<value_undefined>},
        {"min", empty_value_fn<value_undefined>},
        {"reject", empty_value_fn<value_array>},
        {"rejectattr", empty_value_fn<value_array>},
        {"replace", empty_value_fn<value_string>},
        {"reverse", empty_value_fn<value_array>},
        {"safe", empty_value_fn<value_string>},
        {"select", empty_value_fn<value_array>},
        {"selectattr", empty_value_fn<value_array>},
        {"sort", empty_value_fn<value_array>},
        {"string", empty_value_fn<value_string>},
        {"strip", empty_value_fn<value_string>},
        {"sum", empty_value_fn<value_int>},
        {"title", empty_value_fn<value_string>},
        {"truncate", empty_value_fn<value_string>},
        {"unique", empty_value_fn<value_array>},
        {"upper", empty_value_fn<value_string>},
        {"wordcount", empty_value_fn<value_int>},
    };
    return builtins;
}


//////////////////////////////////


static value from_json(const common_json & j, bool mark_input) {
    if (j.is_null()) {
        return mk_val<value_none>();
    } else if (j.is_boolean()) {
        return mk_val<value_bool>(j.get<bool>());
    } else if (j.is_number_integer()) {
        return mk_val<value_int>(j.get<int64_t>());
    } else if (j.is_number_float()) {
        return mk_val<value_float>(j.get<double>());
    } else if (j.is_string()) {
        auto str = mk_val<value_string>(j.get<std::string>());
        if (mark_input) {
            str->mark_input();
        }
        return str;
    } else if (j.is_array()) {
        auto arr = mk_val<value_array>();
        for (const auto & item : j) {
            arr->push_back(from_json(item, mark_input));
        }
        return arr;
    } else if (j.is_object()) {
        auto obj = mk_val<value_object>();
        for (auto it = j.begin(); it != j.end(); ++it) {
            obj->insert(it.key(), from_json(it.value(), mark_input));
        }
        return obj;
    } else {
        throw std::runtime_error("Unsupported JSON value type");
    }
}

// compare operator for value_t
bool value_compare(const value & a, const value & b, value_compare_op op) {
    auto cmp = [&]() {
        // compare numeric types
        if ((is_val<value_int>(a) || is_val<value_float>(a)) &&
            (is_val<value_int>(b) || is_val<value_float>(b))){
            try {
                if (op == value_compare_op::eq) {
                    return a->as_float() == b->as_float();
                } else if (op == value_compare_op::ge) {
                    return a->as_float() >= b->as_float();
                } else if (op == value_compare_op::gt) {
                    return a->as_float() > b->as_float();
                } else if (op == value_compare_op::le) {
                    return a->as_float() <= b->as_float();
                } else if (op == value_compare_op::lt) {
                    return a->as_float() < b->as_float();
                } else if (op == value_compare_op::ne) {
                    return a->as_float() != b->as_float();
                } else {
                    throw std::runtime_error("Unsupported comparison operator for numeric types");
                }
            } catch (...) {}
        }
        // compare string and number
        // TODO: not sure if this is the right behavior
        if ((is_val<value_string>(b) && (is_val<value_int>(a) || is_val<value_float>(a))) ||
            (is_val<value_string>(a) && (is_val<value_int>(b) || is_val<value_float>(b))) ||
            (is_val<value_string>(a) && is_val<value_string>(b))) {
            try {
                if (op == value_compare_op::eq) {
                    return a->as_string().str() == b->as_string().str();
                } else if (op == value_compare_op::ge) {
                    return a->as_string().str() >= b->as_string().str();
                } else if (op == value_compare_op::gt) {
                    return a->as_string().str() > b->as_string().str();
                } else if (op == value_compare_op::le) {
                    return a->as_string().str() <= b->as_string().str();
                } else if (op == value_compare_op::lt) {
                    return a->as_string().str() < b->as_string().str();
                } else if (op == value_compare_op::ne) {
                    return a->as_string().str() != b->as_string().str();
                } else {
                    throw std::runtime_error("Unsupported comparison operator for string/number types");
                }
            } catch (...) {}
        }
        // compare boolean simple
        if (is_val<value_bool>(a) && is_val<value_bool>(b)) {
            if (op == value_compare_op::eq) {
                return a->as_bool() == b->as_bool();
            } else if (op == value_compare_op::ne) {
                return a->as_bool() != b->as_bool();
            } else {
                throw std::runtime_error("Unsupported comparison operator for bool type");
            }
        }
        // compare by type
        if (a->type() != b->type()) {
            return false;
        }
        return false;
    };
    auto result = cmp();
    JJ_DEBUG("Comparing types: %s and %s result=%d", a->type().c_str(), b->type().c_str(), result);
    return result;
}

template<>
void global_from_json(context & ctx, const common_json & json_obj, bool mark_input) {
    // printf("global_from_json: %s\n" , json_obj.dump(2).c_str());
    if (json_obj.is_null() || !json_obj.is_object()) {
        throw std::runtime_error("global_from_json: input JSON value must be an object");
    }
    for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
        JJ_DEBUG("global_from_json: setting key '%s'", it.key().c_str());
        ctx.set_val(it.key(), from_json(it.value(), mark_input));
    }
}

static void append_json_string(std::string & out, const std::string & text) {
    out.push_back('"');
    for (char c : text) {
        switch (c) {
            case '"': out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\b': out.append("\\b"); break;
            case '\f': out.append("\\f"); break;
            case '\n': out.append("\\n"); break;
            case '\r': out.append("\\r"); break;
            case '\t': out.append("\\t"); break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out.append(buf);
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

// recursively convert value to JSON string
// TODO: avoid circular references
static void value_to_json_internal(std::string & out, const value & val, int curr_lvl, int indent, const std::string_view item_sep, const std::string_view key_sep, bool sort_keys) {
    auto append_indent = [&](int extra) {
        if (indent > 0) {
            out.append(static_cast<size_t>((curr_lvl + extra) * indent), ' ');
        }
    };
    auto append_newline = [&]() {
        if (indent >= 0) {
            out.push_back('\n');
        }
    };

    if (is_val<value_none>(val) || val->is_undefined()) {
        out.append("null");
    } else if (is_val<value_bool>(val)) {
        out.append(val->as_bool() ? "true" : "false");
    } else if (is_val<value_int>(val)) {
        out.append(std::to_string(val->as_int()));
    } else if (is_val<value_float>(val)) {
        std::ostringstream oss;
        oss << val->as_float();
        out.append(oss.str());
    } else if (is_val<value_string>(val)) {
        append_json_string(out, val->as_string().str());
    } else if (is_val<value_array>(val)) {
        const auto & arr = val->as_array();
        out.push_back('[');
        if (!arr.empty()) {
            append_newline();
            for (size_t i = 0; i < arr.size(); ++i) {
                append_indent(1);
                value_to_json_internal(out, arr[i], curr_lvl + 1, indent, item_sep, key_sep, sort_keys);
                if (i < arr.size() - 1) {
                    out.append(item_sep);
                }
                append_newline();
            }
            append_indent(0);
        }
        out.push_back(']');
    } else if (is_val<value_object>(val)) {
        const auto & obj = val->as_ordered_object(); // IMPORTANT: need to keep exact order unless sort_keys
        out.push_back('{');
        if (!obj.empty()) {
            append_newline();
            std::vector<size_t> order(obj.size());
            for (size_t i = 0; i < obj.size(); ++i) {
                order[i] = i;
            }
            if (sort_keys) {
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return obj[a].first->as_string().str() < obj[b].first->as_string().str();
                });
            }
            for (size_t i = 0; i < order.size(); ++i) {
                const auto & pair = obj[order[i]];
                append_indent(1);
                append_json_string(out, pair.first->as_string().str());
                out.append(key_sep);
                value_to_json_internal(out, pair.second, curr_lvl + 1, indent, item_sep, key_sep, sort_keys);
                if (i < order.size() - 1) {
                    out.append(item_sep);
                }
                append_newline();
            }
            append_indent(0);
        }
        out.push_back('}');
    } else {
        out.append("null");
    }
}

std::string value_to_json(const value & val, int indent, const std::string_view item_sep, const std::string_view key_sep, bool sort_keys) {
    std::string out;
    out.reserve(64);
    value_to_json_internal(out, val, 0, indent, item_sep, key_sep, sort_keys);
    JJ_DEBUG("value_to_json: result=%s", out.c_str());
    return out;
}

// TODO: avoid circular references
std::string value_to_string_repr(const value & val) {
    if (is_val<value_string>(val)) {
        const std::string val_str = val->as_string().str();

        if (val_str.find('\'') != std::string::npos) {
            return value_to_json(val);
        } else {
            return "'" + val_str + "'";
        }
    } else {
        return val->as_repr();
    }
}

// stats utility
void value_t::stats_t::mark_used(value & val, bool deep) {
    val->stats.used = true;
    if (deep) {
        if (is_val<value_array>(val)) {
            for (auto & item : val->val_arr) {
                mark_used(item, deep);
            }
        } else if (is_val<value_object>(val)) {
            for (auto & pair : val->val_obj) {
                mark_used(pair.first, deep);
                mark_used(pair.second, deep);
            }
        }
    }
}

} // namespace jinja
