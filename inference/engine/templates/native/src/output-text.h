#pragma once

#include "json.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Immutable output fragments. Source leaves store offsets, never pointers into
// a growing input buffer. Concatenation shares completed output; escaping remains
// a leaf operation so streaming need encode only newly published source bytes.
class common_peg_text {
    struct node {
        std::shared_ptr<const node> left, right;
        std::string literal;
        size_t start = 0, end = 0;
        unsigned escaping = 0;
        bool source = false;
        int height = 1;
    };
    using ptr = std::shared_ptr<const node>;
    ptr root;
    explicit common_peg_text(ptr root) : root(std::move(root)) {}
    static int height(const ptr & p) { return p ? p->height : 0; }
    static ptr branch(ptr a, ptr b) {
        if (!a) { return b; }
        if (!b) { return a; }
        auto p = std::make_shared<node>();
        p->height = 1 + std::max(height(a), height(b));
        p->left = std::move(a); p->right = std::move(b);
        return p;
    }
    static ptr join(ptr a, ptr b) {
        if (height(a) > height(b) + 1) {
            auto right = join(a->right, b);
            if (height(right) > height(a->left) + 1) {
                return branch(branch(a->left, right->left), right->right);
            }
            return branch(a->left, right);
        }
        if (height(b) > height(a) + 1) {
            auto left = join(a, b->left);
            if (height(left) > height(b->right) + 1) {
                return branch(left->left, branch(left->right, b->right));
            }
            return branch(left, b->right);
        }
        return branch(a, b);
    }
    static std::string escape(std::string value, unsigned depth) {
        while (depth--) {
            auto encoded = common_json(value).dump();
            value = encoded.substr(1, encoded.size() - 2);
        }
        return value;
    }
    static ptr escaped(const ptr & p) {
        if (!p) { return {}; }
        if (p->left) { return branch(escaped(p->left), escaped(p->right)); }
        auto copy = std::make_shared<node>(*p);
        if (copy->source) { ++copy->escaping; }
        else { copy->literal = escape(copy->literal, 1); }
        return copy;
    }
    static void expand(std::vector<ptr> & stack) {
        auto p = stack.back(); stack.pop_back();
        stack.push_back(p->right); stack.push_back(p->left);
    }
  public:
    common_peg_text() = default;
    common_peg_text(const char * value) : common_peg_text(std::string(value)) {}
    common_peg_text(std::string value) {
        if (!value.empty()) { auto p = std::make_shared<node>(); p->literal = std::move(value); root = p; }
    }
    static common_peg_text source(size_t start, size_t end) {
        if (start == end) { return {}; }
        auto p = std::make_shared<node>(); p->source = true; p->start = start; p->end = end;
        return common_peg_text(p);
    }
    bool empty() const { return !root; }
    common_peg_text escaped() const { return common_peg_text(escaped(root)); }
    common_peg_text & operator+=(const common_peg_text & other) { root = join(root, other.root); return *this; }
    common_peg_text & operator+=(char c) { return *this += common_peg_text(std::string(1, c)); }
    friend common_peg_text operator+(common_peg_text a, const common_peg_text & b) { return a += b; }

    // Verify the old output is still a prefix, skipping shared subtrees in O(1).
    // A changed leaf can extend its immutable source range or literal prefix.
    // Returns only the newly emitted bytes, using the exact upstream JSON encoder.
    std::string suffix(const common_peg_text & previous, std::string_view input) const {
        std::vector<ptr> old, current;
        if (previous.root) { old.push_back(previous.root); }
        if (root) { current.push_back(root); }
        size_t old_skip = 0, new_skip = 0;
        std::string result;
        while (!current.empty()) {
            auto p = current.back();
            if (!old.empty() && !old_skip && !new_skip && old.back() == p) {
                old.pop_back(); current.pop_back(); continue;
            }
            if (p->left && (old.empty() || !old.back()->left || p->height >= old.back()->height)) {
                expand(current); continue;
            }
            if (!old.empty() && old.back()->left) { expand(old); continue; }
            const size_t length = p->source ? p->end - p->start : p->literal.size();
            if (old.empty()) {
                auto text = p->source ? std::string(input.substr(p->start + new_skip, length - new_skip))
                                      : p->literal.substr(new_skip);
                result += escape(std::move(text), p->source ? p->escaping : 0);
                current.pop_back(); new_skip = 0; continue;
            }
            auto q = old.back();
            const size_t old_length = q->source ? q->end - q->start : q->literal.size();
            const auto count = std::min(old_length - old_skip, length - new_skip);
            if (p->source != q->source || p->escaping != q->escaping ||
                (p->source ? p->start + new_skip != q->start + old_skip
                           : p->literal.compare(new_skip, count, q->literal, old_skip, count) != 0)) {
                // Literal and source representations can meet at a provisional
                // delimiter. Compare their actual bytes in that uncommon case.
                auto a = p->source ? escape(std::string(input.substr(p->start + new_skip, count)), p->escaping)
                                   : p->literal.substr(new_skip, count);
                auto b = q->source ? escape(std::string(input.substr(q->start + old_skip, count)), q->escaping)
                                   : q->literal.substr(old_skip, count);
                if (a != b) { throw std::runtime_error("Output parser would change published text"); }
            }
            new_skip += count; old_skip += count;
            if (new_skip == length) { current.pop_back(); new_skip = 0; }
            if (old_skip == old_length) { old.pop_back(); old_skip = 0; }
        }
        if (!old.empty()) { throw std::runtime_error("Output parser would retract published text"); }
        return result;
    }
    std::string str(std::string_view input) const { return suffix({}, input); }
};
