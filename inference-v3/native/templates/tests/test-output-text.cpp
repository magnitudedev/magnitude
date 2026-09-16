#include "output-text.h"
#include <cassert>
#include <iostream>

static std::string escape(const std::string & text) {
    auto value = common_json(text).dump();
    return value.substr(1, value.size() - 2);
}

int main() {
    std::string input;
    common_peg_text previous, current;
    std::string emitted;
    for (size_t i = 0; i < 10000; ++i) {
        auto start = input.size();
        input += i % 2 ? "a\n\"\\😀" : "β\t";
        current += common_peg_text::source(start, input.size()).escaped();
        auto delta = current.suffix(previous, input);
        assert(delta == escape(input.substr(start)));
        emitted += delta;
        previous = current;
    }
    assert(emitted == escape(input));
    assert(current.str(input) == emitted);

    // An open source leaf grows; old source pointers must not survive reallocations.
    previous = {};
    for (size_t i = 0; i <= input.size(); ++i) {
        // Only split at complete UTF-8 boundaries, as the native stream does.
        if (i < input.size() && (static_cast<unsigned char>(input[i]) & 0xc0) == 0x80) { continue; }
        current = common_peg_text("{") + common_peg_text::source(0, i).escaped();
        current.suffix(previous, input);
        previous = current;
    }
    auto completed = current + "}";
    assert(completed.suffix(current, input) == "}");
    bool rejected = false;
    try { current.suffix(completed, input); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { (common_peg_text("[") + common_peg_text::source(0, input.size()).escaped()).suffix(current, input); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    assert(common_peg_text::source(0, input.size()).escaped().escaped().str(input) == escape(escape(input)));
    std::cout << "Shared output fragments preserve prefixes, escaping, and source relocation\n";
}
