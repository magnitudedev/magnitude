#pragma once

#include "chat.h"
#include "templates.h"

namespace templates_native {

using text_spans = std::vector<std::pair<size_t, size_t>>;
class output_stream {
  public:
    output_stream(const common_chat_params & plan, uint64_t limit);
    templates_events feed(const uint8_t * bytes, uint64_t size);
    templates_events finish(uint32_t cause);

  private:
    struct event_value {
        uint32_t kind;
        uint32_t index;
        std::string text;
        std::string id;
    };
    struct call_cursor {
        std::string name;
        std::string id;
        common_peg_text argument_output;
        bool started = false;
        bool complete = false;
    };
    common_peg_arena parser;
    common_peg_parse_context context;
    common_chat_format format;
    uint64_t limit;
    uint64_t received = 0;
    bool terminal = false;
    bool has_explicit_ids = false;
    std::string utf8_pending;
    common_peg_text content_output;
    common_peg_text reasoning_output;
    std::string pending_reasoning;
    bool visible_reasoning = false;
    std::vector<call_cursor> calls;
    std::vector<event_value> values;
    std::vector<templates_event> events;

    void parse(bool natural, bool baseline = false);

    templates_events batch();
};

}
