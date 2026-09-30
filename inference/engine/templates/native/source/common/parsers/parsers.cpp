#include "parsers.h"

#include "templates-log.h"

void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

void foreach_parameter(const json & function, const std::function<void(const common_chat_schema_property &, const common_chat_schema_document_ptr &)> & fn) {
    auto                       params = common_chat_tool_parameters(function);
    auto                       doc    = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));
    const common_chat_schema * root   = doc->root.get();
    for (size_t hops = 0; root->kind() == common_chat_schema::KIND_REF && hops <= doc->refs.size(); hops++) {
        root = static_cast<const common_chat_schema_ref *>(root)->target;
    }
    const auto * object = dynamic_cast<const common_chat_schema_object *>(root);
    if (!object) {
        // Named arguments express one object's properties; under any other
        // argument schema they are written as none, which it may reject.
        templates_native::relax(function.at("name"), *doc->root, "type", common_chat_schema_relaxation::REASON_UNENFORCED);
        return;
    }
    for (const auto & prop : object->properties) {
        fn(prop, doc);
    }
}
