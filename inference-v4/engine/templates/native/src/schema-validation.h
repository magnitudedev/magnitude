#pragma once
#include "json.h"

namespace templates_native {
// Validate only the explicitly qualified input subset; unknown constraints fail.
void validate_schema(const common_json & schema);
}
