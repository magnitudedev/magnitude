#pragma once

#include <string>
#include <vector>

namespace templates_native {
// Drained by the ABI at operation boundaries; no prompt text is logged implicitly.
extern thread_local std::vector<std::string> diagnostics;
void diagnostic(const char * format, ...);
}
#define LOG_DBG(...) do {} while (false)
#define LOG_INF(...) do {} while (false)
#define LOG_WRN(...) templates_native::diagnostic(__VA_ARGS__)
#define LOG_ERR(...) templates_native::diagnostic(__VA_ARGS__)
