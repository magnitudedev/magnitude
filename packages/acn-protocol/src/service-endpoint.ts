export const MAGNITUDE_SERVICE_ORIGIN = "http://127.0.0.1:10100"
export const MAGNITUDE_INFERENCE_BASE_URL = `${MAGNITUDE_SERVICE_ORIGIN}/inference/`
export const MAGNITUDE_ANTHROPIC_BASE_URL =
  `${MAGNITUDE_SERVICE_ORIGIN}/inference/anthropic`
export const MAGNITUDE_CODEX_PROXY_BASE_URL =
  `${MAGNITUDE_SERVICE_ORIGIN}/inference/v1/proxies/codex`
export const MAGNITUDE_CLAUDE_CODE_PROXY_BASE_URL =
  `${MAGNITUDE_SERVICE_ORIGIN}/inference/anthropic/proxies/claude-code`
/**
 * Bearer credential the Claude Code connection writes when Claude Code has no Anthropic login. The
 * Claude Code proxy serves every model that carries it from a local model.
 */
export const MAGNITUDE_CLAUDE_CODE_PLACEHOLDER_CREDENTIAL = "magnitude-local"
