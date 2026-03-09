import { BamlClientHttpError } from '@magnitudedev/llm-core'

/**
 * Detects context-limit errors from LLM providers.
 *
 * Based on empirical testing across all supported providers:
 * - Anthropic (direct + Bedrock): "prompt is too long: X tokens > Y maximum" (status 400 or 401)
 * - Google Gemini: "The input token count exceeds the maximum number of tokens allowed" (status 400)
 * - OpenRouter: "This endpoint's maximum context length is X tokens" (status 400)
 * - OpenAI: error code "context_length_exceeded" (status 400)
 *
 * Note: Status codes are unreliable (Bedrock returns 401 for context errors).
 * Detection is based on error message content only.
 */
export function isContextLimitError(error: unknown): boolean {
  if (!(error instanceof BamlClientHttpError)) return false

  const text = [
    error.message,
    error.detailed_message,
    error.raw_response
  ].filter(Boolean).join(' ').toLowerCase()

  return (
    text.includes('prompt is too long') ||              // Anthropic + Bedrock
    text.includes('token count exceeds the maximum') || // Google Gemini
    text.includes('maximum context length') ||          // OpenRouter / OpenAI-compat
    text.includes('context_length_exceeded')            // OpenAI (structured error code)
  )
}
