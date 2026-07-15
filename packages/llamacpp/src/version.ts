/** GitHub release repo. */
export const LLAMACPP_RELEASE_REPO = "ggml-org/llama.cpp"

/**
 * Minimum build number we accept.
 * Below this, we refuse to use the binary and prompt for update.
 * b8868 is the first release with --fit-print, the newest capability we require.
 */
export const MINIMUM_LLMACPP_VERSION = 8868

/**
 * Build number we download for fresh installs.
 * Can lag ahead of minimum — decouples "will we refuse this" from "what do we ship".
 */
export const RECOMMENDED_LLMACPP_VERSION = 10011

/**
 * Parse version from `llama-server --version` output.
 * Output format: `version: 10011 (bf2c86ddc)`
 */
export function parseVersionNumber(output: string): number {
  const match = output.match(/version:\s*(\d+)/)
  if (!match) throw new Error(`Could not parse llama-server version from: ${output}`)
  return parseInt(match[1], 10)
}

/** Convert a GitHub tag (e.g. "b10011") to a build number. */
export function tagToBuildNumber(tag: string): number {
  return parseInt(tag.replace(/^b/, ""), 10)
}

/** Convert a build number to a GitHub tag. */
export function buildNumberToTag(build: number): string {
  return `b${build}`
}

/** Check if a build number meets the minimum. */
export function meetsMinimum(buildNumber: number): boolean {
  return buildNumber >= MINIMUM_LLMACPP_VERSION
}
