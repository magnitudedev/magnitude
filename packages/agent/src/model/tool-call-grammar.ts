/**
 * Build a bounded-countdown BNF grammar that limits tool calls per response.
 *
 * Size is O(maxCalls + toolCount) — compact regardless of scale.
 * Uses a countdown chain: each `callN` nonterminal either terminates with a
 * single `<tool>` or continues to `<tool> <callN+1>`, bounded by `maxCalls`.
 *
 * @param toolNames - Names of available tools
 * @param maxCalls  - Maximum tool calls allowed per response
 * @param mode      - "auto" allows 0..N (includes empty alternative),
 *                    "required" requires 1..N (no empty alternative)
 */
export function buildMaxToolCallsGrammar(
  toolNames: readonly string[],
  maxCalls: number,
  mode: "auto" | "required",
): string {
  if (maxCalls < 1 || toolNames.length === 0) return ""

  const toolAlts = toolNames.map((n) => `"${n}"`).join(" | ")

  const lines: string[] = []

  // Root rule
  if (mode === "auto") {
    lines.push("<turn> ::= | <call1>")
  } else {
    lines.push("<turn> ::= <call1>")
  }

  // Countdown chain: callN can terminate or continue to callN+1
  for (let i = 1; i < maxCalls; i++) {
    lines.push(`<call${i}> ::= <tool> | <tool> <call${i + 1}>`)
  }

  // Final call: terminates only
  lines.push(`<call${maxCalls}> ::= <tool>`)

  // Shared tool alternatives
  lines.push(`<tool> ::= ${toolAlts}`)

  return lines.join("\n")
}
