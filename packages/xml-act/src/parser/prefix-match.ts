/**
 * Shared prefix-match utility for context-specific tag matching.
 *
 * Instead of generic [a-zA-Z0-9_-] tag-name accumulation, this module
 * provides incremental prefix matching against a known set of valid
 * tag candidates. If the input diverges from all candidates, it
 * immediately reports NoMatch so the caller can flush literal text.
 */

export interface PrefixMatchState {
  readonly candidates: readonly string[]
  readonly matched: string
  readonly raw: string
}

export type PrefixAdvance =
  | { readonly _tag: 'Continue'; readonly candidates: readonly string[]; readonly matched: string; readonly raw: string }
  | { readonly _tag: 'Matched'; readonly tagName: string; readonly raw: string; readonly delimiter: string }
  | { readonly _tag: 'NoMatch'; readonly literal: string }

/**
 * Advance a prefix match by one character.
 *
 * - If `ch` matches the next expected char of at least one candidate, Continue.
 * - If `ch` is a tag delimiter (>, /, whitespace) and at least one candidate
 *   exactly equals `matched`, Matched.
 * - Otherwise, NoMatch — flush `raw + ch` as literal.
 */
export function advancePrefixMatch(state: PrefixMatchState, ch: string): PrefixAdvance {
  const raw = state.raw + ch
  const pos = state.matched.length

  // Try to extend the match: filter candidates whose char at `pos` equals `ch`
  const remaining = state.candidates.filter(c => pos < c.length && c[pos] === ch)
  if (remaining.length > 0) {
    return { _tag: 'Continue', candidates: remaining, matched: state.matched + ch, raw }
  }

  // No candidate continues with this char.
  // Check if this char is a delimiter and we have an exact match.
  if (ch === '>' || ch === '/' || /\s/.test(ch)) {
    const exact = state.candidates.find(c => c === state.matched)
    if (exact !== undefined) {
      return { _tag: 'Matched', tagName: exact, raw, delimiter: ch }
    }
  }

  // No match at all
  return { _tag: 'NoMatch', literal: raw }
}

/**
 * Filter candidates to those starting with a given character.
 * Used by pending-< dispatch to decide whether to enter prefix matching at all.
 */
export function candidatesStartingWith(candidates: readonly string[], ch: string): string[] {
  return candidates.filter(c => c.length > 0 && c[0] === ch)
}