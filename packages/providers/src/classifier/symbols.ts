/**
 * Symbol factory functions for the model family classifier DSL.
 *
 * Patterns are built from typed symbols. All symbols are constructed via
 * factory functions (no constants). The matcher interprets these symbols
 * against an atomized model ID.
 */

export type PatternSymbol =
  | LitSymbol
  | SepSymbol
  | DotSymbol
  | NumSymbol
  | VerSymbol
  | OptSymbol

interface LitSymbol {
  readonly _tag: "lit"
  readonly text: string
}

interface SepSymbol {
  readonly _tag: "sep"
}

interface DotSymbol {
  readonly _tag: "dot"
}

interface NumSymbol {
  readonly _tag: "num"
}

interface VerSymbol {
  readonly _tag: "ver"
}

interface OptSymbol {
  readonly _tag: "opt"
  readonly text: string
}

/** Exact literal text match (case-insensitive). */
export function lit(text: string): LitSymbol {
  return { _tag: "lit", text: text.toLowerCase() }
}

/** Matches a separator atom (`-`/`_`) or nothing. */
export function sep(): SepSymbol {
  return { _tag: "sep" }
}

/**
 * Matches a decimal-point atom (`.`, `p`), a separator atom (`-`/`_`),
 * or nothing. Used between major and minor version numbers.
 */
export function dot(): DotSymbol {
  return { _tag: "dot" }
}

/** Matches any all-digit literal atom. */
export function num(): NumSymbol {
  return { _tag: "num" }
}

/**
 * Matches any version number: a single all-digit literal, or
 * `digit + dot + digit` (e.g. `3.5`, `3p5` → `3`, `dot`, `5`).
 */
export function ver(): VerSymbol {
  return { _tag: "ver" }
}

/** Optional literal text match (case-insensitive). */
export function opt(text: string): OptSymbol {
  return { _tag: "opt", text: text.toLowerCase() }
}
