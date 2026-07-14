import { type PatternSymbol } from "./symbols"
import { type Atom, isAllDigits } from "./atomizer"

export interface PatternEntry {
  readonly pattern: readonly PatternSymbol[]
  readonly priority: number
  readonly exclude?: readonly PatternSymbol[]
}

export interface ModelMetadataPattern {
  readonly architectures?: readonly string[]
  readonly tokenizerModels?: readonly string[]
  readonly tokenizerPres?: readonly string[]
}

export interface Family {
  readonly familyId: string
  readonly patterns: readonly PatternEntry[]
  /** Structured evidence used when a provider exposes model-file metadata. */
  readonly metadataPatterns?: readonly ModelMetadataPattern[]
}

function matchPattern(
  atoms: readonly Atom[],
  pattern: readonly PatternSymbol[],
  startIndex: number,
): boolean {
  let ai = startIndex
  let pi = 0

  while (pi < pattern.length) {
    const sym = pattern[pi]!

    switch (sym._tag) {
      case "lit": {
        if (ai >= atoms.length) return false
        const atom = atoms[ai]!
        if (atom.type === "lit" && atom.value === sym.text) {
          ai++
          pi++
        } else {
          return false
        }
        break
      }
      case "sep": {
        if (ai < atoms.length && atoms[ai]!.type === "sep") {
          ai++
        }
        pi++
        break
      }
      case "dot": {
        if (ai < atoms.length && atoms[ai]!.type === "dot") {
          ai++
        }
        pi++
        break
      }
      case "num": {
        if (ai >= atoms.length) return false
        const atom = atoms[ai]!
        if (atom.type === "lit" && isAllDigits(atom.value)) {
          ai++
          pi++
        } else {
          return false
        }
        break
      }
      case "ver": {
        if (ai >= atoms.length) return false
        const atom = atoms[ai]!
        if (atom.type === "lit" && isAllDigits(atom.value)) {
          if (
            ai + 2 < atoms.length &&
            atoms[ai + 1]!.type === "dot" &&
            atoms[ai + 2]!.type === "lit" &&
            isAllDigits(atoms[ai + 2]!.value)
          ) {
            ai += 3
          } else {
            ai += 1
          }
          pi++
        } else {
          return false
        }
        break
      }
      case "opt": {
        if (ai < atoms.length && atoms[ai]!.type === "lit" && atoms[ai]!.value === sym.text) {
          ai++
        }
        pi++
        break
      }
    }
  }

  return true
}

function patternMatches(
  atoms: readonly Atom[],
  pattern: readonly PatternSymbol[],
): boolean {
  for (let start = 0; start < atoms.length; start++) {
    if (matchPattern(atoms, pattern, start)) return true
  }
  return pattern.length === 0
}

export interface MatchResult {
  readonly familyId: string
  readonly priority: number
}

export function match(
  atoms: readonly Atom[],
  families: readonly Family[],
): MatchResult | null {
  let best: MatchResult | null = null

  for (const family of families) {
    for (const entry of family.patterns) {
      if (entry.exclude && patternMatches(atoms, entry.exclude)) continue
      if (patternMatches(atoms, entry.pattern)) {
        if (!best || entry.priority > best.priority) {
          best = { familyId: family.familyId, priority: entry.priority }
        }
      }
    }
  }

  return best
}
