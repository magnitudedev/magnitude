/**
 * Atomizer — converts a raw model ID into a sequence of typed atoms.
 *
 * This is the preprocessing layer for the classifier. It does NOT strip
 * provider-specific noise (no capability suffixes, no quantization lists,
 * no Bedrock version handling, no Ollama tag processing). It only:
 *
 * 1. Strips a terminal model artifact extension and path prefix
 *    (last segment after `/`, `\\`, `@`, or non-decimal `.`)
 * 2. Lowercases
 * 3. Splits into typed atoms: literals, separators, decimal points
 *
 * The pattern matcher handles everything else — it searches for patterns
 * at any position in the atom stream, so extra atoms (size tokens, dates,
 * capability markers, version suffixes) are simply ignored.
 */

export type AtomType = "lit" | "sep" | "dot"

export interface Atom {
  readonly type: AtomType
  readonly value: string
}

function isAllDigits(value: string): boolean {
  return /^\d+$/.test(value)
}

/**
 * Strip known artifact extensions, then take the last segment after a path
 * separator, `@`, or `.` (the Bedrock provider separator). A `.` followed by
 * a digit is a decimal point, not a provider separator, so it's preserved.
 */
function stripPathPrefix(id: string): string {
  const withoutArtifactExtension = id.replace(/\.(?:gguf|ggml|bin)$/i, "")
  const lastSlash = Math.max(
    withoutArtifactExtension.lastIndexOf("/"),
    withoutArtifactExtension.lastIndexOf("\\"),
  )
  const lastAt = withoutArtifactExtension.lastIndexOf("@")
  const lastDot = withoutArtifactExtension.lastIndexOf(".")
  let index = Math.max(lastSlash, lastAt)
  if (lastDot !== -1 && lastDot < withoutArtifactExtension.length - 1) {
    const nextChar = withoutArtifactExtension[lastDot + 1]
    if (nextChar && /\d/.test(nextChar)) {
      // Decimal point followed by digit — not a provider separator.
    } else {
      index = Math.max(index, lastDot)
    }
  }
  return index === -1
    ? withoutArtifactExtension
    : withoutArtifactExtension.slice(index + 1)
}

/** Characters that act as segment separators. */
const SEPARATOR_CHARS = new Set(["-", "_", ":"])

/**
 * Split a normalized ID string into typed atoms.
 *
 * Within each segment (delimited by separators), alpha and digit runs are
 * split into separate literal atoms. A `.` or `p` between two digit runs
 * becomes a `dot` atom; otherwise `.` becomes a `sep` atom.
 */
function atomize(normalized: string): Atom[] {
  const raw: Atom[] = []
  let current = ""
  let currentIsDigit: boolean | null = null

  function flush(): void {
    if (current !== "") {
      raw.push({ type: "lit", value: current })
      current = ""
      currentIsDigit = null
    }
  }

  for (const char of normalized) {
    if (SEPARATOR_CHARS.has(char)) {
      flush()
      raw.push({ type: "sep", value: char })
      continue
    }

    const isDigit = /\d/.test(char)
    const isAlpha = /[a-z]/.test(char)

    if (isDigit || isAlpha) {
      if (currentIsDigit === null) {
        currentIsDigit = isDigit
        current = char
      } else if (currentIsDigit === isDigit) {
        current += char
      } else {
        flush()
        current = char
        currentIsDigit = isDigit
      }
    } else if (char === ".") {
      // Decide later whether this is a decimal point or a separator.
      flush()
      raw.push({ type: "lit", value: "." })
    } else if (char === "p") {
      // `p` between digits (e.g. `3p5`) is a decimal point.
      flush()
      raw.push({ type: "lit", value: "p" })
    } else {
      flush()
    }
  }
  flush()

  // Classify `.` and `p` atoms: between two digit literals → dot, else → sep.
  const result: Atom[] = []
  for (let i = 0; i < raw.length; i++) {
    const atom = raw[i]!
    if (atom.type === "lit" && (atom.value === "." || atom.value === "p")) {
      const prev = result[result.length - 1] ?? null
      const next = raw[i + 1] ?? null
      if (
        prev &&
        prev.type === "lit" &&
        isAllDigits(prev.value) &&
        next &&
        next.type === "lit" &&
        isAllDigits(next.value)
      ) {
        result.push({ type: "dot", value: atom.value })
      } else if (atom.value === ".") {
        result.push({ type: "sep", value: "." })
      } else {
        // `p` not between digits is a regular literal.
        result.push(atom)
      }
    } else {
      result.push(atom)
    }
  }

  // Merge consecutive separators.
  const merged: Atom[] = []
  for (const atom of result) {
    if (atom.type === "sep") {
      const last = merged[merged.length - 1]
      if (last && last.type === "sep") continue
    }
    merged.push(atom)
  }

  return merged
}

export function atomizeModelId(id: string): Atom[] {
  const stripped = stripPathPrefix(id)
  const lower = stripped.toLowerCase()
  return atomize(lower)
}

export { isAllDigits }
