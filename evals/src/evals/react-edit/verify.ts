/**
 * Verification — compare output file content against expected fixture.
 * Shared across all edit formats.
 */

export interface VerifyResult {
  passed: boolean
  error?: string
  diff?: string
  linesChanged: number
  charsChanged: number
}

function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
}

function normalizeBlankLines(text: string): string {
  return text.replace(/\n{3,}/g, '\n\n')
}

/**
 * If lines differ only in whitespace, restore the expected version.
 */
function restoreWhitespaceOnlyDiffs(expected: string, actual: string): string {
  const expectedLines = expected.split('\n')
  const actualLines = actual.split('\n')
  const max = Math.max(expectedLines.length, actualLines.length)
  const out: string[] = new Array(max)

  for (let i = 0; i < max; i++) {
    const e = expectedLines[i]
    const a = actualLines[i]
    if (e === undefined || a === undefined) {
      out[i] = a ?? ''
      continue
    }
    if (e !== a && e.replace(/\s+/g, '') === a.replace(/\s+/g, '')) {
      out[i] = e
    } else {
      out[i] = a
    }
  }

  return out.join('\n')
}

function createCompactDiff(expected: string, actual: string, contextLines = 3): string {
  const expLines = expected.split('\n')
  const actLines = actual.split('\n')
  const output: string[] = []
  const maxLen = Math.max(expLines.length, actLines.length)

  // Find differing line ranges
  const diffs: { start: number; end: number }[] = []
  let i = 0
  while (i < maxLen) {
    if (expLines[i] !== actLines[i]) {
      const start = i
      while (i < maxLen && expLines[i] !== actLines[i]) i++
      diffs.push({ start, end: i })
    } else {
      i++
    }
  }

  for (const { start, end } of diffs) {
    const ctxStart = Math.max(0, start - contextLines)
    const ctxEnd = Math.min(maxLen, end + contextLines)

    output.push(`@@ line ${start + 1} @@`)
    for (let j = ctxStart; j < ctxEnd; j++) {
      if (j >= start && j < end) {
        if (j < expLines.length) output.push(`-${expLines[j]}`)
        if (j < actLines.length) output.push(`+${actLines[j]}`)
      } else {
        output.push(` ${expLines[j] ?? actLines[j] ?? ''}`)
      }
    }
  }

  return output.join('\n')
}

function computeDiffStats(expected: string, actual: string): { linesChanged: number; charsChanged: number } {
  const expLines = expected.split('\n')
  const actLines = actual.split('\n')
  let linesChanged = 0
  let charsChanged = 0
  const maxLen = Math.max(expLines.length, actLines.length)

  for (let i = 0; i < maxLen; i++) {
    const e = expLines[i] ?? ''
    const a = actLines[i] ?? ''
    if (e !== a) {
      linesChanged++
      charsChanged += Math.abs(e.length - a.length) + Math.min(e.length, a.length)
    }
  }

  return { linesChanged, charsChanged }
}

export function verify(expected: string, actual: string): VerifyResult {
  const expNorm = normalizeBlankLines(normalizeLineEndings(expected))
  const actNorm = normalizeBlankLines(normalizeLineEndings(actual))

  // Exact match after normalization
  if (expNorm === actNorm) {
    return { passed: true, linesChanged: 0, charsChanged: 0 }
  }

  // Whitespace-tolerant match
  const actRestored = restoreWhitespaceOnlyDiffs(expNorm, actNorm)
  if (expNorm === actRestored) {
    return { passed: true, linesChanged: 0, charsChanged: 0 }
  }

  // Failed — produce diff
  const diff = createCompactDiff(expNorm, actNorm)
  const stats = computeDiffStats(expNorm, actNorm)

  return {
    passed: false,
    error: `Content mismatch (${stats.linesChanged} lines differ)`,
    diff,
    ...stats
  }
}
