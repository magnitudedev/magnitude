export function createCompactDiff(expected: string, actual: string, contextLines = 3): string {
  const expLines = expected.split('\n')
  const actLines = actual.split('\n')
  const output: string[] = []
  const maxLen = Math.max(expLines.length, actLines.length)

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

export function computeDiffStats(expected: string, actual: string): { linesChanged: number; charsChanged: number } {
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

export function shouldUseDiff(linesChanged: number, totalLines: number): boolean {
  if (totalLines <= 0) return false
  return linesChanged / totalLines <= 0.3
}