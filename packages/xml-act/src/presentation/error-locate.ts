function getLines(responseText: string): string[] {
  return responseText.split('\n')
}

function countLineForIndex(responseText: string, index: number): number {
  let line = 1
  for (let i = 0; i < index; i += 1) {
    if (responseText[i] === '\n') line += 1
  }
  return line
}

function formatLine(lineNumber: number, content: string): string {
  return `${lineNumber}|${content}`
}

function sliceFormattedLines(lines: string[], startLine: number, endLine: number): string[] {
  const result: string[] = []

  for (let lineNumber = startLine; lineNumber <= endLine; lineNumber += 1) {
    const content = lines[lineNumber - 1]
    if (content === undefined) continue
    result.push(formatLine(lineNumber, content))
  }

  return result
}

export function findErrorLine(responseText: string, anchor: string): number | null {
  if (anchor.length === 0 || responseText.length === 0) return null

  const index = responseText.indexOf(anchor)
  if (index === -1) return null

  return countLineForIndex(responseText, index)
}

export function buildSnippet(
  responseText: string,
  errorLine: number,
  strategy: 'point' | 'block',
  blockStartLine?: number,
): string {
  if (responseText === '') return ''

  const lines = getLines(responseText)

  const clampedErrorLine = Math.min(Math.max(errorLine, 1), lines.length)

  if (strategy === 'point') {
    const startLine = Math.max(1, clampedErrorLine - 1)
    const endLine = Math.min(lines.length, clampedErrorLine + 1)
    return sliceFormattedLines(lines, startLine, endLine).join('\n')
  }

  const startLine = Math.min(Math.max(blockStartLine ?? clampedErrorLine, 1), clampedErrorLine)
  const endLine = Math.min(lines.length, clampedErrorLine + 1)
  const totalLines = endLine - startLine + 1

  if (totalLines <= 10) {
    return sliceFormattedLines(lines, startLine, endLine).join('\n')
  }

  const first = sliceFormattedLines(lines, startLine, startLine + 1)
  const lastStartLine = Math.max(startLine + 2, endLine - 2)
  const last = sliceFormattedLines(lines, lastStartLine, endLine)

  return [...first, '...', ...last].join('\n')
}
