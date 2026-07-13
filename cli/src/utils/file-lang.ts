import { createElement, type ReactNode } from 'react'

import type { Span } from '../markdown/blocks'

export function isMarkdownFile(path: string): boolean {
  return path.toLowerCase().endsWith('.md')
}

export function renderCodeLines(line: Span[], idx: number, fallbackFg: string): ReactNode {
  return createElement(
    'text',
    { key: idx, style: { fg: fallbackFg } },
    ...line.map((span, i) => createElement('span', { key: i, fg: span.fg ?? fallbackFg }, span.text)),
  )
}
