import { beforeEach, describe, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

let rendererWidth = 120

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    info: '#88c0d0',
    secondary: '#81a1c1',
    foreground: '#eceff4',
    muted: '#9aa3b2',
  }),
}))

mock.module('../hooks/use-terminal-width', () => ({
  useTerminalWidth: () => rendererWidth,
}))

mock.module('@opentui/react', () => ({
  useRenderer: () => ({
    terminal: { width: rendererWidth },
    clearSelection() {},
  }),
  useTerminalDimensions: () => ({
    width: rendererWidth,
    height: 40,
  }),
}))

mock.module('../markdown/markdown-content', () => ({
  MarkdownContent: ({ content }: { content: string }) => <markdown-content data-content={content} />,
}))

const { AgentCommunicationCard, truncateContentLines } = await import('./agent-communication-card')

function htmlToText(html: string): string {
  return html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()
}

function makeMessage(content: string): any {
  return {
    id: 'm1',
    type: 'agent_communication',
    direction: 'to_agent',
    agentId: 'agent-2',
    agentRole: 'planner',
    forkId: 'fork-2',
    content,
    preview: content.slice(0, 120),
    timestamp: Date.now(),
  }
}

beforeEach(() => {
  rendererWidth = 120
})

describe('AgentCommunicationCard', () => {
  test('renders role and direction', () => {
    const html = renderToStaticMarkup(
      <AgentCommunicationCard message={makeMessage('hello')} />,
    )

    const text = htmlToText(html)
    expect(text).toContain('[planner] agent-2')
    expect(text).toContain('Lead')
  })

  test('does not show truncation indicator for content under 300 lines', () => {
    const content = 'line 1\nline 2\nline 3'
    const html = renderToStaticMarkup(
      <AgentCommunicationCard message={makeMessage(content)} />,
    )

    const text = htmlToText(html)
    expect(text).not.toContain('lines hidden')
    expect(text).not.toContain('Content capped at')
  })

  test('expand/collapse button is present for overflow content', () => {
    const lines = Array.from({ length: 310 }, (_, i) => `line ${i + 1}`)
    const content = lines.join('\n')
    const html = renderToStaticMarkup(
      <AgentCommunicationCard message={makeMessage(content)} />,
    )

    const text = htmlToText(html)
    expect(text).toContain('Expand')
  })

  test('no expand/collapse button for short content', () => {
    const html = renderToStaticMarkup(
      <AgentCommunicationCard message={makeMessage('short content')} />,
    )

    const text = htmlToText(html)
    expect(text).not.toContain('Expand')
    expect(text).not.toContain('Collapse')
  })
})

describe('truncateContentLines', () => {
  test('returns unchanged text when under the cap', () => {
    const result = truncateContentLines('a\nb\nc', 300)
    expect(result.text).toBe('a\nb\nc')
    expect(result.hiddenCount).toBe(0)
    expect(result.wasTruncated).toBe(false)
  })

  test('truncates to cap and reports hidden count', () => {
    const lines = Array.from({ length: 310 }, (_, i) => `line ${i + 1}`)
    const content = lines.join('\n')
    const result = truncateContentLines(content, 300)

    expect(result.text).toBe(lines.slice(0, 300).join('\n'))
    expect(result.hiddenCount).toBe(10)
    expect(result.wasTruncated).toBe(true)
  })

  test('reports correct hidden count for larger content', () => {
    const lines = Array.from({ length: 500 }, (_, i) => `line ${i + 1}`)
    const content = lines.join('\n')
    const result = truncateContentLines(content, 300)

    expect(result.hiddenCount).toBe(200)
    expect(result.wasTruncated).toBe(true)
  })

  test('handles empty content', () => {
    const result = truncateContentLines('', 300)
    expect(result.text).toBe('')
    expect(result.hiddenCount).toBe(0)
    expect(result.wasTruncated).toBe(false)
  })

  test('handles single line content', () => {
    const result = truncateContentLines('only one line', 300)
    expect(result.text).toBe('only one line')
    expect(result.hiddenCount).toBe(0)
    expect(result.wasTruncated).toBe(false)
  })
})
