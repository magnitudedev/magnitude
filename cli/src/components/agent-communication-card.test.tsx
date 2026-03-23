import { describe, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    info: '#88c0d0',
    secondary: '#81a1c1',
    foreground: '#eceff4',
    muted: '#9aa3b2',
  }),
}))

mock.module('../hooks/use-terminal-width', () => ({
  useTerminalWidth: () => 120,
}))

const { AgentCommunicationCard } = await import('./agent-communication-card')

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

describe('AgentCommunicationCard', () => {
  test('renders role emoji prefix before subagent id', () => {
    const html = renderToStaticMarkup(
      <AgentCommunicationCard
        message={{
          id: 'm1',
          type: 'agent_communication',
          direction: 'to_agent',
          agentId: 'agent-2',
          agentRole: 'planner',
          forkId: 'fork-2',
          content: 'hello',
          preview: 'hello',
          timestamp: Date.now(),
        }}
      />,
    )

    const text = htmlToText(html)
    expect(text).toContain('⚙ agent-2 → Main agent')
  })
})
