import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: '#888888',
    secondary: '#5e81ac',
    warning: '#ebcb8b',
    success: '#a3be8c',
    error: '#bf616a',
    border: '#4c566a',
    terminalBg: '#2e3440',
    primary: '#88c0d0',
  }),
}))

const { ThinkBlock } = await import('./think-block')

const noop = () => {}

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

function render(node: React.ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

test('ThinkBlock renders subagent started/finished/killed rows with structured fields', () => {
  const html = render(
    <ThinkBlock
      block={{
        id: 'tb-1',
        type: 'think_block',
        status: 'completed',
        timestamp: 1000,
        completedAt: 9000,
        steps: [
          {
            id: 's1',
            type: 'subagent_started',
            subagentType: 'builder',
            subagentId: 'agent-7',
            title: 'Investigate flaky test',
            resumed: false,
          },
          {
            id: 's2',
            type: 'subagent_finished',
            subagentId: 'agent-7',
            cumulativeTotalTimeMs: 125000,
            cumulativeTotalToolsUsed: 3,
            resumed: false,
          },
        ],
      }}
      isCollapsed={false}
      onToggle={noop}
    />
  )

  const text = htmlToText(html)
  expect(text).toContain('▶ Subagent started: agent-7 — Investigate flaky test')
  expect(text).toContain('✓ Subagent finished: agent-7 · 2m 5s · 3 tools')
})

test('ThinkBlock includes resumed marker for subagent lifecycle rows', () => {
  const html = render(
    <ThinkBlock
      block={{
        id: 'tb-2',
        type: 'think_block',
        status: 'completed',
        timestamp: 1000,
        completedAt: 9000,
        steps: [
          {
            id: 's1',
            type: 'subagent_started',
            subagentType: 'researcher',
            subagentId: 'agent-3',
            title: 'Trace root cause',
            resumed: true,
          },
          {
            id: 's2',
            type: 'subagent_finished',
            subagentId: 'agent-3',
            cumulativeTotalTimeMs: 60000,
            cumulativeTotalToolsUsed: 1,
            resumed: true,
          },
        ],
      }}
      isCollapsed={false}
      onToggle={noop}
    />
  )

  const text = htmlToText(html)
  expect(text).toContain('▶ Subagent started: agent-3 (resumed) — Trace root cause')
  expect(text).toContain('✓ Subagent finished: agent-3 (resumed) · ↺ 1m · 1 tool')
})

test('ThinkBlock completed summary includes singular subagent lifecycle counts', () => {
  const html = render(
    <ThinkBlock
      block={{
        id: 'tb-3',
        type: 'think_block',
        status: 'completed',
        timestamp: 1000,
        completedAt: 9000,
        steps: [
          {
            id: 's1',
            type: 'subagent_started',
            subagentType: 'builder',
            subagentId: 'agent-1',
            title: 'Do thing',
            resumed: false,
          },
          {
            id: 's2',
            type: 'subagent_finished',
            subagentId: 'agent-1',
            cumulativeTotalTimeMs: 1000,
            cumulativeTotalToolsUsed: 1,
            resumed: false,
          },
        ],
      }}
      isCollapsed
      onToggle={noop}
    />
  )

  const text = htmlToText(html)
  expect(text).toContain('Completed in 8s (1 subagent started, 1 subagent finished) · Show')
})

test('ThinkBlock completed summary includes plural subagent lifecycle counts', () => {
  const html = render(
    <ThinkBlock
      block={{
        id: 'tb-4',
        type: 'think_block',
        status: 'completed',
        timestamp: 1000,
        completedAt: 9000,
        steps: [
          {
            id: 's1',
            type: 'subagent_started',
            subagentType: 'builder',
            subagentId: 'agent-1',
            title: 'Do thing',
            resumed: false,
          },
          {
            id: 's2',
            type: 'subagent_started',
            subagentType: 'researcher',
            subagentId: 'agent-2',
            title: 'Do another thing',
            resumed: false,
          },
          {
            id: 's3',
            type: 'subagent_finished',
            subagentId: 'agent-1',
            cumulativeTotalTimeMs: 1000,
            cumulativeTotalToolsUsed: 1,
            resumed: false,
          },
          {
            id: 's4',
            type: 'subagent_finished',
            subagentId: 'agent-2',
            cumulativeTotalTimeMs: 2000,
            cumulativeTotalToolsUsed: 2,
            resumed: false,
          },
        ],
      }}
      isCollapsed
      onToggle={noop}
    />
  )

  const text = htmlToText(html)
  expect(text).toContain('Completed in 8s (2 subagents started, 2 subagents finished) · Show')
})

test('ThinkBlock summary includes killed subagent counts from both kill sources', () => {
  const now = Date.now()
  const markup = render(
    <ThinkBlock
      block={{
        id: 't5',
        type: 'think',
        timestamp: now,
        status: 'completed',
        completedAt: now + 8000,
        steps: [
          {
            id: 's1',
            type: 'subagent_started',
            subagentId: 'researcher',
            title: 'gather evidence',
            resumed: false,
            timestamp: now + 1000,
            label: '',
          },
          {
            id: 's2',
            type: 'subagent_killed',
            subagentId: 'researcher',
            title: 'gather evidence',
            timestamp: now + 2000,
            label: '',
          },
          {
            id: 's3',
            type: 'subagent_user_killed',
            subagentId: 'builder',
            title: 'fix tests',
            timestamp: now + 3000,
            label: '',
          },
        ],
      }}
      isCollapsed={true}
      onToggle={() => {}}
    />,
  )

  const text = htmlToText(markup)
  expect(text).toContain('Completed in 8s (1 subagent started, 2 subagents killed) · Show')
})

test('ThinkBlock renders user-killed subagent row with dedicated text', () => {
  const now = Date.now()
  const markup = render(
    <ThinkBlock
      block={{
        id: 't-user-killed',
        type: 'think',
        timestamp: now,
        status: 'completed',
        completedAt: now + 1000,
        steps: [
          {
            id: 's1',
            type: 'subagent_user_killed',
            subagentId: 'researcher',
            title: 'gather evidence',
            timestamp: now + 500,
            label: '',
          },
        ],
      }}
      isCollapsed={false}
      onToggle={() => {}}
    />,
  )

  const text = htmlToText(markup)
  expect(text).toContain('■ Subagent killed by user: researcher - gather evidence')
})

test('ThinkBlock applies spacing around consecutive subagent lifecycle rows, not between each row', () => {
  const html = render(
    <ThinkBlock
      block={{
        id: 'tb-5',
        type: 'think_block',
        status: 'completed',
        timestamp: 1000,
        completedAt: 9000,
        steps: [
          {
            id: 's1',
            type: 'subagent_started',
            subagentType: 'builder',
            subagentId: 'agent-1',
            title: 'First',
            resumed: false,
          },
          {
            id: 's2',
            type: 'subagent_started',
            subagentType: 'researcher',
            subagentId: 'agent-2',
            title: 'Second',
            resumed: false,
          },
          {
            id: 's3',
            type: 'subagent_finished',
            subagentId: 'agent-1',
            cumulativeTotalTimeMs: 1000,
            cumulativeTotalToolsUsed: 1,
            resumed: false,
          },
          {
            id: 's4',
            type: 'subagent_finished',
            subagentId: 'agent-2',
            cumulativeTotalTimeMs: 2000,
            cumulativeTotalToolsUsed: 2,
            resumed: false,
          },
        ],
      }}
      isCollapsed={false}
      onToggle={noop}
    />
  )

  expect((html.match(/margin-top:1/g) ?? []).length).toBe(0)
})
