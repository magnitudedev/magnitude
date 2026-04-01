import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { TaskItem } from './types'

mock.module('../../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: '#ffffff',
    muted: '#888888',
    success: '#00ff00',
    border: '#444444',
  }),
}))

mock.module('../../hooks/use-terminal-width', () => ({
  useTerminalWidth: () => 120,
}))

const { TaskList } = await import('./task-list')

const noop = () => {}

const htmlToText = (html: string): string =>
  html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

function render(node: ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

const makeTask = (overrides: Partial<TaskItem> = {}): TaskItem => ({
  forkId: 'fork-1',
  agentId: 'builder-1',
  role: 'builder',
  name: 'Investigate timer mismatch',
  phase: 'idle',
  activeSince: 1_000,
  completedAt: 11_000,
  accumulatedActiveMs: 65_000,
  resumeCount: 0,
  statusLine: 'Completed',
  toolSummaryLine: '',
  toolCount: 0,
  ...overrides,
})

test('returns null when there are no tasks', () => {
  const html = render(<TaskList tasks={[]} pushForkOverlay={noop} />)
  expect(html).toBe('')
})

test('uses rounded full border with neutral theme border color and transparent background', () => {
  const html = render(
    <TaskList
      tasks={[makeTask()]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('border-style:single')
  expect(html).toContain('border:left,right,top,bottom')
  expect(html).toContain('border-color:#64748b')
  expect(html).toContain('background-color:transparent')
})

test('renders task header with white/bold label and muted non-bold summary', () => {
  const html = render(
    <TaskList
      tasks={[makeTask(), makeTask({ forkId: 'fork-2', phase: 'active' })]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#ffffff" attributes="1">Task</text>')
  expect(html).toContain('<text style="fg:#888888"> (1 completed, 1 active)</text>')
})

test('computes summary counts using idle as completed and active as active', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ forkId: 'fork-1', phase: 'idle' }),
        makeTask({ forkId: 'fork-2', phase: 'idle' }),
        makeTask({ forkId: 'fork-3', phase: 'active' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('(2 completed, 1 active)')
})

test('renders assigned-to header in white/bold', () => {
  const html = render(
    <TaskList
      tasks={[makeTask()]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#ffffff" attributes="1">Assigned To</text>')
})

test('maintains expected usable-width split at terminal width 120', () => {
  const html = render(
    <TaskList
      tasks={[makeTask()]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('width:64px')
  expect(html).toContain('width:50px')
})

test('renders collapsed view by default and supports expand/collapse control labels', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ forkId: 'f1', name: 'Task 1' }),
        makeTask({ forkId: 'f2', name: 'Task 2' }),
        makeTask({ forkId: 'f3', name: 'Task 3' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Expand all ▲')
  expect(text).not.toContain('Collapse all ▼')
})

test('collapsed mode renders only the latest six tasks', () => {
  const tasks = Array.from({ length: 8 }, (_, index) =>
    makeTask({ forkId: `f${index + 1}`, name: `Task ${index + 1}` }),
  )

  const html = render(<TaskList tasks={tasks} pushForkOverlay={noop} />)
  const text = htmlToText(html)

  expect(text).not.toContain('Task 1')
  expect(text).not.toContain('Task 2')
  expect(text).toContain('Task 3')
  expect(text).toContain('Task 8')
})

test('completed task timer uses accumulatedActiveMs directly (no final stint double-count)', () => {
  const html = render(
    <TaskList
      tasks={[makeTask()]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('1:05')
  expect(text).not.toContain('1:15')
})
