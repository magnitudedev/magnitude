import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { TaskListItem } from './types'

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

const makeTask = (overrides: Partial<TaskListItem> = {}): TaskListItem => ({
  taskId: 't-1',
  title: 'Investigate timer mismatch',
  type: 'implement',
  status: 'pending',
  depth: 0,
  parentId: null,
  createdAt: 1_000,
  updatedAt: 11_000,
  completedAt: null,
  assignee: { kind: 'lead' },
  workerForkId: null,
  ...overrides,
})

test('returns null when there are no tasks', () => {
  const html = render(<TaskList tasks={[]} pushForkOverlay={noop} />)
  expect(html).toBe('')
})

test('renders task header summary from task statuses', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 't-1', status: 'completed', completedAt: 10_000 }),
        makeTask({ taskId: 't-2', status: 'working' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#ffffff" attributes="1">Task</text>')
  expect(html).toContain('<text style="fg:#888888"> (1 completed, 1 active)</text>')
})

test('renders status glyphs', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 't-pending', status: 'pending' }),
        makeTask({ taskId: 't-working', status: 'working' }),
        makeTask({ taskId: 't-completed', status: 'completed', completedAt: 10_000 }),
      ]}
      pushForkOverlay={noop}
    />,
  )
  const text = htmlToText(html)
  expect(text).toContain('○')
  expect(text).toContain('◉')
  expect(text).toContain('✓')
})

test('renders type label and depth indentation', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 'root', title: 'Root', depth: 0 }),
        makeTask({ taskId: 'child', title: 'Child', depth: 1, parentId: 'root' }),
      ]}
      pushForkOverlay={noop}
    />,
  )
  const text = htmlToText(html)
  expect(text).toContain('[implement] Root')
  expect(text).toContain('└─ [implement] Child')
})

test('collapsed mode renders only the latest six tasks', () => {
  const tasks = Array.from({ length: 8 }, (_, index) =>
    makeTask({ taskId: `t${index + 1}`, title: `Task ${index + 1}` }),
  )

  const html = render(<TaskList tasks={tasks} pushForkOverlay={noop} />)
  const text = htmlToText(html)

  expect(text).not.toContain('Task 1')
  expect(text).not.toContain('Task 2')
  expect(text).toContain('Task 3')
  expect(text).toContain('Task 8')
})

test('renders lead assignee label for unassigned/lead tasks', () => {
  const html = render(
    <TaskList
      tasks={[makeTask({ assignee: { kind: 'lead' }, workerForkId: null })]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('lead')
})
