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

let measuredWidth: number | null = null

mock.module('../../hooks/use-chat-width', () => ({
  useBoxWidth: () => ({ ref: { current: null }, onSizeChange: () => {}, width: measuredWidth }),
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
  measuredWidth = null
  const html = render(<TaskList tasks={[]} pushForkOverlay={noop} />)
  expect(html).toBe('')
})

test('renders task header summary from completed vs not completed statuses', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 't-1', status: 'completed', completedAt: 10_000 }),
        makeTask({ taskId: 't-2', status: 'pending' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#ffffff" attributes="1">Task</text>')
  expect(html).toContain('<text style="fg:#888888"> (1 completed, 1 active)</text>')
})

test('renders task status glyphs as only completed and not completed', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 't-completed', status: 'completed', completedAt: 10_000 }),
        makeTask({ taskId: 't-working', status: 'working' }),
        makeTask({ taskId: 't-pending', status: 'pending' }),
      ]}
      pushForkOverlay={noop}
    />,
  )
  const text = htmlToText(html)
  expect(text).toContain('✓')
  expect(text).toContain('○')
})

test('renders no tree connectors in task rows', () => {
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
  expect(text).toContain('○ Root')
  expect(text).toContain('○ Child')
  expect(text).not.toContain('└─')
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

test('default header shows Task (X completed, Y active) using not-completed active semantics', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 'root', title: 'Root', depth: 0, status: 'pending' }),
        makeTask({ taskId: 'child-done', title: 'Done', depth: 1, parentId: 'root', status: 'completed', completedAt: 10_000 }),
        makeTask({ taskId: 'child-working', title: 'Working', depth: 1, parentId: 'root', status: 'working' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#ffffff" attributes="1">Task</text>')
  expect(html).toContain('<text style="fg:#888888"> (1 completed, 2 active)</text>')
})

test('renders worker assignee with worker status prefix and timer segment', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: { kind: 'worker', agentId: 'builder-abc123', workerType: 'builder' },
          workerForkId: 'fork-abc123',
          workerExecution: {
            state: 'idle',
            activeSince: null,
            accumulatedActiveMs: 83_000,
            completedAt: 83_000,
            resumeCount: 0,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('◌ [builder] builder-abc123 · 1:23')
  expect(text).not.toContain('(resumed)')
  expect(text).not.toContain('↺')
})

test('fileViewerOpen hides assignee column and expand/collapse controls', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-worker',
          assignee: { kind: 'worker', agentId: 'builder-abc123', workerType: 'builder' },
          workerForkId: 'fork-abc123',
        }),
      ]}
      pushForkOverlay={noop}
      fileViewerOpen
    />,
  )

  const text = htmlToText(html)
  expect(text).not.toContain('Assigned To')
  expect(text).not.toContain('Expand all')
  expect(text).not.toContain('Collapse all')
  expect(text).not.toContain('⚒ [builder] builder-abc123')
})

test('fileViewerOpen uses measured width to truncate task names earlier', () => {
  measuredWidth = 24
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          title: 'This is a very long task title that should truncate in narrow single-column mode',
        }),
      ]}
      pushForkOverlay={noop}
      fileViewerOpen
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('…')
  measuredWidth = null
})

test('sticky root header shows correct subtree progress counts', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 'root-a', title: 'Root A', depth: 0, status: 'pending' }),
        makeTask({ taskId: 'child-a1', title: 'Child A1', depth: 1, parentId: 'root-a', status: 'completed', completedAt: 10_000 }),
        makeTask({ taskId: 'child-a2', title: 'Child A2', depth: 1, parentId: 'root-a', status: 'working' }),
        makeTask({ taskId: 'child-a3', title: 'Child A3', depth: 1, parentId: 'root-a', status: 'pending' }),
        makeTask({ taskId: 'child-a4', title: 'Child A4', depth: 1, parentId: 'root-a', status: 'completed', completedAt: 11_000 }),
        makeTask({ taskId: 'child-a5', title: 'Child A5', depth: 1, parentId: 'root-a', status: 'pending' }),
        makeTask({ taskId: 'root-b', title: 'Root B', depth: 0, status: 'pending' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('Root A')
  expect(html).toContain('(2 completed, 4 active)')
})

test('renders resumed worker layout with glyph only', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: { kind: 'worker', agentId: 'planner-1', workerType: 'planner' },
          workerForkId: 'fork-planner-1',
          workerExecution: {
            state: 'working',
            activeSince: null,
            accumulatedActiveMs: 83_000,
            completedAt: null,
            resumeCount: 1,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('◉ [planner] planner-1 · ↺ 1:23')
  expect(text).not.toContain('(resumed)')
})

test('completed task keeps idle worker rendering', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          status: 'completed',
          completedAt: 99_000,
          assignee: { kind: 'worker', agentId: 'builder-idle', workerType: 'builder' },
          workerForkId: 'fork-builder-idle',
          workerExecution: {
            state: 'idle',
            activeSince: null,
            accumulatedActiveMs: 10_000,
            completedAt: 10_000,
            resumeCount: 0,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('✓')
  expect(text).toContain('◌ [builder] builder-idle · 0:10')
})

test('completed task text remains muted gray while checkmark stays green', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          title: 'Completed task title',
          status: 'completed',
          completedAt: 99_000,
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#00ff00">✓ </text>')
  expect(html).toContain('style="fg:#888888">Completed task title</text>')
  expect(html).not.toContain('style="fg:#00ff00">Completed task title</text>')
})

test('renders killed worker with red kill icon glyph', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: { kind: 'worker', agentId: 'builder-killed', workerType: 'builder' },
          workerForkId: 'fork-builder-killed',
          workerExecution: {
            state: 'killed',
            activeSince: null,
            accumulatedActiveMs: 12_000,
            completedAt: 12_000,
            resumeCount: 0,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('✕ [builder] builder-killed · 0:12')
})

test('renders --- in assigned to column for composite tasks', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-feature',
          type: 'feature',
          assignee: { kind: 'none' },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('---')
})

test('keeps assigned to column blank for non-composite unassigned tasks', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-implement-none',
          type: 'implement',
          assignee: { kind: 'none' },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).not.toContain('---')
})
