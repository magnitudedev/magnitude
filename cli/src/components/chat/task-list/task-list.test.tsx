import { test, expect, vi, beforeAll } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { TaskListItem } from './types'
import { initThemeStore, useThemeStateStore } from '../../../hooks/use-theme'

type TaskRow = Extract<TaskListItem, { kind: 'task' }>

beforeAll(() => {
  initThemeStore()
  useThemeStateStore.setState({
    theme: {
      ...useThemeStateStore.getState().theme,
      foreground: '#ffffff',
      muted: '#888888',
      success: '#00ff00',
      border: '#444444',
    },
  })
})

let measuredWidth: number | null = null

vi.mock('../../../hooks/use-local-width', () => ({
  useLocalWidth: () => ({ ref: { current: null }, onSizeChange: () => {}, width: measuredWidth }),
}))

vi.mock('../../button', () => ({
  Button: ({ children, onClick }: { children?: any; onClick?: () => void }) => (
    <button onClick={onClick}>{children}</button>
  ),
}))

const { TaskList, getVisibleTasks, scheduleInitialTaskListSnap } = await import('./task-list')

const noop = () => {}
const theme = () => useThemeStateStore.getState().theme

const htmlToText = (html: string): string =>
  html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

function render(node: ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

const makeTask = (overrides: Partial<TaskRow> = {}): TaskRow => ({
  rowId: 'task:t-1',
  kind: 'task',
  taskId: 't-1',
  title: 'Investigate timer mismatch',
  taskType: 'implement',
  status: 'pending',
  depth: 0,
  parentId: null,
  updatedAt: 11_000,
  assignee: { kind: 'none' },
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
        makeTask({ taskId: 't-1', status: 'completed' }),
        makeTask({ taskId: 't-2', status: 'pending' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(htmlToText(html)).toContain('Task')
  expect(htmlToText(html)).toContain('(1 completed, 1 active)')
})

test('renders task status glyphs as only completed and not completed', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 't-completed', status: 'completed' }),
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

test('expanded mode preserves all tasks for scrollable rendering', () => {
  const tasks = Array.from({ length: 30 }, (_, index) =>
    makeTask({ taskId: `t${index + 1}`, title: `Task ${index + 1}` }),
  )

  const visible = getVisibleTasks(tasks, true)

  expect(visible).toHaveLength(30)
  const firstTask = visible[0]
  const lastTask = visible[29]
  expect(firstTask?.kind === 'task' ? firstTask.title : null).toBe('Task 1')
  expect(lastTask?.kind === 'task' ? lastTask.title : null).toBe('Task 30')
})

test('expanded mode helper preserves all tasks for scrollable rendering', () => {
  const tasks = Array.from({ length: 30 }, (_, index) =>
    makeTask({ taskId: `t${index + 1}`, title: `Task ${index + 1}` }),
  )

  const visible = getVisibleTasks(tasks, true)

  expect(visible).toHaveLength(30)
  const html = render(
    <TaskList
      tasks={visible}
      pushForkOverlay={noop}
    />,
  )
  expect(htmlToText(html)).toContain('Task 30')
})

test('collapsed mode helper keeps only the last six tasks before expansion', () => {
  const tasks = Array.from({ length: 8 }, (_, index) =>
    makeTask({ taskId: `t${index + 1}`, title: `Task ${index + 1}` }),
  )

  const visible = getVisibleTasks(tasks, false)
  expect(visible).toHaveLength(6)
  const firstTask = visible[0]
  const lastTask = visible[5]
  expect(firstTask?.kind === 'task' ? firstTask.title : null).toBe('Task 3')
  expect(lastTask?.kind === 'task' ? lastTask.title : null).toBe('Task 8')
})

test('initial expanded snap helper schedules immediate + deferred bottom snaps and cleans up', () => {
  const scheduled: Array<{ fn: () => void, delay: number, id: number }> = []
  const canceled: number[] = []
  let nextId = 1
  let snapCount = 0

  const cleanup = scheduleInitialTaskListSnap(
    () => { snapCount += 1 },
    ((fn: (...args: any[]) => void, delay?: number) => {
      const id = nextId++
      scheduled.push({ fn: () => fn(), delay: delay ?? 0, id })
      return id as unknown as ReturnType<typeof setTimeout>
    }) as typeof setTimeout,
    ((id: ReturnType<typeof setTimeout>) => {
      canceled.push(id as unknown as number)
    }) as typeof clearTimeout,
  )

  expect(scheduled.map((t) => t.delay)).toEqual([0, 50])

  scheduled[0]?.fn()
  scheduled[1]?.fn()
  expect(snapCount).toBe(2)

  cleanup()
  expect(canceled).toEqual([scheduled[0]!.id, scheduled[1]!.id])
})

test('default header shows Task (X completed, Y active) using not-completed active semantics', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 'root', title: 'Root', depth: 0, status: 'pending' }),
        makeTask({ taskId: 'child-done', title: 'Done', depth: 1, parentId: 'root', status: 'completed' }),
        makeTask({ taskId: 'child-working', title: 'Working', depth: 1, parentId: 'root', status: 'working' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(htmlToText(html)).toContain('Task')
  expect(htmlToText(html)).toContain('(1 completed, 2 active)')
})

test('renders worker assignee with worker status prefix and timer segment', () => {
  vi.useFakeTimers()
  vi.setSystemTime(0)

  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: {
            kind: 'worker',
            variant: 'idle',
            label: '[builder] builder-abc123',
            icon: '●',
            tone: 'muted',
            interactiveForkId: 'fork-abc123',
            workerState: {
              status: 'idle',
              forkId: 'fork-abc123',
              accumulatedMs: 0,
              completedAt: 0,
              resumeCount: 0,
            },
            resumed: false,
            continuityKey: 'fork-abc123',
            ghostEligible: true,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('● [builder] builder-abc123 · 0:00')
  expect(text).not.toContain('(resumed)')
  expect(text).not.toContain('↺')
  vi.useRealTimers()
})

test('keeps assignee column and expand/collapse controls visible', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-worker',
          assignee: {
            kind: 'worker',
            variant: 'idle',
            label: '[builder] builder-abc123',
            icon: '●',
            tone: 'muted',
            interactiveForkId: 'fork-abc123',
            workerState: {
              status: 'idle',
              forkId: 'fork-abc123',
              accumulatedMs: 0,
              completedAt: 0,
              resumeCount: 0,
            },
            resumed: false,
            continuityKey: 'fork-abc123',
            ghostEligible: true,
          },
        }),
      ]}
      pushForkOverlay={noop}

    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('Expand all')
  expect(text).not.toContain('Collapse all')
  expect(text).toContain('[builder] builder-abc123')
})

test('uses measured width to truncate task names earlier', () => {
  measuredWidth = 24
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          title: 'This is a very long task title that should truncate in narrow single-column mode',
        }),
      ]}
      pushForkOverlay={noop}

    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('…')
  measuredWidth = null
})

test('sticky root header shows correct subtree progress counts', () => {
  measuredWidth = 80
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 'root-a', title: 'Root A', depth: 0, status: 'pending' }),
        makeTask({ taskId: 'child-a1', title: 'Child A1', depth: 1, parentId: 'root-a', status: 'completed' }),
        makeTask({ taskId: 'child-a2', title: 'Child A2', depth: 1, parentId: 'root-a', status: 'working' }),
        makeTask({ taskId: 'child-a3', title: 'Child A3', depth: 1, parentId: 'root-a', status: 'pending' }),
        makeTask({ taskId: 'child-a4', title: 'Child A4', depth: 1, parentId: 'root-a', status: 'completed' }),
        makeTask({ taskId: 'child-a5', title: 'Child A5', depth: 1, parentId: 'root-a', status: 'pending' }),
        makeTask({ taskId: 'root-b', title: 'Root B', depth: 0, status: 'pending' }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('Root…')
  expect(html).toContain('(2 completed, 4 active)')
  measuredWidth = null
})

test('renders resumed worker layout with glyph only', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: {
            kind: 'worker',
            variant: 'working',
            label: '[planner] planner-1',
            icon: '●',
            tone: 'active',
            interactiveForkId: 'fork-planner-1',
            workerState: {
              status: 'working',
              forkId: 'fork-planner-1',
              activeSince: 11_000,
              accumulatedMs: 11_000,
              resumeCount: 1,
            },
            resumed: true,
            continuityKey: 'fork-planner-1',
            ghostEligible: true,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('● [planner] planner-1 · ↺')
  expect(text).not.toContain('(resumed)')
})

test('completed task keeps idle worker rendering', () => {
  vi.useFakeTimers()
  vi.setSystemTime(0)

  const html = render(
    <TaskList
      tasks={[
        makeTask({
          status: 'completed',
          assignee: {
            kind: 'worker',
            variant: 'idle',
            label: '[builder] builder-idle',
            icon: '●',
            tone: 'muted',
            interactiveForkId: 'fork-builder-idle',
            workerState: {
              status: 'idle',
              forkId: 'fork-abc123',
              accumulatedMs: 0,
              completedAt: 0,
              resumeCount: 0,
            },
            resumed: false,
            continuityKey: 'fork-builder-idle',
            ghostEligible: true,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('✓')
  expect(text).toContain('● [builder] builder-idle · 0:00')
  vi.useRealTimers()
})

test('completed task text remains muted gray while checkmark stays green', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          title: 'Completed task title',
          status: 'completed',
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  expect(html).toContain('<text style="fg:#1f9670">✓ </text>')
  expect(html).toContain('style="fg:#94a3b8">Completed task title</text>')
  expect(html).not.toContain('style="fg:#1f9670">Completed task title</text>')
})

test('renders killed worker with red kill icon glyph', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: {
            kind: 'worker',
            variant: 'killing',
            label: '[builder] builder-killed',
            icon: '✕',
            tone: 'danger',
            interactiveForkId: 'fork-builder-killed',
            timer: null,
            resumed: false,
            continuityKey: 'fork-builder-killed',
            ghostEligible: true,
          },
        }),
      ]}
      pushForkOverlay={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('✕ [builder] builder-killed')
})

test('renders blank in assigned to column for composite tasks with no worker', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-feature',
          taskType: 'feature',
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

test('keeps assigned to column blank for non-composite unassigned tasks', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-implement-none',
          taskType: 'implement',
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