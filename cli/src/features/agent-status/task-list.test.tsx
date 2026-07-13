import { test, expect, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { DisplayActor, TaskDisplayRow } from '@magnitudedev/sdk'
import { Option } from 'effect'
const testTheme = {
  foreground: '#ffffff',
  muted: '#888888',
  success: '#00ff00',
  border: '#444444',
}

vi.mock('@opentui/react', async () => {
  const actual = await vi.importActual<typeof import('@opentui/react')>('@opentui/react')
  return {
    ...actual,
    useRenderer: () => ({
      requestRender: () => {},
    }),
  }
})

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => testTheme,
}))

type TaskRow = TaskDisplayRow

let measuredWidth: number | null = null

vi.mock('../../hooks/use-local-width', () => ({
  useLocalWidth: () => ({ ref: { current: null }, onSizeChange: () => {}, width: measuredWidth }),
}))

vi.mock('../../components/button', () => ({
  Button: ({ children, onClick }: { children?: any; onClick?: () => void }) => (
    <button onClick={onClick}>{children}</button>
  ),
}))

const { TaskList, getVisibleTasks } = await import('./task-list')

const noop = () => {}
const theme = () => testTheme

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

  status: 'pending',
  depth: 0,
  parentId: Option.none(),
  updatedAt: 11_000,
  assignee: { kind: 'none' },
  ...overrides,
})

const makeActor = (
  actorKey: string,
  name: string,
  role = 'builder',
  work: Partial<DisplayActor['work']> = {},
): DisplayActor => ({
  kind: 'worker',
  name,
  role,
  parentActorKey: 'root',
  taskId: null,
  context: { tokenEstimate: 0, isCompacting: false },
  work: {
    phase: 'worked',
    activeSince: null,
    lastWorkMs: 0,
    accumulatedMs: 0,
    resumeCount: 0,
    activity: null,
    activeChildCount: 0,
    ...work,
  },
})

test('returns null when there are no tasks', () => {
  measuredWidth = null
  const html = render(<TaskList tasks={[]} pushForkOverlay={noop} slotProfiles={null} />)
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
      slotProfiles={null}
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
        makeTask({ taskId: 't-working', status: 'pending' }),
        makeTask({ taskId: 't-pending', status: 'pending' }),
      ]}
      pushForkOverlay={noop}
      slotProfiles={null}
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
        makeTask({ taskId: 'child', title: 'Child', depth: 1, parentId: Option.some('root') }),
      ]}
      pushForkOverlay={noop}
      slotProfiles={null}
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

  const html = render(<TaskList tasks={tasks} pushForkOverlay={noop} slotProfiles={null} />)
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
      slotProfiles={null}
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

test('initial expanded snap helper is no longer exported', () => {
  expect(typeof (globalThis as Record<string, unknown>).scheduleInitialTaskListSnap).toBe('undefined')
})

test('default header shows Task (X completed, Y active) using not-completed active semantics', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({ taskId: 'root', title: 'Root', depth: 0, status: 'pending' }),
        makeTask({ taskId: 'child-done', title: 'Done', depth: 1, parentId: Option.some('root'), status: 'completed' }),
        makeTask({ taskId: 'child-working', title: 'Working', depth: 1, parentId: Option.some('root'), status: 'pending' }),
      ]}
      pushForkOverlay={noop}
      slotProfiles={null}
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
            kind: 'actor',
            actorKey: 'fork-abc123',
            taskState: 'assigned',
            timer: Option.none(),
          },
        }),
      ]}
      actors={{ 'fork-abc123': makeActor('fork-abc123', '[builder] builder-abc123') }}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('● Builder · 0:00')
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
            kind: 'actor',
            actorKey: 'fork-abc123',
            taskState: 'assigned',
            timer: Option.none(),
          },
        }),
      ]}
      actors={{ 'fork-abc123': makeActor('fork-abc123', '[builder] builder-abc123') }}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).toContain('Expand all')
  expect(text).not.toContain('Collapse all')
  expect(text).toContain('Builder')
})

test('renders assigned actor role instead of actor name when preload actor name is task title', () => {
  const title = 'Review session preload implementation for code quality and edge cases'
  measuredWidth = 180
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          title,
          assignee: {
            kind: 'actor',
            actorKey: 'fork-critic-preload',
            taskState: 'assigned',
            timer: Option.none(),
          },
        }),
      ]}
      actors={{ 'fork-critic-preload': makeActor('fork-critic-preload', title, 'critic') }}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )
  measuredWidth = null

  const text = htmlToText(html)
  expect(text).toContain('● Critic')
  expect(text).not.toContain('● Review session preload')
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
      slotProfiles={null}
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
        makeTask({ taskId: 'child-a1', title: 'Child A1', depth: 1, parentId: Option.some('root-a'), status: 'completed' }),
        makeTask({ taskId: 'child-a2', title: 'Child A2', depth: 1, parentId: Option.some('root-a'), status: 'pending' }),
        makeTask({ taskId: 'child-a3', title: 'Child A3', depth: 1, parentId: Option.some('root-a'), status: 'pending' }),
        makeTask({ taskId: 'child-a4', title: 'Child A4', depth: 1, parentId: Option.some('root-a'), status: 'completed' }),
        makeTask({ taskId: 'child-a5', title: 'Child A5', depth: 1, parentId: Option.some('root-a'), status: 'pending' }),
        makeTask({ taskId: 'root-b', title: 'Root B', depth: 0, status: 'pending' }),
      ]}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  expect(html).toContain('○ </text><text style="fg:#f1f5f9">…</text>')
  expect(html).toContain('(2 completed, 4 active)')
  measuredWidth = null
})

test('renders resumed worker layout with glyph only', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          assignee: {
            kind: 'actor',
            actorKey: 'fork-planner-1',
            taskState: 'assigned',
            timer: Option.none(),
          },
        }),
      ]}
      actors={{
        'fork-planner-1': makeActor('fork-planner-1', '[planner] planner-1', 'planner', {
          phase: 'working',
          activeSince: 11_000,
          accumulatedMs: 11_000,
          resumeCount: 1,
        }),
      }}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  const text = htmlToText(html)
  expect(text).toContain('● Planner · ↺')
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
            kind: 'actor',
            actorKey: 'fork-builder-idle',
            taskState: 'assigned',
            timer: Option.none(),
          },
        }),
      ]}
      actors={{ 'fork-builder-idle': makeActor('fork-builder-idle', '[builder] builder-idle') }}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  const text = htmlToText(html)
  expect(text).toContain('✓')
  expect(text).toContain('● Builder · 0:00')
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
      slotProfiles={null}
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
            kind: 'actor',
            actorKey: 'fork-builder-killed',
            taskState: 'killing',
            timer: Option.none(),
          },
        }),
      ]}
      actors={{ 'fork-builder-killed': makeActor('fork-builder-killed', '[builder] builder-killed') }}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  const text = htmlToText(html)
  expect(text).toContain('✕ Builder')
})

test('renders blank in assigned to column for composite tasks with no worker', () => {
  const html = render(
    <TaskList
      tasks={[
        makeTask({
          taskId: 't-feature',

          assignee: { kind: 'none' },
        }),
      ]}
      pushForkOverlay={noop}
      slotProfiles={null}
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
        
          assignee: { kind: 'none' },
        }),
      ]}
      pushForkOverlay={noop}
      slotProfiles={null}
/>,
  )

  const text = htmlToText(html)
  expect(text).toContain('Assigned To')
  expect(text).not.toContain('---')
})
