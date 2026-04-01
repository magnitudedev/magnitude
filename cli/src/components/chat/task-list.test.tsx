import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { TaskItem } from './types'

mock.module('../../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: '#ffffff',
    muted: '#888888',
    success: '#00ff00',
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

test('completed task timer uses accumulatedActiveMs directly (no final stint double-count)', () => {
  const tasks: TaskItem[] = [
    {
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
    },
  ]

  const html = render(
    <TaskList
      tasks={tasks}
      pushForkOverlay={noop}
      modeColor="#ffffff"
      inputBg="#000000"
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('1:05')
  expect(text).not.toContain('1:15')
})
