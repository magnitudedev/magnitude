import path from 'node:path'
import type { AppEvent } from '../events'

type ToolEvent = Extract<AppEvent, { type: 'tool_event' }>

export function extractWrittenFilePathFromToolEvent(event: ToolEvent): string | null {
  if (event.event._tag !== 'ToolExecutionEnded') return null
  if (event.toolKey !== 'fileWrite' && event.toolKey !== 'fileEdit') return null
  if (event.event.result._tag !== 'Success') return null

  if (event.display?.type === 'write_stats') return event.display.path
  if (event.display?.type === 'edit_diff') return event.display.path
  return null
}

export function isWorkspacePath(resolvedPath: string, workspacePath: string): boolean {
  const normalizedWorkspace = path.resolve(workspacePath)
  const normalizedTarget = path.resolve(resolvedPath)
  const relative = path.relative(normalizedWorkspace, normalizedTarget)
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
}
