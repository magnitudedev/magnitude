import path from 'node:path'
import type { AppEvent } from '../events'

type ToolEvent = Extract<AppEvent, { type: 'tool_event' }>

export function extractWrittenFilePathFromToolEvent(event: ToolEvent): string | null {
  if (event.event._tag !== 'ToolExecutionEnded') return null
  if (event.toolKey !== 'fileWrite' && event.toolKey !== 'fileEdit') return null
  if (event.event.result._tag !== 'Success') return null

  const output: unknown = event.event.result.output
  if (output && typeof output === 'object' && 'path' in output) {
    const { path: filePath } = output as Record<string, unknown>
    if (typeof filePath === 'string') return filePath
  }
  return null
}

export function isScratchpadPath(resolvedPath: string, scratchpadPath: string): boolean {
  const normalizedScratchpad = path.resolve(scratchpadPath)
  const normalizedTarget = path.resolve(resolvedPath)
  const relative = path.relative(normalizedScratchpad, normalizedTarget)
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
}
