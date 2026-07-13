/**
 * Shared file panel types and utilities — used by both CLI and web.
 *
 * The streaming preview types (FilePanelStream, ToolHandle, etc.) and the
 * findActiveFileStream function are shared between client-common's
 * useFilePanelState hook and platform-specific rendering hooks.
 */

// ── Tool-state types for live file-stream preview ────────────────────────
// The file-panel previews in-progress file write/edit tool state to render
// an optimistic diff. These are the minimal shape the preview needs; the
// full typed tool-state schemas live in the agent package (server-side).

export type ToolPhase = 'idle' | 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted'

export interface FileWriteState {
  phase: ToolPhase
  path?: string
  body: string
  charCount: number
  lineCount: number
}

export interface FileEditState {
  phase: ToolPhase
  path?: string
  oldText: string
  newText: string
  replaceAll: boolean
  streamingTarget?: 'old' | 'new'
  diffs: unknown[]
}

export interface ToolHandle {
  toolKey: string
  state?: FileEditState | FileWriteState | { phase: ToolPhase }
}

export interface TurnState {
  handles?: {
    handles?: Map<string, ToolHandle>
  }
}

// ── File panel stream type ───────────────────────────────────────────────

export type FileOperationStatus = 'receiving' | 'applying'

export type FilePanelStream =
  | {
      mode: 'write'
      status: FileOperationStatus
      body: string
      baseContent: string | null
    }
  | {
      mode: 'edit'
      status: FileOperationStatus
      oldText: string
      newText: string
      replaceAll: boolean
      streamingTarget?: 'old' | 'new'
      baseContent: string | null
    }

// ── Optimistic update preview ────────────────────────────────────────────

export interface ChangedRange {
  start: number
  end: number
}

export interface OptimisticUpdatePreview {
  content: string
  changedRanges: ChangedRange[]
}

export function computeOptimisticUpdatePreview(
  baseContent: string | null | undefined,
  oldString: string | undefined,
  newString: string | undefined,
  replaceAll: boolean | undefined,
): OptimisticUpdatePreview | null {
  if (!baseContent || !oldString || newString === undefined) return null
  if (!baseContent.includes(oldString)) return null

  if (!replaceAll) {
    const index = baseContent.indexOf(oldString)
    if (index === -1) return null
    return {
      content: baseContent.slice(0, index) + newString + baseContent.slice(index + oldString.length),
      changedRanges: [{ start: index, end: index + newString.length }],
    }
  }

  const changedRanges: ChangedRange[] = []
  let cursor = 0
  let result = ''

  while (cursor < baseContent.length) {
    const index = baseContent.indexOf(oldString, cursor)
    if (index === -1) {
      result += baseContent.slice(cursor)
      break
    }
    result += baseContent.slice(cursor, index)
    const start = result.length
    result += newString
    changedRanges.push({ start, end: start + newString.length })
    cursor = index + oldString.length
  }

  return { content: result, changedRanges }
}

// ── Active file stream detection ─────────────────────────────────────────

export function findActiveFileStream(
  toolHandles: Record<string, ToolHandle> | undefined,
  targetPath: string,
): { toolCallId: string; state: FileEditState | FileWriteState } | null {
  if (!toolHandles) return null

  let result: { toolCallId: string; state: FileEditState | FileWriteState } | null = null
  for (const [callId, handle] of Object.entries(toolHandles)) {
    const state = getActiveFileState(handle, targetPath)
    if (state) result = { toolCallId: callId, state }
  }
  return result
}

function getActiveFileState(handle: ToolHandle, targetPath: string): FileEditState | FileWriteState | null {
  if (!handle.state) return null
  if (handle.state.phase !== 'streaming' && handle.state.phase !== 'executing') return null

  if (handle.toolKey === 'fileEdit' && isFileEditState(handle.state) && handle.state.path === targetPath) {
    return handle.state
  }

  if (handle.toolKey === 'fileWrite' && isFileWriteState(handle.state) && handle.state.path === targetPath) {
    return handle.state
  }

  return null
}

function isFileEditState(state: ToolHandle['state']): state is FileEditState {
  return !!state
    && 'oldText' in state
    && typeof (state as FileEditState).oldText === 'string'
    && 'newText' in state
    && typeof (state as FileEditState).newText === 'string'
    && 'replaceAll' in state
    && typeof (state as FileEditState).replaceAll === 'boolean'
    && 'diffs' in state
    && Array.isArray((state as FileEditState).diffs)
}

function isFileWriteState(state: ToolHandle['state']): state is FileWriteState {
  return !!state
    && 'body' in state
    && typeof (state as FileWriteState).body === 'string'
    && 'charCount' in state
    && typeof (state as FileWriteState).charCount === 'number'
    && 'lineCount' in state
    && typeof (state as FileWriteState).lineCount === 'number'
}
