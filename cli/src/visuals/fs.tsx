/**
 * Filesystem Tool Visuals — Renderers
 *
 * Pure render functions for filesystem tool visual state.
 * State is pre-reduced by DisplayProjection.
 */



type Phase = 'streaming' | 'executing' | 'awaiting_approval' | 'completed' | 'error' | 'rejected' | 'interrupted' | 'done'
type ToolResult<T> =
  | { _tag: 'Success'; output: T }
  | { _tag: 'Error'; error: string }

interface ReadState {
  phase: Phase
  path?: string
  result?: ToolResult<string> | null
}

interface TreeEntry {
  name: string
  type: 'file' | 'dir'
  depth: number
}

interface TreeState {
  phase: Phase
  path?: string
  result?: ToolResult<readonly TreeEntry[]> | null
}

interface SearchMatch {
  file: string
  match: string
}

interface SearchState {
  phase: Phase
  inputs?: {
    pattern?: string
    path?: string
    glob?: string
    limit?: number
  }
  result?: ToolResult<readonly SearchMatch[]> | null
}



// =============================================================================
// Constants
// =============================================================================

const SHIMMER_INTERVAL_MS = 160

// =============================================================================
// Helpers
// =============================================================================

function parseMatch(m: string): { line: number; text: string } {
  const pipeIdx = m.indexOf('|')
  if (pipeIdx === -1) return { line: 0, text: m }
  const prefix = m.slice(0, pipeIdx)
  const text = m.slice(pipeIdx + 1)
  const colonIdx = prefix.indexOf(':')
  const line = colonIdx !== -1 ? parseInt(prefix.slice(0, colonIdx), 10) || 0 : 0
  return { line, text }
}

function truncateLine(text: string, max: number): string {
  if (!text) return ''
  const firstLine = text.split('\n').find(l => l.trim() !== '') ?? ''
  if (firstLine.length > max) return firstLine.slice(0, max - 3) + '...'
  return firstLine
}

function formatSearchInputs(state: SearchState): string {
  const parts: string[] = []
  if (state.inputs?.pattern !== undefined) parts.push(`pattern="${state.inputs.pattern}"`)
  if (state.inputs?.path !== undefined) parts.push(`path="${state.inputs.path}"`)
  if (state.inputs?.glob !== undefined) parts.push(`glob="${state.inputs.glob}"`)
  if (state.inputs?.limit !== undefined) parts.push(`limit=${state.inputs.limit}`)
  return parts.join(' ')
}



// =============================================================================
// readRender
// =============================================================================

export function readLiveText({ state }: { state: ReadState }): string {
  const target = state.path || 'file'
  return state.phase === 'done' ? `Read ${target}` : `Reading ${target}`
}



// =============================================================================


// =============================================================================
// treeRender
// =============================================================================

export function treeLiveText({ state }: { state: TreeState }): string {
  const target = state.path || 'files'
  if (state.phase !== 'done') return `Listing ${target}`
  return state.result?._tag === 'Success' ? `Listed ${target}` : `List ${target}`
}



// =============================================================================
// searchRender
// =============================================================================

export function searchLiveText({ state }: { state: SearchState }): string {
  const summary = formatSearchInputs(state)
  const target = summary.length > 0 ? summary : 'files'
  return state.phase === 'done' ? `Searched ${target}` : `Searching ${target}`
}


