/**
 * Filesystem Tool Visual Reducers
 *
 * State machines for readTool, writeTool, editTool, treeTool, searchTool.
 * Each reducer processes streaming ToolCallEvents.
 */

import type { ToolCallEvent, XmlToolResult } from '@magnitudedev/xml-act'
import { readTool, writeTool, editTool, treeTool, searchTool } from '../tools/fs'
import { defineToolReducer, defineCluster } from './define'

// =============================================================================
// Shared
// =============================================================================

type Phase = 'streaming' | 'executing' | 'done'

function phaseFromEvent(tag: ToolCallEvent['_tag']): Phase | null {
  switch (tag) {
    case 'ToolInputStarted':
    case 'ToolInputFieldValue':
    case 'ToolInputBodyChunk':
    case 'ToolInputChildStarted':
    case 'ToolInputChildComplete':
    case 'ToolInputReady':
      return 'streaming'
    case 'ToolExecutionStarted':
      return 'executing'
    case 'ToolExecutionEnded':
    case 'ToolInputParseError':
      return 'done'
  }
}

// =============================================================================
// readReducer
// =============================================================================

export interface ReadState {
  readonly phase: Phase
  readonly path: string
  readonly result: XmlToolResult<string> | null
}

export const readReducer = defineToolReducer({
  tool: readTool,
  toolKey: 'fileRead',
  initial: { phase: 'streaming', path: '', result: null } satisfies ReadState,

  reduce(state: ReadState, event): ReadState {
    const phase = phaseFromEvent(event._tag) ?? state.phase

    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'path') return { ...state, phase, path: String(event.value) }
        return phase !== state.phase ? { ...state, phase } : state
      case 'ToolExecutionEnded':
        return { ...state, phase: 'done', result: event.result }
      case 'ToolInputParseError':
        return { ...state, phase: 'done' }
      default:
        return phase !== state.phase ? { ...state, phase } : state
    }
  },
})

// =============================================================================
// writeReducer
// =============================================================================

export interface WriteState {
  readonly phase: Phase
  readonly path: string
  readonly contentChunks: readonly string[]
  readonly result: XmlToolResult<void> | null
}

export const writeReducer = defineToolReducer({
  tool: writeTool,
  toolKey: 'fileWrite',
  initial: { phase: 'streaming', path: '', contentChunks: [], result: null } satisfies WriteState,

  reduce(state: WriteState, event): WriteState {
    const phase = phaseFromEvent(event._tag) ?? state.phase

    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'path') return { ...state, phase, path: String(event.value) }
        return phase !== state.phase ? { ...state, phase } : state
      case 'ToolInputBodyChunk':
        return { ...state, phase, contentChunks: [...state.contentChunks, event.text] }
      case 'ToolExecutionEnded':
        return { ...state, phase: 'done', result: event.result }
      case 'ToolInputParseError':
        return { ...state, phase: 'done' }
      default:
        return phase !== state.phase ? { ...state, phase } : state
    }
  },
})

// =============================================================================
// editReducer — cluster-based (shared state across consecutive edit calls)
// =============================================================================

export interface EditState {
  readonly phase: Phase
  readonly path: string
  readonly result: XmlToolResult<string> | null
}

const editCluster = defineCluster<EditState>({
  cluster: 'edit',
  initial: { phase: 'streaming', path: '', result: null },
})

export const editReducer = editCluster.tool(editTool, 'fileEdit', (state, event) => {
  const phase = phaseFromEvent(event._tag) ?? state.phase

  switch (event._tag) {
    case 'ToolInputFieldValue':
      if (event.field === 'path') return { ...state, phase, path: String(event.value) }
      return phase !== state.phase ? { ...state, phase } : state
    case 'ToolExecutionEnded':
      return { ...state, phase: 'done', result: event.result }
    case 'ToolInputParseError':
      return { ...state, phase: 'done' }
    default:
      return phase !== state.phase ? { ...state, phase } : state
  }
})

// =============================================================================
// treeReducer
// =============================================================================

export interface TreeEntry {
  readonly path: string
  readonly name: string
  readonly type: 'file' | 'dir'
  readonly depth: number
}

export interface TreeState {
  readonly phase: Phase
  readonly path: string
  readonly result: XmlToolResult<readonly TreeEntry[]> | null
}

export const treeReducer = defineToolReducer({
  tool: treeTool,
  toolKey: 'fileTree',
  initial: { phase: 'streaming', path: '', result: null } satisfies TreeState,

  reduce(state: TreeState, event): TreeState {
    const phase = phaseFromEvent(event._tag) ?? state.phase

    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'path') return { ...state, phase, path: String(event.value) }
        return phase !== state.phase ? { ...state, phase } : state
      case 'ToolExecutionEnded':
        return { ...state, phase: 'done', result: event.result }
      case 'ToolInputParseError':
        return { ...state, phase: 'done' }
      default:
        return phase !== state.phase ? { ...state, phase } : state
    }
  },
})

// =============================================================================
// searchReducer
// =============================================================================

export interface SearchMatch {
  readonly file: string
  readonly match: string
}

export interface SearchState {
  readonly phase: Phase
  readonly inputs: {
    readonly pattern?: string
    readonly path?: string
    readonly glob?: string
    readonly limit?: number
  }
  readonly result: XmlToolResult<readonly SearchMatch[]> | null
}

export const searchReducer = defineToolReducer({
  tool: searchTool,
  toolKey: 'fileSearch',
  initial: { phase: 'streaming', inputs: {}, result: null } satisfies SearchState,

  reduce(state: SearchState, event): SearchState {
    const phase = phaseFromEvent(event._tag) ?? state.phase

    switch (event._tag) {
      case 'ToolInputFieldValue': {
        if (event.field === 'pattern') {
          return { ...state, phase, inputs: { ...state.inputs, pattern: String(event.value) } }
        }
        if (event.field === 'path') {
          return { ...state, phase, inputs: { ...state.inputs, path: String(event.value) } }
        }
        if (event.field === 'glob') {
          return { ...state, phase, inputs: { ...state.inputs, glob: String(event.value) } }
        }
        if (event.field === 'limit') {
          return { ...state, phase, inputs: { ...state.inputs, limit: Number(event.value) } }
        }
        return phase !== state.phase ? { ...state, phase } : state
      }
      case 'ToolExecutionEnded':
        return { ...state, phase: 'done', result: event.result }
      case 'ToolInputParseError':
        return { ...state, phase: 'done' }
      default:
        return phase !== state.phase ? { ...state, phase } : state
    }
  },
})
