/**
 * Filesystem Tool Visual Reducers
 *
 * State machines for readTool, writeTool, editTool, treeTool, searchTool.
 * Each reducer processes streaming ToolCallEvents.
 */

import type { ToolImageValue } from '@magnitudedev/tools'
import type { ToolCallEvent, XmlToolResult } from '@magnitudedev/xml-act'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { defineToolReducer } from './define'

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
    default:
      return null
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
  readonly contentSoFar: string
  readonly charCount: number
  readonly lineCount: number
  readonly result: XmlToolResult<void> | null
}

function countLines(text: string): number {
  if (text.length === 0) return 0
  return text.split('\n').length
}

export const writeReducer = defineToolReducer({
  tool: writeTool,
  toolKey: 'fileWrite',
  initial: {
    phase: 'streaming',
    path: '',
    contentChunks: [],
    contentSoFar: '',
    charCount: 0,
    lineCount: 0,
    result: null
  } satisfies WriteState,

  reduce(state: WriteState, event): WriteState {
    const phase = phaseFromEvent(event._tag) ?? state.phase

    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'path') return { ...state, phase, path: String(event.value) }
        return phase !== state.phase ? { ...state, phase } : state
      case 'ToolInputBodyChunk': {
        const contentSoFar = state.contentSoFar + event.text
        return {
          ...state,
          phase,
          contentChunks: [...state.contentChunks, event.text],
          contentSoFar,
          charCount: contentSoFar.length,
          lineCount: countLines(contentSoFar),
        }
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

// =============================================================================
// editReducer
// =============================================================================

export interface EditState {
  readonly phase: Phase
  readonly path: string
  readonly oldStringSoFar: string
  readonly newStringSoFar: string
  readonly replaceAll: boolean
  readonly childParsePhase: 'idle' | 'streaming_old' | 'streaming_new'
  readonly result: XmlToolResult<string> | null
}

export const editReducer = defineToolReducer({
  tool: editTool,
  toolKey: 'fileEdit',
  initial: {
    phase: 'streaming',
    path: '',
    oldStringSoFar: '',
    newStringSoFar: '',
    replaceAll: false,
    childParsePhase: 'idle',
    result: null,
  } satisfies EditState,

  reduce(state: EditState, event): EditState {
  const phase = phaseFromEvent(event._tag) ?? state.phase

  switch (event._tag) {
    case 'ToolInputFieldValue':
      if (event.field === 'path') return { ...state, phase, path: String(event.value) }
      if (event.field === 'replaceAll') return { ...state, phase, replaceAll: Boolean(event.value) }
      return phase !== state.phase ? { ...state, phase } : state
    case 'ToolInputChildStarted':
      if (event.field === 'oldString') return { ...state, phase, childParsePhase: 'streaming_old' }
      if (event.field === 'newString') return { ...state, phase, childParsePhase: 'streaming_new' }
      return phase !== state.phase ? { ...state, phase } : state
    case 'ToolInputBodyChunk':
      if (event.field === 'oldString') return { ...state, phase, oldStringSoFar: state.oldStringSoFar + event.text }
      if (event.field === 'newString') return { ...state, phase, newStringSoFar: state.newStringSoFar + event.text }
      return phase !== state.phase ? { ...state, phase } : state
    case 'ToolInputChildComplete':
      if (event.field === 'oldString' || event.field === 'newString') return { ...state, phase, childParsePhase: 'idle' }
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

// =============================================================================
// viewReducer
// =============================================================================

export interface ViewState {
  readonly phase: Phase
  readonly path: string
  readonly result: XmlToolResult<ToolImageValue> | null
}

export const viewReducer = defineToolReducer({
  tool: viewTool,
  toolKey: 'fileView',
  initial: { phase: 'streaming', path: '', result: null } satisfies ViewState,

  reduce(state: ViewState, event): ViewState {
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
