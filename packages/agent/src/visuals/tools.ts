/**
 * Tool Visual Reducers
 *
 * State machines for every tool (except shell and filesystem).
 * Each reducer processes streaming ToolCallEvents.
 */

import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { Tool } from '@magnitudedev/tools'
import type { ToolVisualReducer } from './registry'
import { defineToolReducer, reducer } from './define'

import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import {
  clickTool, doubleClickTool, rightClickTool, typeTool, scrollTool, dragTool,
  navigateTool, goBackTool, switchTabTool, newTabTool, screenshotTool, evaluateTool,
} from '../tools/browser-tools'

// =============================================================================
// Phases
// =============================================================================

export type Phase = 'streaming' | 'executing' | 'success' | 'error' | 'rejected' | 'interrupted'

export function resolveEndPhase(result: { _tag: string }): Phase {
  switch (result._tag) {
    case 'Success': return 'success'
    case 'Error': return 'error'
    case 'Rejected': return 'rejected'
    case 'Interrupted': return 'interrupted'
    default: return 'error'
  }
}

export function isActive(phase: Phase): boolean {
  return phase === 'streaming' || phase === 'executing'
}

// =============================================================================
// webSearchReducer
// =============================================================================

export interface WebSearchState {
  phase: Phase
  query: string
  sourceCount: number
  sources: Array<{ title: string; url: string }>
}

export const webSearchReducer = defineToolReducer<typeof webSearchTool, WebSearchState>({
  tool: webSearchTool,
  toolKey: 'webSearch',
  cluster: 'webSearch',
  initial: { phase: 'streaming', query: '', sourceCount: 0, sources: [] },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputBodyChunk':
        return { ...state, query: state.query + event.text }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded': {
        const phase = resolveEndPhase(event.result)
        if (event.result._tag === 'Success') {
          const output = event.result.output as { sources?: Array<{ title: string; url: string }> }
          const sources = output?.sources ?? []
          return { ...state, phase, sourceCount: sources.length, sources }
        }
        return { ...state, phase }
      }
    }
    return state
  },
})

// =============================================================================
// webFetchReducer
// =============================================================================

export interface WebFetchState {
  phase: Phase
  url: string
}

export const webFetchReducer = defineToolReducer<typeof webFetchTool, WebFetchState>({
  tool: webFetchTool,
  toolKey: 'webFetch',
  cluster: 'webFetch',
  initial: { phase: 'streaming', url: '' },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputBodyChunk':
        return { ...state, url: state.url + event.text }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded':
        return { ...state, phase: resolveEndPhase(event.result) }
    }
    return state
  },
})

// =============================================================================
// Browser Tool Reducers
// =============================================================================

export interface BrowserState {
  phase: Phase
  label: string
  /** Optional detail text (coordinates, URL, content) — rendered separately in muted color */
  detail: string
  [key: string]: unknown
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s
  return s.slice(0, max - 1) + '\u2026'
}

/** Create a minimal browser tool reducer */
function defineBrowserReducer(config: {
  toolKey: string
  pendingLabel: string
  doneLabel: (state: BrowserState) => string
  doneDetail?: (state: BrowserState) => string
  extractLabel?: (state: BrowserState, event: ToolCallEvent) => BrowserState
}): ToolVisualReducer {
  return reducer<BrowserState>({
    toolKey: config.toolKey,
    cluster: 'browser',
    initial: { phase: 'streaming', label: config.pendingLabel, detail: '' },
    reduce(state, event) {
      if (config.extractLabel) {
        state = config.extractLabel(state, event)
      }
      switch (event._tag) {
        case 'ToolExecutionStarted':
          return { ...state, phase: 'executing' as Phase }
        case 'ToolExecutionEnded':
          return {
            ...state,
            phase: resolveEndPhase(event.result),
            label: config.doneLabel(state),
            detail: config.doneDetail?.(state) ?? '',
          }
      }
      return state
    },
  })
}

function clickExtractLabel(state: BrowserState, event: ToolCallEvent): BrowserState {
  if (event._tag === 'ToolInputFieldValue') {
    if (event.field === 'x') return { ...state, x: event.value }
    if (event.field === 'y') return { ...state, y: event.value }
  }
  return state
}

function clickDoneDetail(state: BrowserState): string {
  if (state.x !== undefined) return ` (${state.x}, ${state.y})`
  return ''
}

export const clickReducer = defineBrowserReducer({
  toolKey: 'click',
  pendingLabel: 'Clicking',
  doneLabel: () => 'Clicked',
  doneDetail: clickDoneDetail,
  extractLabel: clickExtractLabel,
})

export const doubleClickReducer = defineBrowserReducer({
  toolKey: 'doubleClick',
  pendingLabel: 'Double-clicking',
  doneLabel: () => 'Double-clicked',
  doneDetail: clickDoneDetail,
  extractLabel: clickExtractLabel,
})

export const rightClickReducer = defineBrowserReducer({
  toolKey: 'rightClick',
  pendingLabel: 'Right-clicking',
  doneLabel: () => 'Right-clicked',
  doneDetail: clickDoneDetail,
  extractLabel: clickExtractLabel,
})

export const typeReducer = defineBrowserReducer({
  toolKey: 'type',
  pendingLabel: 'Typing',
  doneLabel: () => 'Typed',
  doneDetail: (state) => {
    const content = state.content as string | undefined
    return content ? ` "${truncate(content, 40)}"` : ''
  },
  extractLabel: (state, event) => {
    if (event._tag === 'ToolInputBodyChunk') {
      const prev = (state.content as string | undefined) ?? ''
      return { ...state, content: prev + event.text }
    }
    return state
  },
})

export const scrollReducer = defineBrowserReducer({
  toolKey: 'scroll',
  pendingLabel: 'Scrolling',
  doneLabel: (state) => {
    const deltaY = state.deltaY as number | undefined
    if (deltaY !== undefined && deltaY > 0) return 'Scrolled down'
    if (deltaY !== undefined && deltaY < 0) return 'Scrolled up'
    return 'Scrolled'
  },
  extractLabel: (state, event) => {
    if (event._tag === 'ToolInputFieldValue' && event.field === 'deltaY') {
      return { ...state, deltaY: event.value }
    }
    return state
  },
})

export const dragReducer = defineBrowserReducer({
  toolKey: 'drag',
  pendingLabel: 'Dragging',
  doneLabel: () => 'Dragged',
})

export const navigateReducer = defineBrowserReducer({
  toolKey: 'navigate',
  pendingLabel: 'Navigating to ',
  doneLabel: () => 'Navigated to ',
  doneDetail: (state) => {
    const url = state.url as string | undefined
    return url ? truncate(url, 50) : ''
  },
  extractLabel: (state, event) => {
    if (event._tag === 'ToolInputFieldValue' && event.field === 'url') {
      return { ...state, url: event.value, detail: truncate(String(event.value), 50) }
    }
    return state
  },
})

export const goBackReducer = defineBrowserReducer({
  toolKey: 'goBack',
  pendingLabel: 'Going back',
  doneLabel: () => 'Went back',
})

export const switchTabReducer = defineBrowserReducer({
  toolKey: 'switchTab',
  pendingLabel: 'Switching to tab ',
  doneLabel: () => 'Switched to tab ',
  doneDetail: (state) => {
    const index = state.index as number | undefined
    return index !== undefined ? String(index) : ''
  },
  extractLabel: (state, event) => {
    if (event._tag === 'ToolInputFieldValue' && event.field === 'index') {
      return { ...state, index: event.value, detail: String(event.value) }
    }
    return state
  },
})

export const newTabReducer = defineBrowserReducer({
  toolKey: 'newTab',
  pendingLabel: 'Opening new tab',
  doneLabel: () => 'Opened new tab',
})

export const screenshotReducer = defineBrowserReducer({
  toolKey: 'screenshot',
  pendingLabel: 'Taking screenshot',
  doneLabel: () => 'Screenshot',
})

export const evaluateReducer = defineBrowserReducer({
  toolKey: 'evaluate',
  pendingLabel: 'Evaluating JS',
  doneLabel: () => 'Evaluated JS',
  doneDetail: (state) => {
    const code = state.code as string | undefined
    if (code) {
      const shortCode = code.length > 30 ? code.slice(0, 27) + '...' : code
      return `: ${shortCode}`
    }
    return ''
  },
  extractLabel: (state, event) => {
    if (event._tag === 'ToolInputBodyChunk') {
      const prev = (state.code as string | undefined) ?? ''
      return { ...state, code: prev + event.text }
    }
    return state
  },
})

// =============================================================================
// Artifact Tool Reducers
// =============================================================================

export interface ArtifactWriteStreamPreview {
  mode: 'write'
  contentSoFar: string
  charCount: number
  lineCount: number
}

export interface ArtifactUpdateStreamPreview {
  mode: 'update'
  oldStringSoFar: string
  newStringSoFar: string
  childPhase: 'idle' | 'streaming_old' | 'streaming_new'
  replaceAll: boolean
}

export type ArtifactStreamPreview = ArtifactWriteStreamPreview | ArtifactUpdateStreamPreview

export interface ArtifactVisualState {
  phase: Phase
  name: string
  preview?: ArtifactStreamPreview
}

function countLines(text: string): number {
  if (text.length === 0) return 0
  return text.split('\n').length
}

function artifactReduceBase(state: ArtifactVisualState, event: ToolCallEvent): ArtifactVisualState {
  switch (event._tag) {
    case 'ToolInputFieldValue':
      if (event.field === 'id') return { ...state, name: String(event.value) }
      return state
    case 'ToolExecutionStarted':
      return { ...state, phase: 'executing' as Phase }
    case 'ToolExecutionEnded':
      return { ...state, phase: resolveEndPhase(event.result) }
  }
  return state
}


export interface ArtifactSyncState {
  phase: Phase
  name: string
  path: string
}

export const artifactSyncReducer = reducer<ArtifactSyncState>({
  toolKey: 'artifactSync',
  initial: { phase: 'streaming', name: '', path: '' },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'id') return { ...state, name: String(event.value) }
        if (event.field === 'path') return { ...state, path: String(event.value) }
        return state
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded':
        return { ...state, phase: resolveEndPhase(event.result) }
    }
    return state
  },
})

export const artifactReadReducer = reducer<ArtifactVisualState>({
  toolKey: 'artifactRead',
  initial: { phase: 'streaming', name: '' },
  reduce: artifactReduceBase,
})

export const artifactWriteReducer = reducer<ArtifactVisualState>({
  toolKey: 'artifactWrite',
  initial: {
    phase: 'streaming',
    name: '',
    preview: {
      mode: 'write',
      contentSoFar: '',
      charCount: 0,
      lineCount: 0,
    },
  },
  reduce(state, event) {
    const next = artifactReduceBase(state, event)
    if (event._tag === 'ToolInputBodyChunk' && event.field === 'content') {
      const preview = next.preview?.mode === 'write'
        ? next.preview
        : { mode: 'write' as const, contentSoFar: '', charCount: 0, lineCount: 0 }
      const contentSoFar = preview.contentSoFar + event.text
      return {
        ...next,
        preview: {
          mode: 'write',
          contentSoFar,
          charCount: contentSoFar.length,
          lineCount: countLines(contentSoFar),
        },
      }
    }
    return next
  },
})

export const artifactUpdateReducer = reducer<ArtifactVisualState>({
  toolKey: 'artifactUpdate',
  initial: {
    phase: 'streaming',
    name: '',
    preview: {
      mode: 'update',
      oldStringSoFar: '',
      newStringSoFar: '',
      childPhase: 'idle',
      replaceAll: false,
    },
  },
  reduce(state, event) {
    const next = artifactReduceBase(state, event)
    const preview = next.preview?.mode === 'update'
      ? next.preview
      : {
          mode: 'update' as const,
          oldStringSoFar: '',
          newStringSoFar: '',
          childPhase: 'idle' as const,
          replaceAll: false,
        }

    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'replaceAll') {
          return {
            ...next,
            preview: {
              ...preview,
              replaceAll: Boolean(event.value),
            },
          }
        }
        return next

      case 'ToolInputChildStarted':
        if (event.field === 'oldString') {
          return {
            ...next,
            preview: {
              ...preview,
              childPhase: 'streaming_old',
            },
          }
        }
        if (event.field === 'newString') {
          return {
            ...next,
            preview: {
              ...preview,
              childPhase: 'streaming_new',
            },
          }
        }
        return next

      case 'ToolInputBodyChunk': {
        if (event.field === 'oldString') {
          return {
            ...next,
            preview: {
              ...preview,
              oldStringSoFar: preview.oldStringSoFar + event.text,
            },
          }
        }
        if (event.field === 'newString') {
          return {
            ...next,
            preview: {
              ...preview,
              newStringSoFar: preview.newStringSoFar + event.text,
            },
          }
        }
        return next
      }

      case 'ToolInputChildComplete':
        if (event.field === 'oldString' || event.field === 'newString') {
          return {
            ...next,
            preview: {
              ...preview,
              childPhase: 'idle',
            },
          }
        }
        return next

      default:
        return next
    }
  },
})

// =============================================================================
// Agent Tool Reducers
// =============================================================================

export interface AgentCreateState {
  phase: Phase
  id: string
}

export const agentCreateReducer = reducer<AgentCreateState>({
  toolKey: 'agentCreate',
  initial: { phase: 'streaming', id: '' },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'id') return { ...state, id: String(event.value) }
        return state
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded':
        return { ...state, phase: resolveEndPhase(event.result) }
    }
    return state
  },
})

export interface AgentIdState {
  phase: Phase
  id: string
}

function agentIdReducer(state: AgentIdState, event: ToolCallEvent): AgentIdState {
  switch (event._tag) {
    case 'ToolInputFieldValue':
      if (event.field === 'id') return { ...state, id: String(event.value) }
      return state
    case 'ToolExecutionStarted':
      return { ...state, phase: 'executing' as Phase }
    case 'ToolExecutionEnded':
      return { ...state, phase: resolveEndPhase(event.result) }
  }
  return state
}

export const agentDismissReducer = reducer<AgentIdState>({
  toolKey: 'agentDismiss',
  initial: { phase: 'streaming', id: '' },
  reduce: agentIdReducer,
})

export interface AgentMessageState {
  phase: Phase
  id: string
  message: string
}

export const agentMessageReducer = reducer<AgentMessageState>({
  toolKey: 'agentMessage',
  initial: { phase: 'streaming', id: '', message: '' },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'id') return { ...state, id: String(event.value) }
        return state
      case 'ToolInputBodyChunk':
        return { ...state, message: state.message + event.text }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded':
        return { ...state, phase: resolveEndPhase(event.result) }
    }
    return state
  },
})

// =============================================================================
// parentMessageReducer
// =============================================================================

export interface ParentMessageState {
  phase: Phase
  content: string
}

export const parentMessageReducer = reducer<ParentMessageState>({
  toolKey: 'parentMessage',
  initial: { phase: 'streaming', content: '' },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputBodyChunk':
        return { ...state, content: state.content + event.text }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded':
        return { ...state, phase: resolveEndPhase(event.result) }
    }
    return state
  },
})

// =============================================================================
// skillReducer
// =============================================================================

export interface SkillState {
  phase: Phase
  name: string
}

export const skillReducer = reducer<SkillState>({
  toolKey: 'skill',
  initial: { phase: 'streaming', name: '' },
  reduce(state, event) {
    switch (event._tag) {
      case 'ToolInputFieldValue':
        if (event.field === 'id') return { ...state, name: String(event.value) }
        return state
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'ToolExecutionEnded':
        return { ...state, phase: resolveEndPhase(event.result) }
    }
    return state
  },
})
