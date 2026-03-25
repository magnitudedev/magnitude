import { defineStateModel, type BaseState, type EditDiff, type StreamingPartial } from '@magnitudedev/tools'
import { editTool, editXmlBinding } from '../tools/fs'

export interface FileEditState extends BaseState {
  toolKey: 'fileEdit'
  path?: string
  oldText: string
  newText: string
  replaceAll: boolean
  streamingTarget: 'old' | 'new' | null
  baseContent: string | null
  diffs: EditDiff[]
}

const initial: Omit<FileEditState, 'phase' | 'toolKey'> = {
  path: undefined,
  oldText: '',
  newText: '',
  replaceAll: false,
  streamingTarget: null,
  baseContent: null,
  diffs: [],
}

const CONTEXT_LINES = 5

function findUniqueMatchRange(content: string, needle: string): { start: number; end: number } | null {
  if (!content || !needle) return null
  const first = content.indexOf(needle)
  if (first === -1) return null
  const second = content.indexOf(needle, first + 1)
  if (second !== -1) return null
  return { start: first, end: first + needle.length }
}

export function computeProvisionalEditDiffs(
  baseContent: string | null,
  oldText: string,
  newText: string,
  replaceAll: boolean,
): EditDiff[] {
  if (!baseContent || !oldText || replaceAll) return []

  const match = findUniqueMatchRange(baseContent, oldText)
  if (!match) return []

  const fileLines = baseContent.split('\n')
  const startLine = baseContent.slice(0, match.start).split('\n').length - 1
  const endLine = baseContent.slice(0, match.end).split('\n').length - 1

  return [{
    startLine,
    contextBefore: fileLines.slice(Math.max(0, startLine - CONTEXT_LINES), startLine),
    removedLines: oldText.split('\n'),
    addedLines: newText.length > 0 ? newText.split('\n') : [],
    contextAfter: fileLines.slice(endLine + 1, endLine + 1 + CONTEXT_LINES),
  }]
}

export function applyProvisionalDiffs(state: FileEditState): FileEditState {
  return {
    ...state,
    diffs: computeProvisionalEditDiffs(state.baseContent, state.oldText, state.newText, state.replaceAll),
  }
}

function applyStreamingInputUpdate(
  state: FileEditState,
  streaming: StreamingPartial<{
    path: string
    oldString: string
    newString: string
    replaceAll?: boolean
  }>,
): FileEditState {
  const streamingTarget = streaming.newString !== undefined ? 'new'
    : streaming.oldString !== undefined ? 'old'
    : state.streamingTarget

  const nextState: FileEditState = {
    ...state,
    phase: 'streaming',
    path: streaming.path?.value ?? state.path,
    oldText: streaming.oldString?.value ?? state.oldText,
    newText: streaming.newString?.value ?? state.newText,
    replaceAll: streaming.replaceAll === undefined
      ? state.replaceAll
      : streaming.replaceAll.value === true || streaming.replaceAll.value === 'true',
    streamingTarget,
  }

  return applyProvisionalDiffs(nextState)
}

function applyReadyInputUpdate(
  state: FileEditState,
  input: {
    path: string
    oldString: string
    newString: string
    replaceAll?: boolean
  },
): FileEditState {
  const nextState: FileEditState = {
    ...state,
    phase: 'streaming',
    path: input.path,
    oldText: input.oldString,
    newText: input.newString,
    replaceAll: input.replaceAll ?? false,
    streamingTarget: state.streamingTarget,
  }

  return applyProvisionalDiffs(nextState)
}

export const fileEditModel = defineStateModel('fileEdit', {
  tool: editTool,
  binding: editXmlBinding,
})({
  initial,
  reduce: (state, event): FileEditState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
        return applyStreamingInputUpdate(state, event.streaming)
      case 'inputReady':
        return applyReadyInputUpdate(state, event.input)
      case 'executionStarted':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'emission':
        return event.value.type === 'file_edit_base_content'
          ? applyProvisionalDiffs({
              ...state,
              phase: 'executing',
              path: event.value.path,
              baseContent: event.value.baseContent,
            })
          : state
      case 'completed':
        return applyProvisionalDiffs({ ...state, phase: 'completed', streamingTarget: null })
      case 'error':
        return { ...state, phase: 'error', streamingTarget: null }
      case 'rejected':
        return { ...state, phase: 'rejected', streamingTarget: null }
      case 'interrupted':
        return { ...state, phase: 'interrupted', streamingTarget: null }
    }
  },
})
