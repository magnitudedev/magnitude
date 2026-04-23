import { defineStateModel, type BaseState, type EditDiff } from '@magnitudedev/tools'
import { editTool } from '../tools/fs'

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

function applyFieldUpdate(state: FileEditState, field: string, text: string): FileEditState {
  if (field === 'path') {
    return applyProvisionalDiffs({ ...state, phase: 'streaming', path: (state.path ?? '') + text })
  }
  if (field === 'old') {
    return applyProvisionalDiffs({
      ...state,
      phase: 'streaming',
      oldText: state.oldText + text,
      streamingTarget: 'old',
    })
  }
  if (field === 'new') {
    return applyProvisionalDiffs({
      ...state,
      phase: 'streaming',
      newText: state.newText + text,
      streamingTarget: 'new',
    })
  }
  return state
}

function applyReadyInputUpdate(
  state: FileEditState,
  input: { path: string; old: string; new: string; replaceAll?: boolean },
): FileEditState {
  return applyProvisionalDiffs({
    ...state,
    phase: 'streaming',
    path: input.path,
    oldText: input.old,
    newText: input.new,
    replaceAll: input.replaceAll ?? false,
    streamingTarget: state.streamingTarget,
  })
}

export const fileEditModel = defineStateModel('fileEdit', editTool)({
  initial,
  reduce: (state, event): FileEditState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return applyFieldUpdate(state, event.field, event.delta)
      case 'ToolInputReady':
        return applyReadyInputUpdate(state, event.input)
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolEmission': {
        const v = event.value as { type?: string; path?: string; baseContent?: string }
        return v.type === 'file_edit_base_content'
          ? applyProvisionalDiffs({
              ...state,
              phase: 'executing',
              path: v.path ?? state.path,
              baseContent: v.baseContent ?? state.baseContent,
            })
          : state
      }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return applyProvisionalDiffs({ ...state, phase: 'completed', streamingTarget: null })
          case 'Error':
            return { ...state, phase: 'error', streamingTarget: null }
          case 'Rejected':
            return { ...state, phase: 'rejected', streamingTarget: null }
          case 'Interrupted':
            return { ...state, phase: 'interrupted', streamingTarget: null }
        }
      }
      case 'ToolParseError':
        return { ...state, phase: 'error', streamingTarget: null }
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
