import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { editTool } from '../tools/fs'
import type { EditDiff } from './edit-diff'
import { FileEditStateSchema, type FileEditState } from './tool-state'

export { FileEditStateSchema, type FileEditState } from './tool-state'

/** Detect $M/ or ${M}/ prefix and extract display path */
function detectScratchpad(path: string): { isScratchpad: boolean; scratchpadDisplayPath: Option.Option<string> } {
  const s = path.replace(/^\.\/+/, '').replace(/^\.\.\/+/, '')
  if (s.startsWith('$M/')) {
    const displayPath = s.slice('$M/'.length)
    return { isScratchpad: true, scratchpadDisplayPath: displayPath ? Option.some(displayPath) : Option.none() }
  }
  if (s.startsWith('${M}/')) {
    const displayPath = s.slice('${M}/'.length)
    return { isScratchpad: true, scratchpadDisplayPath: displayPath ? Option.some(displayPath) : Option.none() }
  }
  if (s === '$M' || s === '${M}') {
    return { isScratchpad: true, scratchpadDisplayPath: Option.none() }
  }
  return { isScratchpad: false, scratchpadDisplayPath: Option.none() }
}

const initial: Omit<FileEditState, 'phase' | 'errorMessage'> = {
  path: Option.none(),
  oldText: '',
  newText: '',
  replaceAll: false,
  streamingTarget: Option.none(),
  baseContent: Option.none(),
  diffs: [],
  isScratchpad: false,
  scratchpadDisplayPath: Option.none(),
}

const CONTEXT_LINES = 3

function findUniqueMatchRange(content: string, needle: string): Option.Option<{ start: number; end: number }> {
  if (!content || !needle) return Option.none()
  const first = content.indexOf(needle)
  if (first === -1) return Option.none()
  const second = content.indexOf(needle, first + 1)
  if (second !== -1) return Option.none()
  return Option.some({ start: first, end: first + needle.length })
}

export function computeProvisionalEditDiffs(
  baseContent: Option.Option<string>,
  oldText: string,
  newText: string,
  replaceAll: boolean,
): EditDiff[] {
  const content = Option.getOrElse(baseContent, () => '')
  if (!content || !oldText || replaceAll) return []

  const matchOption = findUniqueMatchRange(content, oldText)
  if (Option.isNone(matchOption)) return []
  const match = matchOption.value

  const fileLines = content.split('\n')
  const startLine = content.slice(0, match.start).split('\n').length - 1
  const endLine = content.slice(0, match.end).split('\n').length - 1

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
    const path = Option.getOrElse(state.path, () => '') + text
    const { isScratchpad, scratchpadDisplayPath } = detectScratchpad(path)
    return applyProvisionalDiffs({ ...state, phase: 'streaming', path: Option.some(path), isScratchpad, scratchpadDisplayPath })
  }
  if (field === 'old') {
    return applyProvisionalDiffs({
      ...state,
      phase: 'streaming',
      oldText: state.oldText + text,
      streamingTarget: Option.some('old' as const),
    })
  }
  if (field === 'new') {
    return applyProvisionalDiffs({
      ...state,
      phase: 'streaming',
      newText: state.newText + text,
      streamingTarget: Option.some('new' as const),
    })
  }
  return state
}

function applyReadyInputUpdate(
  state: FileEditState,
  input: { path: string; old: string; new: string; replaceAll: boolean },
): FileEditState {
  const { isScratchpad, scratchpadDisplayPath } = detectScratchpad(input.path)
  return applyProvisionalDiffs({
    ...state,
    path: Option.some(input.path),
    isScratchpad,
    scratchpadDisplayPath,
    oldText: input.old,
    newText: input.new,
    replaceAll: input.replaceAll,
    streamingTarget: state.streamingTarget,
  })
}

export const fileEditModel = defineStateModel(editTool)({
  state: FileEditStateSchema,
  initial,
  reduce: (state, event): FileEditState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return applyFieldUpdate(state, event.field, event.delta)
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return applyReadyInputUpdate({ ...state, phase: 'executing' }, {
          path: event.input.path,
          old: event.input.old,
          new: event.input.new,
          replaceAll: typeof event.input.replaceAll === 'boolean' ? event.input.replaceAll : false,
        })
      case 'ToolEmission': {
        const v = event.value as { type?: string; path?: string; baseContent?: string }
        return v.type === 'file_edit_base_content'
          ? applyProvisionalDiffs({
              ...state,
              phase: 'executing',
              path: typeof v.path === 'string' ? Option.some(v.path) : state.path,
              baseContent: typeof v.baseContent === 'string' ? Option.some(v.baseContent) : state.baseContent,
            })
          : state
      }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return applyProvisionalDiffs({ ...state, phase: 'completed', streamingTarget: Option.none() })
          case 'Error':
            return { ...state, phase: 'error', streamingTarget: Option.none() }
          case 'Denied':
            return { ...state, phase: 'rejected', streamingTarget: Option.none(), errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted':
            return { ...state, phase: 'interrupted', streamingTarget: Option.none() }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
