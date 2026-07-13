import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { writeTool } from '../tools/fs'
import { FileWriteStateSchema, type FileWriteState } from './tool-state'

export { FileWriteStateSchema, type FileWriteState } from './tool-state'

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

const initial: Omit<FileWriteState, 'phase' | 'errorMessage'> = {
  path: Option.none(),
  body: '',
  charCount: 0,
  lineCount: 0,
  isScratchpad: false,
  scratchpadDisplayPath: Option.none(),
}

export const fileWriteModel = defineStateModel(writeTool)({
  state: FileWriteStateSchema,
  initial,
  reduce: (state, event): FileWriteState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk': {
        if (event.field === 'path') {
          const path = Option.getOrElse(state.path, () => '') + event.delta
          const { isScratchpad, scratchpadDisplayPath } = detectScratchpad(path)
          return { ...state, phase: 'streaming', path: Option.some(path), isScratchpad, scratchpadDisplayPath }
        }
        if (event.field === 'content') {
          const content = state.body + event.delta
          return {
            ...state,
            phase: 'streaming',
            body: content,
            charCount: content.length,
            lineCount: content.length > 0 ? content.split('\n').length : 0,
          }
        }
        return state
      }
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted': {
        const content = typeof event.input.content === 'string' ? event.input.content : state.body
        const path = typeof event.input.path === 'string' ? Option.some(event.input.path) : state.path
        const { isScratchpad, scratchpadDisplayPath } = Option.match(path, {
          onNone: () => ({ isScratchpad: state.isScratchpad, scratchpadDisplayPath: state.scratchpadDisplayPath }),
          onSome: detectScratchpad,
        })
        return {
          ...state,
          phase: 'executing',
          path,
          isScratchpad,
          scratchpadDisplayPath,
          body: content,
          charCount: content.length,
          lineCount: content.length > 0 ? content.split('\n').length : 0,
        }
      }
      case 'ToolEmission': {
        const v = event.value as { type?: string; path?: string; linesWritten?: number }
        return v.type === 'write_stats'
          ? {
              ...state,
              phase: 'executing',
              path: typeof v.path === 'string' ? Option.some(v.path) : state.path,
              lineCount: typeof v.linesWritten === 'number' ? v.linesWritten : state.lineCount,
            }
          : state
      }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed' }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Denied':
            return { ...state, phase: 'rejected', errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
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
