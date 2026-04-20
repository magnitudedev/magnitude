import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import type { ToolKey } from '../catalog'
import {
  clickTool,
  doubleClickTool,
  rightClickTool,
  typeTool,
  scrollTool,
  dragTool,
  navigateTool,
  goBackTool,
  switchTabTool,
  newTabTool,
  screenshotTool,
  evaluateTool,
} from '../tools/browser-tools'
import { formatBrowserActionVisualFromStreaming } from '../tools/browser-action-visuals'

type BrowserToolKey = Extract<
  ToolKey,
  'click' | 'doubleClick' | 'rightClick' | 'type' | 'scroll' | 'drag' | 'navigate' | 'goBack' | 'switchTab' | 'newTab' | 'screenshot' | 'evaluate'
>

export interface BrowserActionState extends BaseState {
  toolKey: BrowserToolKey
  label?: string
  detail?: string
  errorDetail?: string
  /** @internal accumulated raw text per field for streaming visual computation */
  _fields: Record<string, string>
}

const initial: Omit<BrowserActionState, 'phase' | 'toolKey'> = {
  label: undefined,
  detail: undefined,
  errorDetail: undefined,
  _fields: {},
}

export function createBrowserActionModel<
  K extends string,
  TInput,
  TOutput,
  TEmission,
>(
  config: { readonly toolKey: K; readonly tool: { inputSchema: { Type: TInput }; outputSchema: { Type: TOutput }; emissionSchema?: { Type: TEmission } } },
) {
  return defineStateModel(config.toolKey, config.tool)({
    initial,
    reduce: (state, event) => {
      switch (event._tag) {
        case 'ToolInputStarted':
          return { ...state, phase: 'streaming', errorDetail: undefined }
        case 'ToolInputFieldChunk': {
          const fields = { ...state._fields, [event.field]: (state._fields[event.field] ?? '') + event.delta }
          const visual = formatBrowserActionVisualFromStreaming(config.toolKey, fields)
          return { ...state, phase: 'streaming', _fields: fields, label: visual.label, detail: visual.detail }
        }
        case 'ToolInputReady': {
          const input = event.input as Record<string, unknown>
          const fields: Record<string, string> = {}
          for (const [k, v] of Object.entries(input)) {
            if (v !== null && v !== undefined) fields[k] = String(v)
          }
          const visual = formatBrowserActionVisualFromStreaming(config.toolKey, fields)
          return { ...state, phase: 'streaming', _fields: fields, label: visual.label, detail: visual.detail }
        }
        case 'ToolExecutionStarted':
          return { ...state, phase: 'executing' }
        case 'ToolExecutionEnded': {
          switch (event.result._tag) {
            case 'Success':
              return { ...state, phase: 'completed' }
            case 'Error':
              return { ...state, phase: 'error', errorDetail: event.result.error }
            case 'Rejected':
              return { ...state, phase: 'rejected' }
            case 'Interrupted':
              return { ...state, phase: 'interrupted' }
          }
        }
        case 'ToolInputParseError':
          return { ...state, phase: 'error', errorDetail: event.error.detail }
        case 'ToolEmission':
        case 'ToolInputFieldComplete':
        default:
          return state
      }
    },
  })
}

export const clickModel = createBrowserActionModel({ toolKey: 'click', tool: clickTool })
export const doubleClickModel = createBrowserActionModel({ toolKey: 'doubleClick', tool: doubleClickTool })
export const rightClickModel = createBrowserActionModel({ toolKey: 'rightClick', tool: rightClickTool })
export const typeModel = createBrowserActionModel({ toolKey: 'type', tool: typeTool })
export const scrollModel = createBrowserActionModel({ toolKey: 'scroll', tool: scrollTool })
export const dragModel = createBrowserActionModel({ toolKey: 'drag', tool: dragTool })
export const navigateModel = createBrowserActionModel({ toolKey: 'navigate', tool: navigateTool })
export const goBackModel = createBrowserActionModel({ toolKey: 'goBack', tool: goBackTool })
export const switchTabModel = createBrowserActionModel({ toolKey: 'switchTab', tool: switchTabTool })
export const newTabModel = createBrowserActionModel({ toolKey: 'newTab', tool: newTabTool })
export const screenshotModel = createBrowserActionModel({ toolKey: 'screenshot', tool: screenshotTool })
export const evaluateModel = createBrowserActionModel({ toolKey: 'evaluate', tool: evaluateTool })
