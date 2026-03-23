import { defineStateModel, type BaseState, type ToolBinding } from '@magnitudedev/tools'
import {
  clickTool,
  clickXmlBinding,
  doubleClickTool,
  doubleClickXmlBinding,
  rightClickTool,
  rightClickXmlBinding,
  typeTool,
  typeXmlBinding,
  scrollTool,
  scrollXmlBinding,
  dragTool,
  dragXmlBinding,
  navigateTool,
  navigateXmlBinding,
  goBackTool,
  goBackXmlBinding,
  switchTabTool,
  switchTabXmlBinding,
  newTabTool,
  newTabXmlBinding,
  screenshotTool,
  screenshotXmlBinding,
  evaluateTool,
  evaluateXmlBinding,
} from '../tools/browser-tools'
import { formatBrowserActionVisualFromStreaming } from '../tools/browser-action-visuals'

export interface BrowserActionState extends BaseState {
  label?: string
  detail?: string
  errorDetail?: string
}

export interface BrowserActionModelConfig<
  K extends string,
  TInput,
  TOutput,
  TEmission,
  TStreaming extends { fields: Record<string, unknown>; body?: string | undefined }
> {
  readonly toolKey: K
  readonly tool: {
    inputSchema: { Type: TInput }
    outputSchema: { Type: TOutput }
    emissionSchema?: { Type: TEmission }
  }
  readonly binding: ToolBinding<TInput, TStreaming>
}

const initial: Omit<BrowserActionState, 'phase' | 'toolKey'> = {
  label: undefined,
  detail: undefined,
  errorDetail: undefined,
}

export function createBrowserActionModel<
  K extends string,
  TInput,
  TOutput,
  TEmission,
  TStreaming extends { fields: Record<string, unknown>; body?: string | undefined },
>(
  config: BrowserActionModelConfig<K, TInput, TOutput, TEmission, TStreaming>,
) {
  return defineStateModel(config.toolKey, {
    tool: config.tool,
    binding: config.binding,
  })({
    initial,
    reduce: (state, event) => {
      switch (event.type) {
        case 'started':
          return { ...state, phase: 'streaming', errorDetail: undefined }
        case 'inputUpdated':
        case 'inputReady': {
          const fields = event.streaming.fields as Record<string, unknown>
          const visual = formatBrowserActionVisualFromStreaming(config.toolKey, fields, event.streaming.body ?? undefined)
          return {
            ...state,
            phase: 'streaming',
            label: visual.label,
            detail: visual.detail,
          }
        }
        case 'executionStarted':
        case 'emission':
        case 'awaitingApproval':
        case 'approvalGranted':
        case 'approvalRejected':
        case 'parseError':
          return { ...state, phase: 'executing' }
        case 'completed':
          return { ...state, phase: 'completed' }
        case 'error':
          return { ...state, phase: 'error', errorDetail: event.error.message }
        case 'rejected':
          return { ...state, phase: 'rejected' }
        case 'interrupted':
          return { ...state, phase: 'interrupted' }
      }
    },
  })
}

export const clickModel = createBrowserActionModel({
  toolKey: 'click',
  tool: clickTool,
  binding: clickXmlBinding,
})

export const doubleClickModel = createBrowserActionModel({
  toolKey: 'doubleClick',
  tool: doubleClickTool,
  binding: doubleClickXmlBinding,
})

export const rightClickModel = createBrowserActionModel({
  toolKey: 'rightClick',
  tool: rightClickTool,
  binding: rightClickXmlBinding,
})

export const typeModel = createBrowserActionModel({
  toolKey: 'type',
  tool: typeTool,
  binding: typeXmlBinding,
})

export const scrollModel = createBrowserActionModel({
  toolKey: 'scroll',
  tool: scrollTool,
  binding: scrollXmlBinding,
})

export const dragModel = createBrowserActionModel({
  toolKey: 'drag',
  tool: dragTool,
  binding: dragXmlBinding,
})

export const navigateModel = createBrowserActionModel({
  toolKey: 'navigate',
  tool: navigateTool,
  binding: navigateXmlBinding,
})

export const goBackModel = createBrowserActionModel({
  toolKey: 'goBack',
  tool: goBackTool,
  binding: goBackXmlBinding,
})

export const switchTabModel = createBrowserActionModel({
  toolKey: 'switchTab',
  tool: switchTabTool,
  binding: switchTabXmlBinding,
})

export const newTabModel = createBrowserActionModel({
  toolKey: 'newTab',
  tool: newTabTool,
  binding: newTabXmlBinding,
})

export const screenshotModel = createBrowserActionModel({
  toolKey: 'screenshot',
  tool: screenshotTool,
  binding: screenshotXmlBinding,
})

export const evaluateModel = createBrowserActionModel({
  toolKey: 'evaluate',
  tool: evaluateTool,
  binding: evaluateXmlBinding,
})
