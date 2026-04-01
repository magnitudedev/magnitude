import { defineStateModel, type BaseState, type StreamingPartial, type ToolBinding } from '@magnitudedev/tools'
import type { ToolKey } from '../catalog'
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

type BrowserToolKey = Extract<
  ToolKey,
  'click' | 'doubleClick' | 'rightClick' | 'type' | 'scroll' | 'drag' | 'navigate' | 'goBack' | 'switchTab' | 'newTab' | 'screenshot' | 'evaluate'
>

export interface BrowserActionState extends BaseState {
  toolKey: BrowserToolKey
  label?: string
  detail?: string
  errorDetail?: string
}

export interface BrowserActionModelConfig<
  K extends string,
  TInput,
  TOutput,
  TEmission
> {
  readonly toolKey: K
  readonly tool: {
    inputSchema: { Type: TInput }
    outputSchema: { Type: TOutput }
    emissionSchema?: { Type: TEmission }
  }
  readonly binding: ToolBinding<TInput>
}

const initial: Omit<BrowserActionState, 'phase' | 'toolKey'> = {
  label: undefined,
  detail: undefined,
  errorDetail: undefined,
}

function unwrapStreamingLeaf(value: unknown): unknown {
  if (!value || typeof value !== 'object') return value
  if ('value' in value && 'isFinal' in value) {
    return (value as { value: unknown }).value
  }
  return value
}

function normalizeStreamingInput(streaming: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(streaming)) {
    normalized[key] = unwrapStreamingLeaf(value)
  }
  return normalized
}

export function createBrowserActionModel<
  K extends string,
  TInput,
  TOutput,
  TEmission,
>(
  config: BrowserActionModelConfig<K, TInput, TOutput, TEmission>,
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
          const visual = formatBrowserActionVisualFromStreaming(
            config.toolKey,
            normalizeStreamingInput(event.streaming as Record<string, unknown>),
          )
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
          return { ...state, phase: 'executing' }
        case 'parseError':
          return { ...state, phase: 'error', errorDetail: event.error }
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
