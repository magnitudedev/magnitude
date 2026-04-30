import type { ToolCallId } from "@magnitudedev/ai"
import type { HarnessEvent, ToolLifecycleEvent, ToolExecutionEnded } from "../events"
import type { BaseState, StateModel } from "./state-model"

export interface ToolHandle {
  readonly toolCallId: ToolCallId
  readonly toolKey: string
  readonly state: BaseState
  readonly process: (event: HarnessEvent) => ToolHandle
  readonly interrupt: () => ToolHandle
}

export function createToolHandle(
  toolCallId: ToolCallId,
  toolKey: string,
  model: StateModel,
): ToolHandle {
  return buildHandle(toolCallId, toolKey, model.initial, model.reduce)
}

function isToolLifecycleEvent(event: HarnessEvent): event is ToolLifecycleEvent {
  switch (event._tag) {
    case 'ToolInputStarted':
    case 'ToolInputFieldChunk':
    case 'ToolInputFieldComplete':
    case 'ToolInputReady':
    case 'ToolInputDecodeFailure':
    case 'ToolExecutionStarted':
    case 'ToolExecutionEnded':
    case 'ToolEmission':
    case 'ToolResultFormatted':
      return true
    default:
      return false
  }
}

function buildHandle(
  toolCallId: ToolCallId,
  toolKey: string,
  state: BaseState,
  reduce: (state: BaseState, event: ToolLifecycleEvent) => BaseState,
): ToolHandle {
  return {
    toolCallId,
    toolKey,
    get state() { return state },
    process(event: HarnessEvent): ToolHandle {
      if (!isToolLifecycleEvent(event)) return this
      // Map ToolInputDecodeFailure → synthetic ToolExecutionEnded(Error)
      const mapped: ToolLifecycleEvent = event._tag === 'ToolInputDecodeFailure'
        ? {
            _tag: 'ToolExecutionEnded',
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            toolKey: event.toolKey,
            result: { _tag: 'Error', error: typeof event.detail === 'string' ? event.detail : String(event.detail) },
          } satisfies ToolExecutionEnded
        : event
      const reduced = reduce(state, mapped)
      return buildHandle(toolCallId, toolKey, reduced, reduce)
    },
    interrupt(): ToolHandle {
      const interruptEvent: ToolLifecycleEvent = {
        _tag: 'ToolExecutionEnded',
        toolCallId,
        toolName: '',
        toolKey,
        result: { _tag: 'Interrupted' },
      } satisfies ToolExecutionEnded
      return buildHandle(toolCallId, toolKey, reduce(state, interruptEvent), reduce)
    },
  }
}
