import type { ProviderToolCallId, ToolCallId } from "@magnitudedev/ai"
import type { HarnessEvent, ToolLifecycleEvent, ToolExecutionEnded } from "../events"
import { BaseStateSchema, type BaseState, type StateModel } from "./state-model"
import type { Schema } from "effect"

export interface ToolHandle<TState extends BaseState = BaseState> {
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolKey: string
  readonly state: TState
}

export function createToolHandle(
  toolCallId: ToolCallId,
  providerToolCallId: ProviderToolCallId,
  toolKey: string,
  model: StateModel<typeof BaseStateSchema>,
): ToolHandle
export function createToolHandle<TStateSchema extends Schema.Schema.AnyNoContext>(
  toolCallId: ToolCallId,
  providerToolCallId: ProviderToolCallId,
  toolKey: string,
  model: StateModel<TStateSchema>,
): ToolHandle<Schema.Schema.Type<TStateSchema>>
export function createToolHandle<TStateSchema extends Schema.Schema.AnyNoContext>(
  toolCallId: ToolCallId,
  providerToolCallId: ProviderToolCallId,
  toolKey: string,
  model: StateModel<TStateSchema>,
): ToolHandle<Schema.Schema.Type<TStateSchema>> {
  return { toolCallId, providerToolCallId, toolKey, state: model.initial }
}

export function isToolLifecycleEvent(event: HarnessEvent): event is ToolLifecycleEvent {
  switch (event._tag) {
    case 'ToolInputStarted':
    case 'ToolInputFieldChunk':
    case 'ToolInputFieldComplete':
    case 'ToolInputReady':
    case 'ToolInputRejected':
    case 'ToolExecutionStarted':
    case 'ToolExecutionEnded':
    case 'ToolEmission':
      return true
    default:
      return false
  }
}

export function processToolHandle<TToolHandle extends ToolHandle>(
  handle: TToolHandle,
  event: HarnessEvent,
  model: StateModel,
): TToolHandle
export function processToolHandle<TToolHandle extends ToolHandle>(
  handle: TToolHandle,
  event: HarnessEvent,
  model: StateModel,
): TToolHandle {
  if (!isToolLifecycleEvent(event)) return handle
  return { ...handle, state: model.reduce(handle.state, event) } as TToolHandle
}

export function interruptToolHandle<TToolHandle extends ToolHandle>(
  handle: TToolHandle,
  model: StateModel,
): TToolHandle {
  const interruptEvent: ToolLifecycleEvent = {
    _tag: 'ToolExecutionEnded',
    toolCallId: handle.toolCallId,
    providerToolCallId: handle.providerToolCallId,
    toolName: '',
    toolKey: handle.toolKey,
    result: { _tag: 'Interrupted' },
  } satisfies ToolExecutionEnded

  return { ...handle, state: model.reduce(handle.state, interruptEvent) }
}
