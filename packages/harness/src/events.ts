import type { JsonValue } from "@magnitudedev/utils/schema"
import type { ProviderToolCallId, ResponseUsage, ToolCallId, ValidationIssue, ModelStreamTerminal } from "@magnitudedev/ai"

// ── Tool Error ───────────────────────────────────────────────────────

/** Base constraint for tool errors — all tool errors must have a message. */
export interface ToolError {
  readonly message: string
}

// ── Tool Result ──────────────────────────────────────────────────────

type ToolResultErased =
  | { readonly _tag: "Success"; readonly output: JsonValue }
  | { readonly _tag: "Error"; readonly error: ToolError }
  | { readonly _tag: "Denied"; readonly denial: JsonValue }
  | { readonly _tag: "Interrupted" }
  | { readonly _tag: "InputRejected"; readonly issue: ValidationIssue; readonly partialInput: JsonValue }

type ToolResultConcrete<TOutput, TError extends ToolError, TDenial> =
  | { readonly _tag: "Success"; readonly output: TOutput }
  | { readonly _tag: "Error"; readonly error: TError }
  | { readonly _tag: "Denied"; readonly denial: TDenial }
  | { readonly _tag: "Interrupted" }
  | { readonly _tag: "InputRejected"; readonly issue: ValidationIssue; readonly partialInput: JsonValue }

export type ToolResult<
  TOutput = never,
  TError extends ToolError = never,
  TDenial = unknown
> =
  [TOutput] extends [never]
    ? ToolResultErased
    : ToolResultConcrete<TOutput, [TError] extends [never] ? ToolError : TError, TDenial>

// ── Tool Result Entry ────────────────────────────────────────────────

/** A tool result entry — the semantic result plus identifying metadata. */
export interface ToolResultEntry {
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly result: ToolResult
}

// ── Safety Stop ──────────────────────────────────────────────────────

export type SafetyStopReason =
  | { readonly _tag: "IdenticalResponseCircuitBreaker"; readonly threshold: number }
  | { readonly _tag: "Other"; readonly message: string }

// ── Turn Outcome ─────────────────────────────────────────────────────

interface TurnOutcomeBase {
  readonly _tag: "Completed" | "OutputTruncated" | "ContentFiltered" | "SafetyStop" | "ToolInputValidationFailure" | "ToolExecutionError" | "GateRejected" | "StreamFailed" | "EngineDefect" | "Interrupted" | "ThoughtLimitExceeded"
}

export type StreamFailedTerminal = Extract<ModelStreamTerminal, { readonly _tag: "StreamFailed" }>

type WithRequestId<T> = T & { readonly requestId: string | null }

type TurnOutcomeConcrete<TInput> = WithRequestId<
  | { readonly _tag: "Completed"; readonly toolCallsCount: number }
  | { readonly _tag: "OutputTruncated" }
  | { readonly _tag: "ContentFiltered" }
  | { readonly _tag: "SafetyStop"; readonly reason: SafetyStopReason }
  | { readonly _tag: "ToolInputValidationFailure"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly toolKey: string; readonly issue: ValidationIssue }
  | { readonly _tag: "ToolExecutionError"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly toolKey: string; readonly error: ToolError }
  | { readonly _tag: "GateRejected"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string }
  | { readonly _tag: "StreamFailed"; readonly message: string; readonly terminal: StreamFailedTerminal }
  | { readonly _tag: "EngineDefect"; readonly message: string }
  | { readonly _tag: "Interrupted" }
  | { readonly _tag: "ThoughtLimitExceeded"; readonly limit: number }
>

type TurnOutcomeErased = WithRequestId<
  | { readonly _tag: "Completed"; readonly toolCallsCount: number }
  | { readonly _tag: "OutputTruncated" }
  | { readonly _tag: "ContentFiltered" }
  | { readonly _tag: "SafetyStop"; readonly reason: SafetyStopReason }
  | { readonly _tag: "ToolInputValidationFailure"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly toolKey: string; readonly issue: ValidationIssue }
  | { readonly _tag: "ToolExecutionError"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly toolKey: string; readonly error: ToolError }
  | { readonly _tag: "GateRejected"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string }
  | { readonly _tag: "StreamFailed"; readonly message: string; readonly terminal: StreamFailedTerminal }
  | { readonly _tag: "EngineDefect"; readonly message: string }
  | { readonly _tag: "Interrupted" }
  | { readonly _tag: "ThoughtLimitExceeded"; readonly limit: number }
>

export type TurnOutcome<TInput = never> =
  [TInput] extends [never]
    ? TurnOutcomeErased
    : TurnOutcomeConcrete<TInput>

// ── Tool Input Lifecycle ─────────────────────────────────────────────

export interface ToolInputStarted {
  readonly _tag: "ToolInputStarted"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
}

export interface ToolInputFieldChunk {
  readonly _tag: "ToolInputFieldChunk"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly field: string
  readonly path: readonly string[]
  readonly delta: string
}

export interface ToolInputFieldComplete {
  readonly _tag: "ToolInputFieldComplete"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly field: string
  readonly path: readonly string[]
  readonly value: unknown
}

export interface ToolInputReady {
  readonly _tag: "ToolInputReady"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
}

// ── Tool Input Rejected ─────────────────────────────────────────────

export interface ToolInputRejected {
  readonly _tag: "ToolInputRejected"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly issue: ValidationIssue
}

// ── Tool Execution Lifecycle ─────────────────────────────────────────

interface ToolExecutionStartedErased {
  readonly _tag: "ToolExecutionStarted"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly input: JsonValue
  readonly cached: boolean
}

interface ToolExecutionStartedConcrete<TInput> {
  readonly _tag: "ToolExecutionStarted"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly input: TInput
  readonly cached: boolean
}

export type ToolExecutionStarted<TInput = never> =
  [TInput] extends [never]
    ? ToolExecutionStartedErased
    : ToolExecutionStartedConcrete<TInput>

interface ToolExecutionEndedErased {
  readonly _tag: "ToolExecutionEnded"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly result: ToolResult
}

interface ToolExecutionEndedConcrete<TOutput, TError extends ToolError> {
  readonly _tag: "ToolExecutionEnded"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly result: ToolResult<TOutput, TError>
}

export type ToolExecutionEnded<TOutput = never, TError extends ToolError = never> =
  [TOutput] extends [never]
    ? ToolExecutionEndedErased
    : ToolExecutionEndedConcrete<TOutput, [TError] extends [never] ? ToolError : TError>

interface ToolEmissionErased {
  readonly _tag: "ToolEmission"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly value: JsonValue
}

interface ToolEmissionConcrete<TEmission> {
  readonly _tag: "ToolEmission"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly value: TEmission
}

export type ToolEmission<TEmission = never> =
  [TEmission] extends [never]
    ? ToolEmissionErased
    : ToolEmissionConcrete<TEmission>



// ── Reasoning and Messages ───────────────────────────────────────────

export interface ThoughtStart {
  readonly _tag: "ThoughtStart"
  readonly level: "low" | "medium" | "high"
}

export interface ThoughtDelta {
  readonly _tag: "ThoughtDelta"
  readonly text: string
}

export interface ThoughtEnd {
  readonly _tag: "ThoughtEnd"
}

export interface MessageStart {
  readonly _tag: "MessageStart"
}

export interface MessageDelta {
  readonly _tag: "MessageDelta"
  readonly text: string
}

export interface MessageEnd {
  readonly _tag: "MessageEnd"
}

// ── Turn End ─────────────────────────────────────────────────────────

interface TurnEndErased {
  readonly _tag: "TurnEnd"
  readonly outcome: TurnOutcome
  readonly usage: ResponseUsage | null
}

interface TurnEndConcrete<TInput> {
  readonly _tag: "TurnEnd"
  readonly outcome: TurnOutcome<TInput>
  readonly usage: ResponseUsage | null
}

export type TurnEnd<TInput = never> =
  [TInput] extends [never]
    ? TurnEndErased
    : TurnEndConcrete<TInput>

// ── Union Types ──────────────────────────────────────────────────────

type ToolLifecycleEventErased =
  | ToolInputStarted
  | ToolInputFieldChunk
  | ToolInputFieldComplete
  | ToolInputReady
  | ToolInputRejected
  | ToolExecutionStarted
  | ToolExecutionEnded
  | ToolEmission

type ToolLifecycleEventConcrete<TInput, TOutput, TEmission, TError extends ToolError> =
  | ToolInputStarted
  | ToolInputFieldChunk
  | ToolInputFieldComplete
  | ToolInputReady
  | ToolInputRejected
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput, TError>
  | ToolEmission<TEmission>

export type ToolLifecycleEvent<TInput = never, TOutput = never, TEmission = never, TError extends ToolError = never> =
  [TInput] extends [never]
    ? ToolLifecycleEventErased
    : ToolLifecycleEventConcrete<TInput, [TOutput] extends [never] ? unknown : TOutput, [TEmission] extends [never] ? unknown : TEmission, [TError] extends [never] ? ToolError : TError>

type HarnessEventErased =
  | ThoughtStart
  | ThoughtDelta
  | ThoughtEnd
  | MessageStart
  | MessageDelta
  | MessageEnd
  | ToolLifecycleEvent
  | TurnEnd

type HarnessEventConcrete<TInput, TOutput, TEmission, TError extends ToolError> =
  | ThoughtStart
  | ThoughtDelta
  | ThoughtEnd
  | MessageStart
  | MessageDelta
  | MessageEnd
  | ToolLifecycleEvent<TInput, TOutput, TEmission, TError>
  | TurnEnd<TInput>

export type HarnessEvent<TInput = never, TOutput = never, TEmission = never, TError extends ToolError = never> =
  [TInput] extends [never]
    ? HarnessEventErased
    : HarnessEventConcrete<TInput, [TOutput] extends [never] ? unknown : TOutput, [TEmission] extends [never] ? unknown : TEmission, [TError] extends [never] ? ToolError : TError>
