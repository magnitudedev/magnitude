import type { ResponseUsage, ToolCallId, ToolResultPart } from "@magnitudedev/ai"

// ── Tool Result ──────────────────────────────────────────────────────

export type ToolResult<TOutput = unknown, TError = unknown> =
  | { readonly _tag: "Success"; readonly output: TOutput }
  | { readonly _tag: "Error"; readonly error: TError }
  | { readonly _tag: "Rejected"; readonly rejection: unknown }
  | { readonly _tag: "Interrupted" }

// ── Turn Outcome ─────────────────────────────────────────────────────

export type SafetyStopReason =
  | { readonly _tag: "IdenticalResponseCircuitBreaker"; readonly threshold: number }
  | { readonly _tag: "Other"; readonly message: string }

export type TurnOutcome =
  | { readonly _tag: "Completed"; readonly toolCallsCount: number }
  | { readonly _tag: "OutputTruncated" }
  | { readonly _tag: "ContentFiltered" }
  | { readonly _tag: "SafetyStop"; readonly reason: SafetyStopReason }
  | { readonly _tag: "ToolInputDecodeFailure"; readonly toolCallId: ToolCallId; readonly toolName: string; readonly detail: unknown }
  | { readonly _tag: "TurnStructureDecodeFailure"; readonly detail: unknown }
  | { readonly _tag: "GateRejected"; readonly toolCallId: ToolCallId; readonly toolName: string }
  | { readonly _tag: "EngineDefect"; readonly message: string }
  | { readonly _tag: "Interrupted" }

// ── Tool Input Lifecycle ─────────────────────────────────────────────

export interface ToolInputStarted {
  readonly _tag: "ToolInputStarted"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly group: string
}

export interface ToolInputFieldChunk {
  readonly _tag: "ToolInputFieldChunk"
  readonly toolCallId: ToolCallId
  readonly field: string
  readonly path: readonly string[]
  readonly delta: string
}

export interface ToolInputFieldComplete {
  readonly _tag: "ToolInputFieldComplete"
  readonly toolCallId: ToolCallId
  readonly field: string
  readonly path: readonly string[]
  readonly value: unknown
}

export interface ToolInputReady<TInput = unknown> {
  readonly _tag: "ToolInputReady"
  readonly toolCallId: ToolCallId
  readonly input: TInput
}

// ── Tool Execution Lifecycle ─────────────────────────────────────────

export interface ToolExecutionStarted<TInput = unknown> {
  readonly _tag: "ToolExecutionStarted"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly group: string
  readonly input: TInput
  readonly cached: boolean
}

export interface ToolExecutionEnded<TOutput = unknown, TError = unknown> {
  readonly _tag: "ToolExecutionEnded"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly group: string
  readonly result: ToolResult<TOutput, TError>
}

export interface ToolEmission<TEmission = unknown> {
  readonly _tag: "ToolEmission"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly value: TEmission
}

// ── Tool Result Formatting ───────────────────────────────────────────

export interface ToolResultFormatted {
  readonly _tag: "ToolResultFormatted"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly parts: readonly ToolResultPart[]
}

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

// ── Failures ─────────────────────────────────────────────────────────

export interface ToolInputDecodeFailure {
  readonly _tag: "ToolInputDecodeFailure"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly group: string
  readonly detail: unknown
}

export interface TurnStructureDecodeFailure {
  readonly _tag: "TurnStructureDecodeFailure"
  readonly detail: unknown
}

// ── Turn End ─────────────────────────────────────────────────────────

export interface TurnEnd {
  readonly _tag: "TurnEnd"
  readonly outcome: TurnOutcome
  readonly usage: ResponseUsage | null
}

// ── Union Types ──────────────────────────────────────────────────────

export type ToolLifecycleEvent<TInput = unknown, TOutput = unknown, TEmission = unknown, TError = unknown> =
  | ToolInputStarted
  | ToolInputFieldChunk
  | ToolInputFieldComplete
  | ToolInputReady<TInput>
  | ToolInputDecodeFailure
  | ToolExecutionStarted<TInput>
  | ToolExecutionEnded<TOutput, TError>
  | ToolEmission<TEmission>
  | ToolResultFormatted

export type HarnessEvent<TInput = unknown, TOutput = unknown, TEmission = unknown, TError = unknown> =
  | ThoughtStart
  | ThoughtDelta
  | ThoughtEnd
  | MessageStart
  | MessageDelta
  | MessageEnd
  | ToolLifecycleEvent<TInput, TOutput, TEmission, TError>
  | TurnStructureDecodeFailure
  | TurnEnd
