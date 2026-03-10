/**
 * Modular Agent Definition — Core Types
 *
 * Tools + three policy functions (permission, turn, display).
 * Framework-agnostic: depends on @magnitudedev/tools for tool types.
 */

import type { Tool } from '@magnitudedev/tools'
import type { Effect } from 'effect'
import type { ThinkingLens } from './thinking-lens'

// =============================================================================
// Tool Set
// =============================================================================

/** A named collection of tools. Keys are definition keys used in policies. */
export type ToolSet = Record<string, Tool.Any>

/** Extract definition key names from a tool set */
export type ToolNames<T extends ToolSet> = Extract<keyof T, string>

/** Extract the input type for a specific tool in a set */
export type ToolInput<T extends ToolSet, K extends ToolNames<T>> = Tool.Input<T[K]>

/** Extract the output type for a specific tool in a set */
export type ToolOutput<T extends ToolSet, K extends ToolNames<T>> = Tool.Output<T[K]>

// =============================================================================
// Permission Policy
// =============================================================================

export interface PermissionPreview {
  readonly kind: string
  readonly data: unknown
}

export type PermissionResult =
  | { readonly decision: 'allow' }
  | { readonly decision: 'approve'; readonly reason?: string; readonly preview?: PermissionPreview }
  | { readonly decision: 'reject'; readonly reason?: string }

export interface PermissionHelpers {
  allow(): PermissionResult
  approve(opts?: { reason?: string; preview?: PermissionPreview }): PermissionResult
  reject(reason?: string): PermissionResult
}

/** Per-tool permission handlers. Ctx is framework-provided context (generic). */
export type PermissionHandlers<T extends ToolSet, Ctx> = {
  [K in ToolNames<T>]?: (input: ToolInput<T, K>, ctx: Ctx) => PermissionResult
} & {
  _default?: (input: unknown, ctx: Ctx & { tool: string }) => PermissionResult
}

/** Permission policy factory: receives helpers, returns per-tool handlers. */
export type PermissionPolicy<T extends ToolSet, Ctx> =
  (p: PermissionHelpers) => PermissionHandlers<T, Ctx>

// =============================================================================
// Turn Policy
// =============================================================================

export interface TurnContext<Ctx = Record<string, never>> {
  readonly toolsCalled: string[]
  readonly lastTool: string | null
  readonly messagesSent: readonly { readonly id: string; readonly dest: string }[]
  readonly error?: unknown
  readonly cancelled?: boolean
  readonly state: Ctx
}

export type TurnDecision = 'continue' | 'yield' | 'finish'

export type TurnResult = {
  readonly action: TurnDecision
  readonly reminder?: string
}

/** Turn policy: decision function. */
export interface TurnPolicy<T extends ToolSet, Ctx = Record<string, never>> {
  /** Called at end of turn to decide next action. */
  decide: (ctx: TurnContext<Ctx> & { tools: ToolNames<T>[] }) => TurnResult
}

// =============================================================================
// Display Policy (Human UI)
// =============================================================================

export interface DisplayPreview {
  readonly kind: string
  readonly data: unknown
}

export interface DisplayOptions {
  readonly label?: string
  readonly preview?: DisplayPreview
}

export type DisplayResult =
  | { readonly action: 'visible'; readonly options?: DisplayOptions }
  | { readonly action: 'hidden' }

export interface DisplayHelpers {
  hidden(): DisplayResult
  visible(options?: DisplayOptions): DisplayResult
}

/** Per-tool display handlers. Called after execution. */
export type DisplayHandlers<T extends ToolSet> = {
  [K in ToolNames<T>]?: (input: ToolInput<T, K>, output: ToolOutput<T, K>) => DisplayResult
} & {
  _default?: (input: unknown, output: unknown, ctx: { tool: string }) => DisplayResult
}

/** Display policy factory: receives helpers, returns per-tool handlers. */
export type DisplayPolicy<T extends ToolSet> =
  (d: DisplayHelpers) => DisplayHandlers<T>

// =============================================================================
// Observables
// =============================================================================

/** Content produced by an observable — text or image */
export type ObservationPart =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'image'; readonly base64: string; readonly mediaType: string; readonly width: number; readonly height: number }

/** Observable with Effect requirements R — captures observations before each turn */
export interface ObservableConfig<R = never> {
  readonly name: string
  readonly observe: () => Effect.Effect<ObservationPart[], never, R>
}

/** Observable with requirements satisfied via layer binding */
export interface BoundObservable {
  readonly name: string
  readonly observe: () => Effect.Effect<ObservationPart[]>
}

// =============================================================================
// Model
// =============================================================================

export type ModelTier = 'primary' | 'secondary' | 'browser'

// =============================================================================
// Agent Config & Definition
// =============================================================================

export interface AgentConfig<T extends ToolSet, Ctx = Record<string, unknown>> {
  readonly id: string
  readonly model: ModelTier
  readonly systemPrompt: string
  /** Override tool groups per definition key. Takes precedence over tool.group. */
  readonly groupOverrides?: Partial<Record<ToolNames<T>, string>>
  readonly permission: PermissionPolicy<T, Ctx>
  readonly turn: TurnPolicy<T, Ctx>
  readonly display: DisplayPolicy<T>
  readonly observables?: ObservableConfig<any>[]
  readonly thinkingLenses: ThinkingLens[]
}

export interface AgentDefinition<T extends ToolSet = ToolSet, Ctx = unknown> {
  readonly id: string
  readonly model: ModelTier
  readonly systemPrompt: string
  readonly tools: T
  readonly observables: ObservableConfig<any>[]
  readonly thinkingLenses: ThinkingLens[]

  getPermission(tool: string, input: unknown, ctx: Ctx): PermissionResult
  getTurn(ctx: TurnContext<Ctx>): TurnResult
  getDisplay(tool: string, input: unknown, output: unknown): DisplayResult

  /** Get the slug for a definition key (e.g., 'fileRead' → 'fs.read') */
  getSlug(toolKey: string): string | undefined
}
