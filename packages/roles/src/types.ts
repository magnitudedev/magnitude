import type { AnyTool, Tool } from '@magnitudedev/tools'
import { Effect, Layer } from 'effect'
import type { Schema } from '@effect/schema'
import type { ThinkingLens } from './thinking-lens'

// =============================================================================
// Tool Set
// =============================================================================

export type ToolSet = Record<string, AnyTool>

export type ToolNames<T extends ToolSet> = Extract<keyof T, string>

type ExtractToolInput<TTool> =
  TTool extends Tool.Any ? Tool.Input<TTool> :
  TTool extends { readonly inputSchema: Schema.Schema<infer I, infer _E, infer _R> } ? I :
  never

type ExtractToolOutput<TTool> =
  TTool extends Tool.Any ? Tool.Output<TTool> :
  TTool extends { readonly outputSchema: Schema.Schema<infer O, infer _E, infer _R> } ? O :
  never

export type ToolInput<T extends ToolSet, K extends ToolNames<T>> = ExtractToolInput<T[K]>

export type ToolOutput<T extends ToolSet, K extends ToolNames<T>> = ExtractToolOutput<T[K]>

// =============================================================================
// Permission Policy
// =============================================================================

export type PermissionResult =
  | { readonly decision: 'allow' }
  | { readonly decision: 'approve'; readonly reason?: string }
  | { readonly decision: 'reject'; readonly reason?: string }

export interface PermissionHelpers {
  allow(): PermissionResult
  approve(reason?: string): PermissionResult
  reject(reason?: string): PermissionResult
}

export type PermissionHandlers<T extends ToolSet, TCtx> = {
  [K in ToolNames<T>]?: (input: ToolInput<T, K>, ctx: TCtx) => PermissionResult
} & {
  _default?: (input: unknown, ctx: TCtx & { tool: string }) => PermissionResult
}

export type PermissionPolicy<T extends ToolSet, TCtx> =
  (p: PermissionHelpers) => PermissionHandlers<T, TCtx>

// =============================================================================
// Turn Policy
// =============================================================================

export interface TurnContext<TCtx = Record<string, never>> {
  readonly toolsCalled: string[]
  readonly lastTool: string | null
  readonly messagesSent: readonly { readonly id: string; readonly dest: string }[]
  readonly error?: unknown
  readonly cancelled?: boolean
  readonly state: TCtx
}

export type TurnDecision = 'continue' | 'yield' | 'finish'

export type TurnResult = {
  readonly action: TurnDecision
  readonly reminder?: string
}

export interface TurnPolicy<TTools extends ToolSet, TCtx = Record<string, never>> {
  decide: (ctx: TurnContext<TCtx> & { tools: ToolNames<TTools>[] }) => TurnResult
}

// =============================================================================
// Observables
// =============================================================================

export type ObservationPart =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'image'; readonly base64: string; readonly mediaType: string; readonly width: number; readonly height: number }

export interface ObservableConfig<R = never> {
  readonly name: string
  readonly observe: () => Effect.Effect<ObservationPart[], never, R>
}

export interface BoundObservable {
  readonly name: string
  readonly observe: () => Effect.Effect<ObservationPart[]>
}

// =============================================================================
// Roles
// =============================================================================

export interface ForkSetupContext {
  forkId: string
  cwd: string
  workspacePath: string
}

export interface RoleConfig<
  TTools extends ToolSet,
  TSlot extends string,
  TCtx,
  TProvides = never,
  TRequirements = never
> {
  readonly id: string
  readonly slot: TSlot
  readonly tools: TTools
  readonly systemPrompt: string
  readonly lenses: ThinkingLens[]
  readonly observables?: ObservableConfig<any>[]
  readonly defaultRecipient: 'user' | 'parent'
  readonly protocolRole: 'orchestrator' | 'subagent' | 'oneshot-orchestrator'
  readonly permission: PermissionPolicy<TTools, TCtx>
  readonly turn: TurnPolicy<TTools, TCtx>
  readonly initialContext: { parentConversation?: boolean }
  readonly spawnable?: boolean
  readonly setup?: (ctx: ForkSetupContext) => Effect.Effect<Layer.Layer<TProvides>, never, TRequirements>
}

export interface RoleDefinition<
  TTools extends ToolSet = ToolSet,
  TSlot extends string = string,
  TCtx = unknown,
  TProvides = never,
  TRequirements = never
> {
  readonly id: string
  readonly slot: TSlot
  readonly tools: TTools
  readonly systemPrompt: string
  readonly lenses: ThinkingLens[]
  readonly observables: readonly ObservableConfig<any>[]
  readonly defaultRecipient: 'user' | 'parent'
  readonly protocolRole: 'orchestrator' | 'subagent' | 'oneshot-orchestrator'
  readonly initialContext: { parentConversation?: boolean }
  readonly spawnable: boolean
  readonly setup?: (ctx: ForkSetupContext) => Effect.Effect<Layer.Layer<TProvides>, never, TRequirements>

  getPermission(tool: string, input: unknown, ctx: TCtx): PermissionResult
  getTurn(ctx: TurnContext<TCtx>): TurnResult
}

export type SlotOf<R extends Record<string, RoleDefinition<any, any, any, any, any>>> =
  R[keyof R]['slot']

export type RoleId<R extends Record<string, RoleDefinition<any, any, any, any, any>>> =
  keyof R & string

export type ProvidesOf<R extends Record<string, RoleDefinition<any, any, any, any, any>>> =
  R[keyof R] extends RoleDefinition<any, any, any, infer P, any> ? P : never

export type RequirementsOf<R extends Record<string, RoleDefinition<any, any, any, any, any>>> =
  R[keyof R] extends RoleDefinition<any, any, any, any, infer Req> ? Req : never
