import type { ToolCatalog, CatalogKeys, CatalogTool } from '@magnitudedev/tools'
import { Effect, Layer } from 'effect'
import type { Schema } from '@effect/schema'
import type { ThinkingLens } from './thinking-lens'

// =============================================================================
// Tools
// =============================================================================

export type ToolNames<T> = T extends ToolCatalog<infer E> ? keyof E & string : never

type ExtractToolInput<TTool> =
  TTool extends { readonly inputSchema: Schema.Schema<infer I, any, any> } ? I : never

type ExtractToolOutput<TTool> =
  TTool extends { readonly outputSchema: Schema.Schema<infer O, any, any> } ? O : never

export type ToolInput<T, K extends string> = ExtractToolInput<CatalogTool<T, K>>

export type ToolOutput<T, K extends string> = ExtractToolOutput<CatalogTool<T, K>>

// =============================================================================
// Tool Policy
// =============================================================================

/** Terminal policy decision */
export type Decision =
  | { readonly decision: 'allow' }
  | { readonly decision: 'deny'; readonly reason: string }

/** A policy handler — returns Effect because interception may involve async exchanges (e.g. approval) */
export type PolicyHandler<TInput, TCtx> = (input: TInput, ctx: TCtx) => Effect.Effect<Decision | null>

/** A fragment is a partial handler map over a toolset. '*' matches any tool. */
export type PolicyFragment<T extends ToolCatalog, TCtx> = {
  [K in ToolNames<T> | '*']?: K extends '*'
    ? PolicyHandler<unknown, TCtx>
    : K extends ToolNames<T>
      ? PolicyHandler<ToolInput<T, K>, TCtx>
      : never
}

/** A policy is an ordered array of fragments. All fragments evaluate; deny > allow > implicit deny. */
export type Policy<T extends ToolCatalog, TCtx> = PolicyFragment<T, TCtx>[]

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

export interface TurnPolicy<TTools extends ToolCatalog, TCtx = Record<string, never>> {
  decide: (ctx: TurnContext<TCtx> & { tools: CatalogKeys<TTools>[] }) => TurnResult
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



export interface RoleBase<
  TTools extends ToolCatalog,
  TSlot extends string,
  TProvides = never,
  TRequirements = never,
  RObs = never
> {
  readonly id: string
  readonly slot: TSlot
  readonly tools: TTools
  readonly systemPrompt: string
  readonly lenses: ThinkingLens[]
  readonly observables?: readonly ObservableConfig<RObs>[]
  readonly lifecyclePrompts?: {
    readonly parentOnSpawn?: (agentIds: readonly string[]) => string
    readonly parentOnIdle?: (agentIds: readonly string[]) => string
  }
  readonly defaultRecipient: 'user' | 'parent'
  readonly protocolRole: 'lead' | 'subagent' | 'oneshot-lead'
  readonly initialContext: { parentConversation?: boolean }
  readonly spawnable?: boolean
  readonly setup?: (ctx: ForkSetupContext) => Effect.Effect<Layer.Layer<TProvides>, never, TRequirements>
  readonly teardown?: (ctx: ForkSetupContext) => Effect.Effect<void, never, TRequirements>
}

export interface RoleConfig<
  TTools extends ToolCatalog,
  TSlot extends string,
  TCtx,
  TProvides = never,
  TRequirements = never,
  RObs = never
> extends RoleBase<TTools, TSlot, TProvides, TRequirements, RObs> {
  readonly policy: Policy<TTools, TCtx>
  readonly turn: TurnPolicy<TTools, TCtx>
}

export interface RoleDefinitionConcrete<
  TTools extends ToolCatalog,
  TSlot extends string,
  TCtx,
  TProvides,
  TRequirements,
  RObs
> extends RoleBase<TTools, TSlot, TProvides, TRequirements, RObs> {
  readonly observables: readonly ObservableConfig<RObs>[]
  readonly spawnable: boolean

  readonly policy: Policy<TTools, TCtx>
  readonly getTurn: (ctx: TurnContext<unknown>) => TurnResult
}

export interface RoleDefinitionErased {
  readonly id: string
  readonly slot: string
  readonly tools: ToolCatalog
  readonly systemPrompt: string
  readonly lenses: ThinkingLens[]
  readonly observables: readonly ObservableConfig<unknown>[]
  readonly spawnable: boolean
  readonly lifecyclePrompts?: {
    readonly parentOnSpawn?: (agentIds: readonly string[]) => string
    readonly parentOnIdle?: (agentIds: readonly string[]) => string
  }
  readonly defaultRecipient: 'user' | 'parent'
  readonly protocolRole: 'lead' | 'subagent' | 'oneshot-lead'
  readonly initialContext: { parentConversation?: boolean }
  readonly policy: unknown
  getTurn(ctx: TurnContext<unknown>): TurnResult
  setup?(ctx: ForkSetupContext): Effect.Effect<unknown, unknown, unknown>
  teardown?(ctx: ForkSetupContext): Effect.Effect<void, unknown, unknown>
}

export type RoleDefinition<
  TTools = never,
  TSlot extends string = string,
  TCtx = unknown,
  TProvides = unknown,
  TRequirements = unknown,
  RObs = unknown
> = [TTools] extends [never]
  ? RoleDefinitionErased
  : RoleDefinitionConcrete<TTools & ToolCatalog, TSlot, TCtx, TProvides, TRequirements, RObs>

export type SlotOf<R extends Record<string, RoleDefinition>> =
  R[keyof R]['slot']

export type RoleId<R extends Record<string, RoleDefinition>> =
  keyof R & string

export type ProvidesOf<R extends Record<string, RoleDefinition>> =
  R[keyof R] extends RoleDefinition<ToolCatalog, string, unknown, infer P, unknown> ? P : never

export type RequirementsOf<R extends Record<string, RoleDefinition>> =
  R[keyof R] extends RoleDefinition<ToolCatalog, string, unknown, unknown, infer Req> ? Req : never
