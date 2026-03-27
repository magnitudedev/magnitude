import type {
  RoleConfig,
  RoleDefinition,
  RoleDefinitionConcrete,
  ToolNames,
  TurnContext,
  TurnResult,
} from './types'
import type { ToolCatalog } from '@magnitudedev/tools'

export function defineRole<
  TTools extends ToolCatalog,
  TSlot extends string,
  TCtx,
  TProvides = never,
  TRequirements = never,
  RObs = unknown
>(
  config: RoleConfig<TTools, TSlot, TCtx, TProvides, TRequirements, RObs>
): RoleDefinitionConcrete<TTools, TSlot, TCtx, TProvides, TRequirements, RObs> {
  const tools = config.tools.keys as ToolNames<TTools>[]

  const getTurn = (ctx: TurnContext<unknown>): TurnResult =>
    config.turn.decide({ ...(ctx as TurnContext<TCtx>), tools })

  return {
    id: config.id,
    slot: config.slot,
    tools: config.tools,
    systemPrompt: config.systemPrompt,
    lenses: config.lenses,
    observables: config.observables ?? [],
    lifecyclePrompts: config.lifecyclePrompts,
    defaultRecipient: config.defaultRecipient,
    protocolRole: config.protocolRole,
    initialContext: config.initialContext,
    spawnable: config.spawnable ?? false,
    setup: config.setup,
    teardown: config.teardown,
    policy: config.policy,
    getTurn,
  }
}

export function defineRoleSet<R extends Record<string, RoleDefinition>>(roles: R): R {
  return roles
}
