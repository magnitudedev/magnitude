import type {
  ToolSet,
  ToolNames,
  RoleConfig,
  RoleDefinition,
  TurnContext,
  TurnResult,
} from './types'

export function toolSet<T extends ToolSet>(tools: T): T {
  return tools
}

function toolNames<T extends ToolSet>(tools: T): ToolNames<T>[] {
  return Object.keys(tools) as ToolNames<T>[]
}

export function defineRole<
  TTools extends ToolSet,
  TSlot extends string,
  TCtx,
  TProvides = never,
  TRequirements = never
>(
  config: RoleConfig<TTools, TSlot, TCtx, TProvides, TRequirements>
): RoleDefinition<TTools, TSlot, TCtx, TProvides, TRequirements> {
  const tools = toolNames(config.tools)

  function getTurn(ctx: TurnContext<TCtx>): TurnResult {
    return config.turn.decide({ ...ctx, tools })
  }

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

export function defineRoleSet<R extends Record<string, RoleDefinition<any, any, any, any, any>>>(roles: R): R {
  return roles
}
