import type {
  ToolSet,
  ToolNames,
  RoleConfig,
  RoleDefinition,
  PermissionResult,
  PermissionHelpers,
  TurnContext,
  TurnResult,
} from './types'
import { allow, approve, reject } from './helpers'

export function toolSet<T extends ToolSet>(tools: T): T {
  return tools
}

function dispatchHandler(
  handlers: Record<string, Function | undefined>,
  tool: string,
  args: unknown[],
  defaultArgs: unknown[],
  fallback: () => unknown,
): unknown {
  const handler = handlers[tool]
  if (handler) return handler(...args)
  const defaultHandler = handlers._default
  if (defaultHandler) return defaultHandler(...defaultArgs)
  return fallback()
}

const permissionHelpers: PermissionHelpers = { allow, approve, reject }

export function defineRole<
  TTools extends ToolSet,
  TSlot extends string,
  TCtx,
  TProvides = never,
  TRequirements = never
>(
  config: RoleConfig<TTools, TSlot, TCtx, TProvides, TRequirements>
): RoleDefinition<TTools, TSlot, TCtx, TProvides, TRequirements> {
  const permissionHandlers = config.permission(permissionHelpers) as Record<string, Function | undefined>

  function getPermission(tool: string, input: unknown, ctx: TCtx): PermissionResult {
    return dispatchHandler(
      permissionHandlers,
      tool,
      [input, ctx],
      [input, { ...ctx as object, tool }],
      () => allow()
    ) as PermissionResult
  }

  function getTurn(ctx: TurnContext<TCtx>): TurnResult {
    return config.turn.decide(ctx as TurnContext<TCtx> & { tools: ToolNames<TTools>[] })
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
    getPermission,
    getTurn,
  }
}

export function defineRoleSet<R extends Record<string, RoleDefinition<any, any, any, any, any>>>(roles: R): R {
  return roles
}
