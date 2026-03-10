/**
 * Modular Agent Definition — Runtime Functions
 */

import type {
  ToolSet, ToolNames, AgentConfig, AgentDefinition,
  PermissionResult, PermissionHelpers,
  TurnContext, TurnResult,
  DisplayResult, DisplayHelpers,
} from './types'
import { allow, approve, reject, hidden, visible } from './helpers'

// =============================================================================
// toolSet — identity function for type inference
// =============================================================================

export function toolSet<T extends ToolSet>(tools: T): T {
  return tools
}

// =============================================================================
// Policy dispatch — type-safe handler lookup
// =============================================================================

/**
 * Generic handler dispatch. Looks up a handler by tool name,
 * falls back to _default, then to a provided fallback.
 *
 * Handlers are stored as a Record<string, Function> at runtime,
 * but typed per-tool via PermissionHandlers/DisplayHandlers
 * at the call site. The generic typing happens in AgentConfig<T>.
 */
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

// =============================================================================
// Helper objects for policy factories
// =============================================================================

const permissionHelpers: PermissionHelpers = { allow, approve, reject }
const displayHelpers: DisplayHelpers = { hidden, visible }

// =============================================================================
// defineAgent — creates an AgentDefinition with policy dispatch
// =============================================================================

export function defineAgent<T extends ToolSet, Ctx = Record<string, unknown>>(
  tools: T,
  config: AgentConfig<T, Ctx>
): AgentDefinition<T, Ctx> {
  const overrides = config.groupOverrides as Record<string, string> | undefined

  // Derive key → slug map from tool.group + tool.name, with optional overrides
  const keyToSlug = new Map<string, string>()
  for (const [key, tool] of Object.entries(tools)) {
    const t = tool as { name: string; group?: string }
    const group = overrides?.[key] ?? t.group
    const slug = (group && group !== 'default') ? `${group}.${t.name}` : t.name
    keyToSlug.set(key, slug)
  }

  // Invoke policy factories once at definition time
  const permissionHandlers = config.permission(permissionHelpers) as Record<string, Function | undefined>
  const displayHandlers = config.display(displayHelpers) as Record<string, Function | undefined>


  function getPermission(tool: string, input: unknown, ctx: Ctx): PermissionResult {
    return dispatchHandler(
      permissionHandlers,
      tool,
      [input, ctx],
      [input, { ...ctx as object, tool }],
      () => allow()
    ) as PermissionResult
  }

  function getTurn(ctx: TurnContext<Ctx>): TurnResult {
    return config.turn.decide(ctx as TurnContext<Ctx> & { tools: ToolNames<T>[] })
  }

  function getReminder(ctx: TurnContext<Ctx>): string | null {
    return config.turn.reminder?.(ctx) ?? null
  }

  function getDisplay(tool: string, input: unknown, output: unknown): DisplayResult {
    return dispatchHandler(
      displayHandlers,
      tool,
      [input, output],
      [input, output, { tool }],
      () => ({ action: 'visible' as const })
    ) as DisplayResult
  }

  function getSlug(toolKey: string): string | undefined {
    return keyToSlug.get(toolKey)
  }

  return {
    id: config.id,
    model: config.model,
    systemPrompt: config.systemPrompt,
    tools,
    observables: config.observables ?? [],
    thinkingLenses: config.thinkingLenses,
    getPermission,
    getTurn,
    getReminder,
    getDisplay,
    getSlug,
  }
}
