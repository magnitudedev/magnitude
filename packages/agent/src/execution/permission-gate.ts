/**
 * Policy Interceptor
 *
 * Evaluates the agent's tool policy for the current tool call and routes the decision.
 */

import { Effect, Context } from 'effect'
import { Fork } from '@magnitudedev/event-core'
import type { RoleDefinition } from '@magnitudedev/roles'
import { evaluatePolicy } from '@magnitudedev/roles'
import type { ToolCallId } from '@magnitudedev/ai'
import type { ExecuteHookContext, InterceptorDecision } from '@magnitudedev/harness'
import type { PolicyContext as RolesPolicyContext } from '@magnitudedev/roles'
import { PolicyContextProviderTag } from '../agents/types'

const { ForkContext } = Fork

/** Context provided to the interceptor for each tool call. */
export interface InterceptorContext {
  readonly toolName: string
  readonly toolCallId: ToolCallId
  readonly input: unknown
  readonly meta: unknown
}

/** Tool interceptor interface — evaluates policy before tool execution. */
export interface ToolInterceptor {
  readonly beforeExecute: (ctx: InterceptorContext) => Effect.Effect<InterceptorDecision, never, any>
}

/** Service tag for the tool interceptor. */
export class ToolInterceptorTag extends Context.Tag('ToolInterceptor')<
  ToolInterceptorTag, ToolInterceptor
>() {}

/** Resolves the active agent definition for a given fork. */
export type AgentResolver = (forkId: string | null) => RoleDefinition

export function buildPolicyInterceptor(
  resolveAgent: AgentResolver,
) {
  return (ctx: InterceptorContext) =>
    Effect.gen(function* () {
      const { forkId } = yield* ForkContext
      const agentDef = resolveAgent(forkId)
      const policyCtx = yield* (yield* PolicyContextProviderTag).get
      const toolKey = getDefKey(ctx.meta)
      if (toolKey === null) {
        return deny('Invalid tool metadata')
      }

      const hookCtx: ExecuteHookContext & { policyContext: RolesPolicyContext } = {
        toolCallId: ctx.toolCallId,
        toolName: ctx.toolName,
        toolKey,
        input: ctx.input,
        policyContext: {
          cwd: policyCtx.cwd,
          scratchpadPath: policyCtx.scratchpadPath,
          disableShellSafeguards: policyCtx.disableShellSafeguards,
          disableCwdSafeguards: policyCtx.disableCwdSafeguards,
        },
      }

      const decision = yield* evaluatePolicy(agentDef.policy, hookCtx)

      return decision
    })
}

function getDefKey(meta: unknown): string | null {
  const m = meta as { defKey?: unknown }
  return typeof m.defKey === 'string' ? m.defKey : null
}

function deny(message: string): InterceptorDecision<string> {
  return { _tag: 'Deny', denial: message }
}
