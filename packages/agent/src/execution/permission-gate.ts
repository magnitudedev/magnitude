/**
 * Policy Interceptor
 *
 * Evaluates the agent's tool policy for the current tool call and routes the decision.
 */

import { Effect } from 'effect'
import { Fork } from '@magnitudedev/event-core'
import type { InterceptorContext, InterceptorDecision } from '@magnitudedev/xml-act'
import { PermissionRejection } from './permission-rejection'
import type {
  RoleDefinition,
  ToolSet,
} from '@magnitudedev/roles'
import { PolicyContextProviderTag, type PolicyContext } from '../agents/types'
import { evaluate } from '../agents/policy'

const { ForkContext } = Fork

/** Resolves the active agent definition for a given fork. */
export type AgentResolver = (forkId: string | null) => RoleDefinition<ToolSet, string, PolicyContext>

export function buildPolicyInterceptor(
  resolveAgent: AgentResolver,
) {
  return (ctx: InterceptorContext) =>
    Effect.gen(function* () {
      const { forkId } = yield* ForkContext
      const agentDef = resolveAgent(forkId)
      const policyCtx = yield* (yield* PolicyContextProviderTag).get
      const defKey = getDefKey(ctx.meta)
      if (defKey === null) {
        return reject(PermissionRejection.Forbidden({ reason: 'Invalid tool metadata' }))
      }

      if (!(defKey in agentDef.tools)) {
        return reject(PermissionRejection.Forbidden({ reason: `Unknown tool: ${defKey}` }))
      }

      const decision = yield* evaluate(
        agentDef.policy,
        defKey,
        ctx.input,
        policyCtx,
      )

      if (decision.decision === 'allow') {
        return { _tag: 'Proceed' } satisfies InterceptorDecision
      }

      return reject(PermissionRejection.Forbidden({ reason: decision.reason }))
    })
}

function getDefKey(meta: unknown): string | null {
  const m = meta as { defKey?: unknown }
  return typeof m.defKey === 'string' ? m.defKey : null
}

function reject(rejection: unknown): InterceptorDecision {
  return { _tag: 'Reject', rejection }
}
