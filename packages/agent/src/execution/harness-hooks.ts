/**
 * Shared harness hooks used by Cortex and CompactionWorker.
 *
 * Centralises beforeExecute (policy gate) and afterExecute (result persistence).
 * Formatting has moved to the agent's prompt construction layer.
 */

import { Effect } from 'effect'
import { logger } from '@magnitudedev/logger'
import type { ExecuteHookContext, HarnessHooks, InterceptorDecision, ToolResult } from '@magnitudedev/harness'
import * as path from 'path'

import { PolicyContextProviderTag } from '../agents/types'
import { persistResult } from '../runtime/result-persistence'
import type { getAgentDefinition } from '../agents/registry'

export interface StandardHooksContext {
  readonly forkId: string | null
  readonly turnId: string
  readonly agentDef: ReturnType<typeof getAgentDefinition>
  readonly scratchpadPath: string
}

export function buildStandardHooks(ctx: StandardHooksContext): HarnessHooks<PolicyContextProviderTag> {
  const { forkId, turnId, agentDef, scratchpadPath } = ctx
  const resultsDir = path.join(scratchpadPath, 'results')

  return {
    beforeExecute: (hookCtx: ExecuteHookContext) =>
      Effect.gen(function* () {
        const policyCtxProvider = yield* PolicyContextProviderTag
        const policyContext = yield* policyCtxProvider.get

        for (const rule of agentDef.policy) {
          const decision = yield* rule({ ...hookCtx, policyContext })
          if (decision !== null) return decision
        }
        return { _tag: 'Proceed' as const } satisfies InterceptorDecision
      }),

    afterExecute: (hookCtx: ExecuteHookContext & { readonly result: ToolResult }) =>
      Effect.gen(function* () {
        if (hookCtx.result._tag === 'Success') {
          yield* persistResult(hookCtx.result.output, turnId, hookCtx.toolCallId, resultsDir).pipe(
            Effect.catchAll((e) => Effect.gen(function* () {
              logger.warn({ forkId, turnId, toolCallId: hookCtx.toolCallId, e }, '[Harness] persistResult failed')
            })),
          )
        }
      }),
  }
}
