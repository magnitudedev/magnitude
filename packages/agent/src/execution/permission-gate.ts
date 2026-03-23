/**
 * Permission Interceptor
 *
 * Builds a ToolInterceptor for xml-act from the agent's permission policy.
 * Replaces the per-tool js-act Gate system with a single interceptor that:
 * - Resolves the agent definition dynamically per fork
 * - Consults the agent's permission policy (allow/reject/approve)
 * - Enforces universal shell rules (CWD boundary, git allowlist)
 * - Routes approval requests through the ApprovalStateService
 */

import { Effect } from 'effect'
import { Fork } from '@magnitudedev/event-core'
import { join } from 'path'
import type { InterceptorContext, InterceptorDecision } from '@magnitudedev/xml-act'
import { ApprovalStateTag } from './approval-state'
import { PermissionRejection } from './permission-rejection'
import { classifyShellCommand, writesStayWithin, isGitAllowed } from '@magnitudedev/shell-classifier'
import { validateAndApply, toEditDiff } from '../util/edit'
import type { ToolDisplay } from '../events'
import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import { PolicyContextProviderTag, type PolicyContext } from '../agents/types'

const { ForkContext } = Fork

/** Resolves the active agent definition for a given fork. */
export type AgentResolver = (forkId: string | null) => RoleDefinition<ToolSet, string, PolicyContext>

/**
 * Build a ToolInterceptor that enforces the agent's permission policy.
 *
 * The interceptor accesses ForkContext, PolicyContextProviderTag, and ApprovalStateTag
 * from the Effect context — all provided via the fork layers.
 */
export function buildPermissionInterceptor(
  resolveAgent: AgentResolver,
) {
  return (ctx: InterceptorContext) =>
    Effect.gen(function* () {
        const approvalState = yield* ApprovalStateTag
        const { forkId } = yield* ForkContext

        const agentDef = resolveAgent(forkId)
        const defKey = (ctx.meta as { defKey: string }).defKey

        const policyCtx = yield* (yield* PolicyContextProviderTag).get

        // Consult agent definition's permission policy
        const result = agentDef.getPermission(defKey, ctx.input, policyCtx)

        // Allow — apply universal shell enforcement before proceeding
        if (result.decision === 'allow') {
          if (defKey === 'shell') {
            const input = ctx.input as { command: string }
            const result = classifyShellCommand(input.command)

            if (result.tier === 'normal') {
              const allowedPrefixes = [policyCtx.workspacePath]
              if (!policyCtx.disableCwdSafeguards && !writesStayWithin(input.command, policyCtx.cwd, ...(allowedPrefixes ?? []))) {
                return reject(PermissionRejection.Forbidden({
                  reason: 'Non read-only shell commands outside the working directory are not allowed.'
                }))
              }
              if (!policyCtx.disableShellSafeguards && !isGitAllowed(input.command)) {
                return reject(PermissionRejection.Forbidden({
                  reason: 'Only read-only git commands are allowed (status, log, diff, etc).'
                }))
              }
            }
          }
          return { _tag: 'Proceed' } satisfies InterceptorDecision
        }

        const reason = result.reason ?? ''

        // Reject — map to appropriate rejection type
        if (result.decision === 'reject') {
          return reject(PermissionRejection.Forbidden({ reason }))
        }

        // Fork agents — auto-reject instead of requesting approval
        if (forkId !== null) {
          return reject(PermissionRejection.Forbidden({
            reason: `This action requires user approval and cannot be performed in a background agent. ${reason}. Use the tools available to you or find an alternative approach.`
          }))
        }

        // Compute display data for edit before blocking on approval
        let display: ToolDisplay | undefined
        if (defKey === 'fileEdit') {
          try {
            const input = ctx.input as { path: string; oldString: string; newString: string; replaceAll?: boolean }
            const fullPath = join(policyCtx.cwd, input.path)
            const content = yield* Effect.promise(() => Bun.file(fullPath).text())
            const applied = validateAndApply(content, input.oldString, input.newString, input.replaceAll ?? false)
            display = { type: 'edit_diff' as const, path: input.path, diffs: [toEditDiff(applied, applied.result)] }
          } catch {
            // Fail silently — approval card shows without diff
          }
        }

        // Approve — request async approval (orchestrator only)
        const approvalDecision = yield* approvalState.requestApproval(
          ctx.toolCallId,
          forkId,
          defKey,
          ctx.input,
          reason,
          display
        )

        if (approvalDecision === 'approved') return { _tag: 'Proceed' } satisfies InterceptorDecision
        return reject(PermissionRejection.UserRejection({ reason: 'User rejected the action' }))
      })
}

function reject(rejection: unknown): InterceptorDecision {
  return { _tag: 'Reject', rejection }
}
