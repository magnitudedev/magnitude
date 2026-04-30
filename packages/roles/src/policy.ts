import { Effect } from 'effect'
import { resolve } from 'node:path'
import {
  classifyShellCommand,
  isGitAllowed,
  isPathWithin,
  writesStayWithin,
} from '@magnitudedev/shell-classifier'
import type { ExecuteHookContext, InterceptorDecision } from '@magnitudedev/harness'
import type { PolicyRule, PolicyContext } from './types'

type FullContext = ExecuteHookContext & { policyContext: PolicyContext }

const proceed: InterceptorDecision = { _tag: 'Proceed' }
const reject = (reason: string): InterceptorDecision => ({ _tag: 'Reject', rejection: reason })

function expandWorkspacePath(path: string, workspacePath: string): string {
  if (path === '$M' || path === '${M}') return workspacePath
  if (path.startsWith('$M/')) return workspacePath + path.slice(2)
  if (path.startsWith('${M}/')) return workspacePath + path.slice(4)
  return path
}

function agentEnv(cwd: string, workspacePath: string): Record<string, string> {
  return {
    ...(process.env as Record<string, string>),
    NO_COLOR: '1',
    PROJECT_ROOT: cwd,
    M: workspacePath,
  }
}

/** Deny forbidden shell commands (respects disableShellSafeguards). */
export function denyForbiddenCommands(): PolicyRule {
  return (ctx: FullContext) => {
    if (ctx.toolKey !== 'shell') return Effect.succeed(null)
    if (ctx.policyContext.disableShellSafeguards) return Effect.succeed(null)

    const input = ctx.input as { command: string }
    const classification = classifyShellCommand(input.command)
    if (classification.tier === 'forbidden') {
      return Effect.succeed(reject(classification.reason ?? 'Forbidden command'))
    }
    return Effect.succeed(null)
  }
}

/** Deny mutating git commands (respects disableShellSafeguards). */
export function denyMutatingGit(): PolicyRule {
  return (ctx: FullContext) => {
    if (ctx.toolKey !== 'shell') return Effect.succeed(null)
    if (ctx.policyContext.disableShellSafeguards) return Effect.succeed(null)

    const input = ctx.input as { command: string }
    if (isGitAllowed(input.command)) return Effect.succeed(null)

    return Effect.succeed(reject('Only read-only git commands are allowed'))
  }
}

/** Deny file writes outside allowed directories. */
export function denyWritesOutside(
  getDirs: (ctx: PolicyContext) => string[],
): PolicyRule {
  return (ctx: FullContext) => {
    const { policyContext } = ctx
    if (policyContext.disableCwdSafeguards) return Effect.succeed(null)

    const env = agentEnv(policyContext.cwd, policyContext.workspacePath)
    const roots = getDirs(policyContext)

    if (ctx.toolKey === 'shell') {
      const input = ctx.input as { command: string }
      if (!writesStayWithin(input.command, env, ...roots)) {
        return Effect.succeed(reject('Command targets paths outside allowed directories'))
      }
      return Effect.succeed(null)
    }

    if (ctx.toolKey === 'fileWrite' || ctx.toolKey === 'fileEdit') {
      const input = ctx.input as { path: string }
      const expandedPath = expandWorkspacePath(input.path, policyContext.workspacePath)
      const fullPath = resolve(policyContext.cwd, expandedPath)
      if (!isPathWithin(fullPath, env, ...roots)) {
        return Effect.succeed(reject('Cannot write files outside allowed directories'))
      }
      return Effect.succeed(null)
    }

    return Effect.succeed(null)
  }
}

/** Deny mass-destructive shell commands in specified protected directories. */
export function denyMassDestructiveIn(
  getDirs: (ctx: PolicyContext) => string[],
): PolicyRule {
  return (ctx: FullContext) => {
    if (ctx.toolKey !== 'shell') return Effect.succeed(null)
    const { policyContext } = ctx
    if (policyContext.disableShellSafeguards) return Effect.succeed(null)

    const input = ctx.input as { command: string }
    const classification = classifyShellCommand(input.command)
    if (classification.tier !== 'mass-destructive') return Effect.succeed(null)

    const env = agentEnv(policyContext.cwd, policyContext.workspacePath)
    const protectedRoots = getDirs(policyContext)

    const nonProtectedRoots = [policyContext.cwd, policyContext.workspacePath]
    if (writesStayWithin(input.command, env, ...nonProtectedRoots)) {
      return Effect.succeed(null)
    }

    const allRoots = [...nonProtectedRoots, ...protectedRoots]
    if (writesStayWithin(input.command, env, ...allRoots)) {
      return Effect.succeed(reject('Mass-destructive operations are not allowed in protected directories'))
    }

    return Effect.succeed(null)
  }
}

/** Allow all tool executions unconditionally. */
export function allowAll(): PolicyRule {
  return () => Effect.succeed(proceed)
}

/**
 * Evaluate policy rules in order. First non-null result wins.
 * If all rules abstain, default to reject.
 */
export function evaluatePolicy(
  rules: PolicyRule[],
  ctx: FullContext,
): Effect.Effect<InterceptorDecision> {
  return Effect.gen(function* () {
    for (const rule of rules) {
      const result = yield* rule(ctx)
      if (result !== null) return result
    }
    return reject('No matching policy rule')
  })
}
