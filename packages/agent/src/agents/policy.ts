import { Effect } from 'effect'
import type { Decision, Policy, PolicyHandler } from '@magnitudedev/roles'
import type { ToolCatalog } from '@magnitudedev/tools'
import type { Schema } from '@effect/schema'
import {
  classifyShellCommand,
  isGitAllowed,
  isPathWithin,
  writesStayWithin,
} from '@magnitudedev/shell-classifier'
import { resolve } from 'node:path'
import type { PolicyContext } from './types'
import { agentEnv } from '../util/agent-env'
import { editTool, writeTool } from '../tools/fs'
import { shellTool } from '../tools/shell'
import { expandWorkspacePath } from '../workspace/workspace-path'

type ShellInput = Schema.Schema.Type<typeof shellTool.inputSchema>
type FileWriteInput = Schema.Schema.Type<typeof writeTool.inputSchema>
type FileEditInput = Schema.Schema.Type<typeof editTool.inputSchema>

const NO_MATCHING_POLICY_RULE = 'No matching policy rule'
const deny = (reason: string): Decision => ({ decision: 'deny', reason })
const allow: Decision = { decision: 'allow' }

export function evaluate(
  policy: Policy<ToolCatalog, unknown>,
  tool: string,
  input: unknown,
  ctx: unknown,
): Effect.Effect<Decision> {
  const handlers: Array<Effect.Effect<Decision | null>> = []

  for (const fragment of policy) {
    const handlersByKey = fragment as Record<string, PolicyHandler<unknown, unknown> | undefined>
    const toolHandler = handlersByKey[tool]
    const wildcardHandler = handlersByKey['*']

    if (toolHandler) {
      handlers.push(toolHandler(input, ctx))
      continue
    }

    if (wildcardHandler) {
      handlers.push(wildcardHandler(input, ctx))
    }
  }

  if (handlers.length === 0) {
    return Effect.succeed(deny(NO_MATCHING_POLICY_RULE))
  }

  // All handlers evaluate concurrently; deny takes precedence over allow.
  // Current handlers are all synchronous (Effect.succeed). If async handlers
  // are added (e.g. approval flows), consider sequential evaluation with early deny exit.
  return Effect.gen(function* () {
    const results = yield* Effect.all(handlers, { concurrency: 'unbounded' })
    const decisions = results.filter((result): result is Decision => result !== null)

    const denied = decisions.find((d) => d.decision === 'deny')
    if (denied) return denied

    const allowed = decisions.find((d) => d.decision === 'allow')
    if (allowed) return allowed

    return deny(NO_MATCHING_POLICY_RULE)
  })
}

/** Deny forbidden shell commands (respects disableShellSafeguards). */
export function denyForbiddenCommands() {
  return {
    shell: (input: ShellInput, ctx: PolicyContext) => {
      if (ctx.disableShellSafeguards) return Effect.succeed(null)

      const classification = classifyShellCommand(input.command)
      if (classification.tier === 'forbidden') {
        return Effect.succeed(deny(classification.reason ?? 'Forbidden command'))
      }

      return Effect.succeed(null)
    },
  }
}

/** Deny mutating git commands (respects disableShellSafeguards). */
export function denyMutatingGit() {
  return {
    shell: (input: ShellInput, ctx: PolicyContext) => {
      if (ctx.disableShellSafeguards) return Effect.succeed(null)
      if (isGitAllowed(input.command)) return Effect.succeed(null)

      return Effect.succeed(deny('Only read-only git commands are allowed'))
    },
  }
}

/** Explicitly allow readonly shell commands. */
export function allowReadonlyShell() {
  return {
    shell: (input: ShellInput) => {
      const classification = classifyShellCommand(input.command)
      if (classification.tier === 'readonly') {
        return Effect.succeed(allow)
      }
      return Effect.succeed(null)
    },
  }
}

function checkPathBounds(
  path: string,
  ctx: PolicyContext,
  roots: (ctx: PolicyContext) => string[],
  operation: string,
): Decision | null {
  if (ctx.disableCwdSafeguards) return null
  const expandedPath = expandWorkspacePath(path, ctx.workspacePath)
  const fullPath = resolve(ctx.cwd, expandedPath)
  const env = agentEnv(ctx.cwd, ctx.workspacePath)
  if (!isPathWithin(fullPath, env, ...roots(ctx))) {
    return deny(`Cannot ${operation} files outside allowed directories`)
  }
  return null
}

/** Deny shell/fileWrite/fileEdit writes targeting paths outside allowed roots. */
export function denyWritesOutside(
  roots: (ctx: PolicyContext) => string[],
) {
  return {
    shell: (input: ShellInput, ctx: PolicyContext) => {
      if (ctx.disableCwdSafeguards) return Effect.succeed(null)

      const env = agentEnv(ctx.cwd, ctx.workspacePath)
      if (!writesStayWithin(input.command, env, ...roots(ctx))) {
        return Effect.succeed(deny('Command targets paths outside allowed directories'))
      }

      return Effect.succeed(null)
    },

    fileWrite: (input: FileWriteInput, ctx: PolicyContext) =>
      Effect.succeed(checkPathBounds(input.path, ctx, roots, 'write')),

    fileEdit: (input: FileEditInput, ctx: PolicyContext) =>
      Effect.succeed(checkPathBounds(input.path, ctx, roots, 'edit')),
  }
}

/** Deny mass-destructive shell commands that target specific protected roots. */
export function denyMassDestructiveIn(
  roots: (ctx: PolicyContext) => string[],
) {
  return {
    shell: (input: ShellInput, ctx: PolicyContext) => {
      if (ctx.disableShellSafeguards) return Effect.succeed(null)

      const classification = classifyShellCommand(input.command)
      if (classification.tier !== 'mass-destructive') return Effect.succeed(null)

      const env = agentEnv(ctx.cwd, ctx.workspacePath)
      const protectedRoots = roots(ctx)

      const nonProtectedRoots = [ctx.cwd, ctx.workspacePath]
      if (writesStayWithin(input.command, env, ...nonProtectedRoots)) {
        return Effect.succeed(null)
      }

      const allRoots = [...nonProtectedRoots, ...protectedRoots]
      if (writesStayWithin(input.command, env, ...allRoots)) {
        return Effect.succeed(deny('Mass-destructive operations are not allowed in protected directories'))
      }

      return Effect.succeed(null)
    },
  }
}

/** Catch-all allow fragment. */
export function allowAll() {
  return {
    '*': () => Effect.succeed(allow),
  }
}
