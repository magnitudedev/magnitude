/**
 * Shell Tool
 *
 * Execute a shell command. Supports detach_after for long-running commands.
 * Delegates all child process lifecycle to DetachedShellRegistry.
 */

import { Effect, Schema, Stream, Deferred, Fiber } from 'effect'
import { defineHarnessTool } from '@magnitudedev/harness'
import { Fork } from '@magnitudedev/event-core'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { agentEnv } from '../util/agent-env'
import { ToolErrorSchema } from './errors'
import { DetachedShellRegistry } from '../process/detached-process-registry'
import { AgentStateReaderTag } from './fork'
import { TurnContextTag } from '../engine/turn-context'
import { createId } from '../util/id'

const { ForkContext } = Fork

const DEFAULT_DETACH_AFTER = 30
const MIN_DETACH_AFTER = 0
const MAX_DETACH_AFTER = 60

// =============================================================================
// Types
// =============================================================================

const ShellCompletedOutput = Schema.Struct({
  mode: Schema.Literal('completed'),
  stdout: Schema.String,
  stderr: Schema.String,
  exitCode: Schema.Number,
})

const ShellDetachedOutput = Schema.Struct({
  mode: Schema.Literal('detached'),
  pid: Schema.Number,
  command: Schema.String,
  stdoutPath: Schema.String,
  stderrPath: Schema.String,
})

const ShellOutput = Schema.Union(ShellCompletedOutput, ShellDetachedOutput)

type ShellOutput = Schema.Schema.Type<typeof ShellOutput>

const ShellErrorSchema = ToolErrorSchema('ShellError', {})

// =============================================================================
// Tool
// =============================================================================

export const shellTool = defineHarnessTool({
  definition: {
    name: 'shell',
    description: `Execute a shell command. By default, commands are given ${DEFAULT_DETACH_AFTER} seconds to complete. If a command finishes within that time, you receive its output directly (mode: completed). If it takes longer, the command continues running in the background and you receive a detached result with paths to its output files — your turn continues and you can do other work. You will be notified via system message when the command finishes.

Use \`detach_after\` to control when this happens:
- For fast commands (ls, cat) expected to take <1m - it serves at an upper bound to prevent you from hanging on a command
- For slow commands, unknown-duration commands, or non-terminating processes, set it to 0 to detach immediately and unblock yourself

Do not use this for operations covered by built-in tools like read, grep, tree, write, edit, and web_fetch.`,
    inputSchema: Schema.Struct({
      command: Schema.String.annotations({ description: 'Shell command to execute' }),
      detach_after: Schema.optionalWith(
        Schema.Number.annotations({
          description: `Seconds to wait before detaching (default: ${DEFAULT_DETACH_AFTER}, min: ${MIN_DETACH_AFTER}, max: ${MAX_DETACH_AFTER}).`,
        }),
        { default: () => DEFAULT_DETACH_AFTER, exact: true },
      ),
    }),
    outputSchema: ShellOutput,
  },
  errorSchema: ShellErrorSchema,
  emissionSchema: Schema.Struct({
    type: Schema.Literal('shell_output'),
    stdout: Schema.String,
    stderr: Schema.String,
  }),
  execute: ({ command, detach_after }, ctx) =>
    Effect.gen(function* () {
      const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
      const { forkId } = yield* ForkContext
      const { turnId } = yield* TurnContextTag
      const agentStateReader = yield* AgentStateReaderTag
      const registry = yield* DetachedShellRegistry

      const agentState = yield* agentStateReader.getAgentState()
      const agent = forkId !== null
        ? [...agentState.agents.values()].find(a => a.forkId === forkId)
        : undefined
      const ownerAgentId = agent?.agentId

      const effectiveDetachAfter = Math.min(
        Math.max(detach_after, MIN_DETACH_AFTER),
        MAX_DETACH_AFTER
      )

      const handle = yield* registry.executeDetached({
        command,
        forkId,
        turnId,
        toolCallId: createId(),
        scratchpadPath: scratchpadPath ?? cwd,
        cwd,
        agentEnv: agentEnv(cwd, scratchpadPath),
        detachAfter: effectiveDetachAfter,
        ownerAgentId,
      })

      // Consume the output stream concurrently — emit shell_output events
      const streamFiber = yield* Effect.forkScoped(
        handle.outputStream.pipe(
          Stream.runForEach((chunk) =>
            ctx.emit({ type: 'shell_output', stdout: chunk.stdout, stderr: chunk.stderr })
          )
        )
      )

      // Wait for completion or detachment
      const result = yield* Deferred.await(handle.result)

      // Drain remaining stream chunks
      yield* Fiber.join(streamFiber)

      switch (result._tag) {
        case 'Completed':
          return { mode: 'completed' as const, stdout: result.stdout, stderr: result.stderr, exitCode: result.exitCode }
        case 'Detached':
          return { mode: 'detached' as const, pid: result.pid, command, stdoutPath: result.stdoutPath, stderrPath: result.stderrPath }
      }
    }),
})
