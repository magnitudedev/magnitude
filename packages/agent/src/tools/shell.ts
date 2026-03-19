/**
 * Shell Tool
 *
 * Execute shell commands in a subprocess.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { Fork } from '@magnitudedev/event-core'
import { spawn } from 'child_process'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { ToolReminderTag } from '../execution/tool-reminder'
import { ToolExecutionContextTag } from '../execution/tool-execution-context'
import { BackgroundProcessRegistryTag } from '../processes/background-process-registry'

const { ForkContext } = Fork

import { AUTO_KILL_BUFFER_S, BACKGROUND_DETACH_MS, DEFAULT_TIMEOUT_S } from '../processes/constants'

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
  reason: Schema.Literal('background', 'timeout_exceeded'),
  pid: Schema.Number,
  stdout: Schema.String,
  stderr: Schema.String,
})

const ShellOutput = Schema.Union(ShellCompletedOutput, ShellDetachedOutput)

type ShellOutput = Schema.Schema.Type<typeof ShellOutput>

// =============================================================================
// Shell Tool
// =============================================================================

const ShellError = ToolErrorSchema('ShellError', {})


// TODO: If the normal tool result truncation triggers on a detached command's initial output,
// that output is lost to the agent. This should eventually be handled by a generic system
// that demotes large tool results to files automatically.
export const shellTool = createTool({
  name: 'shell',
  group: 'default',
  description: 'Execute a shell command. Long-running commands are automatically detached after the timeout with output tracking. Do NOT use &, nohup, disown, or other explicit backgrounding; these bypass tracking. Use kill <pid> to stop background processes. Do not use this for operations covered by built-in tools like fs-read, fs-search, fs-tree, fs-write, edit, and web-fetch.',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number.annotations({ description: "Expected duration in seconds (default: 10). If the command hasn't finished by this time, it is detached as a background process so you can address whether to kill it or wait." })),
    background: Schema.optional(Schema.Boolean.annotations({ description: "For commands that won't terminate on their own (servers, watchers). Detaches immediately." })),
  }),
  outputSchema: ShellOutput,
  errorSchema: ShellError,
  argMapping: ['command', 'timeout', 'background'],
  bindings: {
    openai: { type: 'custom', format: { type: 'grammar', syntax: 'regex', definition: '[\\s\\S]*' }, inputField: 'command' },
    xmlInput: { type: 'tag', attributes: [{ field: 'timeout', attr: 'timeout' }, { field: 'background', attr: 'background' }], body: 'command' },
    xmlOutput: {
      type: 'tag',
      childTags: [
        { tag: 'mode', field: 'mode' },
        { tag: 'reason', field: 'reason' },
        { tag: 'pid', field: 'pid' },
        { tag: 'stdout', field: 'stdout' },
        { tag: 'stderr', field: 'stderr' },
        { tag: 'exitCode', field: 'exitCode' },
      ]
    } as const,
  } as const,

  execute: ({ command, timeout, background }) => Effect.gen(function* () {
    const { cwd } = yield* WorkingDirectoryTag
    const { forkId } = yield* ForkContext
    const { turnId } = yield* ToolExecutionContextTag
    const registry = yield* BackgroundProcessRegistryTag

    let activeChild: ReturnType<typeof spawn> | null = null

    const result = yield* Effect.onInterrupt(Effect.tryPromise({
      try: async (): Promise<ShellOutput> => {
        const startedAt = Date.now()
        const shellPath = process.env.SHELL ?? '/bin/sh'

        return await new Promise<ShellOutput>((resolve) => {
          let stdout = ''
          let stderr = ''
          let settled = false
          let spawned = false
          const detachReason = background === true ? 'background' as const : 'timeout_exceeded' as const
          const detachAfterMs = background === true ? BACKGROUND_DETACH_MS : (timeout ?? DEFAULT_TIMEOUT_S) * 1000

          const child = spawn(shellPath, ['-c', command], {
            cwd,
            env: { ...process.env, NO_COLOR: '1' }
          })
          activeChild = child

          const finalizeCompleted = (exitCode: number, nextStdout = stdout, nextStderr = stderr) => {
            if (settled) return
            settled = true
            resolve({
              mode: 'completed',
              stdout: nextStdout,
              stderr: nextStderr,
              exitCode,
            })
          }

          child.stdout?.on('data', (chunk: Buffer | string) => {
            stdout += typeof chunk === 'string' ? chunk : chunk.toString('utf8')
          })

          child.stderr?.on('data', (chunk: Buffer | string) => {
            stderr += typeof chunk === 'string' ? chunk : chunk.toString('utf8')
          })

          child.once('spawn', () => {
            spawned = true
          })

          child.once('error', (error) => {
            const message = spawned ? String(error) : (error instanceof Error ? error.message : String(error))
            finalizeCompleted(1, stdout, stderr.length > 0 ? stderr : message)
          })

          child.once('exit', (code) => {
            finalizeCompleted(code ?? 1)
          })

          const detachTimer = setTimeout(async () => {
            if (settled) return

            if (child.exitCode !== null || child.signalCode !== null) {
              finalizeCompleted(child.exitCode ?? 1)
              return
            }

            const pid = child.pid
            if (pid == null) {
              finalizeCompleted(1, stdout, stderr.length > 0 ? stderr : 'Failed to determine process id for detached process')
              return
            }

            const initialStdout = stdout
            const initialStderr = stderr

            await Effect.runPromise(
              registry.register({
                pid,
                forkId,
                turnId,
                command,
                reason: detachReason,
                startedAt,
                child,
                timeoutSeconds: timeout ?? DEFAULT_TIMEOUT_S,
                initialStdout,
                initialStderr,
              })
            )

            settled = true
            resolve({
              mode: 'detached',
              reason: detachReason,
              pid,
              stdout: initialStdout,
              stderr: initialStderr,
            })
          }, detachAfterMs)

          const clear = () => clearTimeout(detachTimer)
          child.once('exit', clear)
          child.once('error', clear)
          child.once('close', clear)
        })
      },
      catch: (e) => ({ _tag: 'ShellError' as const, message: e instanceof Error ? e.message : String(e) }),
    }), () => Effect.sync(() => {
      if (activeChild && activeChild.exitCode === null) {
        try { activeChild.kill('SIGTERM') } catch {}
      }
    }))

    if (result.mode === 'detached') {
      const reminder = yield* ToolReminderTag
      const message = result.reason === 'background'
        ? `Background process started (PID ${result.pid}). You will see its stdout/stderr output in your system context each turn. Use \`kill ${result.pid}\` to stop it.`
        : `Command exceeded expected timeout and was detached (PID ${result.pid}). This process will be automatically terminated in ${(timeout ?? DEFAULT_TIMEOUT_S) + AUTO_KILL_BUFFER_S}s. If the process is hanging, kill it now with \`kill ${result.pid}\`. If it is still working correctly and you want it to keep running, declare it as a background process with \`<shell-bg pid="${result.pid}" />\`.`
      yield* reminder.add(message)
    }

    return result
  }),
})

// Tool slugs
export const SHELL_TOOLS = ['default.shell'] as const