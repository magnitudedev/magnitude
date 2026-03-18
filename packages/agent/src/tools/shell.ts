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

import { DETACH_AFTER_MS } from '../processes/constants'

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
  description: 'Execute a shell command. Long-running commands are automatically detached after 5s with output tracking. Do NOT use &, nohup, disown, or other explicit backgrounding; these bypass tracking. Just run commands normally and the system will handle long-running processes automatically. Use kill <pid> to stop background processes. Do not use this for operations covered by built-in tools like fs-read, fs-search, fs-tree, fs-write, edit, and web-fetch.',
  inputSchema: Schema.Struct({ command: Schema.String }),
  outputSchema: ShellOutput,
  errorSchema: ShellError,
  argMapping: ['command'],
  bindings: {
    openai: { type: 'custom', format: { type: 'grammar', syntax: 'regex', definition: '[\\s\\S]*' }, inputField: 'command' },
    xmlInput: { type: 'tag', body: 'command' },
    xmlOutput: {
      type: 'tag',
      childTags: [
        { tag: 'mode', field: 'mode' },
        { tag: 'pid', field: 'pid' },
        { tag: 'stdout', field: 'stdout' },
        { tag: 'stderr', field: 'stderr' },
        { tag: 'exitCode', field: 'exitCode' },
      ]
    } as const,
  } as const,

  execute: ({ command }) => Effect.gen(function* () {
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
                startedAt,
                child,
                initialStdout,
                initialStderr,
              })
            )

            settled = true
            resolve({
              mode: 'detached',
              pid,
              stdout: initialStdout,
              stderr: initialStderr,
            })
          }, DETACH_AFTER_MS)

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
      yield* reminder.add(`Background process detached (PID ${result.pid}). You will see its stdout/stderr output in your system context each turn. Use \`kill ${result.pid}\` to stop it.`)
    }

    return result
  }),
})

// Tool slugs
export const SHELL_TOOLS = ['default.shell'] as const