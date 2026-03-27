/**
 * Shell Tool
 *
 * Execute shell commands in a subprocess.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { spawn } from 'child_process'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { agentEnv } from '../util/agent-env'

const DEFAULT_TIMEOUT_S = 120
const MAX_TIMEOUT_S = 600

// =============================================================================
// Types
// =============================================================================

const ShellOutput = Schema.Struct({
  mode: Schema.Literal('completed'),
  stdout: Schema.String,
  stderr: Schema.String,
  exitCode: Schema.Number,
})

type ShellOutput = Schema.Schema.Type<typeof ShellOutput>

const ShellErrorSchema = ToolErrorSchema('ShellError', {})

// =============================================================================
// Shell Tool
// =============================================================================

const shortenCommandPreview = (command: string, maxLength = 80): string => {
  const normalized = command.replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, Math.max(0, maxLength - 3))}...`
}

export const shellTool = defineTool({
  name: 'shell',
  group: 'default',
  description: 'Execute a shell command. Do not use this for operations covered by built-in tools like read, grep, tree, write, edit, and web-fetch.',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(
      Schema.Number.annotations({ description: 'Timeout in seconds (default: 120, max: 600).' })
    ),
  }),
  outputSchema: ShellOutput,
  errorSchema: ShellErrorSchema,
  execute: ({ command, timeout }, _ctx) =>
    Effect.gen(function* () {
      const { cwd, workspacePath } = yield* WorkingDirectoryTag
      let activeChild: ReturnType<typeof spawn> | null = null
      const effectiveTimeout = Math.min(Math.max(timeout ?? DEFAULT_TIMEOUT_S, 1), MAX_TIMEOUT_S)

      return yield* Effect.onInterrupt(
        Effect.tryPromise({
          try: async (): Promise<ShellOutput> => {
            const shellPath = process.env.SHELL ?? '/bin/sh'

            return await new Promise<ShellOutput>((resolve) => {
              let stdout = ''
              let stderr = ''
              let settled = false
              let spawned = false
              let killTimer: ReturnType<typeof setTimeout> | null = null
              let graceTimer: ReturnType<typeof setTimeout> | null = null

              const child = spawn(shellPath, ['-c', command], {
                cwd,
                env: agentEnv(cwd, workspacePath),
              })
              activeChild = child

              const clearTimers = () => {
                if (killTimer) {
                  clearTimeout(killTimer)
                  killTimer = null
                }
                if (graceTimer) {
                  clearTimeout(graceTimer)
                  graceTimer = null
                }
              }

              const finalize = (exitCode: number, nextStdout = stdout, nextStderr = stderr) => {
                if (settled) return
                settled = true
                clearTimers()
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
                const message = spawned ? String(error) : error instanceof Error ? error.message : String(error)
                finalize(1, stdout, stderr.length > 0 ? stderr : message)
              })

              child.once('exit', (code) => {
                finalize(code ?? 1)
              })

              killTimer = setTimeout(() => {
                if (settled) return
                try {
                  child.kill('SIGINT')
                } catch {}

                graceTimer = setTimeout(() => {
                  if (settled) return
                  try {
                    child.kill('SIGTERM')
                  } catch {}

                  graceTimer = setTimeout(() => {
                    const timeoutMessage = `Command timed out after ${effectiveTimeout}s and was terminated.`
                    const timedOutStderr =
                      stderr.length > 0 ? `${stderr}\n${timeoutMessage}` : timeoutMessage
                    finalize(124, stdout, timedOutStderr)
                  }, 1000)
                }, 1000)
              }, effectiveTimeout * 1000)

              child.once('close', clearTimers)
            })
          },
          catch: (e) => ({
            _tag: 'ShellError',
            message: e instanceof Error ? e.message : String(e),
          }),
        }),
        () =>
          Effect.sync(() => {
            if (activeChild && activeChild.exitCode === null) {
              try {
                activeChild.kill('SIGTERM')
              } catch {}
            }
          })
      )
    }),
  label: (input) => (input.command ? `$ ${shortenCommandPreview(input.command)}` : 'Running command…'),
})

export const shellXmlBinding = defineXmlBinding(shellTool, {
  input: {
    attributes: [{ attr: 'timeout', field: 'timeout' }],
    body: 'command',
  },
  output: {
    childTags: [
      { tag: 'mode', field: 'mode' },
      { tag: 'stdout', field: 'stdout' },
      { tag: 'stderr', field: 'stderr' },
      { tag: 'exitCode', field: 'exitCode' },
    ],
  },
})

// Tool slugs
export const SHELL_TOOLS = ['default.shell'] as const
