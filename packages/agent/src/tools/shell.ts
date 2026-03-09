/**
 * Shell Tool
 *
 * Execute shell commands in a subprocess.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { exec } from 'child_process'
import { promisify } from 'util'
import { WorkingDirectoryTag } from '../execution/working-directory'

const execAsync = promisify(exec)

// =============================================================================
// Types
// =============================================================================

const ShellOutput = Schema.Struct({
  stdout: Schema.String,
  stderr: Schema.String,
  exitCode: Schema.Number
})

type ShellOutput = Schema.Schema.Type<typeof ShellOutput>

// =============================================================================
// Shell Tool
// =============================================================================

const ShellError = ToolErrorSchema('ShellError', {})

export const shellTool = createTool({
  name: 'shell',
  group: 'default',
  description: 'Execute a shell command. Do not use this for operations covered by built-in tools like fs-read, fs-search, fs-tree, fs-write, edit, and webFetch.',
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
        { tag: 'stdout', field: 'stdout' },
        { tag: 'stderr', field: 'stderr' },
        { tag: 'exitCode', field: 'exitCode' },
      ]
    } as const,
  } as const,

  execute: ({ command }) => Effect.gen(function* () {
    const { cwd } = yield* WorkingDirectoryTag

    return yield* Effect.tryPromise({
      try: async (): Promise<ShellOutput> => {
        try {
          const { stdout, stderr } = await execAsync(command, {
            cwd,
            timeout: 30000,
            maxBuffer: 10 * 1024 * 1024 // 10MB
          })
          return { stdout, stderr, exitCode: 0 }
        } catch (error) {
          // exec throws on non-zero exit code
          const execError = error as {
            stdout?: string
            stderr?: string
            code?: number
            killed?: boolean
            signal?: string
          }

          if (execError.killed || execError.signal === 'SIGTERM') {
            return {
              stdout: execError.stdout ?? '',
              stderr: 'Command timed out',
              exitCode: 124
            }
          }

          return {
            stdout: execError.stdout ?? '',
            stderr: execError.stderr ?? String(error),
            exitCode: execError.code ?? 1
          }
        }
      },
      catch: (e) => ({ _tag: 'ShellError' as const, message: e instanceof Error ? e.message : String(e) }),
    })
  }),
})

// Tool slugs
export const SHELL_TOOLS = ['default.shell'] as const
