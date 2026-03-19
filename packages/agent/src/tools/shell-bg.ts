import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { BackgroundProcessRegistryTag } from '../processes/background-process-registry'

const ShellBgOutput = Schema.Struct({
  status: Schema.String,
  pid: Schema.Number,
})

const ShellBgError = ToolErrorSchema('ShellBgError', {})

export const shellBgTool = createTool({
  name: 'shell-bg',
  group: 'default',
  description: 'Promote a detached process to a background process. Use this when a command exceeded its timeout but is still working correctly and you want it to keep running. This cancels the automatic termination deadline.',
  inputSchema: Schema.Struct({
    pid: Schema.Number,
  }),
  outputSchema: ShellBgOutput,
  errorSchema: ShellBgError,
  argMapping: ['pid'],
  bindings: {
    xmlInput: { type: 'tag', attributes: [{ field: 'pid', attr: 'pid' }] },
    xmlOutput: {
      type: 'tag',
      childTags: [
        { tag: 'status', field: 'status' },
        { tag: 'pid', field: 'pid' },
      ]
    } as const,
  } as const,
  execute: ({ pid }) => Effect.gen(function* () {
    const registry = yield* BackgroundProcessRegistryTag
    const promoted = yield* registry.promote(pid)

    if (!promoted.success) {
      const message = promoted.reason === 'not_found'
        ? `No tracked process with PID ${pid}`
        : promoted.reason === 'already_exited'
          ? `Process PID ${pid} has already exited`
          : `Process PID ${pid} is already a background process`
      return yield* Effect.fail({ _tag: 'ShellBgError' as const, message })
    }

    return {
      status: 'promoted',
      pid,
    }
  }),
})

export const SHELL_BG_TOOLS = ['default.shell-bg'] as const
