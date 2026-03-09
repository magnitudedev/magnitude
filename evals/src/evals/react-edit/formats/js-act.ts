/**
 * JS-ACT edit format — uses the js-act sandbox to apply edits via JavaScript string operations.
 */

import { Effect, Stream } from 'effect'
import { Schema } from '@effect/schema'
import {
  Sandbox,
  createTool,
  createToolGroup,
  createJournal,
  ExecutionEvent,
  type SandboxItem,
} from '@magnitudedev/js-act'
import type { EditFormat } from './types'

export const jsActFormat: EditFormat = {
  id: 'js-act',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    return [
      'You must respond with valid JavaScript code only. No markdown, no prose.',
      'Your code will be executed in a sandbox with two global functions:',
      '',
      '  read(filename)          — returns the file content as a string',
      '  write(filename, content) — writes the new content to the file',
      '',
      'Use JavaScript string operations to transform the file content.',
      'For example: .replace(), .split()/.join(), regex, template literals, etc.',
      '',
      'Use var for variable declarations (not const or let).',
      'Every statement must end with a semicolon.',
      'Do NOT use async/await.',
      'Do NOT use prose delimiters or raw string syntax.',
      'Use regular JavaScript strings (single or double quotes).',
      '',
      'Example:',
      '  var content = read("app.tsx");',
      '  content = content.replace("oldValue", "newValue");',
      '  write("app.tsx", content);',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string | Promise<string> {
    return applyJsAct(response, originalContent)
  },
}

async function applyJsAct(response: string, originalContent: string): Promise<string> {
  let writtenContent: string | null = null

  const mockRead = createTool({
    name: 'read',
    description: 'Read file content',
    inputSchema: Schema.Struct({ path: Schema.String }),
    outputSchema: Schema.String,
    argMapping: ['path'],
    execute: ({ path }: { path: string }) => {
      return Effect.succeed(originalContent)
    },
  })


  const mockWrite = createTool({
    name: 'write',
    description: 'Write content to file',
    inputSchema: Schema.Struct({
      path: Schema.String,
      content: Schema.String,
    }),
    outputSchema: Schema.Void,
    argMapping: ['path', 'content'],
    execute: ({ path, content }: { path: string; content: string }) => {
      writtenContent = content
      return Effect.succeed(undefined as void)
    },
  })


  const globalGroup = createToolGroup({
    name: 'default',
    description: 'File tools',
    tools: [mockRead, mockWrite] as any,

    global: true,
  })

  const tools: SandboxItem[] = [globalGroup]

  const program = Effect.scoped(
    Effect.gen(function* () {
      const journal = createJournal()
      const codeStream = Stream.make(response)

      const eventStream = Sandbox.stream(
        tools as unknown as readonly SandboxItem[],
        codeStream,
        {
          journal,
          nonToolTimeout: 10000,
        }
      )

      yield* eventStream.pipe(
        Stream.runForEach((event: ExecutionEvent) => Effect.sync(() => {
          // just consume events
        }))
      )
    })
  )

  await Effect.runPromise(program as Effect.Effect<void, any>)

  if (writtenContent === null) {
    throw new Error('JS-ACT sandbox completed but no write() call was made')
  }

  return writtenContent
}
