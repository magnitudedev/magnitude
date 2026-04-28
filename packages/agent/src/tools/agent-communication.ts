
/**
 * Agent Communication Tool
 *
 * Tool for sending messages between agent workers.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'

const MessageWorkerOutput = Schema.Struct({
  ok: Schema.Boolean,
})

export const messageWorkerTool = defineTool({
  name: 'messageWorker',
  group: 'agent',
  description: 'Send a message to another agent worker.',
  inputSchema: Schema.Struct({
    workerId: Schema.String.annotations({ description: 'ID of the worker to message' }),
    message: Schema.String.annotations({ description: 'Message content to send' }),
  }),
  outputSchema: MessageWorkerOutput,
  execute: (_input, _ctx) =>
    Effect.succeed({ ok: true }),
  label: (input) =>
    `message ${input.workerId ?? 'worker'}`,
})
