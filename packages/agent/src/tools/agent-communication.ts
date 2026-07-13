/**
 * Agent Communication Tool
 */

import { Effect, Option, Schema } from 'effect'
import { defineHarnessTool } from '@magnitudedev/harness'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import { TurnContextTag } from '../engine/turn-context'
import { createId } from '@magnitudedev/generate-id'
import type { AppEvent } from '../events'

const { ForkContext } = Fork

const MessageWorkerOutput = Schema.Struct({
  ok: Schema.Boolean,
  yield: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
})

export const messageWorkerTool = defineHarnessTool({
  definition: {
    name: 'message_worker',
    description: 'Send a message to a worker. This is the only way to communicate with workers besides the initial message in spawn worker. Messages are queued and will be seen by the worker at the first possible opportunity. Any message in prose is NEVER seen by workers, prose messages are only seen by the user, messages to workers must exclusively be sent via this tool.',
    inputSchema: Schema.Struct({
      agentId: Schema.String.annotations({ description: 'Agent ID of the worker to message' }),
      message: Schema.String.annotations({ description: 'Message content to send' }),
      yield: Schema.optionalWith(Schema.Boolean.annotations({ description: 'When true, yield to this worker — the turn will not retrigger.' }), { as: 'Option', exact: true }),
    }),
    outputSchema: MessageWorkerOutput,
  },
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      const { forkId } = yield* ForkContext
      const { turnId } = yield* TurnContextTag
      const bus = yield* WorkerBusTag<AppEvent>()
      const messageId = createId()

      yield* bus.publish({
        type: 'message_start',
        forkId,
        turnId,
        id: messageId,
        destination: { kind: 'worker', agentId: input.agentId },
      })

      yield* bus.publish({
        type: 'message_chunk',
        forkId,
        turnId,
        id: messageId,
        text: input.message,
      })

      yield* bus.publish({
        type: 'message_end',
        forkId,
        turnId,
        id: messageId,
      })

      return { ok: true, yield: Option.filter(input.yield, Boolean) }
    }),
})
