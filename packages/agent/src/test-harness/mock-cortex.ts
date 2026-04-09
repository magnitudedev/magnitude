import { Cause, Duration, Effect, Queue, Stream } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { TransportError } from '@magnitudedev/providers'
import { ExecutionManager } from '../execution/execution-manager'
import { createTurnStream } from '../execution/types'
import { drainTurnEventStream } from '../workers/turn-event-drain'
import { MockTurnScriptTag, type MockTurnResponse } from './turn-script'
import { TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'

function frameToChunks(frame: MockTurnResponse): readonly string[] {
  if (frame.xmlChunks && frame.xmlChunks.length > 0) return frame.xmlChunks
  if (frame.xml !== undefined) return [frame.xml]
  return [`<message>ok</message>${TURN_CONTROL_IDLE}`]
}

function buildStream(frame: MockTurnResponse): Stream.Stream<string, import('@magnitudedev/providers').ModelError> {
  const chunks = frameToChunks(frame)
  const effective = frame.terminateStreamEarly ? chunks.slice(0, Math.max(0, chunks.length - 1)) : chunks

  return Stream.fromIterable(effective).pipe(
    Stream.zipWithIndex,
    Stream.mapEffect(([chunk, idx]) => Effect.gen(function* () {
      const chunkNum = idx + 1
      if (frame.failAfterChunk !== undefined && chunkNum > frame.failAfterChunk) {
        return yield* Effect.fail(new TransportError({ message: `MockTurnScript failAfterChunk=${frame.failAfterChunk}`, status: null }))
      }
      const delayMs = frame.delayMsBetweenChunks ?? 0
      if (delayMs > 0 && chunkNum > 1) {
        yield* Effect.sleep(Duration.millis(delayMs))
      }
      return chunk
    }))
  )
}

export const MockCortex = Worker.defineForked<AppEvent>()({
  name: 'MockCortex',

  forkLifecycle: {
    activateOn: 'agent_created',
  },

  eventHandlers: {
    turn_started: (event, publish) => {
      const { forkId, turnId, chainId } = event
      const rawCodeChunks: string[] = []

      return Effect.gen(function* () {
        const script = yield* MockTurnScriptTag
        const execManager = yield* ExecutionManager
        const frame = yield* script.dequeue({ forkId, turnId })

        const xmlStream = buildStream(frame).pipe(
          Stream.tap((chunk) => Effect.sync(() => { rawCodeChunks.push(chunk) }))
        )

        const turnStream = createTurnStream((queue) => Effect.gen(function* () {
          const executeResult = yield* execManager.execute(
            xmlStream,
            {
              forkId,
              turnId,
              chainId,
              defaultProseDest: forkId === null ? 'user' : 'parent',
              allowSingleUserReplyThisTurn: false,
            },
            queue,
          )

          const usage = {
            inputTokens: frame.usage?.inputTokens ?? null,
            outputTokens: frame.usage?.outputTokens ?? null,
            cacheReadTokens: frame.usage?.cacheReadTokens ?? null,
            cacheWriteTokens: frame.usage?.cacheWriteTokens ?? null,
            inputCost: null,
            outputCost: null,
            totalCost: null,
          }

          yield* Queue.offer(queue, { _tag: 'TurnResult', value: { executeResult, usage, rawCodeChunks } })
        }))

        const drained = yield* drainTurnEventStream(turnStream, forkId, turnId, publish)
        const { executeResult, usage } = drained.finalResult

        yield* publish({
          type: 'turn_completed',
          forkId,
          turnId,
          chainId,
          strategyId: 'xml-act',
          result: executeResult.result,
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
          cacheReadTokens: usage.cacheReadTokens,
          cacheWriteTokens: usage.cacheWriteTokens,
          providerId: null,
          modelId: null,
        })
      }).pipe(
        Effect.onInterrupt(() => {
          return publish({
            type: 'turn_completed',
            forkId,
            turnId,
            chainId,
            strategyId: 'xml-act',
            result: { success: false, error: 'Interrupted', cancelled: true },
            inputTokens: null,
            outputTokens: null,
            cacheReadTokens: null,
            cacheWriteTokens: null,
            providerId: null,
            modelId: null,
          })
        }),
        Effect.catchAllCause((cause) => {
          const failure = Cause.failureOption(cause)
          const defect = Cause.dieOption(cause)
          const message = failure?._tag === 'Some'
            ? (failure.value instanceof Error ? failure.value.message : String(failure.value))
            : defect?._tag === 'Some'
              ? (defect.value instanceof Error ? defect.value.message : String(defect.value))
              : Cause.pretty(cause)

          return publish({
            type: 'turn_unexpected_error',
            forkId,
            turnId,
            message: `MockCortex turn failed: ${message}`,
          })
        })
      )
    }
  }
})