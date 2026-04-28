import { describe, it, expect } from 'vitest'
import { Effect, Scope, Stream } from 'effect'
import type { TurnEngineCrash } from '@magnitudedev/xml-act'
import { createTurnStream } from '../../src/execution/turn-stream'
import { TurnError } from '../../src/execution/types'
import type { TurnEvent, TurnEventSink, TurnStrategyResult } from '../../src/execution/types'

const finalTurnResult: TurnStrategyResult = {
  executeResult: { result: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [] } } },
  usage: {
    inputTokens: 1,
    outputTokens: 1,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    inputCost: 0,
    outputCost: 0,
    totalCost: 0,
  },
}

function drainForTurnResultSlow(
  turnStream: Stream.Stream<TurnEvent, TurnEngineCrash | TurnError, Scope.Scope>,
): Effect.Effect<{ finalResult: TurnStrategyResult | null; seen: string[] }, TurnEngineCrash | TurnError> {
  return Effect.gen(function* () {
    let finalResult: TurnStrategyResult | null = null
    const seen: string[] = []

    yield* Effect.scoped(
      turnStream.pipe(
        Stream.runForEach((event) =>
          Effect.gen(function* () {
            seen.push(event._tag)
            if (event._tag === 'TurnResult') {
              finalResult = event.value
            }
            yield* Effect.sleep('1 millis')
          }),
        ),
      ),
    )

    return { finalResult, seen }
  })
}

describe('createTurnStream queue race', () => {
  it('should still deliver TurnResult even when shutdown happens with heavy queue backlog', async () => {
    const turnStream = createTurnStream((sink: TurnEventSink) =>
      Effect.gen(function* () {
        yield* sink.emit({ _tag: 'RawResponseChunk', text: 'hello world' })
        yield* sink.emit({ _tag: 'MessageStart', id: 'msg-1', destination: { kind: 'user' } })

        for (let i = 0; i < 10000; i++) {
          yield* sink.emit({ _tag: 'MessageChunk', id: 'msg-1', text: `chunk-${i}` })
        }

        yield* sink.emit({ _tag: 'MessageEnd', id: 'msg-1' })
        yield* sink.emit({ _tag: 'TurnResult', value: finalTurnResult })
      })
    )

    const result = await Effect.runPromise(drainForTurnResultSlow(turnStream))

    expect(result.seen.includes('MessageEnd')).toBe(true)
    expect(result.finalResult).not.toBeNull()
  })
})
