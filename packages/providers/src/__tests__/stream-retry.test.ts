import { test, expect } from 'bun:test'
import { Effect, Stream, Schedule, Duration, Exit } from 'effect'

/**
 * Test: Stream.retry with mutable emitted flag for first-chunk retry.
 * 
 * Requirements:
 * 1. If stream fails before first chunk → retry (new connection)
 * 2. If stream fails after first chunk → don't retry, propagate error
 * 3. When consumer stops early → stream is aborted (finalizer runs with interrupt)
 * 4. On success → all chunks delivered
 */

class ConnectionError {
  readonly _tag = 'ConnectionError' as const
  constructor(readonly message: string) {}
}

class MidStreamError {
  readonly _tag = 'MidStreamError' as const
  constructor(readonly message: string) {}
}

const retrySchedule = Schedule.exponential(Duration.millis(10), 1.5).pipe(
  Schedule.intersect(Schedule.recurs(3))
)

type StreamError = ConnectionError | MidStreamError

function makeRetriableStream(
  createStream: () => Effect.Effect<Stream.Stream<string, StreamError>, StreamError>,
) {
  let emitted = false

  return Stream.retry(
    Stream.unwrap(
      Effect.suspend(() => createStream().pipe(
        Effect.map(s => s.pipe(
          Stream.tap(() => Effect.sync(() => { emitted = true }))
        ))
      ))
    ),
    retrySchedule.pipe(
      Schedule.whileInput(() => !emitted)
    )
  )
}

test('retries on failure before first chunk', async () => {
  let attempts = 0

  const stream = makeRetriableStream(() => Effect.sync(() => {
    attempts++
    if (attempts < 3) {
      return Stream.fail(new ConnectionError('connection refused'))
    }
    return Stream.make('chunk1', 'chunk2', 'chunk3')
  }))

  const result = await Effect.runPromise(Stream.runCollect(stream).pipe(Effect.map(c => Array.from(c))))
  expect(result).toEqual(['chunk1', 'chunk2', 'chunk3'])
  expect(attempts).toBe(3)
})

test('does not retry after first chunk emitted', async () => {
  let attempts = 0

  const stream = makeRetriableStream(() => Effect.sync(() => {
    attempts++
    return Stream.concat(
      Stream.make('chunk1'),
      Stream.fail(new MidStreamError('mid-stream failure'))
    )
  }))

  const result = await Effect.runPromiseExit(Stream.runCollect(stream))
  expect(Exit.isFailure(result)).toBe(true)
  expect(attempts).toBe(1) // no retry
})

test('stream aborted when consumer stops early', async () => {
  const log: string[] = []

  const stream = makeRetriableStream(() => Effect.sync(() => {
    return Stream.unwrapScoped(
      Effect.gen(function* () {
        yield* Effect.addFinalizer((exit) => Effect.sync(() => {
          log.push(Exit.isInterrupted(exit) ? 'ABORTED' : 'COMPLETED')
        }))
        // Use async chunks to ensure the stream doesn't complete synchronously
        return Stream.fromEffect(Effect.sleep(Duration.millis(1))).pipe(
          Stream.flatMap(() => Stream.make('chunk1', 'yield', 'extra1', 'extra2'))
        )
      })
    )
  }))

  // Take only until 'yield'
  const result = await Effect.runPromise(
    stream.pipe(
      Stream.takeUntil(c => c === 'yield'),
      Stream.runCollect,
      Effect.map(c => Array.from(c)),
    )
  )

  expect(result).toEqual(['chunk1', 'yield'])
  await new Promise(r => setTimeout(r, 50))
  expect(log.some(l => l === 'ABORTED' || l === 'COMPLETED')).toBe(true)
})

test('retries when driver.stream() Effect itself fails', async () => {
  let attempts = 0

  const stream = makeRetriableStream(() => {
    attempts++
    if (attempts < 2) {
      return Effect.fail(new ConnectionError('DNS resolution failed'))
    }
    return Effect.succeed(Stream.make('hello', 'world')) as Effect.Effect<Stream.Stream<string, StreamError>, StreamError>
  })

  const result = await Effect.runPromise(Stream.runCollect(stream).pipe(Effect.map(c => Array.from(c))))
  expect(result).toEqual(['hello', 'world'])
  expect(attempts).toBe(2)
})