import { test, expect } from 'bun:test'
import { Effect, Stream, Scope, Exit, Sink, Option, Duration } from 'effect'

/**
 * Test: Stream.peel with scope closed via Stream.ensuring (not manual RELEASE)
 * Does this avoid the hang when consumer stops early?
 */

test('peel + ensuring: consumer stops early', async () => {
  const log: string[] = []

  const makeStream = () => Stream.unwrapScoped(
    Effect.gen(function* () {
      yield* Effect.addFinalizer((exit) => Effect.sync(() => {
        log.push(Exit.isInterrupted(exit) ? 'ABORTED' : 'COMPLETED')
      }))
      // Async stream to avoid synchronous completion
      return Stream.fromEffect(Effect.sleep(Duration.millis(1))).pipe(
        Stream.flatMap(() => Stream.make('chunk1', 'yield', 'extra1', 'extra2'))
      )
    })
  )

  const program = Effect.gen(function* () {
    const scope = yield* Scope.make()
    const [head, tail] = yield* Stream.peel(makeStream(), Sink.head<string>()).pipe(
      Effect.provideService(Scope.Scope, scope),
    )
    const firstChunk = Option.getOrNull(head)

    const fullStream = Stream.concat(
      Stream.make(firstChunk!),
      tail.pipe(
        Stream.ensuring(Scope.close(scope, Exit.void))
      )
    )

    // Consumer stops early (like execManager.execute when a turn idles)
    const result = yield* fullStream.pipe(
      Stream.takeUntil(c => c === 'yield'),
      Stream.runCollect,
      Effect.map(c => Array.from(c)),
    )

    return result
  }).pipe(Effect.timeout('3 seconds'))

  const result = await Effect.runPromise(program.pipe(Effect.option))
  console.log('result:', result)
  console.log('log:', log)
  // If result is None, it hung (timeout). If Some, it worked.
  expect(Option.isSome(result)).toBe(true)
})