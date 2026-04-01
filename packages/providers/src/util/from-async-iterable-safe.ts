import { Effect, Option, Stream } from 'effect'

/**
 * Safe replacement for Effect's `Stream.fromAsyncIterable`.
 *
 * Effect's built-in `fromAsyncIterable` uses `Effect.promise` for iterator
 * cleanup (`iterator.return()`). If the iterator's `return()` rejects — e.g.
 * BAML's native FFI throwing a plain object during stream teardown — the
 * rejection becomes an unhandled defect that crashes the consumer with an
 * unhelpful `[object Object]` error.
 *
 * This version uses `Effect.tryPromise` for both `next()` and `return()`,
 * so rejections are caught and handled instead of becoming defects.
 * Cleanup errors are swallowed — iterator teardown failing should not
 * crash the stream consumer.
 */
export function fromAsyncIterableSafe<A, E>(
  iterable: AsyncIterable<A>,
  onError: (e: unknown) => E,
): Stream.Stream<A, E> {
  return Stream.unwrapScoped(
    Effect.gen(function* () {
      const iterator = iterable[Symbol.asyncIterator]()

      yield* Effect.acquireRelease(
        Effect.void,
        () => {
          if (!iterator.return) return Effect.void
          return Effect.tryPromise({
            try: () => iterator.return!(),
            catch: () => undefined,
          }).pipe(Effect.ignore)
        },
      )

      return Stream.repeatEffectOption(
        Effect.tryPromise({
          try: () => iterator.next(),
          catch: (e) => Option.some(onError(e)),
        }).pipe(
          Effect.flatMap((result) =>
            result.done
              ? Effect.fail(Option.none())
              : Effect.succeed(result.value),
          ),
        ),
      )
    }),
  )
}
