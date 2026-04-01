import { describe, expect, test } from 'vitest'
import { Cause, Effect, Exit, Stream } from 'effect'

describe('stream cleanup guard', () => {
  test('stream with failing cleanup effect still completes with emitted chunks', async () => {
    const stream = Stream.make('chunk1', 'chunk2').pipe(
      Stream.ensuring(
        Effect.catchAllCause(
          Effect.fail('cleanup-failed').pipe(Effect.asVoid),
          () => Effect.void,
        ).pipe(Effect.asVoid),
      ),
    )

    const exit = await Effect.runPromiseExit(Stream.runCollect(stream))

    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(Array.from(exit.value)).toEqual(['chunk1', 'chunk2'])
    }
  })

  test('stream error is preserved when cleanup also fails', async () => {
    const stream = Stream.fail('stream-failed').pipe(
      Stream.ensuring(
        Effect.catchAllCause(
          Effect.fail('cleanup-failed').pipe(Effect.asVoid),
          () => Effect.void,
        ).pipe(Effect.asVoid),
      ),
    )

    const exit = await Effect.runPromiseExit(Stream.runCollect(stream))

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      expect(Cause.isFailType(exit.cause)).toBe(true)
      if (Cause.isFailType(exit.cause)) {
        expect(exit.cause.error).toBe('stream-failed')
      }
    }
  })
})
