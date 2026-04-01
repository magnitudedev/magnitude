import { describe, expect, it, vi } from 'vitest'
import { Cause, Effect, Exit, Option, Stream } from 'effect'
import { fromAsyncIterableSafe } from './from-async-iterable-safe'

describe('fromAsyncIterableSafe', () => {
  it('emits all values on normal completion', async () => {
    const iterable: AsyncIterable<string> = {
      async *[Symbol.asyncIterator]() {
        yield 'a'
        yield 'b'
      },
    }

    const values = await Effect.runPromise(
      Stream.runCollect(fromAsyncIterableSafe(iterable, (e) => new Error(String(e)))),
    )

    expect(Array.from(values)).toEqual(['a', 'b'])
  })

  it('maps next() rejection through onError (typed failure, not defect)', async () => {
    const mappedError = { _tag: 'MappedError', message: 'boom-next' } as const
    const rawError = { _tag: 'NativeErr', message: 'boom-next' }

    const iterable: AsyncIterable<string> = {
      [Symbol.asyncIterator]() {
        return {
          next: () => Promise.reject(rawError),
        } as AsyncIterator<string>
      },
    }

    const exit = await Effect.runPromiseExit(
      Stream.runCollect(fromAsyncIterableSafe(iterable, () => mappedError)),
    )

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      const failure = Cause.failureOption(exit.cause)
      expect(Option.isSome(failure)).toBe(true)
      if (Option.isSome(failure)) {
        expect(failure.value).toEqual(mappedError)
      }
    }
  })

  it('swallows return() rejection during finalization (no defect raised)', async () => {
    const iterable: AsyncIterable<string> = {
      [Symbol.asyncIterator]() {
        let i = 0
        return {
          next: async () => {
            i += 1
            return i === 1 ? { done: false, value: 'a' } : { done: false, value: 'b' }
          },
          return: () => Promise.reject({ _tag: 'ReturnErr', message: 'cleanup failed' }),
        } as AsyncIterator<string>
      },
    }

    const values = await Effect.runPromise(
      Stream.runCollect(Stream.take(fromAsyncIterableSafe(iterable, (e) => new Error(String(e))), 1)),
    )

    expect(Array.from(values)).toEqual(['a'])
  })

  it('invokes return() when consumer ends early', async () => {
    const onReturn = vi.fn(() => Promise.resolve({ done: true, value: undefined }))

    const iterable: AsyncIterable<string> = {
      [Symbol.asyncIterator]() {
        let i = 0
        return {
          next: async () => {
            i += 1
            return i === 1 ? { done: false, value: 'a' } : { done: false, value: 'b' }
          },
          return: onReturn,
        } as AsyncIterator<string>
      },
    }

    const values = await Effect.runPromise(
      Stream.runCollect(Stream.take(fromAsyncIterableSafe(iterable, (e) => new Error(String(e))), 1)),
    )

    expect(Array.from(values)).toEqual(['a'])
    expect(onReturn).toHaveBeenCalledTimes(1)
  })
})
