import { Effect, Stream } from 'effect'
import type { BoundModel, ModelSpec, ModelStreamResult, ResponseStreamEvent } from '@magnitudedev/ai'

/**
 * Configuration for a test BoundModel.
 * Each entry in `responses` is the event sequence for one call (in order).
 * If calls exceed the array length, the last entry is reused.
 * If no responses are provided, an empty stream is returned.
 */
export interface TestModelConfig {
  readonly responses?: readonly (readonly ResponseStreamEvent[])[]
}

const testModelSpec: ModelSpec<{}> = {
  modelId: 'test-model',
  endpoint: 'http://test',
  bind: () => { throw new Error('TestModelSpec.bind should not be called') },
  _execute: () => { throw new Error('TestModelSpec._execute should not be called') },
}

export function createTestBoundModel(
  config: TestModelConfig = {},
): BoundModel<{}> {
  let callIndex = 0
  const responses = config.responses ?? []

  return {
    stream: () => {
      const events = callIndex < responses.length
        ? responses[callIndex]
        : responses.length > 0
          ? responses[responses.length - 1]
          : []
      callIndex++
      const result: ModelStreamResult = {
        events: Stream.fromIterable(events),
        parsers: new Map(),
        logprobs: [],
        requestId: null,
      }
      return Effect.succeed(result)
    },
  }
}
