import { Duration, Effect } from "effect"
import {
  RateLimited,
  TransportError,
  type ModelError,
} from "../errors/model-error"

function isRetryableError(error: ModelError): error is RateLimited | TransportError {
  return (
    error instanceof RateLimited ||
    (error instanceof TransportError && error.retryable)
  )
}

function delayFor(error: RateLimited | TransportError, attempt: number): Duration.Duration {
  if (error instanceof RateLimited && error.retryAfterMs !== null) {
    return Duration.millis(error.retryAfterMs)
  }

  const baseMs = Math.min(1000 * 2 ** Math.max(0, attempt - 1), 30_000)
  const jitterMultiplier = 0.5 + Math.random()
  return Duration.millis(Math.round(baseMs * jitterMultiplier))
}

export function retryModelStream<A>(
  effect: Effect.Effect<A, ModelError>,
  maxAttempts = 5,
): Effect.Effect<A, ModelError> {
  const loop = (attempt: number): Effect.Effect<A, ModelError> =>
    effect.pipe(
      Effect.catchAll((error) => {
        if (!isRetryableError(error) || attempt >= maxAttempts) {
          return Effect.fail(error)
        }

        return Effect.sleep(delayFor(error, attempt)).pipe(
          Effect.zipRight(loop(attempt + 1)),
        )
      }),
    )

  return loop(1)
}
