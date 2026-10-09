import { Context, Effect, Layer, Ref, type Scope } from "effect"

/**
 * Inference requests from harnesses and other devices that are still running. A request counts
 * from admission until its response body ends, errors or is cancelled, so a model load and a streamed
 * generation keep the service busy throughout. The application owner installs updates only at zero.
 */
export interface InferenceActivityApi {
  readonly count: Effect.Effect<number>
  /** Counts the request for the lifetime of the surrounding scope. */
  readonly hold: Effect.Effect<void, never, Scope.Scope>
  /** Counts a request until the returned idempotent `finish` is called. */
  readonly begin: Effect.Effect<() => void>
}
export class InferenceActivity extends Context.Tag("InferenceActivity")<InferenceActivity, InferenceActivityApi>() {}

export const makeInferenceActivity = Effect.gen(function* () {
  const active = yield* Ref.make(0)
  const increment = Ref.update(active, value => value + 1)
  const decrement = Ref.update(active, value => Math.max(0, value - 1))
  return InferenceActivity.of({
    count: Ref.get(active),
    hold: Effect.acquireRelease(increment, () => decrement),
    begin: increment.pipe(Effect.as(() => {
      let finished = false
      // Stream callbacks run outside any fiber; updating the Ref needs no services.
      return () => { if (!finished) { finished = true; Effect.runSync(decrement) } }
    }), Effect.map(make => make())),
  })
})

/** Calls `finish` once the body is consumed, errors or is cancelled; a bodiless response finishes at once. */
export const finishWithBody = (response: Response, finish: () => void): Response => {
  if (response.body === null) { finish(); return response }
  const reader = response.body.getReader()
  return new Response(new ReadableStream<Uint8Array>({
    pull: async controller => {
      try {
        const chunk = await reader.read()
        if (chunk.done) { finish(); controller.close() } else controller.enqueue(chunk.value)
      } catch (error) { finish(); controller.error(error) }
    },
    cancel: async reason => { finish(); await reader.cancel(reason) },
  }), { status: response.status, statusText: response.statusText, headers: response.headers })
}

export const InferenceActivityLive = Layer.effect(InferenceActivity, makeInferenceActivity)
