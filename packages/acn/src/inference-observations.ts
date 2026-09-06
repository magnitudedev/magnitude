import { Context, Data, Effect, Layer, Option, Schedule, Schema, Stream } from "effect"
import {
  InferenceObservationGroupIdSchema, InferenceObservationIdSchema,
  InferenceProgressSchema, InferenceTimingsSchema,
  type InferenceObservation, type InferenceObservationGroupId,
} from "@magnitudedev/acn-protocol"
import { SseFramer } from "@magnitudedev/openapi-effect/client-runtime"
import { defineFSM } from "@magnitudedev/utils/fsm"

export const OBSERVATION_ID_HEADER = "magnitude-observation-id"
export const OBSERVATION_GROUP_HEADER = "magnitude-observation-group"
const MAX_RETAINED = 128
const RETENTION_MS = 60_000

const Metadata = Schema.Struct({
  progress: Schema.optionalWith(InferenceProgressSchema, { as: "Option", exact: true }),
  timings: Schema.optionalWith(InferenceTimingsSchema, { as: "Option", exact: true }),
})
type Metadata = typeof Metadata.Type
const decodeMetadata = Schema.decodeUnknownOption(Schema.parseJson(Metadata))
class Active extends Data.TaggedClass("Active")<{ readonly metadata: Metadata }> {}
class Ended extends Data.TaggedClass("Ended")<{
  readonly metadata: Metadata
  readonly endedAt: number
}> {}
const lifetime = defineFSM({ Active, Ended }, { Active: ["Ended"], Ended: [] })
type Entry = {
  readonly requestId: InferenceObservation["requestId"]
  readonly groupId: InferenceObservationGroupId
  readonly startedAt: number
  state: Active | Ended
}

export interface InferenceObservations {
  readonly read: (groupId: InferenceObservationGroupId) => Effect.Effect<ReadonlyArray<InferenceObservation>>
  readonly observe: (request: Request, response: Response) => Effect.Effect<Response>
}
export const InferenceObservations = Context.GenericTag<InferenceObservations>("@magnitudedev/acn/InferenceObservations")

export const makeInferenceObservations = Effect.gen(function* () {
  const clock = yield* Effect.clock
  const entries = new Map<string, Entry>()
  let closed = false
  const now = () => Number(clock.unsafeCurrentTimeNanos()) / 1_000_000
  const prune = () => {
    const time = now()
    for (const [id, entry] of entries) {
      if (entry.state._tag === "Ended" && time - entry.state.endedAt >= RETENTION_MS) entries.delete(id)
    }
  }
  yield* Effect.addFinalizer(() => Effect.sync(() => {
    closed = true
    for (const entry of entries.values()) {
      if (entry.state._tag === "Active") entry.state = lifetime.transition(entry.state, "Ended", { endedAt: now() })
    }
    entries.clear()
  }))
  yield* Effect.sync(prune).pipe(Effect.repeat(Schedule.spaced("10 seconds")), Effect.forkScoped)

  return InferenceObservations.of({
    read: (groupId) => Effect.sync(() => {
      prune()
      const time = now()
      return Array.from(entries.values()).filter(entry => entry.groupId === groupId).map(entry => ({
        requestId: entry.requestId,
        state: entry.state._tag,
        elapsedMs: Math.max(0, (entry.state._tag === "Ended" ? entry.state.endedAt : time) - entry.startedAt),
        ...entry.state.metadata,
      }))
    }),
    observe: (request, response) => Effect.gen(function* () {
      if (closed || request.method !== "POST" || new URL(request.url).pathname !== "/inference/v1/chat/completions" ||
          response.body === null || !response.headers.get("content-type")?.toLowerCase().startsWith("text/event-stream") ||
          response.headers.has("content-encoding")) return response
      const requestId = Schema.decodeUnknownOption(InferenceObservationIdSchema)(request.headers.get(OBSERVATION_ID_HEADER))
      const groupId = Schema.decodeUnknownOption(InferenceObservationGroupIdSchema)(request.headers.get(OBSERVATION_GROUP_HEADER))
      if (Option.isNone(requestId) || Option.isNone(groupId)) return response
      prune()
      if (entries.size >= MAX_RETAINED || entries.has(requestId.value)) return response
      const entry: Entry = { requestId: requestId.value, groupId: groupId.value, startedAt: now(),
        state: new Active({ metadata: { progress: Option.none(), timings: Option.none() } }) }
      entries.set(entry.requestId, entry)
      const close = Effect.sync(() => {
        if (entry.state._tag === "Active") entry.state = lifetime.transition(entry.state, "Ended", { endedAt: now() })
      })
      const decoder = new TextDecoder()
      const framer = new SseFramer(1024 * 1024)
      let observing = true
      const consume = (bytes: Uint8Array) => Effect.sync(() => {
        if (!observing || entry.state._tag !== "Active") return
        try {
          for (const event of framer.push(decoder.decode(bytes, { stream: true }))) {
            if (event.data === "[DONE]") continue
            const decoded = decodeMetadata(event.data)
            if (Option.isNone(decoded)) continue
            const metadata = decoded.value
            entry.state = lifetime.hold(entry.state, { metadata: {
              progress: Option.orElse(metadata.progress, () => entry.state.metadata.progress),
              timings: Option.orElse(metadata.timings, () => entry.state.metadata.timings),
            } })
          }
        } catch {
          // Framing/observer failures must not change the caller's response.
          observing = false
        }
      })
      const body = Stream.fromReadableStream(() => response.body!, cause => cause).pipe(
        Stream.tap(consume), Stream.ensuring(close), Stream.toReadableStream,
      )
      return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers })
    }),
  })
})

export const InferenceObservationsLive = Layer.scoped(InferenceObservations, makeInferenceObservations)
