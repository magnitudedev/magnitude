import { Cache, Context, Data, Effect, Exit, Layer, Redacted, Schema, Stream } from "effect"
import { Digest, InfrastructureFailure, OwnerId } from "./domain"
import { InputManifest, InputRegistry } from "./inputs"
import { sha256 } from "./snapshot"
import { WorkerAccessDenied, WorkerTickets } from "./worker-tickets"

export interface WorkerInputs {
  readonly read: (token: Redacted.Redacted<string>, digest: Digest) => Effect.Effect<Stream.Stream<Uint8Array, InfrastructureFailure>, WorkerAccessDenied | InfrastructureFailure>
}
export const WorkerInputs = Context.GenericTag<WorkerInputs>("@magnitudedev/testing-lab/WorkerInputs")
const GraphKey = Schema.Struct({ owner: OwnerId, digest: Digest, kind: Schema.Literal("source", "artifacts") })

/** The owner's other uploads are not part of a worker's authority, even when their digests are known. */
export const WorkerInputsLive = Layer.effect(WorkerInputs, Effect.gen(function* () {
  const tickets = yield* WorkerTickets
  const inputs = yield* InputRegistry
  const graphs = yield* Cache.makeWith({ capacity: 8, timeToLive: exit => Exit.isSuccess(exit) ? "5 minutes" : 0,
    lookup: (key: typeof GraphKey.Type) => Effect.gen(function* () {
      const content = yield* inputs.read(key.owner, key.digest)
      let size = 0
      const chunks = yield* content.pipe(Stream.tap(chunk => Effect.gen(function* () {
        size += chunk.byteLength
        if (size > 16 * 1024 * 1024) return yield* new InfrastructureFailure({ operation: "worker-input", message: "Assigned manifest exceeds 16 MiB" })
      })), Stream.runCollect)
      const bytes = Buffer.concat(Array.from(chunks))
      if (sha256(bytes) !== key.digest) return yield* new InfrastructureFailure({ operation: "worker-input", message: "Assigned manifest digest mismatch" })
      const manifest = yield* Schema.decodeUnknown(Schema.parseJson(InputManifest))(bytes.toString("utf8"))
      if (manifest.kind !== key.kind) return yield* new InfrastructureFailure({ operation: "worker-input", message: "Assigned manifest kind mismatch" })
      return new Set(manifest.kind === "source" ? manifest.entries.flatMap(entry => entry.kind === "file" ? [entry.sha256] : [])
        : manifest.release.artifacts.map(artifact => Digest.make(artifact.sha256)))
    }),
  })
  return {
    read: (token, digest) => Effect.gen(function* () {
      const { assignment } = yield* tickets.authorize(token)
      const { owner, input } = assignment.plan.request
      yield* inputs.require(owner, input)
      if (digest === input.digest) return yield* inputs.read(owner, digest)
      const key = Data.struct({ owner, digest: input.digest, kind: input.kind })
      // Explicit invalidation also covers a second request in the same clock tick as a zero-TTL failure.
      const allowed = yield* graphs.get(key).pipe(Effect.tapError(() => graphs.invalidate(key)))
      if (!allowed.has(digest)) return yield* new WorkerAccessDenied({})
      // Only the immutable graph is cached, never current credential authority.
      yield* tickets.authorize(token)
      return yield* inputs.read(owner, digest)
    }).pipe(Effect.mapError(error => error._tag === "InputDenied" ? new WorkerAccessDenied({}) : error._tag === "ParseError"
      ? new InfrastructureFailure({ operation: "worker-input", message: "Malformed assigned input manifest" }) : error)),
  } satisfies WorkerInputs
}))
