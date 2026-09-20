import { assignmentInputs } from "./work-store"
import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Context, DateTime, Effect, Layer, Option, Schema, Stream } from "effect"
import { dirname, join, resolve } from "node:path"
import { outputObjects } from "./build-output"
import { ArtifactStore, fileArtifactStore } from "./artifact-store"
import { Digest, InfrastructureFailure } from "./domain"
import { GuestExecutor } from "./guest-executor"
import { InputManifest } from "./inputs"
import { ProcessExecutorLive } from "./process"
import { assertRuntime } from "./runtime"
import { validateTargetResult, WorkClaim } from "./work-store"
import { GuestExecutorLive } from "./worker-entry"
import { WorkerClient, workerClientLayer } from "./worker-client"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"
import { NetworkControlPlane } from "./network-fault"

export const OutwardWorkerConfig = Schema.Struct({ root: Schema.NonEmptyString, pollMs: Schema.Int.pipe(Schema.between(10, 30_000)) })
const fail = (message: string) => new InfrastructureFailure({ operation: "outward-worker", message })

/** Fresh workspace is the execution guard. A failed upload leaves a reply on disk, never an automatic rerun. */
export const runOutwardWorker = (config: typeof OutwardWorkerConfig.Type) => Effect.gen(function* () {
  const client = yield* WorkerClient
  const executor = yield* GuestExecutor
  const fs = yield* FileSystem.FileSystem
  const invocation = yield* client.assignment
  const deadline = DateTime.toEpochMillis(invocation.assignment.deadline)
  if (deadline <= Date.now()) return yield* fail("Worker assignment has expired")
  const root = resolve(config.root)
  if (yield* fs.exists(root)) return yield* fail("Outward worker requires a fresh workspace; existing attempts must not be rerun")
  yield* fs.makeDirectory(dirname(root), { recursive: true })
  yield* fs.makeDirectory(root, { mode: 0o700 })
  yield* fs.writeFileString(join(root, "invocation.json"), yield* Schema.encode(Schema.parseJson(WorkerInvocation))(invocation), { flag: "wx", mode: 0o600 })
  const store = Context.get(yield* Layer.build(fileArtifactStore(join(root, "objects"))), ArtifactStore)
  const journey = Effect.gen(function* () {
    for (const input of assignmentInputs(invocation.assignment)) {
      let length = 0
      yield* store.put(input.digest, client.download(input.digest).pipe(Stream.tap(chunk => Effect.gen(function* () {
        length += chunk.byteLength
        if (length > 16 * 1024 * 1024) return yield* fail("Input manifest exceeds 16 MiB")
      }))))
      const bytes = yield* store.get(input.digest).pipe(Stream.runCollect)
      const manifest = yield* Schema.decodeUnknown(Schema.parseJson(InputManifest))(Buffer.concat(Array.from(bytes)).toString("utf8"))
      if (manifest.kind !== input.kind) return yield* fail("Assigned input kind differs from its manifest")
      const digests = [...new Set(manifest.kind === "source" ? manifest.entries.flatMap(entry => entry.kind === "file" ? [entry.sha256] : [])
        : manifest.release.artifacts.map(artifact => Digest.make(artifact.sha256)))]
      yield* Effect.forEach(digests, digest => store.put(digest, client.download(digest)), { concurrency: 4, discard: true })
    }
    const reply = yield* executor.run(invocation, root)
    if (!Schema.equivalence(WorkClaim)(reply.claim, invocation.assignment.claim)) return yield* fail("Executor replied for another attempt")
    yield* validateTargetResult(invocation.assignment.target, reply.result)
    const temporary = join(root, "reply.json.tmp")
    yield* fs.writeFileString(temporary, yield* Schema.encode(Schema.parseJson(WorkerReply))(reply), { mode: 0o600, flag: "wx" })
    yield* fs.rename(temporary, join(root, "reply.json"))
    return yield* deliverReply(client, store, fs, root, invocation, reply)
  })
  const authority = Effect.forever(Effect.sleep(config.pollMs).pipe(Effect.zipRight(client.assignment), Effect.flatMap(current =>
    Schema.equivalence(WorkerInvocation)(invocation, current) ? Effect.void : Effect.fail(fail("Worker assignment changed during execution")))))
  return yield* Effect.raceFirst(journey, authority).pipe(Effect.timeoutFail({ duration: Math.max(1, deadline - Date.now()), onTimeout: () => fail("Worker assignment deadline expired") }))
})

const deliverReply = (client: WorkerClient, store: ArtifactStore, fs: FileSystem.FileSystem, root: string,
  invocation: typeof WorkerInvocation.Type, reply: typeof WorkerReply.Type) => Effect.gen(function* () {
  if (!Schema.equivalence(WorkClaim)(reply.claim, invocation.assignment.claim)) return yield* fail("Saved reply belongs to another attempt")
  yield* validateTargetResult(invocation.assignment.target, reply.result)
  const evidence = new Map<Digest, number>()
  for (const item of reply.result.cases.flatMap(test => test.evidence)) {
    if (evidence.has(item.sha256) && evidence.get(item.sha256) !== item.bytes) return yield* fail("Conflicting evidence byte counts")
    evidence.set(item.sha256, item.bytes)
  }
  if (Option.isSome(reply.result.output)) {
    const packages = yield* outputObjects(reply.result.output.value).pipe(Effect.provideService(ArtifactStore, store))
    for (const item of packages) {
      if (evidence.has(item.digest) && evidence.get(item.digest) !== item.bytes) return yield* fail("Conflicting output byte counts")
      evidence.set(item.digest, item.bytes)
    }
  }
  for (const [digest, size] of evidence) {
    if (Number((yield* fs.stat(join(root, "objects", digest))).size) !== size) return yield* fail("Local evidence length differs from reply")
    yield* client.upload(digest, size, store.get(digest))
  }
  yield* client.submit(reply)
  return reply
})

/** Explicit delivery-only recovery never acquires a GuestExecutor or downloads input objects. */
export const deliverOutwardWorkerResult = (config: typeof OutwardWorkerConfig.Type) => Effect.gen(function* () {
  const client = yield* WorkerClient
  const fs = yield* FileSystem.FileSystem
  const invocation = yield* client.assignment
  const deadline = DateTime.toEpochMillis(invocation.assignment.deadline)
  if (deadline <= Date.now()) return yield* fail("Worker assignment has expired")
  const root = resolve(config.root)
  const readSaved = (name: string) => fs.stream(join(root, name)).pipe(
    Stream.runFoldEffect({ bytes: 0, chunks: [] as Uint8Array[] }, (state, chunk) => state.bytes + chunk.byteLength > 16 * 1024 * 1024
      ? Effect.fail(fail("Saved worker document exceeds 16 MiB"))
      : Effect.succeed({ bytes: state.bytes + chunk.byteLength, chunks: [...state.chunks, chunk] })),
    Effect.map(value => Buffer.concat(value.chunks).toString("utf8")),
  )
  const delivery = Effect.gen(function* () {
    const saved = yield* readSaved("invocation.json").pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WorkerInvocation))))
    if (!Schema.equivalence(WorkerInvocation)(saved, invocation)) return yield* fail("Saved invocation differs from the live assignment")
    const reply = yield* readSaved("reply.json").pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WorkerReply))))
    const store = Context.get(yield* Layer.build(fileArtifactStore(join(root, "objects"))), ArtifactStore)
    return yield* deliverReply(client, store, fs, root, invocation, reply)
  })
  const authority = Effect.forever(Effect.sleep(config.pollMs).pipe(Effect.zipRight(client.assignment), Effect.flatMap(current =>
    Schema.equivalence(WorkerInvocation)(invocation, current) ? Effect.void : Effect.fail(fail("Worker assignment changed during delivery")))))
  return yield* Effect.raceFirst(delivery, authority).pipe(Effect.timeoutFail({ duration: Math.max(1, deadline - Date.now()), onTimeout: () => fail("Worker delivery deadline expired") }))
})

export const outwardGuestMain = Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const config = yield* Schema.decodeUnknown(OutwardWorkerConfig)({ root: yield* Config.string("LAB_WORKER_ROOT"), pollMs: 10_000 })
  const origin = yield* Config.string("LAB_URL")
  const client = workerClientLayer(origin, yield* Config.redacted("LAB_WORKER_TOKEN")).pipe(Layer.provide(FetchHttpClient.layer))
  const action = yield* Config.literal("execute", "deliver")("LAB_WORKER_ACTION").pipe(Config.withDefault("execute"))
  if (action === "deliver") yield* deliverOutwardWorkerResult(config).pipe(Effect.provide(client))
  else yield* runOutwardWorker(config).pipe(Effect.provide([client, GuestExecutorLive.pipe(Layer.provide(Layer.succeed(NetworkControlPlane, { origin })))]))
}))
if (import.meta.main) BunRuntime.runMain(outwardGuestMain.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
