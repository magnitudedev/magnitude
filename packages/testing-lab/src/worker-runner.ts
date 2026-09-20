import { workerObjectLimits } from "./worker-evidence"
import { outputObjects } from "./build-output"
import { assignmentInputs } from "./work-store"
import { FileSystem } from "@effect/platform"
import { Context, DateTime, Effect, Layer, Option, Schema, Stream } from "effect"
import { join, posix, win32 } from "node:path"
import { ArtifactStore, fileArtifactStore } from "./artifact-store"
import { Digest, InfrastructureFailure, Provider, Target } from "./domain"
import { artifactObjects, InputManifest, InputRegistry } from "./inputs"
import { WorkerTransport } from "./machines"
import { WorkerRunner } from "./scheduler"
import { sha256 } from "./snapshot"
import { WorkClaim, validateTargetResult } from "./work-store"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"
import { retainWorkerDiagnostic } from "./worker-diagnostics"

export const GuestRuntime = Schema.Struct({ provider: Provider, artifactHost: Target.fields.artifactHost,
  executable: Schema.NonEmptyString, args: Schema.Array(Schema.String), root: Schema.NonEmptyString,
  disposable: Schema.Boolean, port: Schema.Int.pipe(Schema.between(1024, 65535)), model: Schema.NonEmptyString })
export interface WorkerTransports { readonly transports: ReadonlyMap<typeof Provider.Type, WorkerTransport> }
export const WorkerTransports = Context.GenericTag<WorkerTransports>("@magnitudedev/testing-lab/WorkerTransports")
const fail = (message: string) => new InfrastructureFailure({ operation: "worker-transport", message })

/** Deliver only the admitted object graph; guest execution never receives coordinator credentials. */
export const transportWorkerRunner = (runtimes: readonly (typeof GuestRuntime.Type)[]) => Layer.effect(WorkerRunner, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const inputs = yield* InputRegistry
  const transports = yield* WorkerTransports
  return {
    run: (machine, assignment) => Effect.scoped(Effect.gen(function* () {
      const selected = runtimes.filter(r => r.provider === machine.provider && r.artifactHost === assignment.target.target.artifactHost)
      const transport = transports.transports.get(machine.provider)
      if (selected.length !== 1 || !transport) return yield* fail("Exactly one qualified guest runtime and transport must match the worker")
      const runtime = selected[0]!
      if (runtime.disposable && (machine.provider === "local" || machine.provider === "spark")) return yield* fail("Shared hosts cannot grant a disposable OS-user context")
      if (machine.tags.runId !== assignment.claim.runId || assignment.claim.targetId !== assignment.target.target.id) return yield* fail("Worker ownership does not match its assignment")
      if (assignment.plan.request.trust === "untrusted-ci" && (!runtime.disposable || machine.provider === "local" || machine.provider === "spark")) return yield* fail("Untrusted work requires a disposable cloud worker")
      const deadline = Math.min(DateTime.toEpochMillis(machine.tags.expiresAt), DateTime.toEpochMillis(assignment.deadline))
      if (deadline <= Date.now()) return yield* fail("Worker assignment has expired")
      const remotePath = assignment.target.target.os === "windows" ? win32 : posix
      const base = machine.provider === "local" ? machine.root : runtime.root
      if (!remotePath.isAbsolute(base)) return yield* fail("Guest runtime root must be absolute")
      const directory = remotePath.join(base, machine.tags.leaseId, `attempt-${assignment.claim.fence}`)
      const local = yield* fs.makeTempDirectoryScoped({ prefix: "lab-worker-transfer-" })
      const owner = assignment.plan.request.owner
      const program = Effect.gen(function* () {
        const store = yield* ArtifactStore
        const copyInput = (digest: Digest) => Effect.gen(function* () {
          yield* store.put(digest, yield* inputs.read(owner, digest))
          yield* transport.upload(machine, join(local, "objects", digest), remotePath.join(directory, "objects", digest))
        })
        for (const input of assignmentInputs(assignment)) {
          yield* inputs.require(owner, input)
          const manifestStream = yield* inputs.read(owner, input.digest)
          let size = 0
          const chunks = yield* manifestStream.pipe(Stream.tap(chunk => Effect.gen(function* () {
            size += chunk.byteLength
            if (size > 16 * 1024 * 1024) return yield* fail("Input manifest exceeds 16 MiB")
          })), Stream.runCollect)
          const bytes = Buffer.concat(Array.from(chunks))
          if (sha256(bytes) !== input.digest) return yield* fail("Input manifest digest differs from admitted input")
          const manifest = yield* Schema.decodeUnknown(Schema.parseJson(InputManifest))(bytes.toString("utf8"))
          if (manifest.kind !== input.kind) return yield* fail("Input manifest kind differs from assignment")
          const digests = [...new Set(manifest.kind === "source" ? manifest.entries.flatMap(e => e.kind === "file" ? [e.sha256] : [])
            : artifactObjects(manifest).map(a => a.sha256))]
          yield* store.put(input.digest, Stream.make(bytes))
          yield* transport.upload(machine, join(local, "objects", input.digest), remotePath.join(directory, "objects", input.digest))
          yield* Effect.forEach(digests, copyInput, { concurrency: 4, discard: true })
        }
        const invocation = WorkerInvocation.make({ schemaVersion: 1, assignment, disposable: runtime.disposable, port: runtime.port, model: runtime.model })
        const job = join(local, "invocation.json")
        yield* fs.writeFileString(job, yield* Schema.encode(Schema.parseJson(WorkerInvocation))(invocation), { mode: 0o600 })
        const remoteJob = remotePath.join(directory, "invocation.json")
        yield* transport.upload(machine, job, remoteJob)
        const response = yield* transport.execute(machine, runtime.executable, [...runtime.args, remoteJob], Math.max(1, deadline - Date.now()))
        if (response.exitCode !== 0) {
          const diagnostic = yield* Effect.gen(function* () {
            const item = yield* retainWorkerDiagnostic(machine, "execution", `stdout:\n${response.stdout}\nstderr:\n${response.stderr}`)
            yield* inputs.upload(owner, item.sha256, fs.stream(join(local, "objects", item.sha256)).pipe(
              Stream.mapError(() => fail("Cannot retain transported worker diagnostics"))))
            return item
          }).pipe(Effect.either)
          return yield* new InfrastructureFailure({ operation: "worker-transport",
            message: `Guest worker exited ${response.exitCode} without an accepted result; ${diagnostic._tag === "Right" ? "execution diagnostics retained in run evidence" : "could not retain execution diagnostics"}`,
            evidence: diagnostic._tag === "Right" ? Option.some([diagnostic.right]) : Option.none() })
        }
        const reply = yield* Schema.decodeUnknown(Schema.parseJson(WorkerReply))(response.stdout)
        if (!Schema.equivalence(WorkClaim)(assignment.claim, reply.claim)) return yield* fail("Guest result belongs to another assignment or attempt")
        yield* validateTargetResult(assignment.target, reply.result)
        const evidence = new Map<Digest, number>()
        for (const item of reply.result.cases.flatMap(c => c.evidence)) {
          if (evidence.has(item.sha256) && evidence.get(item.sha256) !== item.bytes) return yield* fail("Evidence has conflicting byte counts")
          evidence.set(item.sha256, item.bytes)
        }
        if (Option.isSome(reply.result.output)) {
          if (assignment.work.kind !== "build") return yield* fail("A consumer cannot publish build output")
          const digest = reply.result.output.value.artifactDigest
          const file = join(local, "produced-manifest.json")
          yield* transport.download(machine, remotePath.join(directory, "objects", digest), file)
          if (Number((yield* fs.stat(file)).size) > 16 * 1024 * 1024) return yield* fail("Build manifest exceeds 16 MiB")
          yield* store.put(digest, fs.stream(file).pipe(Stream.mapError(() => fail("Cannot read produced manifest"))))
          for (const item of yield* outputObjects(reply.result.output.value)) {
            if (evidence.has(item.digest) && evidence.get(item.digest) !== item.bytes) return yield* fail("Conflicting build output byte counts")
            evidence.set(item.digest, item.bytes)
          }
        }
        const limits = workerObjectLimits(assignment.work.kind)
        if (evidence.size > limits.objectCount || [...evidence.values()].some(bytes => bytes > limits.objectBytes) ||
          [...evidence.values()].reduce((sum, bytes) => sum + bytes, 0) > limits.attemptBytes) return yield* fail("Worker output exceeds its stage transfer budget")
        for (const [digest, length] of evidence) {
          const file = join(local, `evidence-${digest}`)
          yield* transport.download(machine, remotePath.join(directory, "objects", digest), file)
          if (Number((yield* fs.stat(file)).size) !== length) return yield* fail("Evidence length does not match guest result")
          // Persisting verifies SHA-256 before granting the owner access to evidence.
          yield* inputs.upload(owner, digest, fs.stream(file).pipe(Stream.mapError(() => fail("Cannot read transferred evidence"))))
        }
        return reply.result
      }).pipe(Effect.provide(fileArtifactStore(join(local, "objects")).pipe(Layer.provide(Layer.succeed(FileSystem.FileSystem, fs)))))
      return yield* program.pipe(Effect.timeoutFail({ duration: Math.max(1, deadline - Date.now()), onTimeout: () => fail("Worker transfer or execution exceeded the admitted deadline") }))
    })).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail(error.message))),
  } satisfies WorkerRunner
}))
