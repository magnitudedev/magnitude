import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Layer, Schema } from "effect"
import { dirname, join, resolve } from "node:path"
import { fileArtifactStore } from "./artifact-store"
import { runCandidateWorker } from "./candidate-worker"
import { HostInspectorLive } from "./host-inspector"
import { nativeInstaller } from "./installer"
import { InfrastructureFailure } from "./domain"
import { ProcessExecutorLive } from "./process"
import { nativeSourceBuilder } from "./source-builder"
import { assertRuntime } from "./runtime"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"
import { configuredHarnessTools } from "./harnesses/suite"

export const guestMain = (args: readonly string[]) => Effect.gen(function* () {
  yield* assertRuntime
  if (args.length !== 1) return yield* new InfrastructureFailure({ operation: "worker-entry", message: "Expected exactly one invocation file" })
  const fs = yield* FileSystem.FileSystem
  const file = resolve(args[0]!)
  if (Number((yield* fs.stat(file)).size) > 16 * 1024 * 1024) return yield* new InfrastructureFailure({ operation: "worker-entry", message: "Invocation exceeds 16 MiB" })
  const invocation = yield* fs.readFileString(file).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WorkerInvocation))))
  const root = dirname(file)
  const environment = Object.fromEntries(["PATH", "HOME", "TMPDIR", "USER", "LOGNAME", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA", "DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const result = yield* runCandidateWorker(invocation.assignment, { root: join(root, "workspace"), port: invocation.port, model: invocation.model, environment }).pipe(
    Effect.provide([fileArtifactStore(join(root, "objects")), HostInspectorLive, configuredHarnessTools,
      nativeInstaller({ disposable: invocation.disposable, root: join(root, "installation"), environment }),
      nativeSourceBuilder({ root: join(root, "build"), objects: join(root, "objects"), environment }).pipe(Layer.provide(fileArtifactStore(join(root, "objects"))))]),
    Effect.catchAll(error => Effect.sync(() => {
      const now = new Date().toISOString()
      const detail = `Worker setup: ${error.message}`.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]").slice(0, 2400)
      return { cleanupErrors: error._tag === "CandidateWorkerFailure" ? error.cleanupErrors : [], cases: invocation.assignment.target.cases.map(test => ({ targetId: invocation.assignment.target.target.id,
        caseId: test.id, harness: test.harness, startedAt: now, endedAt: now, evidence: [], outcome: { status: "blocked" as const, detail } })) }
    })))
  // stdout is one bounded protocol reply; product logs stay in the evidence workspace.
  yield* Console.log(yield* Schema.encode(Schema.parseJson(WorkerReply))({ schemaVersion: 1, claim: invocation.assignment.claim, result }))
})
if (import.meta.main) BunRuntime.runMain(guestMain(process.argv.slice(2)).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
