import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Layer, Option, Schema } from "effect"
import { dirname, join, resolve } from "node:path"
import { fileArtifactStore } from "./artifact-store"
import { runCandidateWorker } from "./candidate-worker"
import { HostInspectorLive } from "./host-inspector"
import { nativeInstaller } from "./installer"
import { InfrastructureFailure } from "./domain"
import { ProcessExecutor, ProcessExecutorLive } from "./process"
import { GuestExecutor } from "./guest-executor"
import { nativeSourceBuilder } from "./source-builder"
import { assertRuntime } from "./runtime"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"
import { configuredHarnessTools } from "./harnesses/suite"
import { qualifyGuestUser } from "./guest-user"
import { DisposableDesktopUser } from "./desktop-environment"

export const executeGuestInvocation = (invocation: typeof WorkerInvocation.Type, root: string) => Effect.gen(function* () {
  const environment = Object.fromEntries(["PATH", "HOME", "USERPROFILE", "TMPDIR", "USER", "LOGNAME", "SystemRoot", "TEMP", "APPDATA", "LOCALAPPDATA", "DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "LAB_EXPECTED_APPLE_TEAM_ID", "LAB_EXPECTED_WINDOWS_PUBLISHER", "LAB_WINDOWS_SIGNTOOL"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const result = yield* Effect.gen(function* () {
    const user = yield* qualifyGuestUser(invocation.assignment.target.target.provider, invocation.disposable, environment)
    if (Option.isSome(user)) environment.HOME = user.value.home
    const userLayer = Option.match(user, { onNone: () => Layer.empty, onSome: value => Layer.succeed(DisposableDesktopUser, value) })
    const installationRoot = Option.isSome(user) && process.platform === "darwin" ? "/Applications" : join(root, "installation")
    return yield* runCandidateWorker(invocation.assignment, { root: join(root, "workspace"), port: invocation.port, model: invocation.model, environment }).pipe(
    Effect.provide([fileArtifactStore(join(root, "objects")), HostInspectorLive, configuredHarnessTools,
      nativeInstaller({ disposable: invocation.disposable, root: installationRoot, environment }),
      nativeSourceBuilder({ root: join(root, "build"), objects: join(root, "objects"), environment }).pipe(Layer.provide(fileArtifactStore(join(root, "objects"))))]), Effect.provide(userLayer))
  }).pipe(
    Effect.catchAll(error => Effect.sync(() => {
      const now = new Date().toISOString()
      const detail = `Worker setup: ${error.message}`.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]").slice(0, 2400)
      return { cleanupErrors: error._tag === "CandidateWorkerFailure" ? error.cleanupErrors : [], cases: invocation.assignment.target.cases.map(test => ({ targetId: invocation.assignment.target.target.id,
        caseId: test.id, harness: test.harness, startedAt: now, endedAt: now, evidence: [], outcome: { status: "blocked" as const, detail } })) }
    })))
  return WorkerReply.make({ schemaVersion: 1, claim: invocation.assignment.claim, result })
})
export const GuestExecutorLive = Layer.effect(GuestExecutor, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const processes = yield* ProcessExecutor
  return { run: (invocation, root) => executeGuestInvocation(invocation, root).pipe(
    Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, processes)) } satisfies GuestExecutor
}))

export const guestMain = (args: readonly string[]) => Effect.gen(function* () {
  yield* assertRuntime
  if (args.length !== 1) return yield* new InfrastructureFailure({ operation: "worker-entry", message: "Expected exactly one invocation file" })
  const fs = yield* FileSystem.FileSystem
  const file = resolve(args[0]!)
  if (Number((yield* fs.stat(file)).size) > 16 * 1024 * 1024) return yield* new InfrastructureFailure({ operation: "worker-entry", message: "Invocation exceeds 16 MiB" })
  const invocation = yield* fs.readFileString(file).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WorkerInvocation))))
  const reply = yield* executeGuestInvocation(invocation, dirname(file))
  // stdout is one bounded protocol reply; product logs stay in the evidence workspace.
  yield* Console.log(yield* Schema.encode(Schema.parseJson(WorkerReply))(reply))
})
if (import.meta.main) BunRuntime.runMain(guestMain(process.argv.slice(2)).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
