import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schedule, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { ProcessExecutorLive } from "../src/process"
import { linuxWorkerFault, NativeProcessStart, WorkerFault, WorkerFaultRequest } from "../src/worker-fault"

test.skipIf(process.platform !== "linux")("Linux pidfd fault rejects foreign processes and preserves the resident parent", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-native-worker-fault-" })
  const runtime = join(root, "releases", "fixture"), executable = join(runtime, "magnitude-inference")
  yield* fs.makeDirectory(runtime, { recursive: true })
  // A renamed native Python executable provides real process identities without requiring a compiler.
  yield* fs.copyFile(yield* fs.realPath("/usr/bin/python3"), executable)
  yield* fs.chmod(executable, 0o700)
  yield* fs.writeFileString(join(root, "inference-worker"), "import time\ntime.sleep(120)\n")
  yield* fs.writeFileString(join(root, "serve"), "import subprocess,sys,json,os,time\np=subprocess.Popen([sys.executable,'inference-worker'])\nwith open('identity.tmp','w') as f:json.dump({'parent':os.getpid(),'worker':p.pid},f)\nos.rename('identity.tmp','identity.json')\np.wait()\ntime.sleep(120)\n")
  const parent = yield* Effect.acquireRelease(Effect.sync(() => Bun.spawn([executable, "serve"], { cwd: root, stdout: "ignore", stderr: "pipe", detached: true })), child => Effect.gen(function* () {
    yield* Effect.sync(() => { try { process.kill(-child.pid, "SIGKILL") } catch (error) { if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) throw error } })
    yield* Effect.promise(() => child.exited)
  }))
  const identityPath = join(root, "identity.json")
  yield* fs.exists(identityPath).pipe(Effect.repeat({ until: Boolean, schedule: Schedule.spaced("20 millis") }), Effect.timeout("5 seconds"))
  const identity = yield* fs.readFileString(identityPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ parent: Schema.Int, worker: Schema.Int })))))
  const request = yield* Schema.decodeUnknown(WorkerFaultRequest)({ owner: { applicationPid: process.pid, servicePid: process.pid, serviceInstance: "native-test" }, profile: root, workerPid: identity.worker })
  const fault = yield* WorkerFault
  for (const invalid of [{ ...request, profile: join(root, "foreign") }, { ...request, workerPid: WorkerFaultRequest.fields.workerPid.make(identity.parent) },
    { ...request, workerPid: WorkerFaultRequest.fields.workerPid.make(process.pid) }]) {
    expect((yield* fault.crash(invalid).pipe(Effect.either))._tag).toBe("Left")
    expect(() => process.kill(identity.worker, 0)).not.toThrow()
    expect(parent.exitCode).toBeNull()
  }
  const receipt = yield* fault.crash(request)
  expect(receipt.parentPid).toBe(identity.parent)
  yield* fault.verifyParent(receipt)
  expect(parent.exitCode).toBeNull()
  expect((yield* fault.verifyParent({ ...receipt, parentStart: NativeProcessStart.make("wrong") }).pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide([linuxWorkerFault.pipe(Layer.provide(ProcessExecutorLive)), BunContext.layer]))), 20000)
