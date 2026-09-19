import { BunContext } from "@effect/platform-bun"
import { FileSystem } from "@effect/platform"
import { Deferred, Effect, Fiber, Layer } from "effect"
import { expect, test } from "vitest"
import { ApplicationIdentity, LabProcessId, ServiceInstanceId } from "../src/application-identity"
import { InfrastructureFailure } from "../src/domain"
import { checkedCommand, ProcessExecutor, ProcessExecutorLive } from "../src/process"
import { verifyCliInterruption } from "../src/suites/cli-interruption"

for (const mode of ["observation-error", "early-exit", "cancelled"] as const) test.skipIf(process.platform === "win32")(`CLI interruption resumes its service on ${mode}`, () =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-interrupt-" })
    const executable = `${root}/cli`
    yield* fs.writeFileString(executable, mode === "early-exit" ? "#!/bin/sh\nexit 0\n" : "#!/bin/sh\nexec /bin/sleep 60\n", { mode: 0o700 })
    const service = yield* Effect.acquireRelease(Effect.sync(() => Bun.spawn([process.execPath, "-e", "setInterval(() => {}, 1000)"], {
      stdin: "ignore", stdout: "ignore", stderr: "ignore",
    })), child => Effect.gen(function* () {
      child.kill("SIGKILL")
      yield* Effect.promise(() => child.exited)
    }))
    const observed = yield* Deferred.make<void>()
    const inspector = Layer.succeed(ProcessExecutor, { run: () => Deferred.succeed(observed, undefined).pipe(Effect.zipRight(
      mode === "observation-error" ? Effect.fail(new InfrastructureFailure({ operation: "fixture", message: "inspection failed" }))
        : Effect.succeed({ exitCode: 1, stdout: "", stderr: "" }),
    )) })
    const run = verifyCliInterruption({ executable, port: 11449, environment: { PATH: "/usr/bin:/bin:/usr/sbin:/sbin" } },
      ApplicationIdentity.make({ applicationPid: LabProcessId.make(process.pid), servicePid: LabProcessId.make(service.pid), serviceInstance: ServiceInstanceId.make("fixture") }))
      .pipe(Effect.provide(inspector))
    if (mode === "cancelled") {
      const fiber = yield* Effect.forkScoped(run)
      yield* Deferred.await(observed)
      yield* Fiber.interrupt(fiber)
    } else {
      const result = yield* Effect.either(run)
      expect(result._tag).toBe("Left")
    }
    const state = yield* checkedCommand("ps", ["-o", "state=", "-p", String(service.pid)])
    expect(state.stdout.trim().length).toBeGreaterThan(0)
    expect(state.stdout).not.toContain("T")
  })).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))), 30_000)
