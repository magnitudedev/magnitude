import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join, resolve } from "node:path"
import { acquireUpdateInstallationLease, MacUpdateHandoffRequest, nativeHostLayer, observeMacApplicationInstallation, relaunchMacAfterUpdate,
  MacUpdateHandoffFailed } from "@magnitudedev/daemon-management/desktop-native"
import { readUpdateHandoff, UpdateHandoffChannelFailed } from "./update-handoff-channel"

export const runMacUpdateHandoff = () => Effect.runPromise(Effect.gen(function* () {
  const channel = yield* readUpdateHandoff(process.stdin, MacUpdateHandoffRequest)
  const request = channel.request
  if (process.platform !== "darwin" || resolve(process.execPath) !== join(request.helperDirectory, "magnitude-update")) return yield* new UpdateHandoffChannelFailed()
  yield* Effect.scoped(Effect.gen(function* () {
    yield* acquireUpdateInstallationLease(request.stateDirectory)
    const active = yield* observeMacApplicationInstallation(request.bundle)
    yield* Effect.async<void, UpdateHandoffChannelFailed>(resume => {
      process.stdout.write("ready\n", error => resume(error ? new UpdateHandoffChannelFailed() : Effect.void))
    })
    yield* channel.awaitOwnerExit
    yield* Effect.gen(function* () {
      while (yield* active) yield* Effect.sleep("100 millis")
    }).pipe(Effect.timeoutFail({ duration: "5 minutes", onTimeout: () => new MacUpdateHandoffFailed({ message: "The native update is still running." }) }))
  })).pipe(Effect.provide(nativeHostLayer(join(request.helperDirectory, "desktop-host.node"))))
  yield* relaunchMacAfterUpdate(request)
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.sync(() => { process.exitCode = 1 }))))
