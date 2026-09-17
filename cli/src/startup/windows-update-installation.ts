import { BunContext } from "@effect/platform-bun"
import { acquireUpdateInstallationLease, completeWindowsUpdateHandoff, nativeHostLayer, relaunchWindowsAfterUpdate,
  WindowsUpdateHandoffRequest, windowsPrivateFilePermissions } from "@magnitudedev/daemon-management/desktop-native"
import { Effect } from "effect"
import { win32 } from "node:path"
import { readUpdateHandoff, UpdateHandoffChannelFailed } from "./update-handoff-channel"

export const runWindowsUpdateHandoff = () => Effect.runPromise(Effect.gen(function* () {
  const channel = yield* readUpdateHandoff(process.stdin, WindowsUpdateHandoffRequest)
  const request = channel.request
  if (process.platform !== "win32" || win32.resolve(process.execPath) !== win32.join(request.helperDirectory, "magnitude.exe")) {
    return yield* new UpdateHandoffChannelFailed()
  }
  const addon = win32.join(request.helperDirectory, "desktop-host.node")
  const completed = yield* Effect.scoped(Effect.gen(function* () {
    yield* acquireUpdateInstallationLease(request.stateDirectory)
    yield* Effect.async<void, UpdateHandoffChannelFailed>(resume => {
      process.stdout.write("ready\n", error => resume(error ? new UpdateHandoffChannelFailed() : Effect.void))
    })
    yield* channel.awaitOwnerExit
    return yield* completeWindowsUpdateHandoff(request).pipe(Effect.exit)
  })).pipe(Effect.provide([nativeHostLayer(addon), windowsPrivateFilePermissions(addon)]))
  // A recording failure still relaunches the old app, which reconciles the retained Attempted record.
  yield* relaunchWindowsAfterUpdate(request)
  yield* completed
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.sync(() => { process.exitCode = 1 }))))
