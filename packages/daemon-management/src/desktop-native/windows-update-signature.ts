import { Context, Effect, Layer, Schema } from "effect"
import { createRequire } from "node:module"

export class WindowsInstallerSignatureFailed extends Schema.TaggedError<WindowsInstallerSignatureFailed>()("WindowsInstallerSignatureFailed", {}) {}
export interface WindowsInstallerVerifier {
  readonly verify: (path: string) => Effect.Effect<void, WindowsInstallerSignatureFailed>
}
export const WindowsInstallerVerifier = Context.GenericTag<WindowsInstallerVerifier>("@magnitudedev/daemon-management/WindowsInstallerVerifier")

/** Publisher identity comes from the installed build, never from an update response. */
export const nativeWindowsInstallerVerifier = (addonPath: string, organization: string) => Layer.effect(WindowsInstallerVerifier, Effect.gen(function* () {
  const bindings = yield* Effect.try({
    try: () => createRequire(import.meta.url)(addonPath) as { readonly verifyInstallerSignature: (path: string, organization: string) => Promise<void> },
    catch: () => new WindowsInstallerSignatureFailed(),
  })
  return WindowsInstallerVerifier.of({
    // Native verification retains a file handle; let it close before staged-file cleanup.
    verify: path => Effect.tryPromise({ try: () => bindings.verifyInstallerSignature(path, organization), catch: () => new WindowsInstallerSignatureFailed() }).pipe(Effect.uninterruptible),
  })
}))
