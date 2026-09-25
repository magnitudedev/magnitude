import { Effect } from "effect"
import { createRequire } from "node:module"
import { NativeHostUnavailable } from "./index"

/** Keep the launcher's shared lease in Electron, never in service or installer children. */
export const adoptLinuxInstallationLease = (addonPath: string) => Effect.try({
  try: () => {
    const bindings = createRequire(import.meta.url)(addonPath) as { readonly adoptInstallationLease: () => void }
    bindings.adoptInstallationLease()
  },
  catch: () => new NativeHostUnavailable({ message: "Start Magnitude through its installed desktop launcher." }),
})

/** Retain admission in the foreground owner; the native descriptor is close-on-exec. */
export const acquireLinuxInstallationLease = (addonPath: string) => Effect.gen(function* () {
  const bindings = yield* Effect.try({
    try: () => createRequire(import.meta.url)(addonPath) as {
      readonly acquireInstallationLease: () => object
      readonly releaseInstallationLease: (lease: object) => void
    },
    catch: () => new NativeHostUnavailable({ message: "The Magnitude native installation adapter could not be loaded." }),
  })
  yield* Effect.acquireRelease(
    Effect.try({ try: () => bindings.acquireInstallationLease(), catch: error => new NativeHostUnavailable({ message: String(error) }) }),
    lease => Effect.sync(() => bindings.releaseInstallationLease(lease)),
  )
})
