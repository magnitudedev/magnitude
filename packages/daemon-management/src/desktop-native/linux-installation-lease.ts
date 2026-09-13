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
