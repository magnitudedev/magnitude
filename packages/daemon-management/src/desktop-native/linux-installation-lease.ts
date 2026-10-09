import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
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

/**
 * Process ids holding, or waiting for, a lock on `inode` according to `/proc/locks`. A running owner
 * holds its own shared lease, so asking whether *others* hold the lock is the only probe that works
 * from inside it.
 */
export const linuxLockHolders = (procLocks: string, inode: bigint): ReadonlyArray<number> =>
  procLocks.split("\n").flatMap(line => {
    // "1: FLOCK  ADVISORY  READ  1234 08:01:131074 0 EOF", or "1: -> FLOCK ..." for a waiter.
    const fields = line.replace(/^\s*\d+:\s*(->\s*)?/, "").trim().split(/\s+/)
    if (fields.length < 5) return []
    const pid = Number(fields[3]), file = fields[4]!.split(":")
    return Number.isInteger(pid) && file.length === 3 && BigInt(Number.parseInt(file[2]!, 10)) === inode ? [pid] : []
  })

/** Whether any process other than this one holds or awaits the Linux installation lock. */
export const linuxInstallationLockHeldByOthers = (lockPath: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const inode = (yield* fs.stat(lockPath)).ino
  if (Option.isNone(inode)) return false
  return linuxLockHolders(yield* fs.readFileString("/proc/locks"), BigInt(inode.value)).some(pid => pid !== process.pid)
})
