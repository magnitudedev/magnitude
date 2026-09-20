import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { basename, dirname, join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "../domain"
import { sha256 } from "../snapshot"

export const RemovalLoginEntry = Schema.Struct({ path: Schema.String, executable: Schema.String, sha256: Digest })
export const RemovedLoginEntry = Schema.Struct({ path: Schema.String, state: Schema.Literal("Absent", "Dormant") })
/** The UI must first enable this real installed-user entry; an initially absent entry cannot qualify cleanup. */
export const captureRemovalLogin = (environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (process.platform !== "linux" || !environment.HOME) return yield* new InfrastructureFailure({ operation: "uninstall-login", message: "Native login removal verification requires a qualified Linux desktop user" })
  const path = join(environment.XDG_CONFIG_HOME || join(environment.HOME, ".config"), "autostart/dev.magnitude.desktop")
  if ((yield* fs.realPath(path)) !== path || Number((yield* fs.stat(path)).size) > 16 * 1024) return yield* new AssertionFailure({ message: "Login entry is not an owned regular bounded file" })
  const wire = yield* fs.readFileString(path)
  const lines = wire.split(/\r?\n/)
  const executable = "/usr/bin/magnitude-desktop"
  for (const entry of ["Type=Application", "Hidden=false", `TryExec=${executable}`, `Exec=/usr/bin/env "${executable}" --background`]) {
    if (lines.filter(line => line === entry).length !== 1) return yield* new AssertionFailure({ message: "Enabled login entry does not guard the installed desktop executable" })
  }
  if (!(yield* fs.exists(executable))) return yield* new AssertionFailure({ message: "Enabled login entry has no installed executable" })
  return RemovalLoginEntry.make({ path, executable, sha256: sha256(wire) })
})

export const verifyRemovedLogin = (before: typeof RemovalLoginEntry.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const entries = yield* fs.readDirectory(dirname(before.path)).pipe(Effect.catchTag("SystemError", error => error.reason === "NotFound" ? Effect.succeed([] as string[]) : Effect.fail(error)))
  if (!entries.includes(basename(before.path))) return RemovedLoginEntry.make({ path: before.path, state: "Absent" })
  if ((yield* fs.realPath(before.path)) !== before.path || sha256(yield* fs.readFileString(before.path)) !== before.sha256) {
    return yield* new AssertionFailure({ message: "Retained login preference changed during uninstall" })
  }
  if (yield* fs.exists(before.executable)) return yield* new AssertionFailure({ message: "Uninstalled application still has a runnable login entry" })
  return RemovedLoginEntry.make({ path: before.path, state: "Dormant" })
})
