import { FileSystem } from "@effect/platform"
import { Effect, Schedule } from "effect"
import { lstat, readlink, rename, symlink } from "node:fs/promises"
import { posix, win32 } from "node:path"

/** The Linux package's command; the desktop's CLI on Linux. */
export const LINUX_DESKTOP_COMMAND = "/usr/bin/magnitude"

const paths = (platform: string) => platform === "win32" ? win32 : posix

/**
 * The desktop app replaces a CLI-only install in this account: its version folders go, and on Linux the
 * `~/.magnitude/bin/magnitude` link is pointed at the package's command, so shells and background services
 * that call it keep working (macOS points the same link at the app's CLI through CLI registration; Windows
 * keeps the launcher, which already prefers the desktop). Failures are logged; they never stop the app.
 */
export const absorbStandaloneInstallation = (options: { readonly home: string; readonly platform: NodeJS.Platform }) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = paths(options.platform)
  const root = path.join(options.home, ".magnitude", "cli")
  if (!(yield* fs.exists(root).pipe(Effect.orElseSucceed(() => false)))) return
  if (options.platform === "linux") {
    const link = path.join(options.home, ".magnitude", "bin", "magnitude")
    const current = yield* Effect.tryPromise(() => readlink(link)).pipe(Effect.option)
    const present = yield* Effect.tryPromise(() => lstat(link)).pipe(Effect.as(true), Effect.orElseSucceed(() => false))
    if (present && !(current._tag === "Some" && current.value === LINUX_DESKTOP_COMMAND)) {
      const temporary = `${link}.desktop-${process.pid}`
      yield* Effect.tryPromise(async () => {
        await symlink(LINUX_DESKTOP_COMMAND, temporary)
        await rename(temporary, link)
      })
    }
  }
  // A CLI-only serve that just yielded may still hold its files open for a few seconds (Windows refuses the delete).
  yield* fs.remove(root, { recursive: true }).pipe(Effect.retry(Schedule.spaced("1 second").pipe(Schedule.compose(Schedule.recurs(15)))))
  yield* Effect.logInfo("Replaced the CLI-only installation with the desktop app's CLI")
}).pipe(Effect.catchAll(error => Effect.logWarning(`Could not replace the CLI-only installation: ${String(error)}`)))
