import { access, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises"
import { constants } from "node:fs"
import { homedir } from "node:os"
import { dirname, isAbsolute, join } from "node:path"
import { randomUUID } from "node:crypto"
import { Effect, Option } from "effect"
import { LoginStartupFailed, type LoginStartupState } from "@magnitudedev/sdk/desktop-host"

const fileName = "dev.magnitude.desktop"
const failed = (error: unknown) => new LoginStartupFailed({ message: `Could not access desktop login startup: ${String(error)}` })
const absent = (error: unknown) => typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT"
export const renderXdgLoginStartup = (executable: string, enabled: boolean) => {
  if (!isAbsolute(executable) || /[\r\n\0=]/.test(executable)) throw new Error("Login startup requires an absolute executable path without control characters or equals signs")
  // Desktop-entry string escaping is applied before Exec argument unquoting.
  const argument = executable.replace(/[\\"`$]/g, character => `\\${character}`).replaceAll("\\", "\\\\").replaceAll("%", "%%")
  // GLib checks argv[0] before expanding %% escapes. A literal percent in the app
  // path would therefore fail admission. env execs the absolute path after expansion,
  // preserving the environment and PID without introducing shell interpretation.
  return `[Desktop Entry]\nType=Application\nName=Magnitude\nExec=/usr/bin/env "${argument}" --background\nTerminal=false\nHidden=${!enabled}\n`
}

export const makeXdgLoginStartup = (options: {
  readonly executable: string
  readonly configHome?: string
  readonly configDirectories?: readonly string[]
  readonly currentDesktops?: readonly string[]
}) => Effect.gen(function* () {
  const configHome = options.configHome ?? (process.env.XDG_CONFIG_HOME && isAbsolute(process.env.XDG_CONFIG_HOME) ? process.env.XDG_CONFIG_HOME : join(homedir(), ".config"))
  const systemDirectories = options.configDirectories ?? (process.env.XDG_CONFIG_DIRS ?? "/etc/xdg").split(":").filter(isAbsolute)
  const desktops = options.currentDesktops ?? (process.env.XDG_CURRENT_DESKTOP ?? "").split(":")
  const path = join(configHome, "autostart", fileName)
  const expected = yield* Effect.try({ try: () => renderXdgLoginStartup(options.executable, true).split("\n").find(line => line.startsWith("Exec="))!.slice(5), catch: failed })
  const read = Effect.gen(function* () {
    let document = Option.none<string>()
    for (const directory of [configHome, ...systemDirectories]) {
      document = yield* Effect.tryPromise({ try: async () => {
        try { return Option.some(await readFile(join(directory, "autostart", fileName), "utf8")) }
        catch (error) { if (absent(error)) return Option.none<string>(); throw error }
      }, catch: failed })
      if (Option.isSome(document)) break
    }
    if (Option.isNone(document)) return { _tag: "Disabled" } as const
    const fields = new Map<string, string>()
    let inEntry = false
    for (const raw of document.value.split(/\r?\n/)) {
      const line = raw.trim()
      if (line.startsWith("[")) { inEntry = line === "[Desktop Entry]"; continue }
      if (!inEntry || line.startsWith("#") || !line.includes("=")) continue
      const split = line.indexOf("=")
      fields.set(line.slice(0, split).trim(), line.slice(split + 1))
    }
    if (fields.get("Hidden") === "true" || fields.get("X-GNOME-Autostart-enabled") === "false") return { _tag: "Disabled" } as const
    const only = fields.get("OnlyShowIn")?.split(";").filter(Boolean)
    const excluded = fields.get("NotShowIn")?.split(";").filter(Boolean)
    if ((only && !only.some(value => desktops.includes(value))) || excluded?.some(value => desktops.includes(value))) return { _tag: "Disabled" } as const
    if (fields.get("Type") !== "Application" || fields.get("Exec") !== expected || fields.get("DBusActivatable") === "true" || fields.has("AutostartCondition")) {
      return { _tag: "Unavailable", message: "The desktop login entry has changed. Enable launch at login again to restore it." } as const
    }
    const tryExec = fields.get("TryExec")
    if (tryExec) {
      if (!isAbsolute(tryExec)) return { _tag: "Unavailable", message: "The login entry has an unverified executable condition." } as const
      const exists = yield* Effect.tryPromise({ try: () => access(tryExec, constants.X_OK).then(() => true), catch: failed }).pipe(Effect.orElseSucceed(() => false))
      if (!exists) return { _tag: "Disabled" } as const
    }
    return { _tag: "Enabled" } as const
  })
  const lock = yield* Effect.makeSemaphore(1)
  return {
    read: read as Effect.Effect<LoginStartupState, LoginStartupFailed>,
    set: (enabled: boolean) => lock.withPermits(1)(Effect.tryPromise({ try: async () => {
      await mkdir(dirname(path), { recursive: true })
      const temporary = `${path}.${randomUUID()}.tmp`
      try { await writeFile(temporary, renderXdgLoginStartup(options.executable, enabled), { mode: 0o600, flag: "wx" }); await rename(temporary, path) }
      finally { await unlink(temporary).catch(error => { if (!absent(error)) throw error }) }
    }, catch: failed }).pipe(Effect.zipRight(read))),
  }
})
