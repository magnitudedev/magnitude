import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { execFile } from "node:child_process"
import { promisify } from "node:util"
import { Effect, Schedule } from "effect"
import { describe, expect, it } from "vitest"
import { makeXdgLoginStartup, renderXdgLoginStartup } from "./login-startup"

const setup = Effect.acquireRelease(Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-login-"))), root => Effect.promise(() => rm(root, { recursive: true, force: true })))
const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))
const executable = "/opt/Magnitude/magnitude"
describe("graphical-session login startup", () => {
  it("is observational until changed and always launches the desktop in background", () => run(Effect.gen(function* () {
    const root = yield* setup
    const configHome = join(root, "config")
    const startup = yield* makeXdgLoginStartup({ executable, configHome, configDirectories: [] })
    expect((yield* startup.read)._tag).toBe("Disabled")
    expect(yield* Effect.promise(() => readFile(join(configHome, "autostart/dev.magnitude.desktop"), "utf8").catch(() => null))).toBeNull()
    expect((yield* startup.set(true))._tag).toBe("Enabled")
    const entry = yield* Effect.promise(() => readFile(join(configHome, "autostart/dev.magnitude.desktop"), "utf8"))
    expect(entry).toContain('Exec=/usr/bin/env "/opt/Magnitude/magnitude" --background')
    expect(entry).not.toContain("magnitude-service")
    expect((yield* startup.set(false))._tag).toBe("Disabled")
  })))
  it("disabling overrides a system entry rather than exposing it again", () => run(Effect.gen(function* () {
    const root = yield* setup
    const system = join(root, "system")
    yield* Effect.promise(() => mkdir(join(system, "autostart"), { recursive: true }))
    yield* Effect.promise(() => writeFile(join(system, "autostart/dev.magnitude.desktop"), renderXdgLoginStartup(executable, true)))
    const startup = yield* makeXdgLoginStartup({ executable, configHome: join(root, "user"), configDirectories: [system] })
    expect((yield* startup.read)._tag).toBe("Enabled")
    expect((yield* startup.set(false))._tag).toBe("Disabled")
    expect(yield* Effect.promise(() => readFile(join(system, "autostart/dev.magnitude.desktop"), "utf8"))).toContain("Hidden=false")
  })))
  it.each([
    ["Hidden=true", "Disabled"],
    ["X-GNOME-Autostart-enabled=false", "Disabled"],
    ["OnlyShowIn=KDE;", "Disabled"],
    ["NotShowIn=GNOME;", "Disabled"],
    ["OnlyShowIn=GNOME;Unity;", "Enabled"],
    ["DBusActivatable=true", "Unavailable"],
    ["AutostartCondition=if-exists config", "Unavailable"],
    ['Exec="/different/app" --background', "Unavailable"],
  ])("observes desktop control %s", (condition, expected) => run(Effect.gen(function* () {
    const root = yield* setup
    const startup = yield* makeXdgLoginStartup({ executable, configHome: root, configDirectories: [], currentDesktops: ["GNOME"] })
    yield* startup.set(true)
    const path = join(root, "autostart/dev.magnitude.desktop")
    yield* Effect.promise(() => writeFile(path, renderXdgLoginStartup(executable, true) + condition + "\n"))
    expect((yield* startup.read)._tag).toBe(expected)
    expect(yield* Effect.promise(() => readFile(path, "utf8"))).toContain(condition)
  })))
  it("escapes both desktop-entry and argv quoting and rejects malformed paths", () => {
    expect(renderXdgLoginStartup('/opt/My $App/100%/a\\b"c', true)).toContain('Exec=/usr/bin/env "/opt/My \\\\$App/100%%/a\\\\\\\\b\\\\"c" --background')
    expect(() => renderXdgLoginStartup("relative/app", true)).toThrow()
    expect(() => renderXdgLoginStartup("/opt/app\nExec=other", true)).toThrow()
  })
})

describe.skipIf(process.platform !== "linux" || process.env.MAGNITUDE_TEST_XDG !== "1")("native desktop-entry launch", () => {
  it.each(["Magnitude", "My App", "My $App", "100%", "a\\b", 'a"c'])("launches %s with the exact background argument", name => run(Effect.gen(function* () {
    const root = yield* setup
    const executable = join(root, name)
    const receipt = join(root, "argv")
    yield* Effect.promise(() => writeFile(executable, `#!/bin/sh\nprintf '%s\\n' "$@" > "${receipt}"\n`, { mode: 0o700 }))
    const startup = yield* makeXdgLoginStartup({ executable, configHome: root, configDirectories: [] })
    expect((yield* startup.set(true))._tag).toBe("Enabled")
    const entry = join(root, "autostart/dev.magnitude.desktop")
    yield* Effect.promise(() => promisify(execFile)("desktop-file-validate", [entry]))
    yield* Effect.promise(() => promisify(execFile)("gio", ["launch", entry]))
    const argument = yield* Effect.tryPromise(() => readFile(receipt, "utf8")).pipe(
      Effect.retry(Schedule.intersect(Schedule.spaced("20 millis"), Schedule.recurs(50))),
    )
    expect(argument).toBe("--background\n")
    expect((yield* startup.set(false))._tag).toBe("Disabled")
    yield* Effect.promise(() => promisify(execFile)("desktop-file-validate", [entry]))
  })))
})
