import { CommandExecutor } from "@effect/platform"
import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { nativeHostLayerFromLoader } from "../desktop-native/index"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { WindowsInstallerVerifier } from "../desktop-native/windows-update-signature"
import { completeWindowsForegroundUpdate } from "./windows-foreground-update"

const events: string[] = []
const test = it.runIf(process.platform === "win32")
const keys = generateKeyPairSync("ed25519")
const release = await Effect.runPromise(signUpdateRelease({ version: "0.1.6", bytes: 1,
  sha256: createHash("sha256").update("x").digest("hex") }, { os: "windows", arch: "x64", package: "windows-exe" }, keys.privateKey))
beforeEach(() => { events.length = 0; vi.spyOn(process, "chdir").mockImplementation(() => {}) })
afterEach(() => vi.restoreAllMocks())
const run = async (options: { attempted?: boolean; automatic?: boolean; protocol?: string; installerCode?: number; version?: string } = {}) => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-windows-foreground-"))
  const locks = new Set<string>()
  const host = nativeHostLayerFromLoader(() => ({
    acquireLock: (path: string) => {
      if (locks.has(path)) return null
      locks.add(path)
      events.push(path.endsWith("application.lock") ? "owner acquired" : "installation acquired")
      return { path }
    },
    releaseLock: ({ path }: { path: string }) => {
      locks.delete(path)
      events.push(path.endsWith("application.lock") ? "owner released" : "installation released")
    },
  }))
  const step = (event: string) => Effect.sync(() => { events.push(event) })
  const store = PreparedUpdateStore.of({
    read: Effect.succeed(Option.some({ release, installation: options.attempted ? { _tag: "Attempted" } : { _tag: "Unattempted" } })),
    verify: () => step("verify").pipe(Effect.as("installer.exe")), recordAttempt: () => step("attempt"),
    recordFailure: () => step("failure"), discard: step("discard"), removeAbandonedTransfers: Effect.void, outcome: Effect.succeed(Option.none()), recordOutcome: () => Effect.void, markOutcomeReported: Effect.void,
    prepare: () => Effect.die("Unexpected download"),
  })
  return Effect.runPromise(completeWindowsForegroundUpdate({ resources: fileURLToPath(new URL("../../dist/native/win32-x64", import.meta.url)), dataDirectory: root,
    stateDirectory: join(root, "state"), version: "0.1.5", automatic: options.automatic ?? true, launcherProtocol: options.protocol ?? "1" }).pipe(
    Effect.provideService(PreparedUpdateStore, store), Effect.provideService(WindowsInstallerVerifier, { verify: () => step("publisher") }),
    Effect.provideService(CommandExecutor.CommandExecutor, {
      ...CommandExecutor.makeExecutor(() => Effect.die("Unexpected start")),
      exitCode: command => Effect.sync(() => {
        expect(command._tag === "StandardCommand" && command.command).toBe("installer.exe")
        expect(events).toContain("owner released")
        expect(locks.has(join(root, "state/update-installation.lock"))).toBe(true)
        events.push("installer")
        return CommandExecutor.ExitCode(options.installerCode ?? 0)
      }),
      string: () => step("version").pipe(Effect.as(options.version ?? "0.1.6\n")),
    }), Effect.provide(host), Effect.provide(BunFileSystem.layer), Effect.either)).finally(() => rm(root, { recursive: true, force: true }))
}
test("releases application ownership while retaining installation admission through completion", async () => {
  expect(await run()).toMatchObject({ _tag: "Right", right: true })
  expect(events).toEqual(["owner acquired", "installation acquired", "installation released", "installation acquired", "verify", "publisher", "attempt", "owner released", "installer", "version", "discard", "installation released"])
})
test.each([{ attempted: true }, { protocol: "unsupported" }])("defers startup without an eligible preparation and launcher: %j", async options => {
  expect(await run(options)).toMatchObject({ _tag: "Right", right: false })
  expect(events).toEqual(["owner acquired", "installation acquired", "installation released", "owner released"])
})
test("permits an explicit finite retry without a foreground launcher", async () => {
  expect(await run({ automatic: false, attempted: true, protocol: "unsupported" })).toMatchObject({ _tag: "Right", right: true })
})
test.each([{ installerCode: 1 }, { version: "0.1.5" }])("retains failed preparation instead of continuing: %j", async options => {
  expect(await run(options)).toMatchObject({ _tag: "Left" })
  expect(events).toContain("failure")
  expect(events).not.toContain("discard")
  expect(events.at(-1)).toBe("installation released")
})
