import { CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Either, Layer } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { chmod, chown, mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { updateInstallerFilename } from "../../../release/src/hosted-update"
import { GuardedCommand } from "@magnitudedev/utils/guarded-command"
import { installLinuxApplicationUpdate, parseLinuxUpdateInstallation } from "./linux-update-maintenance"

const installed = "/usr/lib/magnitude-desktop"
const caller = 1000

describe("privileged update installer grammar", () => {
  const parse = (argv: readonly string[]) => Effect.runPromise(Effect.either(parseLinuxUpdateInstallation(argv)))
  it("accepts the request path, optionally followed by --parent-stdin", async () => {
    expect(await parse(["_install-application-update", "/var/lib/magnitude/updates/update.json"]))
      .toEqual(Either.right({ request: "/var/lib/magnitude/updates/update.json", parentStdin: false }))
    expect(await parse(["_install-application-update", "/var/lib/magnitude/updates/update.json", "--parent-stdin"]))
      .toEqual(Either.right({ request: "/var/lib/magnitude/updates/update.json", parentStdin: true }))
  })
  it.each([
    ["no request", ["_install-application-update"]],
    ["an extra argument", ["_install-application-update", "/a/updates/update.json", "--parent-stdin", "x"]],
    ["another flag", ["_install-application-update", "/a/updates/update.json", "--force"]],
    ["a flag before the request", ["_install-application-update", "--parent-stdin", "/a/updates/update.json"]],
    ["an option as the request", ["_install-application-update", "--help"]],
    ["a relative request", ["_install-application-update", "updates/update.json"]],
    ["a request with dot segments", ["_install-application-update", "/a/../etc/updates/update.json"]],
    ["a request with a trailing slash", ["_install-application-update", "/a/updates/update.json/"]],
    ["a request with NUL", ["_install-application-update", "/a/updates/update.json\0x"]],
    ["a leading program option", ["-v", "_install-application-update", "/a/updates/update.json"]],
    ["the end-of-options marker", ["_install-application-update", "--", "/a/updates/update.json"]],
    ["a flag with a value", ["_install-application-update", "/a/updates/update.json", "--parent-stdin=1"]],
    ["another command", ["_server-install", "/a/updates/update.json"]],
  ])("refuses %s", async (_, argv) => {
    expect(Either.isLeft(await parse(argv))).toBe(true)
  })
})

describe("privileged update installer refusals", () => {
  let root: string
  let system: string
  let data: string
  let installs: number
  const key = generateKeyPairSync("ed25519")
  const bytes = Buffer.from("signed magnitude package")
  const target = { os: "linux", arch: process.arch === "arm64" ? "arm64" : "x64", package: "deb" } as const
  const map = (path: string) => path === installed || path.startsWith(`${installed}/`) ? join(system, path) : path
  const unmap = (path: string) => path.startsWith(system) ? path.slice(system.length) : path

  beforeEach(async () => {
    installs = 0
    root = await mkdtemp(join(tmpdir(), "magnitude-installer-grammar-"))
    system = join(root, "system")
    data = join(root, "data")
    await mkdir(join(system, installed, "resources"), { recursive: true, mode: 0o755 })
    await writeFile(join(system, installed, "resources/update-trust.json"), JSON.stringify({ keyId: "test", publicKey: key.publicKey.export({ type: "spki", format: "pem" }) }), { mode: 0o644 })
    await writeFile(join(system, installed, "resources/update-package.json"), JSON.stringify({ format: "deb" }), { mode: 0o644 })
    await mkdir(join(data, "updates"), { recursive: true })
    await chown(data, caller, caller)
    await chown(join(data, "updates"), caller, caller)
  })
  afterEach(async () => { vi.unstubAllGlobals(); await rm(root, { recursive: true, force: true }) })

  const prepare = async (options: { readonly version?: string; readonly tag?: string; readonly signer?: typeof key.privateKey; readonly packageBytes?: Buffer } = {}) => {
    const release = await Effect.runPromise(signUpdateRelease({ version: options.version ?? "2.0.0", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex") }, target, options.signer ?? key.privateKey))
    const request = join(data, "updates/update.json")
    await writeFile(request, JSON.stringify({ release, installation: { _tag: options.tag ?? "Attempted" } }))
    await writeFile(join(data, "updates", updateInstallerFilename(target)), options.packageBytes ?? bytes)
    for (const path of [request, join(data, "updates", updateInstallerFilename(target))]) await chown(path, caller, caller)
    return request
  }

  const run = (requestPath: string, currentVersion = "1.0.0") => {
    vi.stubGlobal("process", { ...process, platform: "linux", getuid: () => 0, env: { SUDO_UID: String(caller) } })
    const real = Effect.runSync(Effect.provide(FileSystem.FileSystem, BunContext.layer))
    const fs = FileSystem.make({ ...real,
      stat: path => real.stat(map(path)),
      readFile: path => real.readFile(map(path)),
      realPath: path => path === process.execPath ? Effect.succeed(`${installed}/resources/magnitude`) : real.realPath(map(path)).pipe(Effect.map(unmap)),
    })
    const executor = CommandExecutor.makeExecutor(() => Effect.die("unexpected process"))
    return Effect.runPromise(installLinuxApplicationUpdate(requestPath, currentVersion).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(CommandExecutor.CommandExecutor, { ...executor,
        string: () => Effect.succeed(`magnitude-desktop\t2.0.0-1\t${target.arch === "arm64" ? "arm64" : "amd64"}`) }),
      Effect.provideService(GuardedCommand, { run: () => Effect.sync(() => { installs += 1; return { code: 0, stdout: "", stderr: "" } }) } as never),
      Effect.provide(Layer.empty), Effect.either))
  }

  it("installs a valid signed newer package requested by the caller", async () => {
    expect(Either.isRight(await run(await prepare()))).toBe(true)
    expect(installs).toBe(1)
  })
  it.each([
    ["an older version", async () => run(await prepare({ version: "0.9.0" }))],
    ["the installed version", async () => run(await prepare({ version: "1.0.0" }))],
    ["a tampered package", async () => run(await prepare({ packageBytes: Buffer.from("tampered magnitude packag") }))],
    ["a package signed by another key", async () => run(await prepare({ signer: generateKeyPairSync("ed25519").privateKey }))],
    ["an unattempted record", async () => run(await prepare({ tag: "Unattempted" }))],
    ["a foreign-owned request", async () => { const request = await prepare(); await chown(request, caller + 1, caller + 1); return run(request) }],
    ["a symlinked request", async () => {
      await prepare()
      await mkdir(join(root, "elsewhere/updates"), { recursive: true })
      await symlink(join(data, "updates/update.json"), join(root, "elsewhere/updates/update.json"))
      return run(join(root, "elsewhere/updates/update.json"))
    }],
    ["a request with another name", async () => {
      const request = await prepare()
      await writeFile(join(dirname(request), "other.json"), "{}")
      await chown(join(dirname(request), "other.json"), caller, caller)
      return run(join(dirname(request), "other.json"))
    }],
    ["writable publisher trust", async () => { await chmod(join(system, installed, "resources/update-trust.json"), 0o666); return run(await prepare()) }],
  ])("refuses %s and installs nothing", async (_, scenario) => {
    expect(Either.isLeft(await scenario())).toBe(true)
    expect(installs).toBe(0)
  })
})
