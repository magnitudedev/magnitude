import { CommandExecutor } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it, vi } from "vitest"
import { PublisherKeyId, signUpdateManifest, UpdateManifest } from "../../../release/src/hosted-update/manifest"
import { makeLinuxPackageInstaller } from "./linux-update-package"

const uid = process.getuid!()
afterEach(() => vi.unstubAllGlobals())

describe("privileged Linux application package verification", () => {
  it.each(["valid", "hash", "signature", "target", "downgrade", "identity", "package-manager"] as const)("checks %s before admitting installation", async scenario => {
    vi.stubGlobal("process", { ...process, platform: "linux", arch: "arm64", getuid: () => 0 })
    const directory = await mkdtemp(join(tmpdir(), "linux-package-verification-"))
    const archive = join(directory, "update.deb")
    const bytes = Buffer.from("publisher-owned package bytes")
    await writeFile(archive, bytes)
    const key = generateKeyPairSync("ed25519")
    const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "desktop-linux-arm64-gnu-deb", target: { os: "linux", arch: scenario === "target" ? "x64" : "arm64", package: "deb" },
      path: "releases/2.0.0/magnitude.deb", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, PublisherKeyId.make("test"), key.privateKey))
    if (scenario === "hash") await writeFile(archive, Buffer.alloc(bytes.length))
    let queried = false
    let installed = false
    const executor = CommandExecutor.makeExecutor(command => Effect.gen(function* () {
      throw new Error(`Unexpected process start: ${command._tag}`)
    }))
    try {
      const result = await Effect.runPromise(Effect.gen(function* () {
        const installer = yield* makeLinuxPackageInstaller({ package: "deb", callerUid: uid, currentVersion: scenario === "downgrade" ? "3.0.0" : "1.0.0",
          trustedPublishers: new Map([["test", scenario === "signature" ? generateKeyPairSync("ed25519").publicKey : key.publicKey]]) })
        return yield* installer.install({ envelope, packagePath: archive }).pipe(Effect.either)
      }).pipe(Effect.provideService(CommandExecutor.CommandExecutor, {
        ...executor,
        string: command => Effect.promise(async () => {
          queried = true
          expect(command._tag).toBe("StandardCommand")
          if (command._tag !== "StandardCommand") throw new Error("Expected native package query")
          expect(command.command).toBe("/usr/bin/dpkg-deb")
          expect(command.args.at(-1)).not.toBe(archive)
          expect(await readFile(command.args.at(-1)!)).toEqual(bytes)
          return `${scenario === "identity" ? "another-package" : "magnitude-desktop"}\t2.0.0-42\tarm64`
        }),
        exitCode: command => Effect.sync(() => {
          installed = true
          expect(command._tag).toBe("StandardCommand")
          if (command._tag === "StandardCommand") {
            expect(command.command).toBe("/usr/bin/apt-get")
            expect(command.args.slice(0, 4)).toEqual(["install", "--yes", "--no-remove", "--"])
          }
          return CommandExecutor.ExitCode(scenario === "package-manager" ? 100 : 0)
        }),
      }), Effect.provide(BunContext.layer)))
      expect(result._tag).toBe(scenario === "valid" ? "Right" : "Left")
      expect(queried).toBe(["valid", "identity", "package-manager"].includes(scenario))
      expect(installed).toBe(["valid", "package-manager"].includes(scenario))
      expect(await readdir(directory)).toEqual(["update.deb"])
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
})
