import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { PrivateFilePermissions, WindowsInstallerVerifier } from "@magnitudedev/daemon-management/desktop-native"
import { WindowsInstallerSignatureFailed } from "../../packages/daemon-management/src/desktop-native/windows-update-signature"
import { UpdateClientMetadata, UpdateManifest } from "@magnitudedev/release/hosted-update"
import { PublisherKeyId, signUpdateManifest } from "../../packages/release/src/hosted-update/manifest"
import { makeWindowsUpdateSource } from "./windows-update-source"

// File copying is exercised on Windows; native ACL and Authenticode have separate executable tests.
describe.skipIf(process.platform !== "win32")("Windows installer staging", () => {
  it.each(["valid", "changed", "unsigned"] as const)("handles a %s installer before allowing handoff", async scenario => {
    const directory = await mkdtemp(join(tmpdir(), "windows-update-stage-"))
    const archive = join(directory, "download.exe")
    const cli = join(directory, "magnitude.exe")
    const bytes = Buffer.from("publisher-verified installer fixture")
    await writeFile(archive, scenario === "changed" ? Buffer.from("changed installer bytes") : bytes)
    await writeFile(cli, "bundled CLI fixture")
    const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "windows", target: { os: "windows", arch: "x64", package: "windows-exe" }, path: "releases/2.0.0/desktop.exe",
      bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, PublisherKeyId.make("test"), generateKeyPairSync("ed25519").privateKey))
    let signatures = 0
    try {
      await Effect.runPromise(Effect.gen(function* () {
        const fs = yield* FileSystem.FileSystem
        const windows = yield* makeWindowsUpdateSource({ origin: "https://magnitude.dev", storageOrigin: "https://storage.example", trustedPublishers: new Map(),
          metadata: yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: "1.0.0", os: "windows", os_version: "10", arch: "x64", package: "windows-exe" }),
          sign: () => Effect.succeed("unused"), userAgent: "fixture", cacheDirectory: join(directory, "cache"), stateDirectory: directory,
          applicationPath: join(directory, "Magnitude.exe"), cliPath: cli,
        }).pipe(Effect.provideService(PrivateFilePermissions, {
          prepareDirectory: path => fs.makeDirectory(path).pipe(Effect.orDie),
          createFile: path => fs.writeFileString(path, "", { flag: "wx" }).pipe(Effect.orDie),
          protectFile: () => Effect.void,
        }), Effect.provideService(WindowsInstallerVerifier, {
          verify: path => Effect.gen(function* () {
            signatures++
            expect(yield* fs.readFile(path).pipe(Effect.orDie)).toEqual(new Uint8Array(bytes))
            if (scenario === "unsigned") return yield* new WindowsInstallerSignatureFailed()
          }),
        }))
        const outcome = yield* windows.source.stage(archive, { manifest, envelope }).pipe(Effect.either)
        expect(outcome._tag).toBe(scenario === "valid" ? "Right" : "Left")
        expect(signatures).toBe(scenario === "changed" ? 0 : 1)
        const entries = yield* fs.readDirectory(join(directory, "application-updates"))
        if (scenario === "valid") {
          expect(entries).toHaveLength(1)
          const prepared = join(directory, "application-updates", entries[0]!)
          expect(yield* fs.readFileString(join(prepared, "magnitude-update.exe"))).toBe("bundled CLI fixture")
          expect(yield* fs.readFileString(join(prepared, "magnitude-setup.exe"))).toBe(bytes.toString())
          yield* windows.discard
        } else expect(entries).toEqual([])
        expect(yield* fs.readDirectory(join(directory, "application-updates"))).toEqual([])
      }).pipe(Effect.provide(NodeContext.layer)))
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
})
