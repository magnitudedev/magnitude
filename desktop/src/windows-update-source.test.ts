import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { makePreparedUpdateStore, PreparedUpdateStore, PrivateFilePermissions, WindowsInstallerVerifier } from "@magnitudedev/daemon-management/desktop-native"
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
    const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "windows", target: { os: "windows", arch: "x64", package: "windows-exe" }, filename: "desktop.exe",
      bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const publisher = generateKeyPairSync("ed25519")
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, publisher.privateKey))
    let signatures = 0
    try {
      await Effect.runPromise(Effect.gen(function* () {
        const fs = yield* FileSystem.FileSystem
        const permissions = PrivateFilePermissions.of({
          prepareDirectory: path => fs.makeDirectory(path, { recursive: true }).pipe(Effect.orDie),
          createFile: path => fs.writeFileString(path, "", { flag: "wx" }).pipe(Effect.orDie),
          protectFile: () => Effect.void,
        })
        const store = yield* makePreparedUpdateStore({ dataDirectory: directory, target: manifest.artifact.target,
          trustedPublishers: new Map([["test", publisher.publicKey]]) }).pipe(Effect.provideService(PrivateFilePermissions, permissions))
        const windows = yield* makeWindowsUpdateSource({ origin: "https://magnitude.dev", trustedPublishers: new Map(),
          metadata: yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: "1.0.0", os: "windows", os_version: "10", arch: "x64", package: "windows-exe" }),
          sign: () => Effect.succeed("unused"), userAgent: "fixture", cacheDirectory: join(directory, "cache"), stateDirectory: directory,
          applicationPath: join(directory, "Magnitude.exe"), cliPath: cli, dataDirectory: directory, addonPath: join(directory, "desktop-host.node"),
        }).pipe(Effect.provideService(PreparedUpdateStore, store), Effect.provideService(PrivateFilePermissions, permissions), Effect.provideService(WindowsInstallerVerifier, {
          verify: path => Effect.gen(function* () {
            signatures++
            if (scenario === "unsigned") return yield* new WindowsInstallerSignatureFailed()
          }),
        }))
        const outcome = yield* windows.source.stage(archive, envelope.release).pipe(Effect.either)
        expect(outcome._tag).toBe(scenario === "valid" ? "Right" : "Left")
        expect(signatures).toBe(1)
        const pending = yield* store.read
        expect(pending._tag).toBe(scenario === "valid" ? "Some" : "None")
        if (scenario === "valid") {
          expect(yield* fs.readFileString(join(directory, "updates", "magnitude-setup.exe"))).toBe(bytes.toString())
          expect(yield* fs.exists(join(directory, "update-helpers"))).toBe(false)
        }
      }).pipe(Effect.provide(NodeContext.layer)))
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
})
