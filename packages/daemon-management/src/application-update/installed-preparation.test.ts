import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { PrivateFilePermissions } from "../desktop-native/private-files"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { UpdatePreferences } from "../desktop-native/update-preferences"
import { makeInstalledUpdatePreparation } from "./installed-preparation"
import { readPreparedUpdateState } from "./finite-update"

const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
const noWrites = PrivateFilePermissions.of({
  prepareDirectory: () => Effect.die("Observation attempted to create a directory"),
  createFile: () => Effect.die("Observation attempted to create a file"),
  protectFile: () => Effect.die("Observation attempted to change permissions"),
})

describe("installed update preparation", () => {
  it.each(["darwin", "win32"] as const)("observes %s state without source acquisition or filesystem writes", async platform => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-installed-preparation-"))
    try {
      await writeFile(join(root, "update-configuration.json"), JSON.stringify({ origin: "https://magnitude.dev", keyId: "test", publicKey, acceptance: false }))
      const state = await Effect.runPromise(Effect.gen(function* () {
        const preparation = yield* makeInstalledUpdatePreparation({ resources: root, addonPath: "must-not-load.node",
          dataDirectory: join(root, "missing-profile"), version: "0.1.5", osVersion: "24.0.0", platform,
          architecture: platform === "win32" ? "x64" : "arm64", isolated: true })
        const observed = yield* readPreparedUpdateState.pipe(Effect.provideService(PreparedUpdateStore, preparation.store),
          Effect.provideService(UpdatePreferences, preparation.preferences))
        // Refusal precedes identity acquisition and native verification.
        expect(yield* preparation.makeSource.pipe(Effect.isFailure)).toBe(true)
        return observed
      }).pipe(Effect.provideService(PrivateFilePermissions, noWrites), Effect.provide(BunContext.layer)))
      expect(state).toEqual({ transfer: { _tag: "Idle" }, check: { _tag: "Idle" }, preference: { _tag: "Known", autoDownload: true } })
      expect(await readdir(root)).toEqual(["update-configuration.json"])
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
