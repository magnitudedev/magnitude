import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { decodeApplicationUpdateConfiguration, readInstalledUpdateConfiguration } from "./update-configuration"

const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
const config = { origin: "https://magnitude.dev", keyId: "test", publicKey, acceptance: false }
const read = (root: string) => readInstalledUpdateConfiguration(root).pipe(Effect.provide(BunContext.layer))

describe("installed update configuration", () => {
  it("decodes the shared publisher and optional Windows identity", async () => {
    const decoded = await Effect.runPromise(decodeApplicationUpdateConfiguration(config))
    expect(decoded.trustedPublishers.get("test")?.export({ type: "spki", format: "pem" })).toBe(publicKey)
    expect(Option.isNone(decoded.windowsPublisher)).toBe(true)
    expect((await Effect.runPromise(decodeApplicationUpdateConfiguration({ ...config, windowsPublisher: "Publisher" }))).windowsPublisher)
      .toEqual(Option.some("Publisher"))
  })
  it.each([
    { ...config, acceptance: true },
    { ...config, publicKey: "invalid" },
    { ...config, publicKey: generateKeyPairSync("ec", { namedCurve: "prime256v1" }).publicKey.export({ type: "spki", format: "pem" }).toString() },
    { ...config, unexpected: true },
    { ...config, windowsPublisher: "" },
  ])("rejects invalid trust configuration %#", async value => {
    expect(await Effect.runPromise(decodeApplicationUpdateConfiguration(value).pipe(Effect.isFailure))).toBe(true)
  })
  it("allows a separately built acceptance endpoint", async () => {
    expect((await Effect.runPromise(decodeApplicationUpdateConfiguration({ ...config, acceptance: true, origin: "http://127.0.0.1:8765" }))).acceptance).toBe(true)
  })
  it("reads packaged configuration without creating other files and rejects corrupt or oversized contents", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-update-config-"))
    try {
      expect(await Effect.runPromise(read(root).pipe(Effect.isFailure))).toBe(true)
      expect(await readdir(root)).toEqual([])
      const path = join(root, "update-configuration.json")
      await writeFile(path, JSON.stringify(config))
      expect((await Effect.runPromise(read(root))).keyId).toBe("test")
      expect(await readdir(root)).toEqual(["update-configuration.json"])
      for (const contents of ["{", Buffer.from([0xff]), JSON.stringify(config) + " ".repeat(16384)]) {
        await writeFile(path, contents)
        const result = await Effect.runPromise(read(root).pipe(Effect.flip))
        expect(result._tag).toBe("InstalledUpdateConfigurationFailed")
      }
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
