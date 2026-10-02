import { BunContext } from "@effect/platform-bun"
import { Chunk, Effect, Layer, Option, Stream } from "effect"
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { LOOPBACK_ONLY, resolveNetworkAccess } from "@magnitudedev/storage"
import { AcnChanges, AcnChangesLive } from "./changes"
import { AcnHost, ServerSettings, ServerSettingsLive } from "./server-settings"

const layerFor = (dataDir: string, activeNetwork = LOOPBACK_ONLY) => ServerSettingsLive.pipe(
  Layer.provideMerge(AcnChangesLive),
  Layer.provide(Layer.succeed(AcnHost, { dataDir, port: 10100, activeNetwork })),
  Layer.provide(BunContext.layer),
)
const run = <A, E>(dataDir: string, effect: Effect.Effect<A, E, ServerSettings | AcnChanges>, activeNetwork = LOOPBACK_ONLY) =>
  Effect.runPromise(Effect.scoped(effect.pipe(Effect.provide(layerFor(dataDir, activeNetwork)))))
const fresh = () => mkdtempSync(join(tmpdir(), "magnitude-server-settings-"))

describe("model storage", () => {
  it("reports the default folder until one is configured", async () => {
    const dataDir = fresh()
    const settings = await run(dataDir, Effect.flatMap(ServerSettings, service => service.modelStorage))
    expect(settings).toMatchObject({ path: join(dataDir, "models"), active: join(dataDir, "models"), source: "Default", warning: Option.none() })
  })

  it("rejects a relative folder", async () => {
    const result = await run(fresh(), Effect.flatMap(ServerSettings, service => service.setModelStorage(Option.some("models"))).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })

  it("saves a folder for the next start while the running engine keeps its folder", async () => {
    const dataDir = fresh()
    const next = join(dataDir, "elsewhere")
    const settings = await run(dataDir, Effect.gen(function* () {
      const service = yield* ServerSettings
      yield* service.setModelStorage(Option.some(next))
      return yield* service.modelStorage
    }))
    expect(settings).toMatchObject({ path: next, active: join(dataDir, "models"), source: "Configured" })
  })

  it("tells other clients the setting changed", async () => {
    const pokes = await run(fresh(), Effect.gen(function* () {
      const changes = yield* AcnChanges
      const service = yield* ServerSettings
      const received = yield* changes.stream.pipe(Stream.take(1), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("10 millis")
      yield* service.setModelStorage(Option.none())
      return Chunk.toReadonlyArray(yield* received.await.pipe(Effect.flatten))
    }))
    expect(pokes).toEqual([{ operation: "GetModelStorage" }])
  })
})

describe("network access", () => {
  it("generates a key when first enabled and reports that a restart is pending", async () => {
    const settings = await run(fresh(), Effect.gen(function* () {
      const service = yield* ServerSettings
      yield* service.setNetworkAccess({ enabled: Option.some(true), bind: Option.none(), requireApiKey: Option.none() })
      return yield* service.networkAccess
    }))
    expect(settings.enabled).toBe(true)
    expect(Option.getOrThrow(settings.apiKey)).toMatch(/^mag-/)
    expect(settings.pending).toBe(true)
  })

  it("is not pending when the saved settings match what the service bound", async () => {
    const dataDir = fresh()
    writeFileSync(join(dataDir, "config.json"), JSON.stringify({ network: { enabled: true, apiKey: "mag-fixed", requireApiKey: true, allowedHosts: [] } }))
    const active = resolveNetworkAccess(Option.some({ enabled: true, apiKey: "mag-fixed", requireApiKey: true, allowedHosts: [] }))
    const settings = await run(dataDir, Effect.flatMap(ServerSettings, service => service.networkAccess), active)
    expect(settings.pending).toBe(false)
  })

  it("rejects an address that is not an IP address", async () => {
    const result = await run(fresh(), Effect.flatMap(ServerSettings, service => service.setNetworkAccess({
      enabled: Option.some(true), bind: Option.some({ _tag: "Address", address: "laptop.local" }), requireApiKey: Option.none(),
    })).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })

  it("replaces the key when regenerated", async () => {
    const [before, after] = await run(fresh(), Effect.gen(function* () {
      const service = yield* ServerSettings
      yield* service.setNetworkAccess({ enabled: Option.some(true), bind: Option.none(), requireApiKey: Option.none() })
      const first = yield* service.networkAccess
      yield* service.regenerateNetworkApiKey
      const second = yield* service.networkAccess
      return [first.apiKey, second.apiKey] as const
    }))
    expect(Option.isSome(before) && Option.isSome(after) && before.value !== after.value).toBe(true)
  })
})

describe("directory browsing", () => {
  it("lists visible child folders in name order with the parent", async () => {
    const root = fresh()
    for (const name of ["b", "a10", "a2", ".hidden"]) mkdirSync(join(root, name))
    writeFileSync(join(root, "file.txt"), "")
    const listing = await run(fresh(), Effect.flatMap(ServerSettings, service => service.browseDirectories(Option.some(root))))
    expect(listing.directories.map(entry => entry.name)).toEqual(["a2", "a10", "b"])
    expect(listing.parent).toEqual(Option.some(join(root, "..").replace(/\/$/, "")).pipe(Option.map(path => join(path))))
  })

  it("fails clearly for a folder that cannot be read", async () => {
    const result = await run(fresh(), Effect.flatMap(ServerSettings, service => service.browseDirectories(Option.some(join(fresh(), "missing")))).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })

  it("has no parent at the filesystem root", async () => {
    const listing = await run(fresh(), Effect.flatMap(ServerSettings, service => service.browseDirectories(Option.some("/"))))
    expect(listing.parent).toEqual(Option.none())
  })
})
