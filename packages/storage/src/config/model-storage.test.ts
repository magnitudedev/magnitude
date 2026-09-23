import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, mkdir, realpath, rm, symlink, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { defaultModelStoreRoot, resolveModelStoreLocation, selectModelStorePath } from "./model-storage"

describe("model store selection", () => {
  it("defaults to <dataDir>/models", () => {
    const selection = selectModelStorePath("/data", Option.none())
    expect(selection).toEqual({ path: join("/data", "models"), source: "Default", warning: Option.none() })
    expect(defaultModelStoreRoot("/data")).toBe(join("/data", "models"))
  })

  it("uses an absolute configured path", () => {
    const selection = selectModelStorePath("/data", Option.some("/Volumes/SSD/Models/"))
    expect(selection.source).toBe("Configured")
    expect(selection.path).toBe(join("/Volumes/SSD/Models/"))
    expect(selection.warning).toEqual(Option.none())
  })

  it("falls back with a warning for relative or blank values", () => {
    for (const value of ["models", "./models", "   ", ""]) {
      const selection = selectModelStorePath("/data", Option.some(value))
      expect(selection.source).toBe("Default")
      expect(selection.path).toBe(join("/data", "models"))
      expect(Option.isSome(selection.warning)).toBe(true)
    }
  })
})

describe("model store resolution", () => {
  let dataDir: string
  const resolve = (configured: Option.Option<string>) =>
    Effect.runPromise(resolveModelStoreLocation(dataDir, configured).pipe(Effect.provide(BunFileSystem.layer)))

  beforeEach(async () => {
    dataDir = await realpath(await mkdtemp(join(tmpdir(), "magnitude-model-store-")))
  })
  afterEach(async () => {
    await rm(dataDir, { recursive: true, force: true })
  })

  it("keeps a missing directory as-is so the engine can create it", async () => {
    const location = await resolve(Option.none())
    expect(location).toEqual({ path: join(dataDir, "models"), root: join(dataDir, "models"), source: "Default", warning: Option.none() })
    const target = join(dataDir, "elsewhere")
    const configured = await resolve(Option.some(target))
    expect(configured).toEqual({ path: target, root: target, source: "Configured", warning: Option.none() })
  })

  it("resolves a symlinked default store to its real directory", async () => {
    const real = join(dataDir, "external")
    await mkdir(real)
    await symlink(real, join(dataDir, "models"))
    const location = await resolve(Option.none())
    expect(location.source).toBe("Default")
    expect(location.path).toBe(join(dataDir, "models"))
    expect(location.root).toBe(real)
  })

  it("resolves a symlinked configured store to its real directory", async () => {
    const real = join(dataDir, "external")
    const link = join(dataDir, "link")
    await mkdir(real)
    await symlink(real, link)
    const location = await resolve(Option.some(link))
    expect(location).toEqual({ path: link, root: real, source: "Configured", warning: Option.none() })
  })

  it("falls back to the default when the configured path is a file", async () => {
    const file = join(dataDir, "not-a-directory")
    await writeFile(file, "x")
    const location = await resolve(Option.some(file))
    expect(location.source).toBe("Default")
    expect(location.root).toBe(join(dataDir, "models"))
    expect(Option.isSome(location.warning)).toBe(true)
  })

  it("falls back to the default for a relative configured path", async () => {
    const location = await resolve(Option.some("models"))
    expect(location.source).toBe("Default")
    expect(location.root).toBe(join(dataDir, "models"))
    expect(Option.isSome(location.warning)).toBe(true)
  })
})
