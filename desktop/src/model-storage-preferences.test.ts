import { NodeContext } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { makeAppearancePreferences, makeModelStoragePreferences } from "@magnitudedev/daemon-management/desktop-native"

it("defaults to <dataDir>/models and persists a folder without replacing other configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-model-storage-"))
  const make = () => Effect.runPromise(makeModelStoragePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
  try {
    const preferences = await make()
    expect(preferences.defaultPath).toBe(join(directory, "models"))
    expect(await Effect.runPromise(preferences.read)).toEqual({ path: join(directory, "models"), root: join(directory, "models"), source: "Default", warning: Option.none(), defaultPath: join(directory, "models") })
    const path = join(directory, "config.json")
    await writeFile(path, JSON.stringify({ appearance: "dark", contextLimits: { softCapRatio: 0.8 }, extra: { value: 42 } }))
    const target = join(directory, "external", "Models")
    await Effect.runPromise(preferences.write(Option.some(` ${target} `)))
    expect(await Effect.runPromise((await make()).read)).toEqual({ path: target, root: target, source: "Configured", warning: Option.none(), defaultPath: join(directory, "models") })
    expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({ modelsDirectory: target, appearance: "dark", contextLimits: { softCapRatio: 0.8 }, extra: { value: 42 } })
    const appearance = await Effect.runPromise(makeAppearancePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    await Effect.runPromise(appearance.write("light"))
    expect((await Effect.runPromise(preferences.read)).path).toBe(target)
    await Effect.runPromise(preferences.write(Option.none()))
    expect((await Effect.runPromise(preferences.read)).source).toBe("Default")
    expect(JSON.parse(await readFile(path, "utf8"))).not.toHaveProperty("modelsDirectory")
    expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({ appearance: "light", extra: { value: 42 } })
  } finally { await rm(directory, { recursive: true, force: true }) }
})

it("rejects relative folders, reports a relative saved value as the default, and surfaces failed saves", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-model-storage-"))
  const path = join(directory, "config.json")
  try {
    const preferences = await Effect.runPromise(makeModelStoragePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    for (const value of ["models", "", "   "]) {
      expect((await Effect.runPromise(preferences.write(Option.some(value)).pipe(Effect.either)))._tag).toBe("Left")
    }
    await writeFile(path, JSON.stringify({ modelsDirectory: "relative/models" }))
    const read = await Effect.runPromise(preferences.read)
    expect(read.source).toBe("Default")
    expect(Option.isSome(read.warning)).toBe(true)
    expect(await readFile(path, "utf8")).toBe(JSON.stringify({ modelsDirectory: "relative/models" }))
    await writeFile(path, "broken")
    expect((await Effect.runPromise(preferences.read.pipe(Effect.either)))._tag).toBe("Left")
    const file = join(directory, "file.txt")
    await writeFile(file, "x")
    await writeFile(path, JSON.stringify({ modelsDirectory: file }))
    const fileRead = await Effect.runPromise(preferences.read)
    expect(fileRead.source).toBe("Default")
    expect(fileRead.root).toBe(join(directory, "models"))
    expect(Option.isSome(fileRead.warning)).toBe(true)
    await rm(path)
    await mkdir(path)
    expect((await Effect.runPromise(preferences.write(Option.some(join(directory, "x"))).pipe(Effect.either)))._tag).toBe("Left")
  } finally { await rm(directory, { recursive: true, force: true }) }
})
