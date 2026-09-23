import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { makeAppearancePreferences, makeUpdatePreferences } from "@magnitudedev/daemon-management/desktop-native"

it("defaults to System and persists appearance without replacing other configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-appearance-"))
  const make = () => Effect.runPromise(makeAppearancePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
  try {
    const preferences = await make()
    expect(await Effect.runPromise(preferences.read)).toBe("system")
    const path = join(directory, "config.json")
    await writeFile(path, JSON.stringify({ autoDownloadUpdates: false, contextLimits: { softCapRatio: 0.8 }, extra: { value: 42 } }))
    for (const appearance of ["dark", "light", "system"] as const) {
      await Effect.runPromise(preferences.write(appearance))
      expect(await Effect.runPromise((await make()).read)).toBe(appearance)
      expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({ appearance, autoDownloadUpdates: false, contextLimits: { softCapRatio: 0.8 }, extra: { value: 42 } })
    }
    const updates = await Effect.runPromise(makeUpdatePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    await Effect.runPromise(updates.write(true))
    expect(await Effect.runPromise(preferences.read)).toBe("system")
  } finally { await rm(directory, { recursive: true, force: true }) }
})

it("reports invalid settings and failed saves without modifying configuration on read", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-appearance-"))
  const path = join(directory, "config.json")
  try {
    const preferences = await Effect.runPromise(makeAppearancePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    for (const contents of ['{"appearance":"invalid"}', 'broken']) {
      await writeFile(path, contents)
      expect((await Effect.runPromise(preferences.read.pipe(Effect.either)))._tag).toBe("Left")
      expect(await readFile(path, "utf8")).toBe(contents)
    }
    await rm(path)
    await mkdir(path)
    expect((await Effect.runPromise(preferences.write("dark").pipe(Effect.either)))._tag).toBe("Left")
  } finally { await rm(directory, { recursive: true, force: true }) }
})
