import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { makeAppearancePreferences, makeFoundModelsNoticePreferences } from "@magnitudedev/daemon-management/desktop-native"

it("persists dismissal across restarts and other preference writes", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-found-models-"))
  const make = () => Effect.runPromise(makeFoundModelsNoticePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
  try {
    const preferences = await make()
    expect(await Effect.runPromise(preferences.readDismissed)).toBe(false)
    const path = join(directory, "config.json")
    await writeFile(path, JSON.stringify({ appearance: "dark", extra: { value: 42 } }))
    await Effect.runPromise(preferences.dismiss)
    expect(await Effect.runPromise((await make()).readDismissed)).toBe(true)
    expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({ foundModelsNoticeDismissed: true, appearance: "dark", extra: { value: 42 } })
    const appearance = await Effect.runPromise(makeAppearancePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    await Effect.runPromise(appearance.write("light"))
    expect(await Effect.runPromise((await make()).readDismissed)).toBe(true)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

it("reports invalid saved preferences without changing them on read, and reports failed saves", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-found-models-"))
  try {
    const path = join(directory, "config.json")
    const preferences = await Effect.runPromise(makeFoundModelsNoticePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    for (const contents of ['{"foundModelsNoticeDismissed":"invalid"}', 'broken']) {
      await writeFile(path, contents)
      expect((await Effect.runPromise(preferences.readDismissed.pipe(Effect.either)))._tag).toBe("Left")
      expect(await readFile(path, "utf8")).toBe(contents)
    }
    await rm(path)
    await mkdir(path)
    expect((await Effect.runPromise(preferences.dismiss.pipe(Effect.either)))._tag).toBe("Left")
  } finally { await rm(directory, { recursive: true, force: true }) }
})
