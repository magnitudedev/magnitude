import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { makeUpdatePreferences } from "./update-preferences"

it("defaults to automatic downloads, persists changes, and never silently resets corrupt preferences", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-update-preferences-"))
  const make = () => Effect.runPromise(makeUpdatePreferences(directory).pipe(Effect.provide(NodeContext.layer)))
  try {
    const preferences = await make()
    expect(await Effect.runPromise(preferences.read)).toBe(true)
    await Effect.runPromise(preferences.write(false))
    expect(await Effect.runPromise((await make()).read)).toBe(false)
    const path = join(directory, "updates", "preferences.json")
    await writeFile(path, "broken")
    expect((await Effect.runPromise(preferences.read.pipe(Effect.either)))._tag).toBe("Left")
    expect(await readFile(path, "utf8")).toBe("broken")
    // An explicit user choice repairs the preference; reads alone cannot change it.
    await Effect.runPromise(preferences.write(true))
    expect(await Effect.runPromise((await make()).read)).toBe(true)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
