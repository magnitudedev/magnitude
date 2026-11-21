import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { recordResidentModels, resumeResidentModels } from "./model-resumption"

const directories: string[] = []
afterEach(() => { for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true }) })
const stateDirectory = () => { const directory = mkdtempSync(join(tmpdir(), "model-resumption-")); directories.push(directory); return directory }

describe("loading models again after an update", () => {
  it("does nothing when no models were recorded", async () => {
    const lines: string[] = []
    await Effect.runPromise(resumeResidentModels("http://127.0.0.1:1", stateDirectory(), line => Effect.sync(() => { lines.push(line) })).pipe(Effect.provide(BunContext.layer)))
    expect(lines).toEqual([])
  })
  it("removes a record left by an update that never restarted without loading its models", async () => {
    const directory = stateDirectory()
    const file = join(directory, "models-after-update.json")
    writeFileSync(file, JSON.stringify({ recordedAt: Date.now() - 60 * 60_000, models: ["lfm3-1.7b:gguf:q4-qat"] }))
    const lines: string[] = []
    await Effect.runPromise(resumeResidentModels("http://127.0.0.1:1", directory, line => Effect.sync(() => { lines.push(line) })).pipe(Effect.provide(BunContext.layer)))
    expect(existsSync(file)).toBe(false)
    expect(lines).toEqual([])
  })
  it("keeps the last record while the desktop app does not answer", async () => {
    const directory = stateDirectory()
    const file = join(directory, "models-after-update.json")
    const record = JSON.stringify({ recordedAt: Date.now(), after: "Desktop", models: ["lfm3-1.7b:gguf:q4-qat"] })
    writeFileSync(file, record)
    const result = await Effect.runPromise(recordResidentModels("http://127.0.0.1:1", directory, "Desktop").pipe(Effect.either, Effect.provide(BunContext.layer)))
    expect(result._tag).toBe("Left")
    expect(readFileSync(file, "utf8")).toBe(record)
  })
})
