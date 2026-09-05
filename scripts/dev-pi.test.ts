import { Effect, Fiber, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { awaitPiDevelopmentModel, decodePiDevelopmentStatus, piDevelopmentArgs } from "./dev-pi"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DefaultResourceLoader } from "@earendil-works/pi-coding-agent"
import { parseArgs } from "../node_modules/@earendil-works/pi-coding-agent/dist/cli/args.js"

describe("Pi development resource isolation", () => {
  it("loads only the explicit checkout skill despite a conflicting auto-discovered skill, including reload", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-skills-" })
      const cwd = `${root}/workspace`
      const agentDir = `${root}/pi`
      const ambient = `${root}/.agents/skills/magnitude`
      const explicit = `${root}/checkout/magnitude`
      for (const directory of [cwd, agentDir, ambient, explicit]) yield* fs.makeDirectory(directory, { recursive: true })
      const skill = (description: string) => `---\nname: magnitude\ndescription: ${description}\n---\n${description}\n`
      yield* fs.writeFileString(`${ambient}/SKILL.md`, skill("Ambient copy"))
      yield* fs.writeFileString(`${explicit}/SKILL.md`, skill("Checkout copy"))
      const args = parseArgs(piDevelopmentArgs("test-model", `${explicit}/SKILL.md`))
      expect(args.diagnostics).toEqual([])
      const unisolated = new DefaultResourceLoader({
        cwd, agentDir, additionalSkillPaths: args.skills,
        noExtensions: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      })
      yield* Effect.promise(() => unisolated.reload())
      expect(unisolated.getSkills().diagnostics.some(({ type }) => type === "collision")).toBe(true)
      const loader = new DefaultResourceLoader({
        cwd, agentDir, noSkills: args.noSkills, additionalSkillPaths: args.skills,
        noExtensions: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      })
      for (let reload = 0; reload < 2; reload++) {
        yield* Effect.promise(() => loader.reload())
        const result = loader.getSkills()
        expect(result.skills.map(({ name, filePath }) => ({ name, filePath }))).toEqual([
          { name: "magnitude", filePath: `${explicit}/SKILL.md` },
        ])
        expect(result.diagnostics).toEqual([])
      }
    })).pipe(Effect.provide(BunContext.layer)))
  })
})

describe("Pi development status decoding", () => {
  it("waits through initialization, empty ready snapshots, and uninstalled models", async () => {
    let reads = 0
    const model = { modelId: "gemma:gguf:q4", displayName: "Gemma", installation: "installed" }
    const snapshots = [
      { state: "initializing", models: [] },
      { state: "ready", models: [] },
      { state: "ready", models: [{ ...model, installation: "installing" }] },
      { state: "ready", models: [model] },
    ]
    const result = await Effect.runPromise(awaitPiDevelopmentModel(Effect.suspend(() =>
      decodePiDevelopmentStatus(JSON.stringify({
        schemaVersion: 1, command: "models.status", ok: true,
        data: snapshots[reads++],
      })),
    )))
    expect(result.modelId).toBe(model.modelId)
    expect(reads).toBe(4)
  })

  it("times out if no installed model appears", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* awaitPiDevelopmentModel(decodePiDevelopmentStatus(JSON.stringify({
        schemaVersion: 1, command: "models.status", ok: true,
        data: { state: "ready", models: [] },
      }))).pipe(Effect.fork)
      yield* TestClock.adjust("30 seconds")
      return yield* Fiber.await(fiber)
    }).pipe(Effect.provide(TestContext.TestContext)))
    expect(result._tag).toBe("Failure")
    expect(String(result)).toContain("No installed Magnitude model became available within 30 seconds")
  })

  it("reports command failures without retrying until timeout", async () => {
    let reads = 0
    const result = await Effect.runPromiseExit(awaitPiDevelopmentModel(Effect.suspend(() => {
      reads++
      return Effect.fail("service unavailable")
    })))
    expect(result._tag).toBe("Failure")
    expect(String(result)).toContain("service unavailable")
    expect(reads).toBe(1)
  })

  it("accepts initializing and ready model status envelopes", async () => {
    const initializing = await Effect.runPromise(decodePiDevelopmentStatus(JSON.stringify({
      schemaVersion: 1,
      command: "models.status",
      ok: true,
      data: { state: "initializing", models: [] },
    })))
    expect(initializing.ok).toBe(true)
    if (initializing.ok) expect(initializing.data.state).toBe("initializing")

    const ready = await Effect.runPromise(decodePiDevelopmentStatus(JSON.stringify({
      schemaVersion: 1,
      command: "models.status",
      ok: true,
      data: {
        state: "ready",
        models: [{ modelId: "qwen3.6-35b-a3b:gguf:q6", displayName: "Qwen", installation: "installed" }],
      },
    })))
    expect(ready.ok).toBe(true)
    if (ready.ok) expect(ready.data.models).toHaveLength(1)
  })

  it("preserves the CLI failure message", async () => {
    const error = await Effect.runPromiseExit(decodePiDevelopmentStatus(JSON.stringify({
      schemaVersion: 1,
      command: "models.status",
      ok: false,
      error: { message: "Magnitude service is not running" },
    })))
    expect(error._tag).toBe("Failure")
    expect(String(error)).toContain("Magnitude service is not running")
  })

  it("rejects malformed or incompatible responses", async () => {
    const error = await Effect.runPromiseExit(decodePiDevelopmentStatus("not json"))
    expect(error._tag).toBe("Failure")
    expect(String(error)).toContain("incompatible model status response")
  })
})
