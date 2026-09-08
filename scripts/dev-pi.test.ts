import type { ModelCatalogState } from "@magnitudedev/sdk"
import { makeInstalledCatalogModel, makeCatalogOnlyModel } from "../cli/src/features/local-inference/test-fixtures"
import { Effect, Fiber, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { awaitPiDevelopmentModel, piDevelopmentArgs, resolvePiDevelopmentExecutable } from "./dev-pi"
import { delimiter } from "node:path"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DefaultResourceLoader } from "@earendil-works/pi-coding-agent"
import { parseArgs } from "../node_modules/@earendil-works/pi-coding-agent/dist/cli/args.js"

describe("Pi development resource isolation", () => {
  it("uses installed Pi even when Bun puts a dependency-local Pi first on PATH", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-path-" })
      const dependency = `${root}/node_modules/.bin`
      const installed = `${root}/bin`
      for (const directory of [dependency, installed]) {
        yield* fs.makeDirectory(directory, { recursive: true })
        yield* fs.writeFileString(`${directory}/pi`, "#!/bin/sh\nexit 0\n")
        yield* fs.chmod(`${directory}/pi`, 0o755)
      }
      expect(yield* resolvePiDevelopmentExecutable([dependency, installed].join(delimiter))).toBe(`${installed}/pi`)
      const missing = yield* Effect.exit(resolvePiDevelopmentExecutable(dependency))
      expect(missing._tag).toBe("Failure")
      expect(String(missing)).toContain("Pi is not installed on your PATH")
    })).pipe(Effect.provide(BunContext.layer)))
  })
  it("allows package-first onboarding without a preconfigured model", () => {
    const args = parseArgs(piDevelopmentArgs(undefined, "/checkout/dist/skills/magnitude/SKILL.md"))
    expect(args.model).toBeUndefined()
    expect(args.messages).toEqual([])
    expect(args.noSkills).toBe(true)
    expect(args.skills).toEqual(["/checkout/dist/skills/magnitude/SKILL.md"])
  })
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

const readyCatalog = (models = [makeInstalledCatalogModel()]): ModelCatalogState => ({
  _tag: "Ready", providers: [], failures: [],
  models: models.map(product => ({ _tag: "Local", product, offering: Option.none() })),
  localModelPreparation: { discovery: { complete: true, modelsFound: models.length }, assessment: { complete: true, settledModels: models.length, totalModels: models.length } },
})

describe("Pi development SDK model readiness", () => {
  it("waits through initialization, empty snapshots, and uninstalled models", async () => {
    let reads = 0
    const installed = makeInstalledCatalogModel()
    const snapshots: ModelCatalogState[] = [
      { _tag: "Initializing" }, readyCatalog([]), readyCatalog([makeCatalogOnlyModel()]), readyCatalog([installed]),
    ]
    const result = await Effect.runPromise(awaitPiDevelopmentModel(Effect.suspend(() => Effect.succeed(snapshots[reads++]!))))
    expect(result.modelId).toBe(installed.modelId)
    expect(reads).toBe(4)
  })

  it("times out if no installed model appears", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* awaitPiDevelopmentModel(Effect.succeed(readyCatalog([]))).pipe(Effect.fork)
      yield* TestClock.adjust("30 seconds")
      return yield* Fiber.await(fiber)
    }).pipe(Effect.provide(TestContext.TestContext)))
    expect(result._tag).toBe("Failure")
    expect(String(result)).toContain("No installed Magnitude model became available within 30 seconds")
  })

  it("reports SDK failure without retrying until timeout", async () => {
    let reads = 0
    const result = await Effect.runPromiseExit(awaitPiDevelopmentModel(Effect.suspend(() => {
      reads++
      return Effect.fail("service unavailable")
    })))
    expect(result._tag).toBe("Failure")
    expect(String(result)).toContain("service unavailable")
    expect(reads).toBe(1)
  })
})
