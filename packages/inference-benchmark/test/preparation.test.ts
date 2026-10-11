import { describe, expect, it } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { appendFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { join, resolve } from "node:path"
import { tmpdir } from "node:os"
import { defineExperiment, defineModel, omlx } from "../src/experiment"
import { currentOwnedEnvironmentDigests, planningModelFor, PREPARED_EXPERIMENT_VERSION } from "../src/preparation"

const engine = resolve("engines/omlx")

describe("managed oMLX preparation", () => {
  it("freezes oMLX and every material MLX dependency", async () => {
    const lock = await readFile(resolve(engine, "uv.lock"), "utf8")
    expect(lock).toContain('name = "omlx"\nversion = "0.6.3"')
    expect(lock).toContain("85708e4b9a585df42241c826b6be2b4dba018406")
    expect(lock).toContain('name = "mlx"\nversion = "0.32.0"')
    expect(lock).toContain("ab1806e8f5d6aa035973af194a1b9198ab4754dc")
    expect(lock).toContain("78b96eb5462141447b9a6b4943ef553891da56dd")
    expect(lock).toContain("c55324c86540c369f6818a0f47eae544d405475b")
    expect(PREPARED_EXPERIMENT_VERSION).toBe(5)
  })

  it("invalidates owned evidence when either the lock or adapter changes", async () => {
    const root = await mkdtemp(join(tmpdir(), "omlx-digest-"))
    try {
      await mkdir(join(root, "src"))
      await writeFile(join(root, "pyproject.toml"), "[project]\nname='adapter'\n")
      await writeFile(join(root, "uv.lock"), "revision = 3\n")
      await writeFile(join(root, "src", "adapter.py"), "VALUE = 1\n")
      const digest = () => Effect.runPromise(currentOwnedEnvironmentDigests(root).pipe(Effect.provide(BunContext.layer)))
      const first = await digest()
      await mkdir(join(root, "src", "__pycache__"))
      await writeFile(join(root, "src", "__pycache__", "adapter.pyc"), "generated")
      expect(await digest()).toEqual(first)
      await appendFile(join(root, "src", "adapter.py"), "VALUE = 2\n")
      const adapterChanged = await digest()
      expect(adapterChanged.adapterDigest).not.toBe(first.adapterDigest)
      expect(adapterChanged.lockDigest).toBe(first.lockDigest)
      await appendFile(join(root, "uv.lock"), "# changed\n")
      const lockChanged = await digest()
      expect(lockChanged.lockDigest).not.toBe(first.lockDigest)
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })

  it("plans an all-MLX experiment from logical identity without a GGUF", () => {
    const model = defineModel({
      id: "all-mlx", contextLimit: 32_768,
      source: { repository: "owner/model", revision: "model" },
      artifacts: {
        target: {
          kind: "mlx", repository: "owner/model-mlx", revision: "artifact", manifest: "model.lock.json",
          quantization: { family: "mlx-affine", bits: 4, groupSize: 64 },
        },
      },
    })
    const experiment = defineExperiment({
      id: "all-mlx", title: "All MLX",
      suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: { contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32, temperature: 0, topP: 1, seed: 42, enableThinking: false },
      variants: [{
        id: "baseline", artifact: model.artifacts.target,
        engine: omlx({ pythonProject: "engines/omlx", cache: { kind: "disabled" }, memoryGuard: { kind: "off" }, speculativeDecoding: { kind: "none" } }),
      }],
      execution: { variantOrder: "declared", blocks: 1 },
    })
    expect(planningModelFor(experiment)).toEqual({ id: "all-mlx", contextLimit: 32_768 })
  })

  it("runs the dependency-free adapter unit suite in normal CI", () => {
    const result = Bun.spawnSync(["python3", "-m", "unittest", "discover", "-s", "tests", "-v"], {
      cwd: engine,
      env: { ...process.env, PYTHONPATH: "src" },
    })
    expect(result.exitCode, result.stderr.toString()).toBe(0)
  })
})
