import { afterEach, describe, expect, it } from "vitest"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { prepareCorpus } from "../src/corpus"
import { compileTrialPlanSync } from "../src/plan"
import { sha256 } from "../src/hash"

const roots: string[] = []
afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))))

async function fixtureCorpus() {
  const root = await mkdtemp(join(tmpdir(), "inference-benchmark-"))
  roots.push(root)
  const cache = join(root, "cache")
  const questions = Array.from({ length: 12 }, (_, index) => JSON.stringify({
    id: `simple_python_${index}`,
    question: [[{ role: "user", content: `Return the identifier ${index}.` }]],
    function: [{
      name: `return_identifier_${index}`,
      description: "Return an identifier",
      parameters: { type: "dict", properties: { value: { type: index === 0 ? "float" : "integer" } }, required: ["value"] },
    }],
  })).join("\n") + "\n"
  const answers = Array.from({ length: 12 }, (_, index) => JSON.stringify({
    id: `simple_python_${index}`,
    ground_truth: [{ [`return_identifier_${index}`]: { value: [index] } }],
  })).join("\n") + "\n"
  await mkdir(join(cache, "possible_answer"), { recursive: true })
  await writeFile(join(cache, "questions.json"), questions)
  await writeFile(join(cache, "possible_answer", "questions.json"), answers)
  const lockPath = join(root, "lock.json")
  const selectionPath = join(root, "selection.json")
  await writeFile(lockPath, JSON.stringify({
    repository: "https://github.com/example/example",
    commit: "test-commit",
    dataRoot: "data",
    license: "Apache-2.0",
    files: [
      { path: "questions.json", sha256: sha256(questions) },
      { path: "possible_answer/questions.json", sha256: sha256(answers) },
    ],
  }))
  await writeFile(selectionPath, JSON.stringify({
    version: 1,
    families: ["simple_python"],
    idPrefixes: ["simple_python_"],
    maximumRecords: 12,
    exclusions: [],
    purpose: "test",
  }))
  return Effect.runPromise(prepareCorpus({ root: cache, lockPath, selectionPath, offline: true }).pipe(
    Effect.provide(BunContext.layer),
  ))
}

describe("BFCL corpus and trial planning", () => {
  it("materializes deterministic validated fixtures without Python", async () => {
    const corpus = await fixtureCorpus()
    expect(corpus.fixtures).toHaveLength(12)
    expect(corpus.fixtures[0]?.tools[0]?.function.parameters.type).toBe("object")
    const properties = corpus.fixtures[0]?.tools[0]?.function.parameters.properties as Record<string, { readonly type?: string }>
    expect(properties.value?.type).toBe("number")
    expect(corpus.fixtures[0]?.canonicalToolMessages[0]?.role).toBe("tool")
    expect(corpus.digest).toMatch(/^[a-f0-9]{64}$/)
  })

  it("compiles every workload pattern into one content-addressed plan", async () => {
    const corpus = await fixtureCorpus()
    const model = { id: "test-model", artifactPath: "/test/model.gguf", artifactSha256: "model", chatTemplateDigest: "template", contextLimit: 32_768 }
    const first = compileTrialPlanSync(corpus, model, { profile: "smoke" })
    const second = compileTrialPlanSync(corpus, model, { profile: "smoke" })
    expect(new Set(first.trials.map((trial) => trial.pattern))).toEqual(new Set([
      "single-request",
      "sequential-session",
      "independent-concurrency",
      "forked-concurrency",
      "concurrency-pressure",
      "memory-pressure",
    ]))
    expect(first.digest).toBe(second.digest)
    expect(first.trials.every((trial) => trial.criteria.length === 5)).toBe(true)
    expect(first.servingPolicy).toEqual({ contextTokensPerSequence: 32_768, parallelSequences: 1 })
    expect(compileTrialPlanSync(corpus, model, { profile: "smoke", parallelSequences: 4 }).digest)
      .not.toBe(first.digest)
  })
})
