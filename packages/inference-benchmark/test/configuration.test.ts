import { afterEach, describe, expect, it } from "vitest"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { loadConfiguration } from "../src/configuration"

const roots: string[] = []
afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))))

describe("benchmark configuration", () => {
  it("loads a target-neutral configuration", async () => {
    const root = await mkdtemp(join(tmpdir(), "benchmark-config-"))
    roots.push(root)
    const path = join(root, "benchmark.json")
    await writeFile(path, JSON.stringify({
      suite: "agent-core",
      profile: "smoke",
      model: { id: "test", artifactPath: "/models/test.gguf", contextLimit: 4096 },
      targets: [{ kind: "existing", id: "server", endpoint: "http://localhost:8080", servedModel: "test", parallelSequences: 1 }],
      output: "result.json",
    }))
    const configuration = await Effect.runPromise(loadConfiguration(path).pipe(Effect.provide(BunContext.layer)))
    expect(configuration.targets[0]?.kind).toBe("existing")
    expect(configuration.model.contextLimit).toBe(4096)
  })
})
