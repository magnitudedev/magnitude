import { afterEach, describe, expect, it } from "vitest"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { join, resolve } from "node:path"
import { tmpdir } from "node:os"
import { managedLlamaCppTarget, openTarget, TargetLauncherLive } from "../src/target"
import { EndpointClientLive, executeRequest } from "../src/transport"

const roots: string[] = []
afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))))

const RuntimeLive = TargetLauncherLive.pipe(
  Layer.provideMerge(EndpointClientLive),
  Layer.provideMerge(FetchHttpClient.layer),
  Layer.provideMerge(BunContext.layer),
)

describe("managed target lifecycle", () => {
  it("translates per-sequence context into llama.cpp physical context", () => {
    const target = managedLlamaCppTarget({
      model: {
        id: "model",
        artifactPath: "/model.gguf",
        artifactSha256: "sha",
        contextLimit: 4096,
      },
      maxSequences: 4,
    })
    expect(target.args).toContain("16384")
    expect(target.parallelSequences).toBe(4)
  })

  it("launches, identifies, probes, and stops an owned process", async () => {
    const port = 20_000 + Math.floor(Math.random() * 20_000)
    const script = `const server=Bun.serve({port:${port},routes:{'/health':()=>Response.json({status:'ok'})}});process.on('SIGINT',()=>{server.stop(true);process.exit(0)});await new Promise(()=>{})`
    const session = await Effect.runPromise(Effect.scoped(openTarget({
      kind: "managed",
      engine: "generic",
      id: "managed-test",
      executable: process.execPath,
      args: ["-e", script],
      endpoint: `http://127.0.0.1:${port}`,
      servedModel: "test",
      readiness: { kind: "http", path: "/health" },
      parallelSequences: 1,
    })).pipe(Effect.provide(RuntimeLive)))
    expect(Option.isSome(session.rootPid)).toBe(true)
    if (Option.isSome(session.rootPid)) expect(session.rootPid.value).not.toBe(process.pid)
  })

  it("waits for structured oMLX model/backend readiness and shuts down with SIGINT", async () => {
    const root = await mkdtemp(join(tmpdir(), "fake-omlx-target-"))
    roots.push(root)
    const shutdown = join(root, "shutdown.txt")
    const port = 20_000 + Math.floor(Math.random() * 20_000)
    const target = {
      kind: "managed" as const,
      engine: "omlx" as const,
      id: "mtp",
      executable: process.execPath,
      args: [resolve("test/fixtures/fake-omlx.ts"), `port=${port}`, "model=test-model", "context=4096", "concurrency=2", "backend=mtp", `shutdown=${shutdown}`],
      endpoint: `http://127.0.0.1:${port}`,
      servedModel: "test-model",
      readiness: {
        kind: "omlx" as const,
        path: "/magnitude/benchmark/readiness",
        expected: { servedModel: "test-model", contextCapacity: 4096, parallelSequences: 2, speculativeBackend: "mtp" as const },
      },
      parallelSequences: 2,
    }
    const observation = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const session = yield* openTarget(target)
      return yield* executeRequest(session.endpoint, {
        id: "qualification", fixtureId: "fixture", messages: [{ role: "user", content: "lookup" }],
        tools: [{
          type: "function",
          function: {
            name: "lookup",
            description: "lookup",
            parameters: { type: "object", properties: { key: { type: "string" } }, required: ["key"] },
          },
        }],
        expected: [{ name: "lookup", arguments: { key: ["value"] } }], releaseOffsetMs: 0, dependsOn: [], maxOutputTokens: 8,
      })
    })).pipe(Effect.provide(RuntimeLive)))
    expect(observation.outcome).toBe("valid")
    expect(observation.terminal?.timings.speculativeBackend).toBe("mtp")
    expect(await readFile(shutdown, "utf8")).toBe("SIGINT\n")
  })
})
