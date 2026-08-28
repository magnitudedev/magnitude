import { afterAll, describe, expect, it } from "vitest"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { mkdtemp, rm } from "node:fs/promises"
import { join, resolve } from "node:path"
import { tmpdir } from "node:os"
import { openTarget, TargetLauncherLive } from "../src/target"
import { EndpointClientLive, executeRequest } from "../src/transport"

const model = process.env.MAGNITUDE_OMLX_SMOKE_MODEL
const enabled = process.platform === "darwin" && process.arch === "arm64" && process.env.MAGNITUDE_OMLX_SMOKE === "1" && Boolean(model)
const roots: string[] = []
afterAll(async () => Promise.all(roots.map((root) => rm(root, { recursive: true, force: true }))))

const RuntimeLive = TargetLauncherLive.pipe(
  Layer.provideMerge(EndpointClientLive),
  Layer.provideMerge(FetchHttpClient.layer),
  Layer.provideMerge(BunContext.layer),
)

describe.skipIf(!enabled)("real oMLX Apple Silicon smoke", () => {
  it("loads a pre-prepared tiny MLX snapshot and emits native terminal evidence", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-omlx-smoke-"))
    roots.push(root)
    const port = 20_000 + Math.floor(Math.random() * 20_000)
    const project = resolve("engines/omlx")
    const target = {
      kind: "managed" as const, engine: "omlx" as const, id: "omlx-smoke",
      executable: "uv",
      args: [
        "run", "--frozen", "--no-sync", "--project", project,
        "magnitude-omlx-benchmark-server", "--model", model!, "--served-model", "smoke",
        "--host", "127.0.0.1", "--port", String(port), "--base-path", root,
        "--max-concurrent-requests", "1", "--context-capacity", "2048",
        "--cache-policy", "disabled", "--memory-guard", "off", "--speculative-method", "none",
      ],
      endpoint: `http://127.0.0.1:${port}`, servedModel: "smoke", parallelSequences: 1,
      logPath: join(root, "server.log"),
      readiness: {
        kind: "omlx" as const, path: "/magnitude/benchmark/readiness",
        expected: { servedModel: "smoke", contextCapacity: 2048, parallelSequences: 1, speculativeBackend: "none" as const },
      },
    }
    const observation = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const session = yield* openTarget(target)
      return yield* executeRequest(session.endpoint, {
        id: "smoke", fixtureId: "smoke", messages: [{ role: "user", content: "Reply with OK." }],
        tools: [], expected: [], releaseOffsetMs: 0, dependsOn: [], maxOutputTokens: 8,
      })
    })).pipe(Effect.provide(RuntimeLive)))
    expect(observation.outcome).toBe("valid")
    expect(observation.terminal?.timings.promptMs).toBeGreaterThanOrEqual(0)
    expect(observation.terminal?.timings.generationMs).toBeGreaterThanOrEqual(0)
  }, 1_200_000)
})
