import { describe, expect, it } from "vitest"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { managedLlamaCppTarget, openTarget, TargetLauncherLive } from "../src/target"
import { EndpointClientLive } from "../src/transport"

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
        chatTemplateDigest: "template",
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
      readinessPath: "/health",
      parallelSequences: 1,
    })).pipe(Effect.provide(RuntimeLive)))
    expect(Option.isSome(session.rootPid)).toBe(true)
    if (Option.isSome(session.rootPid)) expect(session.rootPid.value).not.toBe(process.pid)
  })
})
