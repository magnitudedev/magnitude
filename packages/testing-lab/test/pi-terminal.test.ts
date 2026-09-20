import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { dirname, join } from "node:path"
import { expect, test } from "vitest"
import { NativeTerminalDriver } from "../src/terminal"
import { ProcessExecutorLive } from "../src/process"
import { piTerminal } from "../src/harnesses/pi-terminal"

const runtime = process.env.LAB_TERMINAL_NODE_EXECUTABLE
const executable = process.env.LAB_PI_EXECUTABLE
// This qualifies the pinned third-party TUI adapter against synthetic SSE, not Magnitude inference.
test.skipIf(!runtime || !executable).each([false, true])("Pi native TUI verifies real interruption (premature completion: %s)", finishFirst => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-pi-terminal-" })
  const agent = join(root, ".pi", "agent")
  yield* fs.makeDirectory(agent, { recursive: true })
  const environment = { HOME: root, USERPROFILE: root, PI_CODING_AGENT_DIR: agent,
    PATH: `${dirname(runtime!)}${process.platform === "win32" ? ";" : ":"}${process.env.PATH ?? ""}`,
    ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}) }
  const model = "lab-terminal-model", partial = `PARTIAL-${crypto.randomUUID()}`, recovered = `RECOVERED-${crypto.randomUUID()}`
  let requests = 0, cancelled = false
  const server = yield* Effect.acquireRelease(Effect.sync(() => Bun.serve({ hostname: "127.0.0.1", port: 0, idleTimeout: 60,
    async fetch(request) {
      if (new URL(request.url).pathname !== "/v1/chat/completions") return new Response("Unexpected path", { status: 404 })
      const body = await request.json() as { model?: string; stream?: boolean }
      if (body.model !== model || body.stream !== true) return new Response("Unexpected generation request", { status: 400 })
      const index = ++requests
      if (index > 2) return new Response("Unexpected retry", { status: 500 })
      const encoder = new TextEncoder()
      const chunk = (delta: unknown, finish: string | null = null) => encoder.encode(`data: ${JSON.stringify({ id: `fixture-${index}`,
        object: "chat.completion.chunk", created: 1, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`)
      return new Response(new ReadableStream({ start(controller) {
        controller.enqueue(chunk({ role: "assistant", content: index === 1 ? partial.toLowerCase() : recovered.toLowerCase() }))
        if (index === 2 || finishFirst) {
          controller.enqueue(chunk({}, "stop")); controller.enqueue(encoder.encode("data: [DONE]\n\n")); controller.close()
        }
      }, cancel() { if (index === 1) cancelled = true } }), { headers: { "content-type": "text/event-stream" } })
    },
  })), server => Effect.promise(() => server.stop(true)))
  yield* fs.writeFileString(join(agent, "models.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ providers: { magnitude: {
    baseUrl: `http://127.0.0.1:${server.port}/v1`, api: "openai-completions", apiKey: "fixture",
    models: [{ id: "lab-initial-model" }, { id: model }],
  } } }))
  const cleanup: string[] = []
  const receipt = yield* piTerminal({ runtime: runtime!, executable: executable!, cwd: root, environment,
    evidence: join(root, "evidence"), model, initialModel: "lab-initial-model",
    interrupt: { prompt: "Start the first reply.", expected: partial },
    recovery: { prompt: "Give a new reply after the interruption.", expected: recovered },
  }, message => { cleanup.push(message) }).pipe(Effect.provide(ProcessExecutorLive), Effect.either)
  if (finishFirst) {
    expect(receipt._tag).toBe("Left")
    if (receipt._tag === "Left") expect(String(receipt.left)).toContain("interrupted assistant turn")
    expect(requests).toBe(1)
    expect(cancelled).toBe(false)
  } else {
    expect(receipt._tag).toBe("Right")
    if (receipt._tag === "Right") {
      expect(receipt.right.model).toBe(model)
      expect(receipt.right.text).toBe(recovered.toLowerCase())
    }
    expect(requests).toBe(2)
    expect(cancelled).toBe(true)
  }
  expect(cleanup).toEqual([])
})).pipe(Effect.timeout("90 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 100000)
