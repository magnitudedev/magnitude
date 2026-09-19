import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { NativeTerminalDriver } from "../src/terminal"
import { ProcessExecutorLive } from "../src/process"
import { openCodeTerminal } from "../src/harnesses/opencode-terminal"

const runtime = process.env.LAB_TERMINAL_NODE_EXECUTABLE, executable = process.env.LAB_OPENCODE_EXECUTABLE
const json = Schema.encode(Schema.parseJson(Schema.Unknown))

// Native client automation against synthetic SSE; this does not qualify application inference.
test.skipIf(!runtime || !executable).each([false, true])("OpenCode native TUI verifies interruption (premature completion: %s)", finishFirst => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-opencode-terminal-" })
  const environment = { HOME: root, USERPROFILE: root, PATH: process.env.PATH ?? "",
    XDG_CONFIG_HOME: join(root, ".config"), XDG_DATA_HOME: join(root, ".local", "share"), XDG_STATE_HOME: join(root, ".local", "state"), XDG_CACHE_HOME: join(root, ".cache"),
    OPENCODE_DISABLE_AUTOUPDATE: "true", OPENCODE_DISABLE_MODELS_FETCH: "true",
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
        controller.enqueue(chunk({ role: "assistant", content: index === 1 ? partial : recovered }))
        if (index === 2 || finishFirst) { controller.enqueue(chunk({}, "stop")); controller.enqueue(encoder.encode("data: [DONE]\n\n")); controller.close() }
      }, cancel() { if (index === 1) cancelled = true } }), { headers: { "content-type": "text/event-stream" } })
    },
  })), server => Effect.promise(() => server.stop(true)))
  yield* fs.writeFileString(join(root, "opencode.json"), yield* json({ $schema: "https://opencode.ai/config.json", permission: { "*": "deny" },
    agent: { title: { disable: true }, summary: { disable: true } }, enabled_providers: ["magnitude"],
    provider: { magnitude: { npm: "@ai-sdk/openai-compatible", name: "Magnitude", options: { baseURL: `http://127.0.0.1:${server.port}/v1`, apiKey: "fixture" },
      models: { "lab-initial-model": { name: "lab-initial-model" }, [model]: { name: model, variants: { off: {} } } } } } }))
  const cleanup: string[] = []
  const receipt = yield* openCodeTerminal({ runtime: runtime!, executable: executable!, cwd: root, environment,
    evidence: join(root, "evidence"), model, modelName: model, initialModel: "lab-initial-model", initialModelName: "lab-initial-model",
    interrupt: { prompt: "Start the first reply.", expected: partial },
    recovery: { prompt: "Give a new reply after the interruption.", expected: recovered },
  }, message => { cleanup.push(message) }).pipe(Effect.provide(ProcessExecutorLive), Effect.either)
  if (finishFirst) {
    expect(receipt._tag).toBe("Left")
    if (receipt._tag === "Left") expect(String(receipt.left)).toContain("completed before keyboard interruption")
    expect(requests).toBe(1)
    expect(cancelled).toBe(false)
    expect(yield* fs.exists(join(root, "evidence", "failure-screen.json"))).toBe(true)
  } else {
    expect(receipt._tag).toBe("Right")
    if (receipt._tag === "Right") {
      expect(receipt.right.model).toBe(model)
      expect(receipt.right.text).toBe(recovered)
    }
    expect(requests).toBe(2)
    expect(cancelled).toBe(true)
  }
  expect(cleanup).toEqual([])
})).pipe(Effect.timeout("90 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 100000)
