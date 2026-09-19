import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schedule, Schema } from "effect"
import { dirname, join } from "node:path"
import { expect, test } from "vitest"
import { NativeTerminalDriver, TerminalConfig, TerminalDriver, waitForTerminal } from "../src/terminal"
import { command, ProcessExecutorLive } from "../src/process"

const runtime = process.env.LAB_TERMINAL_NODE_EXECUTABLE
const executable = process.env.LAB_PI_EXECUTABLE
const Assistant = Schema.Struct({ type: Schema.Literal("message"), message: Schema.Struct({ role: Schema.Literal("assistant"),
  model: Schema.String, provider: Schema.String, stopReason: Schema.String,
  content: Schema.Array(Schema.Unknown) }) })

// This qualifies the pinned third-party TUI adapter against synthetic SSE, not Magnitude inference.
test.skipIf(!runtime || !executable)("Pi native TUI selects a model, interrupts streaming, recovers and exits", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-pi-terminal-" })
  const agent = join(root, ".pi", "agent"), session = join(root, "session.jsonl")
  yield* fs.makeDirectory(agent, { recursive: true })
  const environment = { HOME: root, USERPROFILE: root, PI_CODING_AGENT_DIR: agent,
    PATH: `${dirname(runtime!)}${process.platform === "win32" ? ";" : ":"}${process.env.PATH ?? ""}`,
    ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}) }
  const version = yield* command(executable!, ["--version"], { env: environment, inheritEnv: false }).pipe(Effect.provide(ProcessExecutorLive))
  expect(version.exitCode).toBe(0)
  expect(version.stdout.trim()).toBe("0.85.1")
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
        if (index === 2) {
          controller.enqueue(chunk({}, "stop")); controller.enqueue(encoder.encode("data: [DONE]\n\n")); controller.close()
        }
      }, cancel() { if (index === 1) cancelled = true } }), { headers: { "content-type": "text/event-stream" } })
    },
  })), server => Effect.promise(() => server.stop(true)))
  yield* fs.writeFileString(join(agent, "models.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ providers: { magnitude: {
    baseUrl: `http://127.0.0.1:${server.port}/v1`, api: "openai-completions", apiKey: "fixture",
    models: [{ id: "lab-initial-model" }, { id: model }],
  } } }))
  const readAssistants = Effect.gen(function* () {
    if (!(yield* fs.exists(session))) return []
    const contents = yield* fs.readFileString(session)
    const complete = contents.slice(0, contents.lastIndexOf("\n") + 1)
    const entries = yield* Effect.forEach(complete.split("\n").filter(Boolean), line => Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(line))
    return entries.filter(Schema.is(Assistant))
  })
  const waitForMessages = (count: number) => readAssistants.pipe(Effect.repeat({ until: entries => entries.length >= count,
    schedule: Schedule.identity<ReadonlyArray<typeof Assistant.Type>>().pipe(Schedule.addDelay(() => "100 millis")) }), Effect.timeout("15 seconds"))
  const cleanup: string[] = []
  yield* Effect.scoped(Effect.gen(function* () {
    const terminal = yield* (yield* TerminalDriver).start(TerminalConfig.make({ runtime: runtime!, executable: executable!,
      args: ["--provider", "magnitude", "--model", "lab-initial-model", "--thinking", "off", "--no-tools", "--offline", "--session", session],
      cwd: root, environment, evidence: join(root, "evidence"), columns: 120, rows: 40,
    }), message => { cleanup.push(message) })
    yield* waitForTerminal(terminal, screen => screen.lines.some(line => line.includes("lab-initial-model")), "show the initial model")
    yield* terminal.write(`/model magnitude/${model}`)
    // Pi can paint its footer before enabling submission. Enter retries only this idempotent
    // selection command; once consumed, an empty Enter is a no-op. Generation is never retried.
    yield* terminal.write("\r").pipe(Effect.zipRight(terminal.screen), Effect.repeat({
      until: screen => screen.lines.some(line => line.includes(`Model: ${model}`)),
      schedule: Schedule.spaced("100 millis"),
    }), Effect.timeout("30 seconds"))
    yield* terminal.write("Start the first reply.\r")
    yield* waitForTerminal(terminal, screen => screen.lines.some(line => line.includes(partial)), "render streamed assistant output")
    yield* terminal.write("\u001b")
    const aborted = yield* waitForMessages(1)
    expect(aborted.map(entry => entry.message.stopReason)).toEqual(["aborted"])
    yield* terminal.write("Give a new reply after the interruption.\r")
    yield* waitForTerminal(terminal, screen => screen.lines.some(line => line.includes(recovered)), "render the follow-up answer")
    const completed = yield* waitForMessages(2)
    expect(completed.map(entry => entry.message.stopReason)).toEqual(["aborted", "stop"])
    expect(completed.every(entry => entry.message.model === model && entry.message.provider === "magnitude")).toBe(true)
    yield* terminal.write("\u0004")
    expect((yield* terminal.exited).code).toBe(0)
  }))
  expect(requests).toBe(2)
  expect(cancelled).toBe(true)
  expect(cleanup).toEqual([])
})).pipe(Effect.timeout("90 seconds"), Effect.provide(Layer.merge(BunContext.layer, NativeTerminalDriver.pipe(Layer.provide(BunContext.layer)))))), 100000)
