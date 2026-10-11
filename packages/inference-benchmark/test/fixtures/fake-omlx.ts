export {}

const args = Object.fromEntries(process.argv.slice(2).map((value) => value.split("=", 2) as [string, string]))
const port = Number(args.port)
const servedModel = args.model ?? "test-model"
const contextCapacity = Number(args.context ?? 4096)
const concurrency = Number(args.concurrency ?? 1)
const backend = args.backend ?? "none"
let loaded = false
setTimeout(() => { loaded = true }, 75)

const server = Bun.serve({
  port,
  routes: {
    "/magnitude/benchmark/readiness": () => loaded
      ? Response.json({
          ready: true,
          discovered_model: servedModel,
          served_model: servedModel,
          loaded: true,
          context_capacity: contextCapacity,
          max_concurrent_requests: concurrency,
          speculative_backend: backend,
          qualification_completed: true,
        })
      : Response.json({ ready: false }, { status: 503 }),
    "/v1/chat/completions": async (request) => {
      const body = await request.json() as { model: string }
      const timings = {
        cache_n: 2,
        prompt_n: 8,
        prompt_ms: 4.5,
        predicted_n: 3,
        predicted_ms: 6.25,
        ...(backend === "none" ? {} : {
          draft_n: 4,
          draft_n_accepted: 2,
          speculative_backend: backend,
        }),
      }
      const chunks = [
        { id: "fake", model: body.model, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] },
        { id: "fake", model: body.model, choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call-1", type: "function", function: { name: "lookup", arguments: "{\"key\":\"value\"}" } }] }, finish_reason: "tool_calls" }] },
        {
          id: "fake", model: body.model, choices: [],
          usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13, prompt_tokens_details: { cached_tokens: 2 } },
          timings,
        },
      ]
      return new Response(`${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`, {
        headers: { "content-type": "text/event-stream" },
      })
    },
  },
})

process.on("SIGINT", async () => {
  if (args.shutdown) await Bun.write(args.shutdown, "SIGINT\n")
  server.stop(true)
  process.exit(0)
})

await new Promise(() => {})
