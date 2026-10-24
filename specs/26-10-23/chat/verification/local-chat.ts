import { FetchHttpClient } from "@effect/platform"
import { Effect, Layer, Option, Stream } from "effect"
import { MagnitudeClient, ChatConversationId, ChatCommandId, ChatModelId, defaultChatSettings, applyChatUpdate, type ChatConversation, type ChatGeneration, type ChatSettings } from "../../../../packages/sdk/src/index"
import { strict as assert } from "node:assert"
import { writeFileSync } from "node:fs"
const layer = MagnitudeClient.layer({ origin: "http://127.0.0.1:11101", autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer))
const report: Record<string, unknown>[] = []
const run = Effect.gen(function* () {
  const client = yield* MagnitudeClient
  const all = yield* client.chat.listModels({})
  console.log(JSON.stringify(all.data.map(m => ({ id: m.id, name: m.name, reasoning: Option.getOrNull(m.reasoning) }))))
  const model = all.data.find(m => m.id === process.env.CHAT_TEST_MODEL) ?? all.data.find(m => /lfm2.5.*2.6/i.test(m.name)) ?? all.data[0]
  assert(model, "An installed model is required")
  const r = Option.getOrNull(model.reasoning)
  const settings: ChatSettings = { ...defaultChatSettings, modelId: Option.some(ChatModelId.make(model.id)), maxOutputTokens: Option.some(512), reasoningEffort: r ? Option.some(r.supported_efforts.includes("none") ? "none" : r.supported_efforts.includes("off") ? "off" : r.default_effort) : Option.none() }
  const conversationId = ChatConversationId.make(crypto.randomUUID())
  yield* client.chat.createConversation({ conversationId, settings })
  console.log(`LOCAL_TEST_CHAT=${conversationId}`)
  let last: ChatGeneration | undefined
  const send = (content: string, current = settings) => Effect.gen(function* () {
    const cmd = ChatCommandId.make(crypto.randomUUID())
    const generationId = yield* client.chat.startGeneration({ conversationId, commandId: cmd, content: Option.some(content), retryMessageId: Option.none(), settings: current })
    assert.equal(yield* client.chat.startGeneration({ conversationId, commandId: cmd, content: Option.some(content), retryMessageId: Option.none(), settings: current }), generationId)
    const phases = new Set<string>()
    const result = yield* client.chat.watchConversation({ conversationId }).pipe(Stream.scan(null as ChatConversation | null, applyChatUpdate), Stream.filterMap(c => c ? Option.some({ _tag: "Snapshot" as const, conversation: c }) : Option.none()), Stream.tap(u => Effect.sync(() => { if (u._tag === "Snapshot") for (const m of u.conversation.messages) if (Option.isSome(m.generation) && m.generation.value.id === generationId && Option.isSome(m.generation.value.activity)) phases.add(m.generation.value.activity.value.phase) })), Stream.filterMap(u => u._tag === "Snapshot" ? Option.fromNullable(u.conversation.messages.flatMap(m => Option.toArray(m.generation)).find(g => g.id === generationId && g.status !== "running")) : Option.none()), Stream.runHead, Effect.map(Option.getOrThrow), Effect.timeout("100 seconds"))
    assert.notEqual(result.status, "failed", Option.getOrNull(result.error) ?? "Generation failed")
    assert(Option.isSome(result.metrics.outputTokens), "Engine output count should be present")
    assert(Option.isSome(result.metrics.ttftMs), "TTFT should be measured")
    report.push({ id: result.id, model: model.name, status: result.status, chars: result.content.length, reasoningChars: result.reasoning.length, outputTokens: Option.getOrNull(result.metrics.outputTokens), ttftMs: Option.getOrNull(result.metrics.ttftMs), tokensPerSecond: Option.getOrNull(result.metrics.tokensPerSecond), phases: [...phases] })
    console.log(JSON.stringify(report.at(-1)))
    last = result
    return result
  })
  yield* send("Answer in one short sentence: what is 2 + 2?")
  yield* send("Remember the code name ORANGE KITE for this conversation. Acknowledge briefly.")
  yield* send("What code name did I just give you? Reply with only the code name.", { ...settings, temperature: Option.some(0), seed: Option.some(42) })
  for (let i = 0; i < 7; i++) yield* send(`Turn ${i}. ${"The test compares local inference latency, prompt caching and generation throughput. ".repeat(i * 4)} Summarize the distinction in one sentence.`, { ...settings, maxOutputTokens: Option.some(40), topP: Option.some(.9), topK: Option.some(20), minP: Option.some(.05), repetitionPenalty: Option.some(1.05), presencePenalty: Option.some(.1), frequencyPenalty: Option.some(.1) })
  yield* send("Write a long explanation of local inference.", { ...settings, maxOutputTokens: Option.some(8), stop: ["\n"] })
  assert(last)
  const old = last
  const retryId = yield* client.chat.startGeneration({ conversationId, commandId: ChatCommandId.make(crypto.randomUUID()), content: Option.none(), retryMessageId: Option.some(old.messageId), settings: { ...settings, maxOutputTokens: Option.some(16) } })
  yield* client.chat.watchConversation({ conversationId }).pipe(Stream.scan(null as ChatConversation | null, applyChatUpdate), Stream.filterMap(c => c ? Option.some({ _tag: "Snapshot" as const, conversation: c }) : Option.none()), Stream.filter(u => u._tag === "Snapshot" && u.conversation.messages.some(m => Option.isSome(m.generation) && m.generation.value.id === retryId && m.generation.value.status !== "running")), Stream.take(1), Stream.runDrain, Effect.timeout("30 seconds"))
  const stoppedId = yield* client.chat.startGeneration({ conversationId, commandId: ChatCommandId.make(crypto.randomUUID()), content: Option.some("Count from one to one thousand, writing every number."), retryMessageId: Option.none(), settings: { ...settings, maxOutputTokens: Option.some(2048) } })
  yield* Effect.sleep("200 millis")
  yield* client.chat.stopGeneration({ conversationId, generationId: stoppedId })
  const saved = yield* client.chat.getConversation({ conversationId })
  assert.equal(Option.getOrThrow(saved.messages.at(-1)!.generation).status, "stopped")
  assert.equal(saved.messages.length, 24)
  yield* client.chat.updateConversation({ conversationId, title: Option.some("Local Chat verification"), settings: Option.none() })
  assert.equal((yield* client.chat.getConversation({ conversationId })).title, "Local Chat verification")
  console.log(`PASS: ${report.length} completed requests, retry, Stop, idempotency, settings and 24-message saved history`)
  writeFileSync(`/tmp/magnitude-chat-local-${model.id.replace(/[^a-z0-9]/gi, "-")}-report.json`, JSON.stringify({ conversationId, report }, null, 2))
})
await Effect.runPromise(Effect.scoped(run.pipe(Effect.provide(layer))))
