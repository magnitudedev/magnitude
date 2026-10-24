import { Effect, Layer, Option, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { strict as assert } from "node:assert"
import { MagnitudeClient, ChatConversationId, ChatCommandId, ChatModelId, defaultChatSettings, applyChatUpdate, type ChatConversation } from "../../../../packages/sdk/src/index"
const layer = MagnitudeClient.layer({ origin: "http://127.0.0.1:11101", autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer))
await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const client = yield* MagnitudeClient
  const settings = { ...defaultChatSettings, modelId: Option.some(ChatModelId.make("minicpm5-1b:gguf:q4")), reasoningEffort: Option.some("none"), maxOutputTokens: Option.some(16) }
  const conversationId = ChatConversationId.make(crypto.randomUUID())
  yield* client.chat.createConversation({ conversationId, settings })
  yield* client.chat.updateConversation({ conversationId, title: Option.some("Context limit verification"), settings: Option.none() })
  const content = "token ".repeat(150_000)
  yield* client.chat.startGeneration({ conversationId, commandId: ChatCommandId.make(crypto.randomUUID()), content: Option.some(content), retryMessageId: Option.none(), settings })
  const result = yield* client.chat.watchConversation({ conversationId }).pipe(Stream.scan(null as ChatConversation | null, applyChatUpdate), Stream.filterMap(c => c?.messages.at(-1)?.generation.pipe(Option.filter(g => g.status !== "running")) ?? Option.none()), Stream.runHead, Effect.map(Option.getOrThrow), Effect.timeout("20 seconds"))
  assert.equal(result.status, "failed")
  assert.match(Option.getOrThrow(result.error), /prompt is too long|context/i)
  const saved = yield* client.chat.getConversation({ conversationId })
  assert.equal(saved.messages[0]?.content, content)
  console.log(JSON.stringify({ result: "PASS", conversationId, status: result.status, error: Option.getOrThrow(result.error), preservedCharacters: content.length }))
}).pipe(Effect.provide(layer))))
