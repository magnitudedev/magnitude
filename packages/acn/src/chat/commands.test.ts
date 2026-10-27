import { Effect, Either, Option } from "effect"
import { describe, expect, it } from "vitest"
import { EventStore } from "@magnitudedev/event-engine"
import type { ChatMessagePayload, ChatRetryPayload, MessageId, SessionEvent, SessionId } from "@magnitudedev/acn-protocol"
import { finished, model, newChat, newId, sent, settings, started } from "./chat.test-support"
import * as Commands from "./commands"
import { ChatEngine } from "./engine"
import { Live, Seen, Transcript } from "./folds"
import { Generation } from "./generation"

const Folds = ChatEngine.make({ name: "commands", projections: [Seen, Transcript, Live], workers: [] })

const chatId: SessionId = newChat()

/** The reader of a session holding these events. */
const reader = (events: ReadonlyArray<SessionEvent>) => Effect.gen(function* () {
  const runtime = yield* Folds.open({ store: yield* EventStore.memory<SessionEvent>() })
  yield* Effect.forEach(events, runtime.accept, { discard: true })
  return runtime.read
})

/** What a command decides over a session holding these events. */
const decide = <J>(events: ReadonlyArray<SessionEvent>, command: Commands.Command<J>) =>
  Effect.runPromise(Effect.scoped(Effect.flatMap(reader(events), command)))

/** The settled state commands decide on, over a session holding these events. */
const states = (events: ReadonlyArray<SessionEvent>) =>
  Effect.runPromise(Effect.scoped(Effect.flatMap(reader(events), Commands.states)))

const message = (ids: { readonly messageId: MessageId; readonly responseId: MessageId }): ChatMessagePayload => ({
  chatId,
  ...ids,
  text: "Hi",
  model,
  reasoningEffort: Option.none(),
  settings,
})
const retrying = (replaces: MessageId, responseId = newId()): ChatRetryPayload =>
  ({ chatId, replaces, responseId, model, reasoningEffort: Option.none(), settings })

const rejection = (result: Either.Either<void, { readonly _tag: string; readonly message: string }>) =>
  Either.match(result, { onLeft: (error) => [error._tag, error.message], onRight: () => "accepted" })

const [m1, r1] = [newId(), newId()]
const answered = [sent(m1, "Hi", 0), started(r1, 0), finished(r1, 10)]
const running = [sent(m1, "Hi", 0), started(r1, 0)]

describe("start", () => {
  it("starts only a session that does not exist, with its first events, and does nothing when replayed", async () => {
    const ids = { messageId: newId(), responseId: newId() }
    const fresh = await decide([], Commands.start(message(ids)))
    expect([fresh.events.map((event) => event._tag), rejection(fresh.result)]).toEqual([["message_sent", "response_started"], "accepted"])
    const taken = await decide(answered, Commands.start(message(ids)))
    expect(taken.events).toEqual([])
    expect(rejection(taken.result)).toEqual(["InvalidChatRequest", "These IDs were already used for other messages."])
    const replay = await decide(answered, Commands.start(message({ messageId: m1, responseId: r1 })))
    expect([replay.events, rejection(replay.result)]).toEqual([[], "accepted"])
  })

  it("is replayed only as the session's first message and its response", async () => {
    const [m2, r2] = [newId(), newId()]
    const later = await decide([...answered, sent(m2, "Again", 20), started(r2, 20)], Commands.start(message({ messageId: m2, responseId: r2 })))
    expect([later.events, rejection(later.result)[0]]).toEqual([[], "InvalidChatRequest"])
    const partly = await decide(answered, Commands.start(message({ messageId: m1, responseId: newId() })))
    expect([partly.events, rejection(partly.result)[0]]).toEqual([[], "InvalidChatRequest"])
  })
})

describe("send", () => {
  it("needs an existing chat", async () => {
    const decision = await decide([], Commands.send(message({ messageId: newId(), responseId: newId() })))
    expect(rejection(decision.result)).toEqual(["ChatNotFound", "This conversation no longer exists."])
  })

  it("does nothing when replayed, rejects partly used IDs, and is busy while a response runs", async () => {
    const replay = await decide(answered, Commands.send(message({ messageId: m1, responseId: r1 })))
    expect([replay.events, rejection(replay.result)]).toEqual([[], "accepted"])
    const partly = await decide(answered, Commands.send(message({ messageId: m1, responseId: newId() })))
    expect(rejection(partly.result)[0]).toBe("InvalidChatRequest")
    const busy = await decide(running, Commands.send(message({ messageId: newId(), responseId: newId() })))
    expect(rejection(busy.result)[0]).toBe("ChatBusy")
    const next = await decide(answered, Commands.send(message({ messageId: newId(), responseId: newId() })))
    expect(next.events.map((event) => event._tag)).toEqual(["message_sent", "response_started"])
  })

  it("is replayed only when its response answers its message", async () => {
    const [m2, r2] = [newId(), newId()]
    const twice = [...answered, sent(m2, "Again", 20), started(r2, 20), finished(r2, 30)]
    for (const ids of [{ messageId: m1, responseId: r2 }, { messageId: r1, responseId: m1 }]) {
      const crossed = await decide(twice, Commands.send(message(ids)))
      expect([crossed.events, rejection(crossed.result)]).toEqual([[], ["InvalidChatRequest", "These IDs were already used for other messages."]])
      expect(Commands.sendReplayed(message(ids))(await states(twice))).toBe(false)
    }
    expect(Commands.sendReplayed(message({ messageId: m2, responseId: r2 }))(await states(twice))).toBe(true)
  })
})

describe("retry", () => {
  it("replaces only the latest ended response, once", async () => {
    const replaced = await decide(answered, Commands.retry(retrying(r1)))
    expect(replaced.events).toMatchObject([{ _tag: "response_started", replaces: Option.some(r1) }])
    const r2 = newId()
    const replay = await decide([...answered, started(r2, 20, Option.some(r1))], Commands.retry(retrying(r1, r2)))
    expect([replay.events, rejection(replay.result)]).toEqual([[], "accepted"])
    const stale = await decide(answered, Commands.retry(retrying(newId())))
    expect(rejection(stale.result)).toEqual(["InvalidChatRequest", "Only the last response can be retried."])
    const busy = await decide(running, Commands.retry(retrying(r1)))
    expect(rejection(busy.result)[0]).toBe("ChatBusy")
  })

  it("is not replayed by a response ID that names a user message", async () => {
    const named = await decide(answered, Commands.retry(retrying(r1, m1)))
    expect([named.events, rejection(named.result)]).toEqual([[], ["InvalidChatRequest", "These IDs were already used for other messages."]])
    expect(Commands.retryReplayed(retrying(r1, m1))(await states(answered))).toBe(false)
  })
})

describe("stop", () => {
  it("interrupts generation only for the running response it names", async () => {
    const stopping = await decide(running, Commands.stop({ chatId, responseId: r1 }))
    expect(stopping.interrupts).toEqual([Generation])
    const ended = await decide(answered, Commands.stop({ chatId, responseId: r1 }))
    expect([ended.events, ended.interrupts]).toEqual([[], undefined])
    const unknown = await decide(running, Commands.stop({ chatId, responseId: newId() }))
    expect(rejection(unknown.result)).toEqual(["InvalidChatRequest", "This chat has no such response."])
    const message = await decide(running, Commands.stop({ chatId, responseId: m1 }))
    expect([message.interrupts, rejection(message.result)]).toEqual([undefined, ["InvalidChatRequest", "This chat has no such response."]])
  })
})
