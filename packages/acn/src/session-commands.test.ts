import { afterEach, describe, expect, it } from "vitest"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Exit, Layer, Option, Ref, Scope, Stream } from "effect"
import type {
  AgentLifecycleState,
  AppEvent,
  CodingAgentSession,
  ForkTurnState,
} from "@magnitudedev/agent"
import { DirectoryPathSchema } from "@magnitudedev/acn-protocol"
import { AgentRuntime, type AgentRuntimeApi } from "./agent-runtime"
import { FileMentionSearcherLive } from "./file-mention-searcher"
import { GitInspectorLive } from "./git-inspector"
import { SessionCommands, SessionCommandsLive } from "./session-commands"
import { testFileSystemManagerLayer, testPlatformLayer } from "./session-test-support"
import type { RuntimeEntry } from "./session-types"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

const roots: string[] = []

afterEach(async () => Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))))

async function scratchpad(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "magnitude-session-commands-"))
  roots.push(root)
  return root
}

const idleTurnState: ForkTurnState = {
  _tag: "idle",
  completedTurns: 0,
  triggers: [],
  pendingInboundCommunications: [],
  parentForkId: null,
  connectionRetryCount: 0,
}

const idleAgentStatus: AgentLifecycleState = {
  agents: new Map(),
  agentByForkId: new Map(),
  rootWork: {
    phase: "idle",
    accumulatedProductiveMs: 0,
    productiveStartedAt: null,
    lastProductiveMs: 0,
    chainStartedAt: null,
    activeChildCount: 0,
    _currentTurn: null,
    _currentChainId: null,
    _isThinking: false,
    _generation: null,
  },
}

const makeSession = (send: CodingAgentSession["send"]): CodingAgentSession => ({
  on: {
    restoreQueuedMessages: Stream.never,
  },
  state: {
    work: {
      get: () => Effect.succeed({ _tag: "Quiescent" as const, workerCount: 0 as const }),
      subscribe: Stream.succeed({
        _tag: "Quiescent" as const,
        workerCount: 0 as const,
      }),
    },
    turn: {
      getFork: () => Effect.succeed(idleTurnState),
      subscribeFork: () => Stream.succeed(idleTurnState),
    },
    agentStatus: {
      get: () => Effect.succeed(idleAgentStatus),
      subscribe: Stream.succeed(idleAgentStatus),
    },
  },
  displayView: {
    stream: () => Stream.die("unused test session displayView.stream"),
    snapshot: () => Effect.die("unused test session displayView.snapshot"),
    setShape: () => Effect.die("unused test session displayView.setShape"),
    close: () => Effect.void,
  },
  send,
  interrupt: () => Effect.die("unused test session interrupt"),
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
})

const makeEntry = Effect.fn("test.make-session-command-entry")(function* (
  sessionId: string,
  session: CodingAgentSession,
  scratchpadPath = "/tmp/magnitude-session-commands-scratchpad",
) {
  const scope = yield* Scope.make()
  return {
    id: sessionId,
    createdAt: 1,
    updatedAt: 1,
    title: "Session",
    cwd: DirectoryPathSchema.make(process.cwd()),
    scratchpadPath,
    session,
    scope,
  } satisfies RuntimeEntry
})

const mentionSearcherLayer = FileMentionSearcherLive.pipe(
  Layer.provide(Layer.mergeAll(
    testFileSystemManagerLayer,
    GitInspectorLive.pipe(Layer.provide(testPlatformLayer)),
    testPlatformLayer,
  )),
)

const makeLayer = (runtime: AgentRuntimeApi) =>
  SessionCommandsLive.pipe(
    Layer.provide(
      Layer.mergeAll(
        Layer.succeed(AgentRuntime, runtime),
        mentionSearcherLayer,
        BunFileSystem.layer,
        BunPath.layer,
      ),
    ),
  )

describe("SessionCommands", () => {
  it("sendUserMessage starts an evicted runtime before sending", async () => {
    await Effect.runPromise(
      Effect.gen(function* () {
        const sentEvents = yield* Ref.make<ReadonlyArray<AppEvent>>([])
        const withSessionCalls = yield* Ref.make(0)
        const entry = yield* makeEntry(
          "session-a",
          makeSession((event) => Ref.update(sentEvents, (events) => [...events, event])),
        )

        const runtime: AgentRuntimeApi = {
          withSession: (sessionId, _label, use) =>
            Ref.update(withSessionCalls, (count) => count + 1).pipe(
              Effect.zipRight(use({ ...entry, id: sessionId }, 1)),
            ),
          withSessionWork: (sessionId, _label, use) =>
            Ref.update(withSessionCalls, (count) => count + 1).pipe(
              Effect.zipRight(use({ ...entry, id: sessionId }, 1)),
            ),
          withSessionRequest: () => Effect.die("unused"),
          tryWithResident: () => Effect.succeed(Option.none()),
          tryWithBusyResident: () => Effect.succeed(Option.none()),
          residentSessions: Effect.succeed([]),
          dispose: () => Effect.void,
          deleteSession: (_sessionId, remove) => remove,
          registerRetirementObserver: () => Effect.succeed(Effect.void),
          changes: Stream.never,
        }

        yield* Effect.gen(function* () {
          const commands = yield* SessionCommands
          yield* commands.sendUserMessage({
            sessionId: "session-a",
            content: "hello after eviction",
            taskMode: false,
            uploads: [],
            mentions: [],
          })
        }).pipe(Effect.provide(makeLayer(runtime)))

        const calls = yield* Ref.get(withSessionCalls)
        const events = yield* Ref.get(sentEvents)

        expect(calls).toBe(1)
        expect(events).toHaveLength(1)
        expect(events[0]?.type).toBe("user_message")
      }),
    )
  })

  it("materializes an uploaded text file before admitting its trailing mention", async () => {
    const root = await scratchpad()
    await Effect.runPromise(
      Effect.gen(function* () {
        const sentEvents = yield* Ref.make<ReadonlyArray<AppEvent>>([])
        const entry = yield* makeEntry(
          "session-with-upload",
          makeSession(event => Ref.update(sentEvents, events => [...events, event])),
          root,
        )
        const runtime: AgentRuntimeApi = {
          withSession: (_sessionId, _label, use) => use(entry, 1),
          withSessionWork: (_sessionId, _label, use) => use(entry, 1),
          withSessionRequest: () => Effect.die("unused"),
          tryWithResident: () => Effect.succeed(Option.none()),
          tryWithBusyResident: () => Effect.succeed(Option.none()),
          residentSessions: Effect.succeed([]),
          dispose: () => Effect.void,
          deleteSession: (_sessionId, remove) => remove,
          registerRetirementObserver: () => Effect.succeed(Effect.void),
          changes: Stream.never,
        }

        yield* Effect.gen(function* () {
          const commands = yield* SessionCommands
          yield* commands.sendUserMessage({
            sessionId: "session-with-upload",
            content: "",
            taskMode: false,
            uploads: [{
              type: "raw_text_file",
              filename: "notes.md",
              data: Buffer.from("# Notes\n").toString("base64"),
            }],
            mentions: [],
          })
        }).pipe(Effect.provide(makeLayer(runtime)))

        const events = yield* Ref.get(sentEvents)
        expect(events[0]).toMatchObject({
          type: "user_message",
          mentions: [{
            attachment: { type: "mention_file", path: "$M/attachments/notes.md" },
            placement: { _tag: "trailing" },
          }],
        })
      }),
    )
  })

  it("removes a materialized upload when message admission fails", async () => {
    const root = await scratchpad()
    await Effect.runPromise(
      Effect.gen(function* () {
        const entry = yield* makeEntry(
          "session-rejecting-upload",
          makeSession(() => Effect.die("admission failed")),
          root,
        )
        const runtime: AgentRuntimeApi = {
          withSession: (_sessionId, _label, use) => use(entry, 1),
          withSessionWork: (_sessionId, _label, use) => use(entry, 1),
          withSessionRequest: () => Effect.die("unused"),
          tryWithResident: () => Effect.succeed(Option.none()),
          tryWithBusyResident: () => Effect.succeed(Option.none()),
          residentSessions: Effect.succeed([]),
          dispose: () => Effect.void,
          deleteSession: (_sessionId, remove) => remove,
          registerRetirementObserver: () => Effect.succeed(Effect.void),
          changes: Stream.never,
        }

        const exit = yield* Effect.gen(function* () {
          const commands = yield* SessionCommands
          yield* commands.sendUserMessage({
            sessionId: "session-rejecting-upload",
            content: "",
            taskMode: false,
            uploads: [{
              type: "raw_text_file",
              filename: "notes.md",
              data: Buffer.from("not admitted").toString("base64"),
            }],
            mentions: [],
          })
        }).pipe(Effect.provide(makeLayer(runtime)), Effect.exit)

        expect(Exit.isFailure(exit)).toBe(true)
      }),
    )

    expect(await readdir(join(root, "attachments"))).toEqual([])
  })
})
