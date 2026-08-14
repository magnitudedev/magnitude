import { describe, expect, it } from "vitest"
import * as RpcGroup from "@effect/rpc/RpcGroup"
import * as RpcTest from "@effect/rpc/RpcTest"
import { Context, Effect, Layer, Option, Stream } from "effect"
import type {
  DisplayMessage,
  DisplayRootStatus,
  DisplayTimelineEntry,
  DisplayViewShape,
  DisplayViewStateEvent,
  SessionMetadata,
  SessionOptions,
  StreamEvent,
} from "@magnitudedev/sdk"
import { MagnitudeRpcs } from "@magnitudedev/sdk"
import { makeDisplayViewSnapshotFixture } from "@magnitudedev/sdk/testing"
import {
  HeadlessSessionClient,
  HeadlessSessionClientFailure,
  HeadlessSessionIdSchema,
  type HeadlessSessionId,
  HeadlessSessionStartFailed,
  HeadlessSessionStreamEnded,
  runHeadlessSession,
  type RunHeadlessSessionRequest,
} from "./session-runner"

const session: SessionMetadata = {
  sessionId: "session-1",
  title: "Headless test",
  cwd: "/repo",
  createdAt: 1,
  updatedAt: 1,
  messageCount: 1,
  lastMessage: "test prompt",
}

const sessionOptions: SessionOptions = {
  headless: true,
  solo: true,
  disableCwdSafeguards: false,
  disableShellSafeguards: false,
}

function displayStateEvent(options: {
  readonly shape: DisplayViewShape
  readonly status: DisplayRootStatus
  readonly messages?: readonly DisplayMessage[]
  readonly entries?: readonly DisplayTimelineEntry[]
  readonly mode?: "idle" | "streaming"
}): DisplayViewStateEvent {
  const messages = options.messages ?? []
  return {
    _tag: "state",
    ...makeDisplayViewSnapshotFixture({
      shape: options.shape,
      session: { sessionId: session.sessionId, title: session.title ?? "", cwd: session.cwd },
      status: options.status,
      messages,
      entries: options.entries,
      mode: options.mode,
      streamingMessageId: options.mode === "streaming" ? "assistant-1" : null,
    }),
  }
}

const userMessage: DisplayMessage = {
  id: "user-1",
  type: "user_message",
  content: "test prompt",
  timestamp: 1,
  taskMode: false,
  attachments: [],
}

const assistantMessage: DisplayMessage = {
  id: "assistant-1",
  type: "assistant_message",
  content: "done",
  timestamp: 2,
}

function entry(message: DisplayMessage, streaming = false): DisplayTimelineEntry {
  return {
    kind: "message",
    id: `entry:${message.id}`,
    messageId: message.id,
    timestamp: message.timestamp,
    role: message.type === "user_message" ? "user" : "assistant",
    streaming,
    interrupted: false,
    nextMessageInterrupted: false,
  }
}

function makeClient(options: {
  readonly terminalEvent?: DisplayViewStateEvent
  readonly createFailure?: string
} = {}): {
  readonly client: HeadlessSessionClient
  readonly created: Array<{
    readonly sessionId: HeadlessSessionId
    readonly cwd: string
    readonly initial: { readonly type: "message" | "goal"; readonly content: string }
    readonly options: SessionOptions
  }>
  readonly getSessionCalls: string[]
} {
  const created: Array<{
    readonly sessionId: HeadlessSessionId
    readonly cwd: string
    readonly initial: { readonly type: "message" | "goal"; readonly content: string }
    readonly options: SessionOptions
  }> = []
  const getSessionCalls: string[] = []
  let shape: DisplayViewShape | null = null

  const client: HeadlessSessionClient = {
    createSession: (request) => Effect.sync(() => {
      created.push(request)
      if (options.createFailure) {
        return { _tag: "failed" as const, error: options.createFailure }
      }
      return { _tag: "created" as const, metadata: session }
    }),
    getSession: (sessionId) => Effect.sync(() => {
      getSessionCalls.push(sessionId)
      return session
    }),
    resyncDisplayView: () => Effect.sync(() => {
      if (!shape) throw new Error("shape must be registered before resync")
      return displayStateEvent({
        shape,
        status: { _tag: "Worked", lastProductiveMs: 20 },
        messages: [userMessage, assistantMessage],
        entries: [entry(userMessage), entry(assistantMessage)],
      })
    }),
    streamDisplayView: (_sessionId, _viewId, requestedShape) => {
      shape = requestedShape
      const initialEvent = displayStateEvent({
        shape: requestedShape,
        status: {
          _tag: "Working",
          chainStartedAt: 1,
          detail: { _tag: "Thinking" },
          activeChildCount: 0,
        },
        messages: [userMessage],
        entries: [entry(userMessage)],
      })
      const terminalEvent = options.terminalEvent ?? displayStateEvent({
        shape: requestedShape,
        status: { _tag: "Worked", lastProductiveMs: 20 },
        messages: [userMessage, assistantMessage],
        entries: [entry(userMessage), entry(assistantMessage)],
      })
      return Stream.make(initialEvent, terminalEvent)
    },
  }

  return { client, created, getSessionCalls }
}

const headlessShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 10_000, live: true, presentation: "default" },
  },
}

const request = {
  sessionId: HeadlessSessionIdSchema.make(session.sessionId),
  cwd: "/repo",
  initial: { type: "message" as const, content: "test prompt" },
  options: sessionOptions,
}

describe("runHeadlessSession", () => {
  it("creates a durable daemon session, observes display state, and returns after completion", async () => {
    const fake = makeClient()
    const snapshots: string[] = []

    const result = await Effect.runPromise(runHeadlessSession(request, {
      onSnapshot: (snapshot) => Effect.sync(() => {
        const root = snapshot.state.actors.root
        snapshots.push(root?.kind === "root" ? root.status._tag : "missing")
      }),
    }).pipe(Effect.provideService(HeadlessSessionClient, fake.client)))

    expect(result.status).toBe("completed")
    expect(result.sessionId).toBe(session.sessionId)
    expect(result.session).toEqual(session)
    expect(fake.created).toEqual([request])
    expect(fake.getSessionCalls).toEqual([session.sessionId])
    expect(snapshots).toEqual(["Working", "Worked"])
  })

  it("returns a failed terminal result when the authoritative display ends in an error", async () => {
    const failure: DisplayMessage = {
      id: "error-1",
      type: "error",
      message: "provider not ready",
      timestamp: 2,
      cta: Option.none(),
    }
    const fake = makeClient({
      terminalEvent: displayStateEvent({
        shape: {
          timelines: {
            root: { kind: "tail", limit: 10_000, live: true, presentation: "default" },
          },
        },
        status: { _tag: "Worked", lastProductiveMs: 20 },
        messages: [userMessage, failure],
        entries: [entry(userMessage), entry(failure)],
      }),
    })

    const result = await Effect.runPromise(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, fake.client)))

    expect(result.status).toBe("failed")
  })

  it("uses authoritative timeline order when success and failure timestamps tie", async () => {
    const tiedAssistant: DisplayMessage = { ...assistantMessage, timestamp: 2 }
    const tiedFailure: DisplayMessage = {
      id: "error-1",
      type: "error",
      message: "provider failed after partial output",
      timestamp: 2,
      cta: Option.none(),
    }
    const fake = makeClient({
      terminalEvent: displayStateEvent({
        shape: {
          timelines: {
            root: { kind: "tail", limit: 10_000, live: true, presentation: "default" },
          },
        },
        status: { _tag: "Worked", lastProductiveMs: 20 },
        messages: [userMessage, tiedAssistant, tiedFailure],
        entries: [entry(userMessage), entry(tiedAssistant), entry(tiedFailure)],
      }),
    })

    const result = await Effect.runPromise(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, fake.client)))

    expect(result.status).toBe("failed")
  })

  it("acquires and releases the display subscription when the materialized state is already terminal", async () => {
    const fake = makeClient()
    let acquired = 0
    let released = 0
    const terminal = displayStateEvent({
      shape: {
        timelines: {
          root: { kind: "tail", limit: 10_000, live: true, presentation: "default" },
        },
      },
      status: { _tag: "Worked", lastProductiveMs: 20 },
      messages: [userMessage, assistantMessage],
      entries: [entry(userMessage), entry(assistantMessage)],
    })
    const client: HeadlessSessionClient = {
      ...fake.client,
      streamDisplayView: () => Stream.acquireRelease(
        Effect.sync(() => {
          acquired += 1
          return terminal
        }),
        () => Effect.sync(() => { released += 1 }),
      ),
    }

    const result = await Effect.runPromise(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))

    expect(result.status).toBe("completed")
    expect(acquired).toBe(1)
    expect(released).toBe(1)
  })

  it("releases the display subscription when the output observer fails", async () => {
    let releases = 0
    const terminal = displayStateEvent({
      shape: {
        timelines: {
          root: { kind: "tail", limit: 10_000, live: true, presentation: "default" },
        },
      },
      status: { _tag: "Worked", lastProductiveMs: 20 },
      messages: [userMessage, assistantMessage],
      entries: [entry(userMessage), entry(assistantMessage)],
    })
    const client: HeadlessSessionClient = {
      ...makeClient().client,
      streamDisplayView: () => Stream.acquireRelease(
        Effect.succeed(terminal),
        () => Effect.sync(() => { releases += 1 }),
      ),
    }

    const exit = await Effect.runPromiseExit(runHeadlessSession(request, {
      onSnapshot: () => Effect.die("output sink closed"),
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))

    expect(exit._tag).toBe("Failure")
    expect(releases).toBe(1)
  })

  it("fails explicitly when durable session creation is rejected", async () => {
    const fake = makeClient({ createFailure: "disk unavailable" })

    const exit = await Effect.runPromiseExit(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, fake.client)))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      expect(exit.cause._tag).toBe("Fail")
      if (exit.cause._tag === "Fail") {
        expect(exit.cause.error).toBeInstanceOf(HeadlessSessionStartFailed)
        expect(exit.cause.error.message).toContain("disk unavailable")
      }
    }
  })

  it("resyncs from authoritative display state when a streamed patch cannot apply", async () => {
    const fake = makeClient()
    let resyncCalls = 0
    const invalidPatch: StreamEvent = {
      _tag: "patch",
      ops: [{
        op: "replace",
        path: ["state", "session", "sessionId", "nested"],
        value: "NotARealStatus",
      }],
    }
    const client: HeadlessSessionClient = {
      ...fake.client,
      streamDisplayView: (sessionId, viewId, shape) => fake.client
        .streamDisplayView(sessionId, viewId, shape)
        .pipe(Stream.take(1), Stream.concat(Stream.make(invalidPatch))),
      resyncDisplayView: (sessionId, viewId) => {
        resyncCalls += 1
        return fake.client.resyncDisplayView(sessionId, viewId)
      },
    }

    const result = await Effect.runPromise(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))

    expect(result.status).toBe("completed")
    expect(resyncCalls).toBe(1)
  })

  it("fails instead of reporting success when Worked has no authoritative root timeline", async () => {
    const fake = makeClient()
    const terminal = displayStateEvent({
      shape: headlessShape,
      status: { _tag: "Worked", lastProductiveMs: 20 },
      messages: [userMessage, assistantMessage],
      entries: [entry(userMessage), entry(assistantMessage)],
    })
    const incomplete: DisplayViewStateEvent = {
      ...terminal,
      state: {
        ...terminal.state,
        timelines: {},
      },
    }
    const client: HeadlessSessionClient = {
      ...fake.client,
      streamDisplayView: () => Stream.make(incomplete),
      resyncDisplayView: () => Effect.succeed(incomplete),
    }

    const exit = await Effect.runPromiseExit(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(HeadlessSessionStreamEnded)
    }
  })

  it("does not accept interrupted root status while its timeline is streaming", async () => {
    const fake = makeClient()
    const terminal = displayStateEvent({
      shape: headlessShape,
      status: { _tag: "Interrupted", lastProductiveMs: 20 },
      messages: [userMessage, assistantMessage],
      entries: [entry(userMessage), entry(assistantMessage)],
    })
    const timeline = terminal.state.timelines.root
    const incomplete: DisplayViewStateEvent = {
      ...terminal,
      state: {
        ...terminal.state,
        timelines: { ...terminal.state.timelines, root: { ...timeline, mode: "streaming" } },
      },
    }
    const client: HeadlessSessionClient = {
      ...fake.client,
      streamDisplayView: () => Stream.make(incomplete),
      resyncDisplayView: () => Effect.succeed(incomplete),
    }
    const exit = await Effect.runPromiseExit(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(HeadlessSessionStreamEnded)
    }
  })

  it.each([
    {
      name: "message IDs do not match their order keys",
      mutate: (event: DisplayViewStateEvent): DisplayViewStateEvent => {
        const [timelineKey] = Object.keys(event.state.timelines)
        const timeline = event.state.timelines[timelineKey]
        if (timelineKey === undefined || timeline === undefined) throw new Error("missing root timeline")
        const messageId = timeline.messages.order.at(-1)
        const message = messageId === undefined ? undefined : timeline.messages.byId[messageId]
        if (message === undefined) throw new Error("missing terminal message")
        return {
          ...event,
          state: {
            ...event.state,
            timelines: {
              ...event.state.timelines,
              [timelineKey]: {
                ...timeline,
                messages: {
                  order: ["alias"],
                  byId: { alias: { ...message, id: "different" } },
                },
              },
            },
          },
        }
      },
    },
    {
      name: "messages.order contains duplicate IDs",
      mutate: (event: DisplayViewStateEvent): DisplayViewStateEvent => {
        const [timelineKey] = Object.keys(event.state.timelines)
        const timeline = event.state.timelines[timelineKey]
        if (timelineKey === undefined || timeline === undefined) throw new Error("missing root timeline")
        const messageId = timeline.messages.order.at(-1)
        const message = messageId === undefined ? undefined : timeline.messages.byId[messageId]
        if (messageId === undefined || message === undefined) throw new Error("missing terminal message")
        return {
          ...event,
          state: {
            ...event.state,
            timelines: {
              ...event.state.timelines,
              [timelineKey]: {
                ...timeline,
                messages: {
                  order: [messageId, messageId],
                  byId: { [messageId]: message },
                },
              },
            },
          },
        }
      },
    },
   {
      name: "a presentation message is absent from authoritative order",
      mutate: (event: DisplayViewStateEvent): DisplayViewStateEvent => {
        const [timelineKey] = Object.keys(event.state.timelines)
        const timeline = event.state.timelines[timelineKey]
        if (timelineKey === undefined || timeline === undefined) throw new Error("missing root timeline")
        const orphan = { ...assistantMessage, id: "orphan-assistant" }
        return {
          ...event,
          state: {
            ...event.state,
            timelines: {
              ...event.state.timelines,
              [timelineKey]: {
                ...timeline,
                messages: { ...timeline.messages, byId: { ...timeline.messages.byId, [orphan.id]: orphan } },
                presentation: {
                  ...timeline.presentation,
                  entries: [...timeline.presentation.entries, {
                    kind: "message" as const,
                    id: "entry:orphan-assistant",
                    messageId: orphan.id,
                    timestamp: orphan.timestamp,
                    role: "assistant" as const,
                    streaming: false,
                    interrupted: false,
                    nextMessageInterrupted: false,
                  }],
                },
              },
            },
          },
        }
      },
    },
  ])("fails closed when $name", async ({ mutate }) => {
    const fake = makeClient()
    const incomplete = mutate(displayStateEvent({
      shape: headlessShape,
      status: { _tag: "Worked", lastProductiveMs: 20 },
      messages: [userMessage, assistantMessage],
      entries: [entry(userMessage), entry(assistantMessage)],
    }))
    const client: HeadlessSessionClient = {
      ...fake.client,
      streamDisplayView: () => Stream.make(incomplete),
      resyncDisplayView: () => Effect.succeed(incomplete),
    }

    const exit = await Effect.runPromiseExit(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(HeadlessSessionStreamEnded)
    }
  })

  it.each([
    {
      name: "the root timeline is still streaming",
      mutate: (timeline: DisplayViewStateEvent["state"]["timelines"][string]) => ({
        ...timeline,
        mode: "streaming" as const,
      }),
    },
    {
      name: "the root timeline still names a streaming message",
      mutate: (timeline: DisplayViewStateEvent["state"]["timelines"][string]) => ({
        ...timeline,
        streamingMessageId: timeline.messages.order.at(-1) ?? "missing",
      }),
    },
    {
      name: "a non-assistant presentation remains streaming",
      mutate: (timeline: DisplayViewStateEvent["state"]["timelines"][string]) => ({
        ...timeline,
        presentation: {
          ...timeline.presentation,
          entries: timeline.presentation.entries.map((entry) =>
            entry.kind === "message" && entry.role !== "assistant"
              ? { ...entry, streaming: true }
              : entry),
        },
      }),
    },
    {
      name: "the authoritative assistant presentation remains streaming",
      mutate: (timeline: DisplayViewStateEvent["state"]["timelines"][string]) => ({
        ...timeline,
        presentation: {
          ...timeline.presentation,
          entries: timeline.presentation.entries.map((entry) =>
            entry.kind === "message" && entry.role === "assistant"
              ? { ...entry, streaming: true }
              : entry),
        },
      }),
    },
  ])("fails closed when $name", async ({ mutate }) => {
    const terminal = displayStateEvent({
      shape: headlessShape,
      status: { _tag: "Worked", lastProductiveMs: 4 },
      messages: [userMessage, {
        id: "assistant-1",
        type: "assistant_message",
        content: "done",
        timestamp: 4,
      }],
      entries: [entry(userMessage), {
        kind: "message",
        id: "entry:assistant-1",
        messageId: "assistant-1",
        timestamp: 4,
        role: "assistant",
        streaming: false,
        interrupted: false,
        nextMessageInterrupted: false,
      }],
    })
    const rootKey = Object.keys(terminal.state.timelines)[0]!
    const timeline = terminal.state.timelines[rootKey]!
    const incomplete = {
      ...terminal,
      state: {
        ...terminal.state,
        timelines: { ...terminal.state.timelines, [rootKey]: mutate(timeline) },
      },
    }
    const fake = makeClient({ terminalEvent: incomplete })
    const result = await Effect.runPromise(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, fake.client), Effect.exit))
    expect(result._tag).toBe("Failure")
  })

  it("fails instead of reporting success when the display subscription ends before terminal state", async () => {
    const fake = makeClient()
    const client: HeadlessSessionClient = {
      ...fake.client,
      streamDisplayView: () => Stream.empty,
    }

    const exit = await Effect.runPromiseExit(runHeadlessSession(request, {
      onSnapshot: () => Effect.void,
    }).pipe(Effect.provideService(HeadlessSessionClient, client)))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(HeadlessSessionStreamEnded)
    }
  })

  it("runs through the typed SDK RPC group and preserves daemon-side session state", async () => {
    const trace: string[] = []
    let persistedRequest: RunHeadlessSessionRequest | null = null
    let registeredShape: DisplayViewShape | null = null

    type MagnitudeRpc = typeof MagnitudeRpcs extends RpcGroup.RpcGroup<infer Rpc> ? Rpc : never
    type HeadlessRpcTag =
      | "CreateSession"
      | "GetSession"
      | "ResyncDisplayView"
      | "StreamDisplayView"
    const rpcByTag = <Tag extends HeadlessRpcTag>(tag: Tag): Extract<MagnitudeRpc, { _tag: Tag }> => {
      const rpc = MagnitudeRpcs.requests.get(tag)
      if (!rpc) throw new Error(`Missing ${tag} from MagnitudeRpcs`)
      return rpc as Extract<MagnitudeRpc, { _tag: Tag }>
    }
    const createSessionRpc = rpcByTag("CreateSession")
    const headlessRpcs = RpcGroup.make(
      createSessionRpc,
      rpcByTag("GetSession"),
      rpcByTag("ResyncDisplayView"),
      rpcByTag("StreamDisplayView"),
    )
    const demandTag = [...createSessionRpc.middlewares][0]!
    const demand: Context.Tag.Service<typeof demandTag> = ({ next }) => next
    const demandLayer = Layer.succeed(demandTag, demand)

    const handlers = Layer.mergeAll(
      headlessRpcs.toLayerHandler("CreateSession", (rpcRequest) => Effect.sync(() => {
        trace.push("CreateSession")
        expect(Option.getOrThrow(rpcRequest.sessionId)).toBe(request.sessionId)
        const initial = Option.getOrThrow(rpcRequest.initial)
        const options = Option.getOrThrow(rpcRequest.options)
        persistedRequest = {
          sessionId: HeadlessSessionIdSchema.make(Option.getOrThrow(rpcRequest.sessionId)),
          cwd: rpcRequest.cwd,
          initial: initial._tag === "goal"
            ? { type: "goal", content: initial.objective }
            : { type: "message", content: initial.content },
          options,
        }
        return { _tag: "created" as const, metadata: session }
      })),
      headlessRpcs.toLayerHandler("GetSession", () => Effect.sync(() => {
        trace.push("GetSession")
        return session
      })),
      headlessRpcs.toLayerHandler("ResyncDisplayView", () => Effect.sync(() => {
        trace.push("ResyncDisplayView")
        if (!registeredShape) throw new Error("display shape was not registered")
        return displayStateEvent({
          shape: registeredShape,
          status: { _tag: "Worked", lastProductiveMs: 20 },
          messages: [userMessage, assistantMessage],
          entries: [entry(userMessage), entry(assistantMessage)],
        })
      })),
      headlessRpcs.toLayerHandler("StreamDisplayView", (rpcRequest) => {
        trace.push("StreamDisplayView")
        expect(rpcRequest.materialize).toBe(true)
        registeredShape = rpcRequest.shape
        return Stream.make(displayStateEvent({
          shape: rpcRequest.shape,
          status: { _tag: "Worked", lastProductiveMs: 20 },
          messages: [userMessage, assistantMessage],
          entries: [entry(userMessage), entry(assistantMessage)],
        }))
      }),
    )

    const mapFailure = (operation: string) => <A, E, R>(effect: Effect.Effect<A, E, R>) =>
      effect.pipe(Effect.mapError((error) => new HeadlessSessionClientFailure({
        operation,
        message: String(error),
      })))

    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const rpc = yield* RpcTest.makeClient(headlessRpcs)
      const client: HeadlessSessionClient = {
        createSession: (rpcRequest) => rpc.CreateSession({
          cwd: rpcRequest.cwd,
          draftOwnerId: Option.none(),
          sessionId: Option.some(rpcRequest.sessionId),
          options: Option.some(rpcRequest.options),
          initial: Option.some(rpcRequest.initial.type === "goal"
            ? { _tag: "goal" as const, objective: rpcRequest.initial.content }
            : {
                _tag: "message" as const,
                messageId: Option.none(),
                content: rpcRequest.initial.content,
                visibleMessage: Option.none(),
                taskMode: false,
                imageAttachments: [],
                mentions: [],
              }),
        }).pipe(mapFailure("CreateSession")),
        getSession: (sessionId) => rpc.GetSession({ sessionId }).pipe(mapFailure("GetSession")),
        resyncDisplayView: (sessionId, viewId) => rpc.ResyncDisplayView({
          sessionId,
          viewId,
        }).pipe(mapFailure("ResyncDisplayView")),
        streamDisplayView: (sessionId, viewId, shape) => rpc.StreamDisplayView({
          sessionId,
          viewId,
          shape,
          materialize: true,
        }).pipe(Stream.mapError((error) => new HeadlessSessionClientFailure({
          operation: "StreamDisplayView",
          message: String(error),
        }))),
      }
      return yield* runHeadlessSession(request, { onSnapshot: () => Effect.void }).pipe(
        Effect.provideService(HeadlessSessionClient, client),
      )
    })).pipe(Effect.provide(Layer.mergeAll(handlers, demandLayer))))

    expect(result.status).toBe("completed")
    expect(persistedRequest).toEqual(request)
    expect(trace).toEqual([
      "CreateSession",
      "StreamDisplayView",
      "GetSession",
    ])
  })
})
