import { mkdtemp } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Effect, Layer, Option, Ref, Stream } from "effect"
import { DirectoryPathSchema, SessionNotFound } from "@magnitudedev/acn-protocol"
import { AgentRuntime, type AgentRuntimeApi } from "./agent-runtime"
import { SessionCommands, type SessionCommandsApi } from "./session-commands"
import { SessionDrafts, type SessionDraftsApi } from "./session-drafts"
import { SessionInspector } from "./session-inspector"
import { SessionLifecycle, SessionLifecycleLive } from "./session-lifecycle"
import { makeTestStorageLayer } from "./session-test-support"
import type { SendUserMessageInput } from "./session-types"

const metadata = {
  sessionId: "session-1",
  title: "New Chat",
  cwd: DirectoryPathSchema.make("/project"),
  archived: false,
  pinnedAt: Option.none<number>(),
  createdAt: 1,
  updatedAt: 1,
  messageCount: 1,
  lastMessage: null,
}

describe("SessionLifecycle initial messages", () => {
  it("creates a session from an attachment-only initial message", async () => {
    const sent = await Effect.runPromise(Effect.gen(function* () {
      const root = yield* Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-lifecycle-")))
      const sent = yield* Ref.make<SendUserMessageInput[]>([])
      const runtime: AgentRuntimeApi = {
        withSession: () => Effect.die("unused"),
        withSessionWork: () => Effect.die("unused"),
        withSessionRequest: () => Effect.die("unused"),
        tryWithResident: () => Effect.succeed(Option.none()),
        tryWithBusyResident: () => Effect.succeed(Option.none()),
        residentSessions: Effect.succeed([]),
        dispose: () => Effect.void,
        deleteSession: (_sessionId, remove) => remove,
        registerRetirementObserver: () => Effect.succeed(Effect.void),
        changes: Stream.never,
      }
      const commands: SessionCommandsApi = {
        sendUserMessage: input => Ref.update(sent, current => [...current, input]),
        sendUserEvent: () => Effect.die("unused"),
        getRuntimeExecutionContext: () => Effect.die("unused"),
        startGoal: () => Effect.die("unused"),
        interrupt: () => Effect.die("unused"),
      }
      const drafts: SessionDraftsApi = {
        preload: () => Effect.die("unused"),
        release: () => Effect.die("unused"),
        claim: () => Effect.succeed({ key: "draft-key", sessionId: "session-1" }),
        promote: () => Effect.succeed(metadata),
        releaseClaim: () => Effect.void,
      }
      const inspector: SessionInspector = {
        get: (sessionId) => Effect.fail(new SessionNotFound({ sessionId })),
        page: () => Effect.die("unused"),
        recentDirectories: () => Effect.die("unused"),
        changes: Stream.never,
      }
      const layer = SessionLifecycleLive.pipe(Layer.provide(Layer.mergeAll(
        Layer.succeed(AgentRuntime, runtime),
        Layer.succeed(SessionCommands, commands),
        Layer.succeed(SessionDrafts, drafts),
        Layer.succeed(SessionInspector, inspector),
        makeTestStorageLayer(root),
      )))

      const result = yield* Effect.gen(function* () {
        const lifecycle = yield* SessionLifecycle
        return yield* lifecycle.createSession("/project", undefined, {
          _tag: "message",
          messageId: Option.some("message-1"),
          content: "",
          visibleMessage: Option.none(),
          taskMode: false,
          uploads: [{ type: "raw_text_file", filename: "notes.md", data: "bm90ZXM=" }],
          mentions: [],
        }, undefined, "owner-1")
      }).pipe(Effect.provide(layer))

      expect(result._tag).toBe("created")
      return yield* Ref.get(sent)
    }))

    expect(sent).toHaveLength(1)
    expect(sent[0]).toMatchObject({
      sessionId: "session-1",
      content: "",
      uploads: [{ type: "raw_text_file", filename: "notes.md" }],
    })
  })
})
