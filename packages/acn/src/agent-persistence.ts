import { Effect, Schema } from "effect"
import {
  DEFAULT_CHAT_NAME,
  PersistenceError,
  type AppEvent,
  type ChatPersistenceService,
  type SessionMetadata as AgentSessionMetadata,
} from "@magnitudedev/agent"
import type { EventCursor, Timestamped } from "@magnitudedev/event-core"
import type { MagnitudeStorageShape, StoredSessionMeta } from "@magnitudedev/storage"
import { defaultStoredMeta } from "./session-store"

export class AcnChatPersistence implements ChatPersistenceService {
  constructor(
    private readonly storage: MagnitudeStorageShape,
    private readonly workingDirectory: string,
    private readonly sessionId: string,
    private readonly version: string,
    private readonly initialVisibility: StoredSessionMeta["visibility"] = "visible",
  ) {}

  loadEvents(): Effect.Effect<Timestamped<AppEvent>[], PersistenceError> {
    return this.storage.sessions.readEvents<Timestamped<AppEvent>>(this.sessionId).pipe(
      Effect.mapError((e) => new PersistenceError({ reason: "LoadFailed", message: String(e) })),
    )
  }

  loadEventsAfterCursor(cursor: EventCursor): Effect.Effect<Timestamped<AppEvent>[] | null, PersistenceError> {
    return this.storage.sessions.readEventsAfterCursor<Timestamped<AppEvent>>(this.sessionId, cursor).pipe(
      Effect.mapError((e) => new PersistenceError({ reason: "LoadFailed", message: String(e) })),
    )
  }

  private buildMetadata(now: string): StoredSessionMeta {
    return defaultStoredMeta(this.sessionId, this.workingDirectory, this.version, now, this.initialVisibility)
  }

  private updateSummaryFromEvents(metadata: StoredSessionMeta, events: readonly Timestamped<AppEvent>[]): StoredSessionMeta {
    let next = metadata
    for (const event of events) {
      if (event.type !== "user_message") continue
      const text = event.text.trim()
      if (!text) continue
      next = {
        ...next,
        messageCount: next.messageCount + 1,
        firstUserMessage: next.firstUserMessage ?? text,
        lastMessage: text,
      }
    }
    return next
  }

  persistNewEvents(events: Timestamped<AppEvent>[]): Effect.Effect<EventCursor | null, PersistenceError> {
    return Effect.gen(this, function* () {
      const cursor = yield* this.storage.sessions.appendEventsWithCursor(this.sessionId, events).pipe(
        Effect.mapError((e) => new PersistenceError({ reason: "SaveFailed", message: String(e) })),
      )
      if (!cursor) return null

      const metadata = yield* this.storage.sessions.readMeta(this.sessionId).pipe(
        Effect.mapError((e) => new PersistenceError({ reason: "LoadFailed", message: String(e) })),
        Effect.catchAll(() => Effect.succeed(null)),
      )
      if (metadata) {
        yield* this.storage.sessions.updateMeta(this.sessionId, (current) => ({
          ...this.updateSummaryFromEvents(current ?? this.buildMetadata(new Date().toISOString()), events),
          updated: new Date().toISOString(),
          lastActiveVersion: this.version,
        })).pipe(
          Effect.mapError((e) => new PersistenceError({ reason: "SaveFailed", message: String(e) })),
          Effect.catchAll(() => Effect.void),
        )
        return cursor
      }
      const now = new Date().toISOString()
      yield* this.storage.sessions.writeMeta(
        this.sessionId,
        this.updateSummaryFromEvents(this.buildMetadata(now), events),
      ).pipe(
        Effect.mapError((e) => new PersistenceError({ reason: "SaveFailed", message: String(e) })),
        Effect.catchAll(() => Effect.void),
      )
      return cursor
    })
  }

  loadProjectionSnapshot(): Effect.Effect<unknown | null, PersistenceError> {
    return this.storage.sessions.readProjectionSnapshot(this.sessionId).pipe(
      Effect.mapError((e) => new PersistenceError({ reason: "LoadFailed", message: String(e) })),
    )
  }

  saveProjectionSnapshot<A>(snapshot: A): Effect.Effect<void, PersistenceError> {
    return this.storage.sessions.writeProjectionSnapshot(this.sessionId, snapshot).pipe(
      Effect.mapError((e) => new PersistenceError({ reason: "SaveFailed", message: String(e) })),
    )
  }

  getSessionMetadata(): Effect.Effect<AgentSessionMetadata, PersistenceError> {
    return Effect.gen(this, function* () {
      const meta = yield* this.storage.sessions.readMeta(this.sessionId).pipe(
        Effect.mapError((e) => new PersistenceError({ reason: "LoadFailed", message: String(e) })),
      )
      const stored = meta ?? this.buildMetadata(new Date().toISOString())
      if (!meta) {
        yield* this.storage.sessions.writeMeta(this.sessionId, stored).pipe(
          Effect.mapError((e) => new PersistenceError({ reason: "SaveFailed", message: String(e) })),
        )
      }
      return {
        sessionId: stored.sessionId,
        chatName: stored.chatName ?? DEFAULT_CHAT_NAME,
        workingDirectory: stored.workingDirectory,
        gitBranch: stored.gitBranch,
        created: stored.created,
        updated: stored.updated,
        initialVersion: stored.initialVersion,
        lastActiveVersion: stored.lastActiveVersion,
      }
    })
  }

  saveSessionMetadata(
    update: Partial<Omit<AgentSessionMetadata, "sessionId" | "created" | "initialVersion" | "lastActiveVersion">>,
  ): Effect.Effect<void, PersistenceError> {
    return this.storage.sessions.updateMeta(this.sessionId, (current) => {
      const now = new Date().toISOString()
      const base = current ?? this.buildMetadata(now)
      return {
        ...base,
        chatName: update.chatName ?? base.chatName,
        workingDirectory: update.workingDirectory ?? base.workingDirectory,
        gitBranch: update.gitBranch ?? base.gitBranch,
        updated: update.updated ?? now,
        lastActiveVersion: this.version,
      }
    }).pipe(
      Effect.mapError((e) => new PersistenceError({ reason: "SaveFailed", message: String(e) })),
    )
  }
}
