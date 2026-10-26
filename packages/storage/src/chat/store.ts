import type { PlatformError } from "@effect/platform/Error"
import { Context, Data, type DateTime, type Effect, Option, type ParseResult } from "effect"
import type { EventStore } from "@magnitudedev/event-engine"
import { ChatSummary, type SessionEvent, type SessionId } from "@magnitudedev/acn-protocol"
import type { SqliteCheckpointIncomplete, SqliteWalRefused } from "../sqlite/database"
import type { SqliteDriverError } from "../sqlite/driver"

/** The store's file could not be opened, read or written, or a delete could not clear the write-ahead log. */
export class ChatStoreUnavailable extends Data.TaggedError("ChatStoreUnavailable")<{
  readonly cause: SqliteDriverError | SqliteWalRefused | SqliteCheckpointIncomplete | PlatformError
}> {}

/** The schema version of the chat store's format, the file's `user_version` (see `store-live.ts`). */
export const chatStoreVersion = 1

/** A SQLite file's schema version and tables, other than SQLite's own. */
export interface SqliteFormat {
  readonly version: number
  readonly tables: ReadonlyArray<string>
}

/** The file is not a chat store of this format or an earlier one: a SQLite file of another format, or not a SQLite database. */
export class ChatStoreIncompatible extends Data.TaggedError("ChatStoreIncompatible")<{
  /** None when the file is not a SQLite database. */
  readonly format: Option.Option<SqliteFormat>
}> {
  /** The file's format, or that it is not a SQLite database. */
  override get message(): string {
    return Option.match(this.format, {
      onNone: () => "Not a chat store: not a SQLite database",
      onSome: ({ version, tables }) =>
        `Not a chat store of this format or an earlier one: version ${version}, tables ${tables.length === 0 ? "none" : tables.join(", ")}`,
    })
  }

  /** Whether a later format wrote the file, rather than something that is not a chat store of any version. */
  get newer(): boolean {
    return Option.exists(this.format, ({ version }) => version > chatStoreVersion)
  }
}

/** Stored data does not decode. */
export class ChatStoreCorrupt extends Data.TaggedError("ChatStoreCorrupt")<{
  readonly cause: ParseResult.ParseError
}> {}

/** A value to store does not encode: it breaks a refinement of its schema that its type does not express. */
export class ChatStoreRejected extends Data.TaggedError("ChatStoreRejected")<{
  readonly cause: ParseResult.ParseError
}> {}

const chatStoreErrors = [ChatStoreUnavailable, ChatStoreIncompatible, ChatStoreCorrupt, ChatStoreRejected] as const

export type ChatStoreError = InstanceType<(typeof chatStoreErrors)[number]>

/** Whether a value is the chat store's error. */
export const isChatStoreError = (value: unknown): value is ChatStoreError =>
  chatStoreErrors.some((error) => value instanceof error)

/** A session's record: its identity, title, creation, and the two fields it caches of its log. */
export const ChatSessionRecord = ChatSummary.omit("running")
export type ChatSessionRecord = typeof ChatSessionRecord.Type

/**
 * How a rename went: `renamed` from `from`; `unchanged` because the record already had the title;
 * `title_changed` because its title was neither; `missing` because the session has no record.
 */
export type ChatRenameOutcome = "renamed" | "unchanged" | "title_changed" | "missing"

/**
 * Every chat session's log and record, in `<storage root>/chats.sqlite`. A session's record is
 * created by the commit of its first message, in that commit's transaction, and deleted with its
 * log, so a session has a record exactly while it has a log. Each error is logged, with its cause,
 * where the store raises it.
 */
export interface ChatStore {
  /**
   * A session's log. Each commit appends its events and, in the same transaction, creates the
   * session's record if it has none (`createdRecord`), then updates the record's `messageCount` and
   * `time.updated`, which cache the log (`committedRecord`).
   */
  readonly session: (id: SessionId) => EventStore<SessionEvent, ChatStoreError>
  /** The session's record, if it exists. */
  readonly record: (id: SessionId) => Effect.Effect<Option.Option<ChatSessionRecord>, ChatStoreError>
  /**
   * Renames the session from the title `from` to `title` at `time`, as a compare-and-set against its
   * current title, in one transaction that also touches `time.updated` (`renamedRecord`).
   */
  readonly rename: (
    id: SessionId,
    change: { readonly from: string; readonly title: string; readonly time: DateTime.Utc }
  ) => Effect.Effect<ChatRenameOutcome, ChatStoreError>
  /** Records by latest activity, most recent first. A row that does not decode fails the page as corrupt. */
  readonly page: (p: { readonly offset: number; readonly limit: number }) => Effect.Effect<ReadonlyArray<ChatSessionRecord>, ChatStoreError>
  /**
   * Deletes a session's record and events in one transaction, then truncates the write-ahead log so
   * nothing of it remains. A reader that blocks the truncation fails it as unavailable; deleting
   * again finishes it.
   */
  readonly delete: (id: SessionId) => Effect.Effect<void, ChatStoreError>
}
export const ChatStore = Context.GenericTag<ChatStore>("@magnitudedev/storage/ChatStore")
