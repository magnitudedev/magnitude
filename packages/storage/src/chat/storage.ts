import { FileSystem, Path } from "@effect/platform"
import { Context, Effect, Exit, Layer, Option, Schema, Scope } from "effect"
import {
  ChatConversation,
  ChatConversationId,
  ChatFailed,
  ChatGeneration,
  ChatGenerationId,
  ChatCommandId,
  ChatHistory,
  ChatMessage,
  ChatMessageId,
  ChatSettings,
} from "@magnitudedev/acn-protocol"
import { GlobalStorage } from "../services/global-storage"
import { SqliteDriver, type SqliteConnection } from "../sqlite/driver"

export const StoredChatGeneration = ChatGeneration.pipe(
  Schema.extend(Schema.Struct({ requestJson: Schema.String }))
)
export type StoredChatGeneration = typeof StoredChatGeneration.Type
export interface ChatStorage {
  readonly list: (
    offset: number,
    limit: number
  ) => Effect.Effect<typeof ChatHistory.Type, ChatFailed>
  readonly get: (
    id: ChatConversationId
  ) => Effect.Effect<ChatConversation, ChatFailed>
  readonly create: (
    id: ChatConversationId,
    settings: ChatSettings
  ) => Effect.Effect<ChatConversation, ChatFailed>
  readonly update: (
    id: ChatConversationId,
    title: Option.Option<string>,
    settings: Option.Option<ChatSettings>
  ) => Effect.Effect<ChatConversation, ChatFailed>
  readonly remove: (id: ChatConversationId) => Effect.Effect<void, ChatFailed>
  readonly findCommand: (
    id: ChatCommandId,
    conversationId: ChatConversationId
  ) => Effect.Effect<Option.Option<StoredChatGeneration>, ChatFailed>
  readonly admit: (
    conversationId: ChatConversationId,
    generation: StoredChatGeneration,
    user: Option.Option<{
      readonly id: ChatMessageId
      readonly content: string
    }>,
    retryMessageId: Option.Option<ChatMessageId>
  ) => Effect.Effect<void, ChatFailed>
  readonly save: (
    generation: StoredChatGeneration
  ) => Effect.Effect<void, ChatFailed>
}
export const ChatStorage = Context.GenericTag<ChatStorage>(
  "@magnitudedev/storage/ChatStorage"
)
const encodeSettings = Schema.encode(Schema.parseJson(ChatSettings))
const decodeSettings = Schema.decodeUnknown(Schema.parseJson(ChatSettings))
const encodeGeneration = Schema.encode(Schema.parseJson(StoredChatGeneration))
const decodeGeneration = Schema.decodeUnknown(
  Schema.parseJson(StoredChatGeneration)
)
const ConversationRow = Schema.Struct({
  id: ChatConversationId,
  title: Schema.String,
  created_at: Schema.Number,
  updated_at: Schema.Number,
  settings: Schema.String,
})
const MessageRow = Schema.Struct({
  id: ChatMessageId,
  role: Schema.Literal("user", "assistant"),
  content: Schema.String,
  data: Schema.NullOr(Schema.String),
})
const SummaryRow = ConversationRow.pipe(
  Schema.extend(
    Schema.Struct({ message_count: Schema.Number, active: Schema.Number })
  )
)
const storageError = () =>
  new ChatFailed({
    code: "Storage",
    message:
      "Chat history could not be read or saved. Check access to the Magnitude data folder.",
  })
const protect = <A, E>(
  effect: Effect.Effect<A, E>
): Effect.Effect<A, ChatFailed> =>
  effect.pipe(
    Effect.mapError((error) =>
      error instanceof ChatFailed ? error : storageError()
    )
  )

export const ChatStorageLive = Layer.scoped(
  ChatStorage,
  Effect.gen(function* () {
    const storage = yield* GlobalStorage
    const driver = yield* SqliteDriver
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const scope = yield* Scope.Scope
    const lock = yield* Effect.makeSemaphore(1)
    const transaction = <A, E>(
      db: SqliteConnection,
      body: Effect.Effect<A, E>
    ) =>
      Effect.uninterruptible(
        Effect.gen(function* () {
          yield* db.execute("BEGIN IMMEDIATE")
          const result = yield* Effect.exit(body)
          if (Exit.isFailure(result)) {
            yield* db.execute("ROLLBACK").pipe(Effect.ignore)
            return yield* result
          }
          yield* db
            .execute("COMMIT")
            .pipe(
              Effect.onError(() => db.execute("ROLLBACK").pipe(Effect.ignore))
            )
          return result.value
        })
      )
    const open = yield* Effect.cached(
      protect(
        Effect.gen(function* () {
          yield* fs.makeDirectory(storage.root, { recursive: true })
          const db = yield* driver
            .open(path.join(storage.root, "chats.sqlite"), { create: true })
            .pipe(Effect.provideService(Scope.Scope, scope))
          for (const sql of [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=FULL",
            "PRAGMA foreign_keys=ON",
            "PRAGMA busy_timeout=5000",
          ])
            yield* db.execute(sql)
          const version = yield* db.query("PRAGMA user_version")
          const v =
            (yield* Schema.decodeUnknown(
              Schema.Array(Schema.Struct({ user_version: Schema.Number }))
            )(version))[0]?.user_version ?? 0
          if (v > 1)
            return yield* new ChatFailed({
              code: "Storage",
              message:
                "Chat history was created by a newer Magnitude version. Update Magnitude to open it.",
            })
          if (v === 0)
            yield* transaction(
              db,
              Effect.gen(function* () {
                for (const sql of [
                  "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, settings TEXT NOT NULL)",
                  "CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, generation_id TEXT, UNIQUE(conversation_id, ordinal))",
                  "CREATE TABLE generations (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE, command_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL, data TEXT NOT NULL)",
                  "CREATE INDEX conversations_recent ON conversations(updated_at DESC, id)",
                  "CREATE INDEX generations_conversation ON generations(conversation_id, status)",
                  "PRAGMA user_version=1",
                ])
                  yield* db.execute(sql)
              })
            )
          // A new service process cannot resume a previous process's native request.
          const abandoned = yield* db.query(
            "SELECT data FROM generations WHERE status='running'"
          )
          yield* transaction(
            db,
            Effect.forEach(
              abandoned,
              (row) =>
                Effect.gen(function* () {
                  const record = yield* Schema.decodeUnknown(
                    Schema.Struct({ data: Schema.String })
                  )(row)
                  const old = yield* decodeGeneration(record.data)
                  const value: StoredChatGeneration = {
                    ...old,
                    status: "interrupted",
                    activity: Option.none(),
                    updatedAt: Date.now(),
                  }
                  yield* db.execute(
                    "UPDATE generations SET status=?, data=? WHERE id=?",
                    [value.status, yield* encodeGeneration(value), value.id]
                  )
                }),
              { discard: true }
            )
          )
          return db
        })
      )
    )
    const read = (db: SqliteConnection, id: ChatConversationId) =>
      Effect.gen(function* () {
        const rows = yield* db.query("SELECT * FROM conversations WHERE id=?", [
          id,
        ])
        if (!rows[0])
          return yield* new ChatFailed({
            code: "NotFound",
            message: "This conversation no longer exists.",
          })
        const record = yield* Schema.decodeUnknown(ConversationRow)(rows[0])
        const messages = yield* db.query(
          "SELECT m.id, m.role, m.content, g.data FROM messages m LEFT JOIN generations g ON g.id=m.generation_id WHERE m.conversation_id=? ORDER BY m.ordinal",
          [id]
        )
        const decoded = yield* Schema.decodeUnknown(Schema.Array(MessageRow))(
          messages
        )
        const projected = yield* Effect.forEach(decoded, (row) =>
          Effect.gen(function* () {
            const generation =
              row.data === null
                ? Option.none<ChatGeneration>()
                : Option.some(yield* decodeGeneration(row.data))
            return {
              id: row.id,
              role: row.role,
              content:
                row.role === "user"
                  ? row.content
                  : Option.match(generation, {
                      onNone: () => "",
                      onSome: (g) => g.content,
                    }),
              generation,
            } satisfies ChatMessage
          })
        )
        return {
          id: record.id,
          title: record.title,
          createdAt: record.created_at,
          updatedAt: record.updated_at,
          settings: yield* decodeSettings(record.settings),
          messages: projected,
        } satisfies ChatConversation
      })
    const run = <A, E>(f: (db: SqliteConnection) => Effect.Effect<A, E>) =>
      protect(
        lock.withPermits(1)(
          open.pipe(Effect.flatMap((db) => Effect.suspend(() => f(db))))
        )
      )
    return ChatStorage.of({
      get: (id) => run((db) => read(db, id)),
      list: (offset, limit) =>
        run((db) =>
          Effect.gen(function* () {
            const rows = yield* db.query(
              "SELECT c.*, (SELECT COUNT(*) FROM messages WHERE conversation_id=c.id) AS message_count, EXISTS(SELECT 1 FROM generations WHERE conversation_id=c.id AND status='running') AS active FROM conversations c ORDER BY updated_at DESC,id LIMIT ? OFFSET ?",
              [limit + 1, offset]
            )
            const decoded = yield* Schema.decodeUnknown(
              Schema.Array(SummaryRow)
            )(rows.slice(0, limit))
            const conversations = yield* Effect.forEach(decoded, (row) =>
              Effect.gen(function* () {
                return {
                  id: row.id,
                  title: row.title,
                  createdAt: row.created_at,
                  updatedAt: row.updated_at,
                  settings: yield* decodeSettings(row.settings),
                  messageCount: row.message_count,
                  active: Boolean(row.active),
                }
              })
            )
            return { conversations, hasMore: rows.length > limit }
          })
        ),
      create: (id, settings) =>
        run((db) =>
          Effect.gen(function* () {
            const now = Date.now()
            yield* db.execute(
              "INSERT OR IGNORE INTO conversations VALUES (?, 'New chat', ?, ?, ?)",
              [id, now, now, yield* encodeSettings(settings)]
            )
            return yield* read(db, id)
          })
        ),
      update: (id, title, settings) =>
        run((db) =>
          Effect.gen(function* () {
            const old = yield* read(db, id)
            yield* db.execute(
              "UPDATE conversations SET title=?, settings=? WHERE id=?",
              [
                Option.getOrElse(title, () => old.title),
                yield* encodeSettings(
                  Option.getOrElse(settings, () => old.settings)
                ),
                id,
              ]
            )
            return yield* read(db, id)
          })
        ),
      remove: (id) =>
        run((db) => db.execute("DELETE FROM conversations WHERE id=?", [id])),
      findCommand: (id, conversationId) =>
        run((db) =>
          Effect.gen(function* () {
            const rows = yield* db.query(
              "SELECT data, conversation_id FROM generations WHERE command_id=?",
              [id]
            )
            if (!rows[0]) return Option.none()
            const row = yield* Schema.decodeUnknown(
              Schema.Struct({
                data: Schema.String,
                conversation_id: ChatConversationId,
              })
            )(rows[0])
            if (row.conversation_id !== conversationId)
              return yield* new ChatFailed({
                code: "InvalidRequest",
                message: "This request ID belongs to another conversation.",
              })
            return Option.some(yield* decodeGeneration(row.data))
          })
        ),
      admit: (id, generation, user, retryMessageId) =>
        run((db) =>
          transaction(
            db,
            Effect.gen(function* () {
              const conversation = yield* read(db, id)
              if (
                conversation.messages.some(
                  (m) =>
                    Option.isSome(m.generation) &&
                    m.generation.value.status === "running"
                )
              )
                return yield* new ChatFailed({
                  code: "Busy",
                  message:
                    "A response is already running in this conversation.",
                })
              if (Option.isSome(user)) {
                const ordinal = conversation.messages.length
                yield* db.execute(
                  "INSERT INTO messages VALUES (?, ?, ?, 'user', ?, NULL)",
                  [user.value.id, id, ordinal, user.value.content]
                )
                yield* db.execute(
                  "INSERT INTO messages VALUES (?, ?, ?, 'assistant', '', ?)",
                  [generation.messageId, id, ordinal + 1, generation.id]
                )
              } else {
                const last = conversation.messages.at(-1)
                if (
                  Option.isNone(retryMessageId) ||
                  !last ||
                  last.id !== retryMessageId.value ||
                  last.role !== "assistant"
                )
                  return yield* new ChatFailed({
                    code: "InvalidRequest",
                    message: "Only the last response can be retried.",
                  })
                yield* db.execute(
                  "UPDATE messages SET generation_id=? WHERE id=?",
                  [generation.id, generation.messageId]
                )
              }
              yield* db.execute(
                "INSERT INTO generations VALUES (?, ?, ?, ?, ?, ?)",
                [
                  generation.id,
                  id,
                  generation.messageId,
                  generation.commandId,
                  generation.status,
                  yield* encodeGeneration(generation),
                ]
              )
              const title =
                conversation.messages.length === 0 && Option.isSome(user)
                  ? user.value.content.trim().replace(/\s+/g, " ").slice(0, 80)
                  : conversation.title
              yield* db.execute(
                "UPDATE conversations SET title=?, updated_at=?, settings=? WHERE id=?",
                [
                  title,
                  generation.createdAt,
                  yield* encodeSettings(generation.settings),
                  id,
                ]
              )
            })
          )
        ),
      save: (generation) =>
        run((db) =>
          transaction(
            db,
            Effect.gen(function* () {
              yield* db.execute(
                "UPDATE generations SET status=?, data=? WHERE id=?",
                [
                  generation.status,
                  yield* encodeGeneration(generation),
                  generation.id,
                ]
              )
              yield* db.execute(
                "UPDATE conversations SET updated_at=? WHERE id=(SELECT conversation_id FROM generations WHERE id=?)",
                [generation.updatedAt, generation.id]
              )
            })
          )
        ),
    })
  })
)
