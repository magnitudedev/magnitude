import { ParseResult, Schema } from "effect"
import { ChatSessionRecord } from "./store"

// The only mapping between a `sessions` row and a session's record: each of the row's five columns
// owns one field. `session_id` owns `id`, `title` and `message_count` own `title` and
// `messageCount`, and `created_at` and `updated_at` own `time.created` and `time.updated` (the list
// order), in epoch ms.

const SessionRow = Schema.Struct({
  session_id: Schema.String,
  title: Schema.String,
  message_count: Schema.Number,
  created_at: Schema.Number,
  updated_at: Schema.Number,
})

/** A row's columns and the record they hold. A row with any other column does not decode. */
export const StoredSession = Schema.transformOrFail(SessionRow, Schema.typeSchema(ChatSessionRecord), {
  strict: true,
  decode: ({ session_id, title, message_count, created_at, updated_at }) =>
    ParseResult.decodeUnknown(ChatSessionRecord)({
      id: session_id,
      title,
      messageCount: message_count,
      time: { created: created_at, updated: updated_at },
    }),
  encode: (record) => ParseResult.map(ParseResult.encode(ChatSessionRecord)(record), ({ id, title, messageCount, time }) => ({
    session_id: id,
    title,
    message_count: messageCount,
    created_at: time.created,
    updated_at: time.updated,
  })),
})
