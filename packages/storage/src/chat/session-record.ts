import { Array, DateTime, Option } from "effect"
import type { MessageSent, SessionEvent, SessionId } from "@magnitudedev/acn-protocol"
import type { ChatRenameOutcome, ChatSessionRecord } from "./store"

// The rules of a session's record, one owner for each: its creation and default title, the two
// fields it caches of its log, and rename's compare-and-set. The store applies them to its rows, and
// a holder of a record in memory applies them to that.

const titleLength = 80

/** A message's text as a title: whitespace collapsed, at most 80 code points. */
const titleOf = (text: string) => Array.take([...text.replace(/\s+/g, " ").trim()], titleLength).join("")

const isMessage = (event: SessionEvent): event is MessageSent => event._tag === "message_sent"

/** The messages the transcript shows: each `message_sent`, and each `response_started` without `replaces`, since a retry takes the place of the response it replaces. */
const messages = (events: ReadonlyArray<SessionEvent>) =>
  events.filter((event) => isMessage(event) || (event._tag === "response_started" && Option.isNone(event.replaces))).length

/**
 * What a commit's events change of the record's caches: the messages they add, and the latest
 * observation, their latest event other than output. Every counted event is observed, so output
 * alone changes neither.
 */
export const cacheChange = (events: ReadonlyArray<SessionEvent>): Option.Option<{ readonly messages: number; readonly updated: DateTime.Utc }> =>
  Option.map(Array.findLast(events, (event) => event._tag !== "response_output"), (latest) => ({
    messages: messages(events),
    updated: latest.time,
  }))

/**
 * The record a commit creates for a session that has none, before its caches apply: titled by the
 * commit's first message, created at its first event, counting nothing yet. A session's log begins
 * with its first message, so the commit holding it is the session's first, and creates its record.
 */
export const createdRecord = (id: SessionId, events: ReadonlyArray<SessionEvent>): Option.Option<ChatSessionRecord> =>
  Option.map(Option.all([Array.head(events), Array.findFirst(events, isMessage)]), ([first, message]) => ({
    id,
    title: titleOf(message.text),
    time: { created: first.time, updated: first.time },
    messageCount: 0,
  }))

/** The record a commit leaves: the session's own, or the one the commit creates, with the commit's caches applied. */
export const committedRecord = (
  id: SessionId,
  current: Option.Option<ChatSessionRecord>,
  events: ReadonlyArray<SessionEvent>
): Option.Option<ChatSessionRecord> =>
  Option.map(Option.orElse(current, () => createdRecord(id, events)), (record) => Option.match(cacheChange(events), {
    onNone: () => record,
    onSome: ({ messages, updated }) => ({
      ...record,
      messageCount: record.messageCount + messages,
      time: { ...record.time, updated: DateTime.max(record.time.updated, updated) },
    }),
  }))

/** How a rename from `from` to `title` goes against the record's current title, if it has a record. */
export const renameOutcome = (current: Option.Option<string>, { from, title }: { readonly from: string; readonly title: string }): ChatRenameOutcome =>
  Option.match(current, {
    onNone: () => "missing",
    onSome: (existing) => existing === title ? "unchanged" : existing === from ? "renamed" : "title_changed",
  })

/**
 * The record a rename at `time` leaves: its title, and its latest observation touched. The latest
 * observation never moves back, since the next event may precede the rename's wall clock.
 */
export const renamedRecord = (record: ChatSessionRecord, { title, time }: { readonly title: string; readonly time: DateTime.Utc }): ChatSessionRecord => ({
  ...record,
  title,
  time: { ...record.time, updated: DateTime.max(record.time.updated, time) },
})
