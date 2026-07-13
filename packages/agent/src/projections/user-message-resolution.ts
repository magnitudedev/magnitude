import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent, Attachment, MentionResolution } from '../events'
import { UserPartSchema, type UserPart } from '@magnitudedev/ai'
import { Schema } from 'effect'
import { AgentMessageAttachmentSchema } from '../attachments'

export interface UserMessageResolvedSignal {
  readonly messageId: string
  readonly forkId: string | null
  readonly timestamp: number
  readonly content: readonly UserPart[]
  readonly attachments: readonly Attachment[]
  readonly mode: 'text' | 'audio'
  readonly synthetic: boolean
  readonly taskMode: boolean
  readonly mentionResolutions: readonly MentionResolution[]
}

const StoredUserMessageRawSchema = Schema.Struct({
  messageId: Schema.String,
  forkId: Schema.NullOr(Schema.String),
  timestamp: Schema.Number,
  content: Schema.Array(UserPartSchema),
  attachments: Schema.Array(AgentMessageAttachmentSchema),
  mode: Schema.Literal('text', 'audio'),
  synthetic: Schema.Boolean,
  taskMode: Schema.Boolean,
})
export type StoredUserMessageRaw = typeof StoredUserMessageRawSchema.Type

export const UserMessageResolutionStateSchema = Schema.Struct({
  rawByMessageId: Schema.ReadonlyMap({ key: Schema.String, value: StoredUserMessageRawSchema }),
})
export type UserMessageResolutionState = typeof UserMessageResolutionStateSchema.Type

export const UserMessageResolutionProjection = Projection.define<AppEvent>()({
  name: 'UserMessageResolution',
  state: UserMessageResolutionStateSchema,

  initial: {
    rawByMessageId: new Map<string, StoredUserMessageRaw>(),
  },

  signals: {
    userMessageResolved: Signal.create<UserMessageResolvedSignal>('UserMessageResolution/userMessageResolved'),
  },

  eventHandlers: {
    user_message: ({ event, state }) => ({
      ...state,
      rawByMessageId: new Map(state.rawByMessageId).set(event.messageId, {
        messageId: event.messageId,
        forkId: event.forkId,
        timestamp: event.timestamp,
        content: event.content,
        attachments: event.attachments,
        mode: event.mode,
        synthetic: event.synthetic,
        taskMode: event.taskMode,
      }),
    }),

    user_message_ready: ({ event, state, emit }) => {
      const raw = state.rawByMessageId.get(event.messageId)
      if (!raw) {
        return state
      }

      emit.userMessageResolved({
        messageId: raw.messageId,
        forkId: raw.forkId,
        timestamp: raw.timestamp,
        content: raw.content,
        attachments: raw.attachments,
        mode: raw.mode,
        synthetic: raw.synthetic,
        taskMode: raw.taskMode,
        mentionResolutions: event.mentionResolutions ?? event.resolvedMentions ?? [],
      })

      const next = new Map(state.rawByMessageId)
      next.delete(event.messageId)
      return { ...state, rawByMessageId: next }
    },
  },
})
