import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent, Attachment, ResolvedMention } from '../events'
import type { ContentPart } from '../content'

export interface StoredUserMessageRaw {
  readonly messageId: string
  readonly forkId: string | null
  readonly timestamp: number
  readonly content: readonly ContentPart[]
  readonly attachments: readonly Attachment[]
  readonly mode: 'text' | 'audio'
  readonly synthetic: boolean
  readonly taskMode: boolean
}

export interface UserMessageResolvedSignal {
  readonly messageId: string
  readonly forkId: string | null
  readonly timestamp: number
  readonly content: readonly ContentPart[]
  readonly attachments: readonly Attachment[]
  readonly mode: 'text' | 'audio'
  readonly synthetic: boolean
  readonly taskMode: boolean
  readonly resolvedMentions: readonly ResolvedMention[]
}

export interface UserMessageResolutionState {
  readonly rawByMessageId: ReadonlyMap<string, StoredUserMessageRaw>
}

export const UserMessageResolutionProjection = Projection.define<AppEvent, UserMessageResolutionState>()({
  name: 'UserMessageResolution',

  initial: {
    rawByMessageId: new Map(),
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
        resolvedMentions: event.resolvedMentions,
      })

      const next = new Map(state.rawByMessageId)
      next.delete(event.messageId)
      return { ...state, rawByMessageId: next }
    },
  },
})
