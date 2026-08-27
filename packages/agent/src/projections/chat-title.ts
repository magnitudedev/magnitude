import { Projection, Signal } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'
import { deriveChatTitle } from '../util/chat-title'

export interface ChatTitleResolvedSignal {
  readonly title: string
}

export const ChatTitleStateSchema = Schema.Union(
  Schema.TaggedStruct('Pending', {}),
  Schema.TaggedStruct('Resolved', {
    chatName: Schema.NullOr(Schema.String),
  }),
)

export type ChatTitleState = typeof ChatTitleStateSchema.Type

export const ChatTitleProjection = Projection.define<AppEvent>()({
  name: 'ChatTitle',
  state: ChatTitleStateSchema,

  initial: { _tag: 'Pending' },

  signals: {
    chatTitleResolved: Signal.create<ChatTitleResolvedSignal>('ChatTitle/chatTitleResolved'),
  },

  eventHandlers: {
    user_message: ({ event, state, emit }) => {
      if (state._tag === 'Resolved' || event.forkId !== null || event.synthetic) {
        return state
      }

      const chatName = deriveChatTitle(event.text)
      if (chatName !== null) {
        emit.chatTitleResolved({ title: chatName })
      }
      return { _tag: 'Resolved' as const, chatName }
    },
  },
})
