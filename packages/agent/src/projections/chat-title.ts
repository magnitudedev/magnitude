import { Projection, Signal } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'

export interface ChatTitleGeneratedSignal {
  readonly title: string
}

export const ChatTitleStateSchema = Schema.Struct({
  chatName: Schema.NullOr(Schema.String),
})

export type ChatTitleState = typeof ChatTitleStateSchema.Type

export const ChatTitleProjection = Projection.define<AppEvent>()({
  name: 'ChatTitle',
  state: ChatTitleStateSchema,

  initial: {
    chatName: null,
  },

  signals: {
    chatTitleGenerated: Signal.create<ChatTitleGeneratedSignal>('ChatTitle/chatTitleGenerated'),
  },

  eventHandlers: {
    chat_title_generated: ({ event, state, emit }) => {
      emit.chatTitleGenerated({ title: event.title })
      return { ...state, chatName: event.title }
    },
  },
})
