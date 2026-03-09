/**
 * ChatTitleProjection
 *
 * Tracks the auto-generated chat title.
 * Emits a signal when a title is generated so the CLI can update.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'

// =============================================================================
// Types
// =============================================================================

export interface ChatTitleState {
  readonly title: string | null
}

export interface ChatTitleGeneratedSignal {
  readonly title: string
}

// =============================================================================
// Projection
// =============================================================================

export const ChatTitleProjection = Projection.define<AppEvent, ChatTitleState>()({
  name: 'ChatTitle',
  initial: { title: null },

  signals: {
    chatTitleGenerated: Signal.create<ChatTitleGeneratedSignal>('ChatTitle/chatTitleGenerated'),
  },

  eventHandlers: {
    chat_title_generated: ({ event, emit }) => {
      logger.info({ title: event.title }, '[ChatTitleProjection] Received chat_title_generated event, emitting signal')
      emit.chatTitleGenerated({ title: event.title })
      return { title: event.title }
    }
  }
})
