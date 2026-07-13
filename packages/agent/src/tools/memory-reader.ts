/**
 * Conversation State Reader Service
 *
 * Service tag for tools to access the ConversationProjection state
 * (clean user↔orchestrator conversation for reviewer context injection).
 */

import { Effect, Context } from 'effect'
import type { ConversationState } from '../projections/conversation'

export interface ConversationStateReader {
  readonly getState: () => Effect.Effect<ConversationState>
}

export class ConversationStateReaderTag extends Context.Tag('ConversationStateReader')<
  ConversationStateReaderTag,
  ConversationStateReader
>() {}
