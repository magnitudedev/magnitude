/**
 * Autopilot Worker
 *
 * When the agent completes a turn and autopilot is enabled, generates a continuation
 * message using a perspective-flipped LLM call. The generated message is
 * published as an `autopilot_message_generated` event for the CLI to
 * preview and send after a countdown.
 */

import { Effect } from 'effect'
import { Image as BamlImage } from '@boundaryml/baml'
import { Worker } from '@magnitudedev/event-core'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { logger } from '@magnitudedev/logger'
import { primary } from '@magnitudedev/providers'
import type { AppEvent } from '../events'
import { ContentPart } from '../content'
import { MemoryProjection, getView } from '../projections/memory'
import type { LLMMessage } from '../projections/memory'
import { SessionContextProjection } from '../projections/session-context'

import { buildAutopilotSystemPrompt } from '../skills/autopilot-prompts'
/** Max recent messages to send to the autopilot LLM */
const CONTEXT_MESSAGE_LIMIT = 10
function toLLMContent(parts: ContentPart[]): (BamlImage | string)[] {
  return parts.map(part => {
    switch (part.type) {
      case 'text': return part.text
      case 'image': return BamlImage.fromBase64(part.mediaType, part.base64)
    }
  })
}

function toBamlMessages(messages: LLMMessage[]): ChatMessage[] {
  return messages.map(m => ({
    role: m.role,
    content: toLLMContent(m.content)
  }))
}

// =============================================================================
// Worker
// =============================================================================

export const Autopilot = Worker.define<AppEvent>()({
  name: 'Autopilot',

  signalHandlers: () => []
})
