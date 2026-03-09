/**
 * Session Replay Tool
 * 
 * Replays events from a Magnitude session's events.jsonl through the actual
 * agent projection system to extract the exact LLM conversation messages.
 * 
 * Uses MemoryProjection + its dependencies (ForkProjection, TaskGraphProjection).
 * to reconstruct the conversation as the model saw it.
 */

import { Agent } from '@magnitudedev/event-core'
import {
  MemoryProjection,
  getView,
  ForkProjection,
  TaskGraphProjection,
  type LLMMessage,
  type ForkMemoryState
} from '@magnitudedev/agent'
import type { AppEvent } from '@magnitudedev/agent'
import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { join } from 'path'

// ---------------------------------------------------------------------------
// Minimal Replay Agent — only projections needed for MemoryProjection
// ---------------------------------------------------------------------------

const ReplayAgent = Agent.define<AppEvent>()({
  name: 'SessionReplay',
  projections: [
    ForkProjection,
    TaskGraphProjection,
    MemoryProjection
  ],
  workers: [],
  expose: {
    state: {
      memory: MemoryProjection
    }
  }
})

// ---------------------------------------------------------------------------
// Replay API
// ---------------------------------------------------------------------------

export interface ReplayResult {
  messages: LLMMessage[]
  totalEvents: number
  timezone: string | null
}

/**
 * Replay a session's events and extract the LLM conversation messages.
 */
export async function replaySession(eventsJsonlPath: string): Promise<ReplayResult> {
  const raw = readFileSync(eventsJsonlPath, 'utf-8')
  const events: AppEvent[] = raw
    .trim()
    .split('\n')
    .map(line => JSON.parse(line))

  // Extract timezone from session_initialized event
  const initEvent = events.find(e => e.type === 'session_initialized') as any
  const timezone: string | null = initEvent?.context?.timezone ?? null

  // Create the replay agent client
  const client = await ReplayAgent.createClient()

  try {
    // Replay all events through the projection system
    for (const event of events) {
      await client.send(event)
    }

    // Read the root fork's memory state
    const memoryState: ForkMemoryState = await client.state.memory.getFork(null)

    // Transform to LLM messages using the actual getView function
    const llmMessages = getView(memoryState.messages, timezone, 'agent')

    return {
      messages: llmMessages,
      totalEvents: events.length,
      timezone
    }
  } finally {
    await client.dispose()
  }
}

/**
 * Replay a session and write messages to an output directory.
 */
export async function extractAndWrite(
  eventsJsonlPath: string,
  outputDir: string,
  options?: { maxMessages?: number }
): Promise<void> {
  const result = await replaySession(eventsJsonlPath)

  const messages = options?.maxMessages
    ? result.messages.slice(0, options.maxMessages)
    : result.messages

  mkdirSync(outputDir, { recursive: true })

  console.log(`Total events: ${result.totalEvents}`)
  console.log(`Total messages: ${messages.length}`)
  console.log(`User messages: ${messages.filter(m => m.role === 'user').length}`)
  console.log(`Assistant messages: ${messages.filter(m => m.role === 'assistant').length}`)
  console.log(`Timezone: ${result.timezone}`)
  console.log()

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    const num = String(i + 1).padStart(2, '0')
    const suffix = msg.role === 'user' ? 'user' : 'assistant'
    const filename = `msg${num}_${suffix}.txt`
    writeFileSync(join(outputDir, filename), msg.content.map(p => typeof p === 'string' ? p : '').join(''))

    const preview = msg.content.map(p => typeof p === 'string' ? p : '').join('').substring(0, 120).replace(/\n/g, '\\n')
    console.log(`  ${filename} (${msg.content.map(p => typeof p === 'string' ? p : '').join('').length} chars): ${preview}...`)
  }

  console.log(`\nWrote ${messages.length} message files to ${outputDir}`)
}
