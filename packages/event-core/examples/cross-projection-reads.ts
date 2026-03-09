/**
 * Cross-Projection Reads Example
 *
 * Demonstrates:
 * 1. Projections reading from other projections via `reads` config
 * 2. Cycle detection when projections have circular dependencies
 *
 * Run with: npx tsx examples/cross-projection-reads.ts
 */

import { Projection, Agent } from '../src'

// =============================================================================
// Events
// =============================================================================

interface SetConfig {
  type: 'set_config'
  timezone: string
  locale: string
}

interface AddMessage {
  type: 'add_message'
  content: string
  timestamp: number
}

interface Increment {
  type: 'increment'
}

type AppEvent = SetConfig | AddMessage | Increment

// =============================================================================
// Example 1: Valid Cross-Projection Reads
// =============================================================================

console.log('=== Example 1: Valid Cross-Projection Reads ===\n')

// ConfigProjection - holds app configuration
interface ConfigState {
  timezone: string | null
  locale: string | null
}

const ConfigProjection = Projection.define<AppEvent, ConfigState>()({
  name: 'Config',
  initial: { timezone: null, locale: null },

  eventHandlers: {
    set_config: ({ event }) => ({
      timezone: event.timezone,
      locale: event.locale
    })
  }
})

// MessageProjection - reads timezone from ConfigProjection
interface MessageState {
  messages: Array<{ content: string; formattedTime: string }>
}

const MessageProjection = Projection.define<AppEvent, MessageState>()({
  name: 'Message',
  initial: { messages: [] },

  reads: [ConfigProjection] as const,

  eventHandlers: {
    add_message: ({ event, state, read }) => {
      const config = read(ConfigProjection)
      const formattedTime = config.timezone
        ? `${new Date(event.timestamp).toLocaleString('en-US', { timeZone: config.timezone })}`
        : new Date(event.timestamp).toISOString()

      return {
        messages: [
          ...state.messages,
          { content: event.content, formattedTime }
        ]
      }
    }
  }
})

// StatsProjection - reads from MessageProjection (chain: Stats -> Message -> Config)
interface StatsState {
  messageCount: number
  lastMessageTime: string | null
}

const StatsProjection = Projection.define<AppEvent, StatsState>()({
  name: 'Stats',
  initial: { messageCount: 0, lastMessageTime: null },

  reads: [MessageProjection] as const,

  eventHandlers: {
    add_message: ({ state, read }) => {
      const messages = read(MessageProjection)
      const lastMessage = messages.messages[messages.messages.length - 1]

      return {
        messageCount: messages.messages.length,
        lastMessageTime: lastMessage?.formattedTime ?? null
      }
    }
  }
})

// Create agent with valid dependencies
const ValidAgent = Agent.define<AppEvent>()({
  name: 'ValidAgent',
  projections: [ConfigProjection, MessageProjection, StatsProjection],
  workers: [],
  expose: {
    state: {
      config: ConfigProjection,
      messages: MessageProjection,
      stats: StatsProjection
    }
  }
})

async function runValidExample() {
  const client = await ValidAgent.createClient()

  // Small delay to ensure workers are ready
  await new Promise(r => setTimeout(r, 10))

  // Set config first
  console.log('Setting config...')
  await client.send({
    type: 'set_config',
    timezone: 'America/New_York',
    locale: 'en-US'
  })

  // Add some messages
  console.log('Adding messages...')
  await client.send({
    type: 'add_message',
    content: 'Hello world!',
    timestamp: Date.now()
  })

  await client.send({
    type: 'add_message',
    content: 'Cross-projection reads work!',
    timestamp: Date.now()
  })

  // Read final state using client.state
  const configState = await client.state.config.get()
  const messagesState = await client.state.messages.get()
  const statsState = await client.state.stats.get()

  console.log('Config state:', configState)
  console.log('Messages:', messagesState.messages)
  console.log('Stats:', statsState)

  await client.dispose()
}

await runValidExample()

// =============================================================================
// Example 2: Cycle Detection with Real Projections
// =============================================================================

console.log('\n=== Example 2: Cycle Detection ===\n')

// Three projections that form a cycle: A reads B, B reads C, C reads A

interface StateA { value: number }
interface StateB { value: number }
interface StateC { value: number }

// Forward declarations to enable circular reads
// We'll define them in order but with reads pointing to each other

const ProjectionC = Projection.define<AppEvent, StateC>()({
  name: 'C',
  initial: { value: 0 },
  eventHandlers: {
    increment: ({ state }) => ({ value: state.value + 1 })
  }
})

const ProjectionB = Projection.define<AppEvent, StateB>()({
  name: 'B',
  initial: { value: 0 },
  reads: [ProjectionC] as const,
  eventHandlers: {
    increment: ({ state, read }) => {
      const c = read(ProjectionC)
      return { value: state.value + c.value }
    }
  }
})

// ProjectionA reads from ProjectionB (which reads from C)
// But we'll also make C read from A to create a cycle
const ProjectionA = Projection.define<AppEvent, StateA>()({
  name: 'A',
  initial: { value: 0 },
  reads: [ProjectionB] as const,
  eventHandlers: {
    increment: ({ state, read }) => {
      const b = read(ProjectionB)
      return { value: state.value + b.value }
    }
  }
})

// Now redefine C to read from A - this creates the cycle!
const ProjectionC_Cyclic = Projection.define<AppEvent, StateC>()({
  name: 'C',
  initial: { value: 0 },
  reads: [ProjectionA] as const,  // A -> B -> C -> A = CYCLE!
  eventHandlers: {
    increment: ({ state, read }) => {
      const a = read(ProjectionA)
      return { value: state.value + a.value }
    }
  }
})

// Create agent with cyclic dependencies
const CyclicAgent = Agent.define<AppEvent>()({
  name: 'CyclicAgent',
  projections: [ProjectionA, ProjectionB, ProjectionC_Cyclic],
  workers: [],
  expose: {}
})

async function demonstrateCycleDetection() {
  console.log('Creating agent with cyclic projections: A -> B -> C -> A\n')

  try {
    const client = await CyclicAgent.createClient()
    console.log('ERROR: Should have thrown on cycle detection!')
    await client.dispose()
  } catch (error) {
    console.log('✓ Cycle detected:', (error as Error).message)
  }
}

await demonstrateCycleDetection()

console.log('\n=== Demo Complete ===')
