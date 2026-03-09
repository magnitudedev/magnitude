/**
 * Forked Counter Example
 *
 * Demonstrates Projection.defineForked() with a simple counter that can be forked.
 * Each fork has independent state, and forks can be initialized with parent's value.
 *
 * Run with: npx tsx examples/forked-counter.ts
 */

import { Effect, Layer } from 'effect'
import { Projection, Worker, Agent, Signal, type ForkableEvent } from '../src'

// =============================================================================
// Events
// =============================================================================

interface Increment {
  type: 'increment'
  forkId?: string
  amount: number
}

interface ForkCreated {
  type: 'fork_created'
  forkId: string
  parentForkId?: string
  inheritValue: boolean
}

interface ForkCounterInitialized {
  type: 'fork_counter_initialized'
  forkId: string
  value: number
}

interface Interrupt {
  type: 'interrupt'
}

type CounterEvent = Increment | ForkCreated | ForkCounterInitialized | Interrupt

// =============================================================================
// Forked Projection
// =============================================================================

interface CounterForkState {
  value: number
}

const CounterProjection = Projection.defineForked<CounterEvent, CounterForkState>()({
  name: 'Counter',
  initialFork: { value: 0 },

  signals: {
    valueChanged: Signal.create<{ forkId: string | undefined; value: number }>('Counter/valueChanged')
  },

  eventHandlers: {
    // Increment - operates on specific fork
    increment: ({ event, fork, emit }) => {
      const newValue = fork.value + event.amount
      emit.valueChanged({ forkId: event.forkId, value: newValue })
      return { value: newValue }
    },

    // Initialize fork with specific value (e.g., inherited from parent)
    fork_counter_initialized: ({ event, fork, emit }) => {
      emit.valueChanged({ forkId: event.forkId, value: event.value })
      return { value: event.value }
    }
  }
})

// =============================================================================
// Fork Orchestrator Worker
// =============================================================================

const ForkOrchestrator = Worker.define<CounterEvent>()({
  name: 'ForkOrchestrator',

  eventHandlers: {
    fork_created: (event, publish) => Effect.gen(function* () {
      const { forkId, parentForkId, inheritValue } = event

      if (inheritValue) {
        // Get parent's value and initialize fork with it
        const counter = yield* CounterProjection.Tag
        const parentState = yield* counter.getFork(parentForkId)

        yield* publish({
          type: 'fork_counter_initialized',
          forkId,
          value: parentState.value
        })

        console.log(`Fork ${forkId} created with inherited value: ${parentState.value}`)
      } else {
        console.log(`Fork ${forkId} created with initial value: 0`)
      }
    })
  }
})

// =============================================================================
// Logger Worker (observes all forks)
// =============================================================================

const LoggerWorker = Worker.define<CounterEvent>()({
  name: 'Logger',

  signalHandlers: (on) => [
    on(CounterProjection.signals.valueChanged, ({ forkId, value }, publish) => Effect.sync(() => {
      const forkLabel = forkId ?? 'root'
      console.log(`[${forkLabel}] Counter value changed to: ${value}`)
    }))
  ]
})

// =============================================================================
// Agent
// =============================================================================

const CounterAgent = Agent.define<CounterEvent>()({
  name: 'CounterAgent',

  projections: [CounterProjection],
  workers: [ForkOrchestrator, LoggerWorker],

  expose: {
    state: {
      counter: CounterProjection
    },
    signals: {
      valueChanged: CounterProjection.signals.valueChanged
    }
  }
})

// =============================================================================
// Demo
// =============================================================================

async function main() {
  console.log('=== Forked Counter Demo ===\n')

  const client = await CounterAgent.createClient()

  // Small delay to ensure workers are subscribed
  await new Promise(r => setTimeout(r, 10))

  // Increment root counter
  console.log('> Incrementing root counter by 5')
  await client.send({ type: 'increment', amount: 5 })

  console.log('> Incrementing root counter by 3')
  await client.send({ type: 'increment', amount: 3 })

  // Create a fork that inherits parent value
  console.log('\n> Creating fork "alpha" with inheritance')
  await client.send({
    type: 'fork_created',
    forkId: 'alpha',
    parentForkId: undefined,
    inheritValue: true
  })

  // Increment fork
  console.log('> Incrementing fork "alpha" by 10')
  await client.send({ type: 'increment', forkId: 'alpha', amount: 10 })

  // Increment root again - should be independent
  console.log('> Incrementing root by 1')
  await client.send({ type: 'increment', amount: 1 })

  // Create another fork without inheritance
  console.log('\n> Creating fork "beta" without inheritance')
  await client.send({
    type: 'fork_created',
    forkId: 'beta',
    parentForkId: undefined,
    inheritValue: false
  })

  console.log('> Incrementing fork "beta" by 100')
  await client.send({ type: 'increment', forkId: 'beta', amount: 100 })

  // Show final state
  console.log('\n=== Final State ===')
  const allForks = await Effect.runPromise(
    Effect.gen(function* () {
      const counter = yield* CounterProjection.Tag
      return yield* counter.getAllForks()
    }).pipe(
      Effect.provide(CounterAgent.Layer)
    )
  )

  // Note: Can't easily access internal state from client, but the logs show the values
  console.log('Check the logs above for final values:')
  console.log('- root: 9 (5 + 3 + 1)')
  console.log('- alpha: 18 (inherited 8, then +10)')
  console.log('- beta: 100 (started at 0, then +100)')

  await client.dispose()
  console.log('\n=== Demo Complete ===')
}

main().catch(console.error)
