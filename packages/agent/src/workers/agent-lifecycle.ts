/**
 * AgentLifecycle Worker
 *
 * Handles agent infrastructure that tools can't own:
 * - Root fork init on session start
 * - Interrupt process cleanup for active forks
 *
 * fork() and task.validate handle their own lifecycle directly.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ExecutionManager } from '../execution/execution-manager'
import { WorkingStateProjection } from '../projections/working-state'

import { buildInterruptedTurnCompleted } from '../util/interrupt-utils'

// =============================================================================
// Worker
// =============================================================================

export const AgentLifecycle = Worker.define<AppEvent>()({
  name: 'AgentLifecycle',

  // interrupt handler must not be interrupted itself
  ignoreInterrupt: ['interrupt'] as const,

  eventHandlers: {

    // Create root fork resources when session starts
    session_initialized: (event, _publish) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      const rootVariant = event.context?.oneshot ? 'lead-oneshot' : 'lead'
      yield* execManager.initFork(null, rootVariant)
    }).pipe(Effect.orDie),

    // Interrupt stops the current turn and also kills tracked background
    // processes for that fork. Child process exit listeners publish
    // background_process_exited events as cleanup completes.
    interrupt: (event, _publish, _read) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.interruptProcesses(event.forkId)
    }).pipe(Effect.orDie),

    soft_interrupt: () => Effect.void,

    agent_killed: (event) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)
    }).pipe(Effect.orDie),

    subagent_user_killed: (event, publish) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)
      yield* publish({ type: 'wake', forkId: event.parentForkId })
    }).pipe(Effect.orDie),

    subagent_idle_closed: (event) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)
    }).pipe(Effect.orDie),
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.turnInterrupted, ({ forkId, turnId, chainId }, publish) => Effect.gen(function* () {
      const turnCompleted = yield* buildInterruptedTurnCompleted({ forkId, turnId, chainId })
      yield* publish(turnCompleted)
    }).pipe(Effect.orDie))
  ]
})
