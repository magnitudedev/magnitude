/**
 * AgentLifecycle Worker
 *
 * Handles agent infrastructure that tools can't own:
 * - Root fork init on session start
 *
 * fork() and task.validate handle their own lifecycle directly.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ExecutionManager } from '../execution/types'


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
      const rootVariant = 'leader'
      yield* execManager.initFork(null, rootVariant)
    }).pipe(Effect.orDie),

    // Interrupt is for stopping the agent's turn, not killing background
    // processes. Detached shell processes are killed on disposeFork (agent
    // killed, worker killed, session end) via killAll.
    interrupt: (_event) => Effect.void,

    agent_killed: (event) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)
    }).pipe(Effect.orDie),

    worker_user_killed: (event, _publish) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)
    }).pipe(Effect.orDie),

    worker_idle_closed: (event) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)
    }).pipe(Effect.orDie),
  },
})
