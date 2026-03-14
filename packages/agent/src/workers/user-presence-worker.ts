/**
 * UserPresenceWorker
 *
 * On focus regained, waits 5s to confirm the user actually stayed.
 * If they were away longer than the threshold and active agents exist,
 * publishes user_return_confirmed (which wakes the orchestrator) and wake.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { UserPresenceProjection } from '../projections/user-presence'
import { AgentRoutingProjection } from '../projections/agent-routing'
import { WorkingStateProjection } from '../projections/working-state'
import { USER_AWAY_RETURN_THRESHOLD_MS, USER_PRESENCE_CONFIRM_DELAY_MS } from '../constants'

export const UserPresenceWorker = Worker.define<AppEvent>()({
  name: 'UserPresenceWorker',

  signalHandlers: (on) => [
    on(UserPresenceProjection.signals.presenceChanged, ({ focused }, publish, read) => Effect.gen(function* () {
      if (!focused) return

      // Wait to confirm the user actually stayed
      yield* Effect.sleep(USER_PRESENCE_CONFIRM_DELAY_MS)

      const presence = yield* read(UserPresenceProjection)

      // If they left again during the wait, bail
      if (!presence.currentFocusState) return

      // Only trigger if they were away long enough
      if (presence.blurredAt === null || presence.focusedAt === null) return
      const awayMs = presence.focusedAt - presence.blurredAt
      if (awayMs <= USER_AWAY_RETURN_THRESHOLD_MS) return

      // Only trigger if there are agents actively working (mid-turn).
      // HACK: We need cross-fork WorkingState reads from a global worker, which
      // the framework doesn't support via read(). Using Tag + getFork() directly.
      const agentState = yield* read(AgentRoutingProjection)
      const workingStateInst = yield* WorkingStateProjection.Tag
      let hasActiveAgents = false
      for (const agent of agentState.agents.values()) {
        if (agent.dismissed) continue
        const ws = yield* workingStateInst.getFork(agent.forkId)
        if (ws.working) {
          hasActiveAgents = true
          break
        }
      }
      if (!hasActiveAgents) return

      yield* publish({ type: 'user_return_confirmed', forkId: null })
      yield* publish({ type: 'wake', forkId: null })
    })),
  ],
})