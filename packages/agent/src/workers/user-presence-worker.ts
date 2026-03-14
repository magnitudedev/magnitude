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
import { AgentStatusProjection } from '../projections/agent-status'
import { USER_AWAY_RETURN_THRESHOLD_MS, USER_PRESENCE_CONFIRM_DELAY_MS } from '../constants'

export const UserPresenceWorker = Worker.define<AppEvent>()({
  name: 'UserPresenceWorker',

  signalHandlers: (on) => [
    on(UserPresenceProjection.signals.presenceChanged, ({ focused }, publish, read) => Effect.gen(function* () {
      if (!focused) return

      yield* Effect.sleep(USER_PRESENCE_CONFIRM_DELAY_MS)

      const presence = yield* read(UserPresenceProjection)
      if (!presence.currentFocusState) return

      if (presence.blurredAt === null || presence.focusedAt === null) return
      const awayMs = presence.focusedAt - presence.blurredAt
      if (awayMs <= USER_AWAY_RETURN_THRESHOLD_MS) return

      const agentState = yield* read(AgentStatusProjection)
      const hasActiveAgents = [...agentState.agents.values()].some(agent => agent.status === 'working')
      if (!hasActiveAgents) return

      yield* publish({ type: 'user_return_confirmed', forkId: null })
      yield* publish({ type: 'wake', forkId: null })
    })),
  ],
})