/**
 * CompactionWorker — PHASE 5 STUB
 *
 * Compaction is disabled during native paradigm migration.
 * The full compaction rewrite (using the new Codec/TurnEngine pipeline) is
 * deferred to a later phase. This stub exists so the worker is still registered
 * in the worker registry without crashing.
 *
 * TODO: rebuild compaction in the native-paradigm follow-up phase.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

export const CompactionWorker = Worker.defineForked<AppEvent>()({
  name: 'CompactionWorker',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: ['agent_killed', 'subagent_user_killed', 'subagent_idle_closed'],
  },

  eventHandlers: {
    // No-op: compaction is disabled in this phase.
    // All events are silently ignored.
  },
})
