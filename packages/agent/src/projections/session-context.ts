/**
 * SessionContextProjection
 *
 * Holds session-level configuration that doesn't change after initialization.
 * Other projections can read from this via the `reads` mechanism.
 */

import { Projection } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'

const GitContextSchema = Schema.Struct({
  branch: Schema.String,
  status: Schema.String,
  recentCommits: Schema.String,
})

const AgentsFileSchema = Schema.Struct({
  filename: Schema.String,
  content: Schema.String,
})

const SkillSchema = Schema.Struct({
  name: Schema.String,
  description: Schema.String,
  path: Schema.String,
})

export const SessionContextSchema = Schema.Struct({
  cwd: Schema.String,
  scratchpadPath: Schema.String,
  platform: Schema.Literal('macos', 'linux', 'windows'),
  shell: Schema.String,
  timezone: Schema.String,
  username: Schema.String,
  fullName: Schema.NullOr(Schema.String),
  git: Schema.NullOr(GitContextSchema),
  folderStructure: Schema.String,
  agentsFile: Schema.NullOr(AgentsFileSchema),
  skills: Schema.NullOr(Schema.Array(SkillSchema)),
})

export const SessionContextStateSchema = Schema.Struct({
  initialized: Schema.Boolean,
  context: Schema.NullOr(SessionContextSchema),
})
export type SessionContextState = typeof SessionContextStateSchema.Type

// =============================================================================
// Projection
// =============================================================================

export const SessionContextProjection = Projection.define<AppEvent>()({
  name: 'SessionContext',
  state: SessionContextStateSchema,
  initial: { initialized: false, context: null },

  eventHandlers: {
    session_initialized: ({ event }) => ({
      initialized: true,
      context: event.context
    }),

    compaction_injected: ({ state }) => {
      // Minimal event — refreshedContext flows via CompactionProjection's compactionInjected signal.
      // WindowProjection handles session_context replacement in its signal handler.
      return state
    }
  }
})
