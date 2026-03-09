/**
 * ArtifactProjection
 *
 * Global projection holding artifact state.
 * Artifacts are content stores used for context flow between agents.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

// =============================================================================
// Types
// =============================================================================

export interface ArtifactItem {
  readonly id: string
  readonly content: string
  readonly syncPath: string | null
}

export interface ArtifactState {
  readonly artifacts: ReadonlyMap<string, ArtifactItem>
}

// =============================================================================
// Projection
// =============================================================================

export const ArtifactProjection = Projection.define<AppEvent, ArtifactState>()({
  name: 'Artifact',

  initial: {
    artifacts: new Map(),
  },

  signals: {
    artifactChanged: Signal.create<{ id: string; previousContent: string | null; content: string; changedByForkId: string | null }>('Artifact/artifactChanged'),
    artifactSynced: Signal.create<{ id: string; path: string }>('Artifact/artifactSynced'),
  },

  eventHandlers: {
    artifact_changed: ({ event, state, emit }) => {
      const existing = state.artifacts.get(event.id)
      const previousContent = existing?.content ?? null

      const artifacts = new Map(state.artifacts)
      artifacts.set(event.id, {
        id: event.id,
        content: event.content,
        syncPath: existing?.syncPath ?? null,
      })

      emit.artifactChanged({ id: event.id, previousContent, content: event.content, changedByForkId: event.forkId })
      return { ...state, artifacts }
    },

    artifact_synced: ({ event, state, emit }) => {
      const existing = state.artifacts.get(event.id)
      if (!existing) return state

      const artifacts = new Map(state.artifacts)
      artifacts.set(event.id, {
        ...existing,
        syncPath: event.path,
      })
      emit.artifactSynced({ id: event.id, path: event.path })
      return { ...state, artifacts }
    },
  },
})
