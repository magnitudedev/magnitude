/**
 * ArtifactAwarenessProjection (Forked)
 *
 * Per-fork awareness of artifact IDs.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { extractArtifactRefs } from '../util/artifact-links'
import { OutboundMessagesProjection } from './outbound-messages'
import { AgentRegistryProjection } from './agent-registry'
import { ArtifactProjection } from './artifact'
import type { ArtifactState } from './artifact'
import { createCompactDiff, computeDiffStats, shouldUseDiff } from '../util/compact-diff'

export interface ForkArtifactAwarenessState {
  readonly awareArtifactIds: ReadonlySet<string>
  readonly pendingRefs: ReadonlyMap<string, ReadonlySet<string | null>>
}

export function isAwareArtifact(state: ForkArtifactAwarenessState, artifactId: string): boolean {
  return state.awareArtifactIds.has(artifactId)
}

export function getAwareArtifacts(state: ForkArtifactAwarenessState): string[] {
  return [...state.awareArtifactIds]
}

function formatArtifactUpdateNotification(previousContent: string | null, content: string, artifactId: string): string {
  const oldText = previousContent ?? ''
  const { linesChanged } = computeDiffStats(oldText, content)
  const totalLines = Math.max(1, content.split('\n').length)
  const body = shouldUseDiff(linesChanged, totalLines)
    ? (createCompactDiff(oldText, content) || content)
    : content

  return `Artifact "${artifactId}" was updated.\n\n${body}`
}

function clonePendingRefs(pendingRefs: ReadonlyMap<string, ReadonlySet<string | null>>): Map<string, Set<string | null>> {
  return new Map([...pendingRefs.entries()].map(([artifactId, forkIds]) => [artifactId, new Set(forkIds)]))
}

function processArtifactRefs(
  refs: Array<{ id: string }>,
  forkId: string | null,
  artifactState: ArtifactState,
  emitFirstMentioned: (value: { forkId: string | null; artifactId: string; content: string }) => void,
  forkState: ForkArtifactAwarenessState
): { awareArtifactIds: Set<string>; pendingRefs: Map<string, Set<string | null>>; changed: boolean; pendingChanged: boolean } {
  const nextAware = new Set(forkState.awareArtifactIds)
  const nextPendingRefs = clonePendingRefs(forkState.pendingRefs)

  let changed = false
  let pendingChanged = false

  for (const ref of refs) {
    if (nextAware.has(ref.id)) continue
    const artifact = artifactState.artifacts.get(ref.id)

    if (!artifact) {
      const existingPending = nextPendingRefs.get(ref.id)
      const pending = existingPending ?? new Set<string | null>()
      const beforeSize = pending.size
      // forkId can be null (root/orchestrator fork) — must be included so
      // the artifactChanged handler grants awareness when the artifact is created later
      pending.add(forkId)
      nextPendingRefs.set(ref.id, pending)
      if (!existingPending || pending.size !== beforeSize) pendingChanged = true
      continue
    }

    emitFirstMentioned({ forkId, artifactId: ref.id, content: artifact.content })
    nextAware.add(ref.id)
    changed = true
  }

  return { awareArtifactIds: nextAware, pendingRefs: nextPendingRefs, changed, pendingChanged }
}

export const ArtifactAwarenessProjection = Projection.defineForked<AppEvent, ForkArtifactAwarenessState>()({
  name: 'ArtifactAwareness',
  reads: [OutboundMessagesProjection, AgentRegistryProjection, ArtifactProjection] as const,
  initialFork: {
    awareArtifactIds: new Set(),
    pendingRefs: new Map(),
  },
  signals: {
    artifactFirstMentioned: Signal.create<{ forkId: string | null; artifactId: string; content: string }>('ArtifactAwareness/artifactFirstMentioned'),
    artifactUpdateNotification: Signal.create<{ forkId: string | null; artifactId: string; text: string }>('ArtifactAwareness/artifactUpdateNotification'),
  },
  eventHandlers: {
    artifact_changed: ({ event, fork }) => {
      // Writing an artifact grants awareness to the writing fork (no injection needed — they just wrote it)
      if (fork.awareArtifactIds.has(event.id)) return fork
      const awareArtifactIds = new Set(fork.awareArtifactIds)
      awareArtifactIds.add(event.id)
      return { ...fork, awareArtifactIds }
    },

    user_message: ({ event, fork, emit, read }) => {
      const text = event.content
        .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
        .map(p => p.text)
        .join('')

      const refs = extractArtifactRefs(text)
      if (refs.length === 0) return fork

      const artifactState = read(ArtifactProjection)
      const result = processArtifactRefs(refs, event.forkId, artifactState, emit.artifactFirstMentioned, fork)

      if (!result.changed && !result.pendingChanged) return fork
      return { ...fork, awareArtifactIds: result.awareArtifactIds, pendingRefs: result.pendingRefs }
    },

    fork_started: ({ event, fork, emit, read }) => {
      const refs = extractArtifactRefs(event.context)
      if (refs.length === 0) return fork

      const artifactState = read(ArtifactProjection)
      const result = processArtifactRefs(refs, event.forkId, artifactState, emit.artifactFirstMentioned, fork)

      if (!result.changed && !result.pendingChanged) return fork
      return { ...fork, awareArtifactIds: result.awareArtifactIds, pendingRefs: result.pendingRefs }
    },
  },
  signalHandlers: (on) => [
    on(OutboundMessagesProjection.signals.messageCompleted, ({ value, state, read, emit }) => {
      const refs = extractArtifactRefs(value.text)
      if (refs.length === 0) return state

      const targetForkId = value.targetForkId
      const artifactState = read(ArtifactProjection)
      const forks = new Map(state.forks)
      const rootFork = forks.get(null)
      if (!rootFork) return state
      let pendingRefs = clonePendingRefs(rootFork.pendingRefs)

      const applyToFork = (forkId: string | null) => {
        const forkState = forks.get(forkId)
        if (!forkState) return

        const result = processArtifactRefs(refs, forkId, artifactState, emit.artifactFirstMentioned, {
          ...forkState,
          pendingRefs,
        })

        pendingRefs = result.pendingRefs
        if (result.changed) forks.set(forkId, { ...forkState, awareArtifactIds: result.awareArtifactIds })
      }

      applyToFork(value.forkId)
      if (targetForkId !== undefined) applyToFork(targetForkId)

      const updatedRootFork = forks.get(null) ?? rootFork
      forks.set(null, { ...updatedRootFork, pendingRefs })

      return { ...state, forks }
    }),

    on(ArtifactProjection.signals.artifactChanged, ({ value, state, emit }) => {
      const forks = new Map(state.forks)

      for (const [forkId, fork] of forks) {
        if (forkId === value.changedByForkId) continue
        if (!fork.awareArtifactIds.has(value.id)) continue

        emit.artifactUpdateNotification({
          forkId,
          artifactId: value.id,
          text: formatArtifactUpdateNotification(value.previousContent, value.content, value.id),
        })
      }

      const applyToFork = (forkId: string | null, artifactId: string, content: string) => {
        const forkState = forks.get(forkId)
        if (!forkState) return
        if (forkState.awareArtifactIds.has(artifactId)) return

        const awareArtifactIds = new Set(forkState.awareArtifactIds)
        awareArtifactIds.add(artifactId)
        emit.artifactFirstMentioned({ forkId, artifactId, content })
        forks.set(forkId, { ...forkState, awareArtifactIds })
      }

      for (const [forkId, fork] of forks) {
        const pendingForkIds = fork.pendingRefs.get(value.id)
        if (!pendingForkIds) continue

        if (forkId === null) {
          for (const targetForkId of pendingForkIds) applyToFork(targetForkId, value.id, value.content)
        } else {
          applyToFork(forkId, value.id, value.content)
        }

        const pendingRefs = clonePendingRefs(fork.pendingRefs)
        pendingRefs.delete(value.id)
        forks.set(forkId, { ...fork, pendingRefs })
      }

      return { ...state, forks }
    }),
  ],
})