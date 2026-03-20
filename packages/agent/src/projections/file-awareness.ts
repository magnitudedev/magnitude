/**
 * FileAwarenessProjection (Forked)
 *
 * Per-fork awareness of referenced files.
 */

import path from 'node:path'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { SessionContextProjection } from './session-context'
import { scanFileRefs } from '../workspace/file-refs'
import { resolveFileRef } from '../workspace/file-ref-resolution'
import { OutboundMessagesProjection } from './outbound-messages'
import { createCompactDiff, computeDiffStats, shouldUseDiff } from '../util/compact-diff'
import { extractWrittenFilePathFromToolEvent } from '../workspace/file-tracking'

export interface ForkFileAwarenessState {
  readonly injectedFiles: ReadonlyMap<string, string>
  readonly pendingRefs: ReadonlyMap<string, ReadonlySet<string | null>>
}

function contentHash(content: string): string {
  return createHash('sha256').update(content).digest('hex')
}

function formatFileUpdateNotification(previousContent: string, content: string, filePath: string): string {
  const { linesChanged } = computeDiffStats(previousContent, content)
  const totalLines = Math.max(1, content.split('\n').length)
  const body = shouldUseDiff(linesChanged, totalLines)
    ? (createCompactDiff(previousContent, content) || content)
    : content

  return `File "${filePath}" was updated.\n\n${body}`
}

function clonePendingRefs(pendingRefs: ReadonlyMap<string, ReadonlySet<string | null>>): Map<string, Set<string | null>> {
  return new Map([...pendingRefs.entries()].map(([resolvedPath, forkIds]) => [resolvedPath, new Set(forkIds)]))
}

function readResolvedFile(filePath: string): string | null {
  try {
    return readFileSync(filePath, 'utf8')
  } catch {
    return null
  }
}

function processFileRefs(
  text: string,
  forkId: string | null,
  sessionContext: { cwd: string; workspacePath: string },
  emitFirstMentioned: (value: { forkId: string | null; path: string; content: string }) => void,
  forkState: ForkFileAwarenessState
): { injectedFiles: Map<string, string>; pendingRefs: Map<string, Set<string | null>>; changed: boolean; pendingChanged: boolean } {
  const refs = scanFileRefs(text)
  if (refs.length === 0) {
    return {
      injectedFiles: new Map(forkState.injectedFiles),
      pendingRefs: clonePendingRefs(forkState.pendingRefs),
      changed: false,
      pendingChanged: false,
    }
  }

  const nextInjectedFiles = new Map(forkState.injectedFiles)
  const nextPendingRefs = clonePendingRefs(forkState.pendingRefs)

  let changed = false
  let pendingChanged = false

  for (const ref of refs) {
    const explicitWorkspacePrefix = ref.path.startsWith('$M/') || ref.path.startsWith('${M}/')
    const innerPath = ref.path.startsWith('${M}/')
      ? ref.path.slice('${M}/'.length)
      : ref.path.startsWith('$M/')
        ? ref.path.slice('$M/'.length)
        : ref.path

    const resolvedRef = resolveFileRef(ref.path, sessionContext.cwd, sessionContext.workspacePath)
    const existingResolvedPath = resolvedRef?.resolvedPath ?? null

    if (existingResolvedPath) {
      const content = readResolvedFile(existingResolvedPath)
      if (content === null) continue

      const hash = contentHash(content)
      if (nextInjectedFiles.get(existingResolvedPath) === hash) continue

      emitFirstMentioned({
        forkId,
        path: innerPath,
        content,
      })

      nextInjectedFiles.set(existingResolvedPath, hash)
      nextPendingRefs.delete(existingResolvedPath)
      changed = true
      continue
    }

    const pendingPath = explicitWorkspacePrefix
      ? path.resolve(sessionContext.workspacePath, innerPath)
      : path.resolve(sessionContext.cwd, innerPath)

    const existingPending = nextPendingRefs.get(pendingPath)
    const pending = existingPending ?? new Set<string | null>()
    const beforeSize = pending.size
    pending.add(forkId)
    nextPendingRefs.set(pendingPath, pending)
    if (!existingPending || pending.size !== beforeSize) pendingChanged = true
  }

  return { injectedFiles: nextInjectedFiles, pendingRefs: nextPendingRefs, changed, pendingChanged }
}

function toText(parts: readonly { type: string; text?: string }[]): string {
  return parts
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text' && typeof p.text === 'string')
    .map(p => p.text)
    .join('')
}

const fileFirstMentionedSignal = Signal.create<{ forkId: string | null; path: string; content: string }>('FileAwareness/fileFirstMentioned')
const fileUpdateNotificationSignal = Signal.create<{ forkId: string | null; path: string; notificationText: string }>('FileAwareness/fileUpdateNotification')

export const FileAwarenessProjection = Projection.defineForked<AppEvent, ForkFileAwarenessState>()({
  name: 'FileAwareness',
  reads: [SessionContextProjection, OutboundMessagesProjection] as const,
  initialFork: {
    injectedFiles: new Map(),
    pendingRefs: new Map(),
  },
  signals: {
    fileFirstMentioned: fileFirstMentionedSignal,
    fileUpdateNotification: fileUpdateNotificationSignal,
  },
  eventHandlers: {
    user_message: ({ event, fork, emit, read }) => {
      const text = toText(event.content)
      if (!text) return fork

      const context = read(SessionContextProjection).context
      if (!context) return fork

      const result = processFileRefs(text, event.forkId, context, emit.fileFirstMentioned, fork)
      if (!result.changed && !result.pendingChanged) return fork

      return { ...fork, injectedFiles: result.injectedFiles, pendingRefs: result.pendingRefs }
    },

    agent_created: ({ event, fork, emit, read }) => {
      const text = `${event.context}\n${event.message}`
      const context = read(SessionContextProjection).context
      if (!context) return fork

      const result = processFileRefs(text, event.forkId, context, emit.fileFirstMentioned, fork)
      if (!result.changed && !result.pendingChanged) return fork

      return { ...fork, injectedFiles: result.injectedFiles, pendingRefs: result.pendingRefs }
    },

    tool_event: ({ event, fork, emit }) => {
      const resolvedPath = extractWrittenFilePathFromToolEvent(event)
      if (!resolvedPath) return fork

      const content = readResolvedFile(resolvedPath)
      if (content === null) return fork

      const nextHash = contentHash(content)
      let injectedFiles = fork.injectedFiles
      let pendingRefs = fork.pendingRefs
      let changed = false

      if (fork.injectedFiles.has(resolvedPath)) {
        const previousHash = fork.injectedFiles.get(resolvedPath)
        if (previousHash && previousHash !== nextHash) {
          emit.fileUpdateNotification({
            forkId: event.forkId,
            path: resolvedPath,
            notificationText: formatFileUpdateNotification('', content, resolvedPath),
          })
          const next = new Map(fork.injectedFiles)
          next.set(resolvedPath, nextHash)
          injectedFiles = next
          changed = true
        }
      }

      const pendingForkIds = fork.pendingRefs.get(resolvedPath)
      if (pendingForkIds && pendingForkIds.has(event.forkId)) {
        emit.fileFirstMentioned({
          forkId: event.forkId,
          path: resolvedPath,
          content,
        })
        const next = new Map(injectedFiles)
        next.set(resolvedPath, nextHash)
        injectedFiles = next

        const nextPending = clonePendingRefs(pendingRefs)
        nextPending.delete(resolvedPath)
        pendingRefs = nextPending
        changed = true
      }

      return changed ? { ...fork, injectedFiles, pendingRefs } : fork
    },
  },

  signalHandlers: (on) => [
    on(OutboundMessagesProjection.signals.messageCompleted, ({ value, state, read, emit }) => {
      const context = read(SessionContextProjection).context
      if (!context) return state

      const forks = new Map(state.forks)
      const rootFork = forks.get(null)
      if (!rootFork) return state

      let pendingRefs = clonePendingRefs(rootFork.pendingRefs)

      const applyToFork = (forkId: string | null) => {
        const forkState = forks.get(forkId)
        if (!forkState) return

        const result = processFileRefs(
          value.text,
          forkId,
          context,
          emit.fileFirstMentioned,
          { ...forkState, pendingRefs }
        )
        pendingRefs = result.pendingRefs
        if (result.changed || result.pendingChanged) {
          forks.set(forkId, {
            ...forkState,
            injectedFiles: result.injectedFiles,
          })
        }
      }

      applyToFork(value.forkId)
      if (value.targetForkId !== undefined) applyToFork(value.targetForkId)

      const updatedRootFork = forks.get(null) ?? rootFork
      forks.set(null, { ...updatedRootFork, pendingRefs })

      return { ...state, forks }
    }),


  ],
})