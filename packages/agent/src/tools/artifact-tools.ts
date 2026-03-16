/**
 * Artifact Tools
 *
 * Tool groups for managing artifacts — content stores for context flow between agents.
 *
 * Orchestrator tools: artifact.sync
 * Agent tools: artifact.read, artifact.write, artifact.update
 */

import { Effect, Context } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import type { ArtifactState } from '../projections/artifact'
import { ArtifactProjection } from '../projections/artifact'
import { ArtifactAwarenessProjection, isAwareArtifact } from '../projections/artifact-awareness'
import { validateAndApply } from '../util/edit'
import { ChatPersistence } from '../persistence/chat-persistence-service'

// =============================================================================
// Artifact State Reader Service
// =============================================================================

export interface ArtifactStateReader {
  readonly getState: () => Effect.Effect<ArtifactState>
}

export class ArtifactStateReaderTag extends Context.Tag('ArtifactStateReader')<
  ArtifactStateReaderTag,
  ArtifactStateReader
>() {}

// =============================================================================
// Errors
// =============================================================================

const ArtifactError = ToolErrorSchema('ArtifactError', {})

function ensureAware(forkId: string | null, id: string) {
  return Effect.gen(function* () {
    const awarenessProjection = yield* ArtifactAwarenessProjection.Tag
    const awarenessState = yield* awarenessProjection.getFork(forkId)
    const aware = isAwareArtifact(awarenessState, id)
    if (!aware) {
      return yield* Effect.fail({
        _tag: 'ArtifactError' as const,
        message: `Artifact "${id}" is not in this agent's awareness set. Mention [[${id}]] in a message first.`,
      })
    }
  })
}

// =============================================================================
// artifact.sync — Bind artifact to a file path with bidirectional bootstrap
// =============================================================================

export const artifactSyncTool = createTool({
  name: 'sync',
  group: 'artifact',
  description: 'Bind artifact ID to file path. If artifact exists, write artifact to file. If artifact is missing and file exists, load file into artifact. After binding, future artifact changes auto-sync to this path.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Artifact ID (logical name, e.g. "plan"; not a file path).' }),
    path: Schema.String.annotations({ description: 'File path to sync to' }),
  }),
  outputSchema: Schema.Struct({ id: Schema.String, path: Schema.String }),
  errorSchema: ArtifactError,
  argMapping: ['id', 'path'],
  bindings: {
    xmlInput: { type: 'tag', attributes: [{ field: 'id', attr: 'id' }, { field: 'path', attr: 'path' }], selfClosing: true },
    xmlOutput: { type: 'tag', childTags: [{ tag: 'id', field: 'id' }, { tag: 'path', field: 'path' }] } as const,
  } as const,
  execute: ({ id, path }) =>
    Effect.gen(function* () {
      const { forkId } = yield* Fork.ForkContext

      const stateReader = yield* ArtifactStateReaderTag
      const state = yield* stateReader.getState()
      const bus = yield* WorkerBusTag<AppEvent>()

      const existing = state.artifacts.get(id)

      if (!existing) {
        // Artifact doesn't exist — try loading from file
        const fileContent = yield* Effect.tryPromise({
          try: () => Bun.file(path).text(),
          catch: (e) => e instanceof Error ? e : new Error(String(e)),
        }).pipe(
          Effect.catchAll(() => Effect.succeed<string | null>(null))
        )

        if (fileContent == null) {
          return yield* Effect.fail({
            _tag: 'ArtifactError' as const,
            message: `Artifact "${id}" does not exist and file "${path}" does not exist (or cannot be read). At least one must exist.`,
          })
        }

        yield* bus.publish({
          type: 'artifact_changed',
          forkId,
          id,
          previousContent: null,
          content: fileContent,
        })
      }

      // Bind sync path — worker handles writing to disk
      yield* bus.publish({
        type: 'artifact_synced',
        forkId,
        id,
        path,
      })

      return { id, path }
    }),
})

// =============================================================================
// artifact.read — Read artifact content
// =============================================================================

export const artifactReadTool = createTool({
  name: 'read',
  group: 'artifact',
  description: 'Read the current content of an artifact.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Artifact ID (logical name, e.g. "plan"; not a file path).' }),
  }),
  outputSchema: Schema.String,
  errorSchema: ArtifactError,
  argMapping: ['id'],
  bindings: {
    xmlInput: { type: 'tag', attributes: [{ field: 'id', attr: 'id' }], selfClosing: true },
    xmlOutput: { type: 'tag' } as const,
  } as const,
  execute: ({ id }) =>
    Effect.gen(function* () {
      const { forkId } = yield* Fork.ForkContext
      yield* ensureAware(forkId, id)

      const stateReader = yield* ArtifactStateReaderTag
      const state = yield* stateReader.getState()
      const artifact = state.artifacts.get(id)

      if (!artifact) {
        return yield* Effect.fail({ _tag: 'ArtifactError' as const, message: `Artifact "${id}" does not exist` })
      }

      return artifact.content
    }),
})

// =============================================================================
// artifact.write — Write artifact content (full replace)
// =============================================================================

export const artifactWriteTool = createTool({
  name: 'write',
  group: 'artifact',
  description: 'Write content to an artifact (creates if missing, full replace).',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Artifact ID (logical name, e.g. "plan"; not a file path).' }),
    content: Schema.String.annotations({ description: 'New content for the artifact' }),
  }),
  outputSchema: Schema.String,
  errorSchema: ArtifactError,
  argMapping: ['id', 'content'],
  bindings: {
    xmlInput: { type: 'tag', attributes: [{ field: 'id', attr: 'id' }], body: 'content' },
    xmlOutput: { type: 'tag' } as const,
  } as const,
  execute: ({ id, content }) =>
    Effect.gen(function* () {
      const { forkId } = yield* Fork.ForkContext

      const stateReader = yield* ArtifactStateReaderTag
      const state = yield* stateReader.getState()
      const previousContent = state.artifacts.get(id)?.content ?? null

      const bus = yield* WorkerBusTag<AppEvent>()
      yield* bus.publish({
        type: 'artifact_changed',
        forkId,
        id,
        previousContent,
        content,
      })

      const persistence = yield* ChatPersistence
      yield* persistence.saveArtifact(id, content).pipe(
        Effect.mapError((e) => ({ _tag: 'ArtifactError' as const, message: e.message })),
      )
      return content
    }),
})

// =============================================================================
// artifact.update — LLM-powered edit of artifact content
// =============================================================================

export const artifactUpdateTool = createTool({
  name: 'update',
  group: 'artifact',
  description: 'Update an artifact using exact find/replace. The <old> content must match the artifact exactly.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Artifact ID (logical name, e.g. "plan"; not a file path).' }),
    oldString: Schema.String.annotations({ description: 'Exact text to find in the artifact' }),
    newString: Schema.String.annotations({ description: 'Replacement text' }),
    replaceAll: Schema.optional(Schema.Boolean.annotations({ description: 'Replace all occurrences instead of requiring uniqueness' })),
  }),
  outputSchema: Schema.String,
  errorSchema: ArtifactError,
  argMapping: ['id', 'oldString', 'newString', 'replaceAll'],
  bindings: {
    xmlInput: {
      type: 'tag',
      attributes: [{ field: 'id', attr: 'id' }, { field: 'replaceAll', attr: 'replaceAll' }],
      childTags: [
        { tag: 'old', field: 'oldString' },
        { tag: 'new', field: 'newString' },
      ],
    },
    xmlOutput: { type: 'tag' } as const,
  } as const,
  execute: ({ id, oldString, newString, replaceAll }) =>
    Effect.gen(function* () {
      const { forkId } = yield* Fork.ForkContext
      yield* ensureAware(forkId, id)

      const stateReader = yield* ArtifactStateReaderTag
      const state = yield* stateReader.getState()
      const artifact = state.artifacts.get(id)

      if (!artifact) {
        return yield* Effect.fail({ _tag: 'ArtifactError' as const, message: `Artifact "${id}" does not exist` })
      }

      let applied
      try {
        applied = validateAndApply(artifact.content, oldString, newString, replaceAll ?? false)
      } catch (e) {
        return yield* Effect.fail({ _tag: 'ArtifactError' as const, message: e instanceof Error ? e.message : String(e) })
      }

      const newContent = applied.result
      logger.info({ id, oldLen: artifact.content.length, newLen: newContent.length }, '[artifact.update] find/replace applied')

      const bus = yield* WorkerBusTag<AppEvent>()
      yield* bus.publish({
        type: 'artifact_changed',
        forkId,
        id,
        previousContent: artifact.content,
        content: newContent,
      })

      const persistence = yield* ChatPersistence
      yield* persistence.saveArtifact(id, newContent).pipe(
        Effect.mapError((e) => ({ _tag: 'ArtifactError' as const, message: e.message })),
      )
      return newContent
    }),
})

// =============================================================================
// Tool Group Exports
// =============================================================================

/** Orchestrator artifact tools */
export const artifactOrchestratorTools = [artifactSyncTool, artifactReadTool, artifactWriteTool, artifactUpdateTool]

/** Agent artifact tools: read, write, update */
export const artifactAgentTools = [artifactReadTool, artifactWriteTool, artifactUpdateTool]