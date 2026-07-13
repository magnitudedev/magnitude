import { Context, Effect, Stream } from 'effect'

export type RuntimeProjectionKind = 'global' | 'forked'

export interface ProjectionIntrospection {
  readonly name: string
  readonly kind: RuntimeProjectionKind
  readonly forkId: string | null
  readonly reads: readonly string[]
  readonly signalSubscriptions: readonly string[]
  readonly state: unknown
  readonly timestamp: number
}

export interface RuntimeIntrospection {
  readonly engineName: string
  readonly schemaVersion: string
  readonly timestamp: number
  readonly projections: readonly ProjectionIntrospection[]
}

export interface RuntimeProjectionInspector {
  readonly name: string
  readonly kind: RuntimeProjectionKind
  readonly reads: readonly string[]
  readonly signalSubscriptions: readonly string[]
  readonly read: (forkId: string | null) => Effect.Effect<unknown>
  readonly changes: Stream.Stream<void>
}

export interface RuntimeIntrospectionService {
  readonly current: (forkId?: string | null) => Effect.Effect<RuntimeIntrospection>
  readonly changes: (forkId?: string | null) => Stream.Stream<RuntimeIntrospection>
}

export class RuntimeIntrospector extends Context.Tag('RuntimeIntrospector')<
  RuntimeIntrospector,
  RuntimeIntrospectionService
>() {}

const currentTime = () => Date.now()

export const makeRuntimeIntrospectionService = (
  engineName: string,
  schemaVersion: string,
  inspectors: readonly RuntimeProjectionInspector[]
): RuntimeIntrospectionService => {
  const current = (forkId: string | null = null) =>
    Effect.gen(function* () {
      const timestamp = currentTime()
      const projections = yield* Effect.forEach(inspectors, (inspector) =>
        Effect.map(inspector.read(forkId), (state): ProjectionIntrospection => ({
          name: inspector.name,
          kind: inspector.kind,
          forkId: inspector.kind === 'forked' ? forkId : null,
          reads: inspector.reads,
          signalSubscriptions: inspector.signalSubscriptions,
          state,
          timestamp,
        }))
      )

      return {
        engineName,
        schemaVersion,
        timestamp,
        projections,
      } satisfies RuntimeIntrospection
    })

  const changeStream = inspectors.length === 0
    ? Stream.never
    : Stream.mergeAll(
        inspectors.map((inspector) => inspector.changes),
        { concurrency: 'unbounded' },
      )

  return {
    current,
    changes: (forkId: string | null = null) =>
      Stream.concat(
        Stream.fromEffect(current(forkId)),
        Stream.mapEffect(changeStream, () => current(forkId)),
      ),
  }
}
