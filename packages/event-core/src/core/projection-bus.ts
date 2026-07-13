/**
 * ProjectionBus
 *
 * Handles ALL synchronous projection communication:
 * - Events: dispatched to registered event handlers
 * - Signals: dispatched to registered signal handlers
 *
 * Implements two-phase event processing:
 * - Phase 1: Event handlers run in dependency order, signals are buffered
 * - Phase 2: Buffered signals are flushed iteratively
 *
 * ALL handlers run synchronously before dispatch returns.
 * Projections cannot publish events (only emit signals).
 *
 * Dependency graph tracks:
 * - Explicit read dependencies (via `reads` config)
 * - Implicit signal dependencies (subscribing to another projection's signal)
 *
 * Generic over E - the application's event union type.
 */

import { Effect, Context, Layer, Ref, Cause, PubSub, Stream } from 'effect'
import { type BaseEvent, type Timestamped } from './event-bus-core'
import { FrameworkErrorReporter, FrameworkError, type FrameworkErrorReporterService } from './framework-error'
import {
  trackRead,
  wrapConsumersWithTracking,
  wrapStateWithProxies,
  type AddressedReadTracker,
  type ProjectionAddressedConsumers,
  type ProjectionAddressedDescriptors
} from '../projection/addressed'
import type { AddressedError } from '../addressed/errors'

const MAX_SIGNAL_FLUSH_ITERATIONS = 100
type ProjectionHandlerError = unknown

/**
 * Minimal structural shape that forked projection states must satisfy.
 * Avoids importing ForkedState from defineForked (which would create a circular import).
 */
interface ForkedProjectionState {
  readonly forks: Map<string | null, unknown>
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProjectionBusService<E extends BaseEvent> {
  /**
   * Register an event handler for this projection.
   * Handlers run in dependency order based on reads and signal subscriptions.
   */
  register: (
    handler: (event: E) => Effect.Effect<void, ProjectionHandlerError>,
    eventTypes: readonly E['type'][],
    name: string
  ) => Effect.Effect<void>

  /**
   * Register a signal handler for this projection.
   * Automatically registers a dependency on the signal's source projection.
   */
  registerSignalHandler: (
    signalName: string,
    handler: (value: unknown, sourceState: unknown) => Effect.Effect<void, ProjectionHandlerError>,
    projectionName: string
  ) => Effect.Effect<void>

  /**
   * Register an ambient handler for this projection.
   * Handlers run in dependency order using the same dependency graph as events/signals.
   */
  registerAmbientHandler: (
    ambientName: string,
    handler: (value: unknown) => Effect.Effect<void, ProjectionHandlerError>,
    projectionName: string
  ) => Effect.Effect<void>

  /**
   * Queue a signal for emission (called by projections).
   * Signals are not dispatched immediately - they're buffered until flush phase.
   */
  queueSignal: (signalName: string, value: unknown, sourceState: unknown) => Effect.Effect<void>

  /**
   * Process an event through all projections with two-phase handling.
   * Phase 1: Run event handlers in dependency order
   * Phase 2: Flush all queued signals iteratively (handlers sorted per-signal)
   */
  processEvent: (event: Timestamped<E>) => Effect.Effect<void>

  /**
   * Process an ambient change through all registered ambient handlers.
   * Ambient handlers run in dependency order and can emit signals, which are flushed iteratively.
   */
  processAmbientChange: (ambientName: string, value: unknown) => Effect.Effect<void>

  /**
   * Register a dependency edge: `from` depends on `to`.
   * Used by projections to declare read dependencies.
   */
  registerDependency: (from: string, to: string) => Effect.Effect<void>

  /**
   * Register a state getter for a projection.
   * Used by read() to access projection state synchronously.
   */
  registerStateGetter: (
    name: string,
    getter: () => unknown,
    isForked: boolean
  ) => Effect.Effect<void>

  /**
   * Register addressed state info for a projection so that consuming
   * projections can read its addressed fields through Proxies.
   */
  registerAddressedState: (
    name: string,
    info: AddressedStateInfo
  ) => Effect.Effect<void>

  /**
   * Get current state of a projection (for read()).
   * Returns the full state for standard projections.
   */
  getProjectionState: (name: string) => unknown

  /**
   * Get projection state with addressed fields wrapped as Proxies.
   * The tracker records which addresses were read during this handler,
   * keyed by (source projection, property).
   */
  getProjectionStateWithTracker: (name: string, tracker: AddressedReadTracker) => unknown

  /**
   * Get fork state of a forked projection.
   * Returns the specific fork's state, or full state if not forked.
   */
  getForkState: (name: string, forkId: string | null) => unknown

  /**
   * Get fork state with addressed fields wrapped as Proxies.
   */
  getForkStateWithTracker: (name: string, forkId: string | null, tracker: AddressedReadTracker) => unknown

  /**
   * Get addressed consumers whose direct reads record the same semantic
   * addresses as read() state Proxies.
   */
  getProjectionAddressedConsumersWithTracker: (
    name: string,
    tracker: AddressedReadTracker
  ) => unknown

  /**
   * Register a runtime consumer observer. Runtime observers are external
   * materializations: they rerun when tracked projection state or addressed
   * entries change, without becoming projections and without app events.
   */
  registerRuntimeObserver: (observerName: string) => Effect.Effect<Stream.Stream<void>>

  /**
   * Atomically replace all dependencies and pins for one runtime observer.
   * This acquires new addressed pins before releasing obsolete ones.
   */
  updateRuntimeObserverDependencies: (
    observerName: string,
    projectionNames: ReadonlySet<string>,
    addressedReads: AddressedReadTracker
  ) => Effect.Effect<void, AddressedError>

  /**
   * Release a runtime observer and all addressed pins owned by it.
   */
  releaseRuntimeObserver: (observerName: string) => Effect.Effect<void, AddressedError>

  /**
   * Notify the bus that addressed entries changed for a source projection's
   * property. Called during publish(). The flush cycle triggers a rebuild of
   * every observer whose tracked set intersects and that hasn't already run
   * since the change was recorded. Signals are never gated by this.
   */
  notifyAddressedChange: (
    sourceProjection: string,
    property: string,
    changedAddresses: ReadonlySet<string>
  ) => void

  /**
   * Replace the set of addresses a consuming projection is tracking for one
   * (source projection, property). Called after each handler invocation that
   * read that property through Proxies.
   */
  updateAddressedDependencies: (
    observerName: string,
    sourceProjection: string,
    property: string,
    addresses: ReadonlySet<string>
  ) => void

  /**
   * Replace a consumer's pins for one (source projection, property) so the
   * segments it read stay resident. Routed to the source's addressed runtime.
   */
  pinAddressedConsumer: (
    sourceProjection: string,
    property: string,
    owner: string,
    addresses: ReadonlySet<string>
  ) => Effect.Effect<void, AddressedError>

  /**
   * Validate no cycles exist in the dependency graph.
   * Should be called after all projections are registered.
   * Throws if a cycle is detected.
   */
  validateNoCycles: () => Effect.Effect<void>
}

/**
 * Info needed to wrap a projection's addressed fields with Proxies and to
 * pin consumer reads in its address spaces.
 */
export interface AddressedStateInfo {
  readonly descriptors: ProjectionAddressedDescriptors
  readonly consumers: ProjectionAddressedConsumers<ProjectionAddressedDescriptors>
  readonly consumersForScope: (scope: Iterable<string>) => ProjectionAddressedConsumers<ProjectionAddressedDescriptors>
  readonly pinConsumer: (
    property: string,
    owner: string,
    addresses: ReadonlySet<string>
  ) => Effect.Effect<void, AddressedError>
  readonly isForked: boolean
}

// Create a tag for a specific event type E
export const ProjectionBusTag = <E extends BaseEvent>() =>
  Context.GenericTag<ProjectionBusService<E>>('ProjectionBus')

const runtimeProjectionName = (handlerName: string): string =>
  handlerName.endsWith('Projection')
    ? handlerName.slice(0, -'Projection'.length)
    : handlerName

// ---------------------------------------------------------------------------
// Topological Sort
// ---------------------------------------------------------------------------

/**
 * Topologically sort handler names based on dependency graph.
 * Returns names in order such that dependencies come before dependents.
 */
function topologicalSort(
  handlerNames: readonly string[],
  dependencyGraph: Map<string, Set<string>>
): string[] {
  const names = new Set(handlerNames)
  const inDegree = new Map<string, number>()
  const edges = new Map<string, string[]>()

  // Initialize
  for (const name of names) {
    inDegree.set(name, 0)
    edges.set(name, [])
  }

  // Build in-degree counts and edge lists
  // If A depends on B, then B -> A (B must come before A)
  for (const name of names) {
    const deps = dependencyGraph.get(name) ?? new Set()
    for (const dep of deps) {
      if (names.has(dep)) {
        edges.get(dep)!.push(name)
        inDegree.set(name, (inDegree.get(name) ?? 0) + 1)
      }
    }
  }

  // Kahn's algorithm
  const queue: string[] = []
  for (const [name, degree] of inDegree) {
    if (degree === 0) queue.push(name)
  }

  const sorted: string[] = []
  while (queue.length > 0) {
    const name = queue.shift()!
    sorted.push(name)
    for (const dependent of edges.get(name)!) {
      const newDegree = inDegree.get(dependent)! - 1
      inDegree.set(dependent, newDegree)
      if (newDegree === 0) queue.push(dependent)
    }
  }

  // If we couldn't sort all names, there's a cycle
  // (This shouldn't happen if validateNoCycles was called, but handle gracefully)
  if (sorted.length !== names.size) {
    // Return original order as fallback
    return [...handlerNames]
  }

  return sorted
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

export function makeProjectionBusLayer<E extends BaseEvent>(): Layer.Layer<
  ProjectionBusService<E>,
  never,
  FrameworkErrorReporterService
> {
  const Tag = ProjectionBusTag<E>()

  return Layer.scoped(Tag, Effect.gen(function* () {
    const reporter = yield* FrameworkErrorReporter
    // Event handlers in registration order (will be sorted by dependencies)
    type EventHandlerItem = {
      name: string
      eventTypes: readonly E['type'][]
      handler: (event: E) => Effect.Effect<void, ProjectionHandlerError>
    }
    const eventHandlersRef = yield* Ref.make<EventHandlerItem[]>([])

    // Signal handlers: Map<signalName, Array<{name, handler}>>
    type SignalHandlerItem = {
      name: string
      handler: (value: unknown, sourceState: unknown) => Effect.Effect<void, ProjectionHandlerError>
    }
    const signalHandlersRef = yield* Ref.make<Map<string, SignalHandlerItem[]>>(new Map())

    // Ambient handlers: Map<ambientName, Array<{name, handler}>>
    type AmbientHandlerItem = {
      name: string
      handler: (value: unknown) => Effect.Effect<void, ProjectionHandlerError>
    }
    const ambientHandlersRef = yield* Ref.make<Map<string, AmbientHandlerItem[]>>(new Map())

    // Global signal queue for two-phase processing
    type QueuedSignal = {
      signalName: string
      value: unknown
      sourceState: unknown
      eventTimestamp: number
    }
    const signalQueueRef = yield* Ref.make<QueuedSignal[]>([])

    // Dependency graph: Map<projectionName, Set<dependsOnProjectionNames>>
    const dependencyGraphRef = yield* Ref.make<Map<string, Set<string>>>(new Map())

    // State getters: Map<projectionName, { getter, isForked }>
    type StateGetterEntry = {
      getter: () => unknown
      isForked: boolean
    }
    const stateGetters = new Map<string, StateGetterEntry>()

    // Addressed state info: Map<projectionName, AddressedStateInfo>
    const addressedStateInfo = new Map<string, AddressedStateInfo>()

    // Addressed change tracking:
    // - pendingAddressedChanges: per (source, property), the changed addresses
    //   and the tick of the latest change — consumed by the trigger phase
    // - addressedDependencies: Map<observer, Map<source, Map<property, Set>>>
    //   — each observer's tracked reads
    // - observerLastRunTick: when each observer's handler last completed.
    //   Reads always see committed state, so an observer that ran after a
    //   change was recorded has already read the fresh content — triggering
    //   it again would be a pure duplicate.
    // - firstDeclaredSignalHandler: the trigger callback per observer. The
    //   contract: signal handlers of a projection that reads addressed
    //   dependency content are source-agnostic rebuilds, safe to re-run.
    interface PendingAddressedChange {
      readonly addresses: Set<string>
      tick: number
    }
    const pendingAddressedChanges = new Map<string, Map<string, PendingAddressedChange>>()
    const addressedDependencies = new Map<string, Map<string, Map<string, ReadonlySet<string>>>>()
    const observerLastRunTick = new Map<string, number>()
    const firstDeclaredSignalHandler = new Map<string, SignalHandlerItem['handler']>()
    let changeTick = 0

    interface RuntimeObserverState {
      readonly projectionNames: Set<string>
      readonly addressedReads: AddressedReadTracker
      readonly pubsub: PubSub.PubSub<void>
    }
    const runtimeObservers = new Map<string, RuntimeObserverState>()
    const runtimeChangedProjections = new Set<string>()
    const runtimeAddressedChanges = new Map<string, Map<string, Set<string>>>()

    // Cache for sorted handler order (invalidated when handlers registered)
    let cachedEventHandlerOrder: string[] | null = null
    const signalHandlerOrderCache = new Map<string, string[]>()

    const recordObserverRun = (name: string): void => {
      observerLastRunTick.set(name, changeTick)
    }

    const recordRuntimeProjectionChange = (name: string): void => {
      runtimeChangedProjections.add(runtimeProjectionName(name))
    }

    const observerTracksAddressedChange = (
      reads: AddressedReadTracker,
      source: string,
      property: string,
      addresses: ReadonlySet<string>
    ): boolean => {
      const tracked = reads.get(source)?.get(property)
      if (!tracked) return false
      for (const address of addresses) {
        if (tracked.has(address)) return true
      }
      return false
    }

    const publishRuntimeObserverChanges = Effect.gen(function* () {
      if (runtimeChangedProjections.size === 0 && runtimeAddressedChanges.size === 0) return

      const observers = [...runtimeObservers.values()]
      for (const observer of observers) {
        let shouldPublish = false

        for (const projectionName of observer.projectionNames) {
          if (runtimeChangedProjections.has(projectionName)) {
            shouldPublish = true
            break
          }
        }

        if (!shouldPublish) {
          for (const [source, perProperty] of runtimeAddressedChanges) {
            for (const [property, addresses] of perProperty) {
              if (observerTracksAddressedChange(observer.addressedReads, source, property, addresses)) {
                shouldPublish = true
                break
              }
            }
            if (shouldPublish) break
          }
        }

        if (shouldPublish) {
          yield* PubSub.publish(observer.pubsub, undefined)
        }
      }

      runtimeChangedProjections.clear()
      runtimeAddressedChanges.clear()
    })

    const copyReadTracker = (tracker: AddressedReadTracker): AddressedReadTracker => {
      const copy = new Map<string, Map<string, Set<string>>>()
      for (const [source, perProperty] of tracker) {
        const perPropertyCopy = new Map<string, Set<string>>()
        for (const [property, addresses] of perProperty) {
          perPropertyCopy.set(property, new Set(addresses))
        }
        copy.set(source, perPropertyCopy)
      }
      return copy
    }

    const addressedReadKeys = (
      left: AddressedReadTracker,
      right: AddressedReadTracker
    ): readonly (readonly [string, string])[] => {
      const keys = new Map<string, readonly [string, string]>()
      for (const tracker of [left, right]) {
        for (const [source, perProperty] of tracker) {
          for (const property of perProperty.keys()) {
            keys.set(`${source}\u0000${property}`, [source, property])
          }
        }
      }
      return [...keys.values()]
    }

    const pinRuntimeAddressedReads = (
      owner: string,
      reads: AddressedReadTracker
    ): Effect.Effect<void, AddressedError> =>
      Effect.forEach(
        reads,
        ([source, perProperty]) =>
          Effect.forEach(
            perProperty,
            ([property, addresses]) => {
              const info = addressedStateInfo.get(source)
              if (!info) {
                return Effect.dieMessage(
                  `No addressed state registered for projection "${source}"`
                )
              }
              return info.pinConsumer(property, owner, addresses)
            },
            { discard: true }
          ),
        { discard: true }
      )

    const replaceRuntimeAddressedReads = (
      owner: string,
      previous: AddressedReadTracker,
      next: AddressedReadTracker
    ): Effect.Effect<void, AddressedError> =>
      Effect.gen(function* () {
        const nextOwner = `${owner}:next`
        yield* pinRuntimeAddressedReads(nextOwner, next).pipe(
          Effect.onError(() =>
            Effect.forEach(
              addressedReadKeys(new Map(), next),
              ([source, property]) => {
                const info = addressedStateInfo.get(source)
                return info ? info.pinConsumer(property, nextOwner, new Set()) : Effect.void
              },
              { discard: true }
            ).pipe(Effect.ignore)
          )
        )

        for (const [source, property] of addressedReadKeys(previous, next)) {
          const info = addressedStateInfo.get(source)
          if (!info) {
            return yield* Effect.dieMessage(
              `No addressed state registered for projection "${source}"`
            )
          }
          const addresses = next.get(source)?.get(property) ?? new Set<string>()
          yield* info.pinConsumer(property, owner, addresses)
          yield* info.pinConsumer(property, nextOwner, new Set())
        }
      })

    /**
     * Run addressed-change triggers for the current pending set. For each
     * observer whose tracked set for a changed (source, property) intersects
     * the changed addresses — and whose last run predates the change — its
     * first-declared signal handler is re-invoked to rebuild from fresh deps.
     * Consumes the pending set; triggers may queue signals or record new
     * pending changes, which the enclosing flush loop picks up.
     */
    const runAddressedTriggers = Effect.gen(function* () {
      const graph = yield* Ref.get(dependencyGraphRef)
      const pending = new Map(pendingAddressedChanges)
      pendingAddressedChanges.clear()

      const observerNames = topologicalSort([...addressedDependencies.keys()], graph)

      for (const observerName of observerNames) {
        const perSource = addressedDependencies.get(observerName)
        if (!perSource) continue
        const lastRun = observerLastRunTick.get(observerName) ?? -1

        let triggeringSource: string | null = null
        for (const [source, perProperty] of pending) {
          const trackedPerProperty = perSource.get(source)
          if (!trackedPerProperty) continue
          for (const [property, change] of perProperty) {
            if (change.tick <= lastRun) continue
            const tracked = trackedPerProperty.get(property)
            if (!tracked) continue
            for (const address of change.addresses) {
              if (tracked.has(address)) {
                triggeringSource = source
                break
              }
            }
            if (triggeringSource) break
          }
          if (triggeringSource) break
        }
        if (!triggeringSource) continue

        const handler = firstDeclaredSignalHandler.get(observerName)
        if (!handler) continue

        yield* handler({ timestamp: currentEventTimestamp }, undefined).pipe(
          Effect.catchAllCause((cause) =>
            reporter.report(FrameworkError.ProjectionSignalHandlerError({
              projectionName: observerName,
              signalName: `${triggeringSource}/addressedChange`,
              cause
            }))
          )
        )
        recordObserverRun(observerName)
        recordRuntimeProjectionChange(observerName)
      }
    })

    const flushSignalQueue = Effect.gen(function* () {
      let iterations = 0
      const graph = yield* Ref.get(dependencyGraphRef)

      while (true) {
        if (iterations++ >= MAX_SIGNAL_FLUSH_ITERATIONS) {
          yield* Effect.logWarning(`Signal flush exceeded ${MAX_SIGNAL_FLUSH_ITERATIONS} iterations, possible infinite loop`)
          break
        }

        const queue = yield* Ref.getAndSet(signalQueueRef, [])
        if (queue.length > 0) {
          const signalHandlers = yield* Ref.get(signalHandlersRef)

          for (const { signalName, value, sourceState, eventTimestamp } of queue) {
            const timestampedValue = Object.assign({}, value, { timestamp: eventTimestamp })
            const handlers = signalHandlers.get(signalName) ?? []
            if (handlers.length === 0) continue

            let sortedNames = signalHandlerOrderCache.get(signalName)
            if (!sortedNames) {
              const handlerNames = handlers.map(h => h.name)
              sortedNames = topologicalSort(handlerNames, graph)
              signalHandlerOrderCache.set(signalName, sortedNames)
            }

            const nameToSignalHandler = new Map(handlers.map(h => [h.name, h]))

            for (const name of sortedNames) {
              const handlerItem = nameToSignalHandler.get(name)
              if (handlerItem) {
                yield* handlerItem.handler(timestampedValue, sourceState).pipe(
                  Effect.catchAllCause((cause) =>
                    reporter.report(FrameworkError.ProjectionSignalHandlerError({
                      projectionName: name,
                      signalName,
                      cause
                    }))
                  )
                )
                recordObserverRun(name)
                recordRuntimeProjectionChange(name)
              }
            }
          }
          continue
        }

        // Queue drained: run addressed-change triggers. These cover content
        // changes that arrived with no accompanying signal (e.g. a chunk
        // appended to a tracked segment, or a rollover marking the sequence
        // sentinel). Triggers may queue signals — loop back for them.
        if (pendingAddressedChanges.size > 0) {
          yield* runAddressedTriggers
          continue
        }

        break
      }
    })

    let currentEventTimestamp: number = Date.now()

    return {
      register: (
        handler: (event: E) => Effect.Effect<void, ProjectionHandlerError>,
        eventTypes: readonly E['type'][],
        name: string
      ) => Effect.gen(function* () {
        yield* Ref.update(eventHandlersRef, (handlers) => [
          ...handlers,
          { name, eventTypes, handler }
        ])
        // Invalidate cache
        cachedEventHandlerOrder = null
      }),

      registerSignalHandler: (
        signalName: string,
        handler: (value: unknown, sourceState: unknown) => Effect.Effect<void, ProjectionHandlerError>,
        projectionName: string
      ) => Effect.gen(function* () {
        // Extract source projection from signal name (e.g., "Fork/forkCreated" -> "Fork")
        const sourceProjection = signalName.split('/')[0]

        // Register dependency: this projection depends on the signal's source
        yield* Ref.update(dependencyGraphRef, (graph) => {
          const deps = graph.get(projectionName) ?? new Set()
          deps.add(sourceProjection)
          return new Map(graph).set(projectionName, deps)
        })

        // Register the handler
        yield* Ref.update(signalHandlersRef, (map) => {
          const existing = map.get(signalName) ?? []
          const newMap = new Map(map)
          newMap.set(signalName, [...existing, { name: projectionName, handler }])
          return newMap
        })

        // The first handler a projection registers is its first-declared
        // signal handler (registration follows declaration order). It doubles
        // as the projection's addressed-change trigger.
        if (!firstDeclaredSignalHandler.has(projectionName)) {
          firstDeclaredSignalHandler.set(projectionName, handler)
        }

        // Invalidate signal handler cache for this signal
        signalHandlerOrderCache.delete(signalName)
      }),

      registerAmbientHandler: (
        ambientName: string,
        handler: (value: unknown) => Effect.Effect<void, ProjectionHandlerError>,
        projectionName: string
      ) =>
        Ref.update(ambientHandlersRef, (map) => {
          const existing = map.get(ambientName) ?? []
          const newMap = new Map(map)
          newMap.set(ambientName, [...existing, { name: projectionName, handler }])
          return newMap
        }),

      queueSignal: (signalName: string, value: unknown, sourceState: unknown) =>
        Ref.update(signalQueueRef, (queue) => [...queue, { signalName, value, sourceState, eventTimestamp: currentEventTimestamp }]),

      registerDependency: (from: string, to: string) => Effect.gen(function* () {
        yield* Ref.update(dependencyGraphRef, (graph) => {
          const deps = graph.get(from) ?? new Set()
          deps.add(to)
          return new Map(graph).set(from, deps)
        })
        // Invalidate caches
        cachedEventHandlerOrder = null
        signalHandlerOrderCache.clear()
      }),

      registerStateGetter: (name: string, getter: () => unknown, isForked: boolean) =>
        Effect.sync(() => {
          stateGetters.set(name, { getter, isForked })
        }),

      registerAddressedState: (name: string, info: AddressedStateInfo) =>
        Effect.sync(() => {
          addressedStateInfo.set(name, info)
        }),

      getProjectionState: (name: string) => {
        const entry = stateGetters.get(name)
        if (!entry) {
          throw new Error(`No state getter registered for projection "${name}"`)
        }
        return entry.getter()
      },

      getProjectionStateWithTracker: (name: string, tracker: AddressedReadTracker) => {
        const entry = stateGetters.get(name)
        if (!entry) {
          throw new Error(`No state getter registered for projection "${name}"`)
        }
        const info = addressedStateInfo.get(name)
        const state = entry.getter()
        if (!info) return state
        const record = (property: string, address: string) => trackRead(tracker, name, property, address)
        if (info.isForked) {
          // Wrap each fork's addressed fields with Proxies
          const forkedState = state as ForkedProjectionState
          const wrappedForks = new Map<string | null, unknown>()
          for (const [forkId, forkState] of forkedState.forks) {
            const consumers = info.consumersForScope(['forks', forkId ?? 'root'])
            wrappedForks.set(forkId, wrapStateWithProxies(forkState, info.descriptors, consumers, record))
          }
          return { ...forkedState, forks: wrappedForks }
        }
        return wrapStateWithProxies(state, info.descriptors, info.consumers, record)
      },

      getForkState: (name: string, forkId: string | null) => {
        const entry = stateGetters.get(name)
        if (!entry) {
          throw new Error(`No state getter registered for projection "${name}"`)
        }
        if (!entry.isForked) {
          return entry.getter()
        }
        const state = entry.getter() as ForkedProjectionState
        return state.forks.get(forkId)
      },

      getForkStateWithTracker: (name: string, forkId: string | null, tracker: AddressedReadTracker) => {
        const entry = stateGetters.get(name)
        if (!entry) {
          throw new Error(`No state getter registered for projection "${name}"`)
        }
        const info = addressedStateInfo.get(name)
        const record = (property: string, address: string) => trackRead(tracker, name, property, address)
        if (!entry.isForked) {
          const state = entry.getter()
          if (!info) return state
          return wrapStateWithProxies(state, info.descriptors, info.consumers, record)
        }
        const state = entry.getter() as ForkedProjectionState
        const forkState = state.forks.get(forkId)
        if (forkState === undefined) return undefined
        if (!info) return forkState
        const consumers = info.consumersForScope(['forks', forkId ?? 'root'])
        return wrapStateWithProxies(forkState, info.descriptors, consumers, record)
      },

      getProjectionAddressedConsumersWithTracker: (name: string, tracker: AddressedReadTracker) => {
        const info = addressedStateInfo.get(name)
        if (!info) {
          return {}
        }
        const trackingConsumers = (scope: Iterable<string>) => {
          const consumers = info.consumersForScope(scope)
          return wrapConsumersWithTracking(
            info.descriptors,
            consumers,
            (property, address) => trackRead(tracker, name, property, address)
          )
        }
        if (info.isForked) {
          return {
            ...trackingConsumers([]),
            forFork: (forkId: string | null) =>
              trackingConsumers(['forks', forkId ?? 'root'])
          }
        }
        return trackingConsumers([])
      },

      registerRuntimeObserver: (observerName: string) =>
        Effect.gen(function* () {
          const existing = runtimeObservers.get(observerName)
          if (existing) {
            yield* PubSub.shutdown(existing.pubsub)
          }
          const pubsub = yield* PubSub.unbounded<void>()
          runtimeObservers.set(observerName, {
            projectionNames: new Set(),
            addressedReads: new Map(),
            pubsub
          })
          return Stream.fromPubSub(pubsub)
        }),

      updateRuntimeObserverDependencies: (observerName, projectionNames, addressedReads) =>
        Effect.gen(function* () {
          const observer = runtimeObservers.get(observerName)
          if (!observer) {
            return yield* Effect.dieMessage(`Unknown runtime observer "${observerName}"`)
          }
          const owner = `runtime:${observerName}`
          const nextAddressedReads = copyReadTracker(addressedReads)
          yield* replaceRuntimeAddressedReads(owner, observer.addressedReads, nextAddressedReads)
          runtimeObservers.set(observerName, {
            projectionNames: new Set(projectionNames),
            addressedReads: nextAddressedReads,
            pubsub: observer.pubsub
          })
        }),

      releaseRuntimeObserver: (observerName) =>
        Effect.gen(function* () {
          const observer = runtimeObservers.get(observerName)
          if (!observer) return

          const owner = `runtime:${observerName}`
          yield* replaceRuntimeAddressedReads(owner, observer.addressedReads, new Map())
          runtimeObservers.delete(observerName)
          yield* PubSub.shutdown(observer.pubsub)
        }),

      notifyAddressedChange: (sourceProjection: string, property: string, changedAddresses: ReadonlySet<string>) => {
        if (changedAddresses.size === 0) return
        changeTick += 1
        let perProperty = pendingAddressedChanges.get(sourceProjection)
        if (!perProperty) {
          perProperty = new Map()
          pendingAddressedChanges.set(sourceProjection, perProperty)
        }
        const existing = perProperty.get(property)
        if (existing) {
          for (const addr of changedAddresses) existing.addresses.add(addr)
          existing.tick = changeTick
        } else {
          perProperty.set(property, { addresses: new Set(changedAddresses), tick: changeTick })
        }

        let runtimePerProperty = runtimeAddressedChanges.get(sourceProjection)
        if (!runtimePerProperty) {
          runtimePerProperty = new Map()
          runtimeAddressedChanges.set(sourceProjection, runtimePerProperty)
        }
        const runtimeAddresses = runtimePerProperty.get(property) ?? new Set<string>()
        for (const address of changedAddresses) runtimeAddresses.add(address)
        runtimePerProperty.set(property, runtimeAddresses)
      },

      updateAddressedDependencies: (observerName: string, sourceProjection: string, property: string, addresses: ReadonlySet<string>) => {
        let perSource = addressedDependencies.get(observerName)
        if (!perSource) {
          perSource = new Map()
          addressedDependencies.set(observerName, perSource)
        }
        let perProperty = perSource.get(sourceProjection)
        if (!perProperty) {
          perProperty = new Map()
          perSource.set(sourceProjection, perProperty)
        }
        perProperty.set(property, new Set(addresses))
      },

      pinAddressedConsumer: (sourceProjection: string, property: string, owner: string, addresses: ReadonlySet<string>) => {
        const info = addressedStateInfo.get(sourceProjection)
        if (!info) {
          return Effect.dieMessage(
            `No addressed state registered for projection "${sourceProjection}"`
          )
        }
        return info.pinConsumer(property, owner, addresses)
      },

      validateNoCycles: () => Effect.gen(function* () {
        const graph = yield* Ref.get(dependencyGraphRef)
        const visited = new Set<string>()
        const inStack = new Set<string>()

        function dfs(node: string, path: string[]): string[] | null {
          if (inStack.has(node)) {
            return [...path, node]
          }
          if (visited.has(node)) {
            return null
          }

          visited.add(node)
          inStack.add(node)

          for (const dep of graph.get(node) ?? []) {
            const cycle = dfs(dep, [...path, node])
            if (cycle) return cycle
          }

          inStack.delete(node)
          return null
        }

        for (const node of graph.keys()) {
          const cycle = dfs(node, [])
          if (cycle) {
            throw new Error(`Circular dependency detected: ${cycle.join(' → ')}`)
          }
        }
      }),

      processEvent: (event: Timestamped<E>) => Effect.gen(function* () {
        currentEventTimestamp = event.timestamp
        const handlers = yield* Ref.get(eventHandlersRef)
        const graph = yield* Ref.get(dependencyGraphRef)

        // Get or compute sorted order for event handlers
        if (!cachedEventHandlerOrder) {
          const handlerNames = handlers.map(h => h.name)
          cachedEventHandlerOrder = topologicalSort(handlerNames, graph)
        }

        // Build name -> handler map
        const nameToHandler = new Map(handlers.map(h => [h.name, h]))

        // Phase 1: Run event handlers in dependency order
        for (const name of cachedEventHandlerOrder) {
          const handlerItem = nameToHandler.get(name)
          if (handlerItem && handlerItem.eventTypes.includes(event.type)) {
            yield* handlerItem.handler(event).pipe(
              Effect.catchAllCause((cause) =>
                reporter.report(FrameworkError.ProjectionEventHandlerError({
                  projectionName: name,
                  eventType: event.type,
                  cause
                }))
              )
            )
            recordObserverRun(name)
            recordRuntimeProjectionChange(name)
          }
        }

        // Phase 2: Flush signals iteratively
        yield* flushSignalQueue
        yield* publishRuntimeObserverChanges
      }),

      processAmbientChange: (ambientName: string, value: unknown) => Effect.gen(function* () {
        const graph = yield* Ref.get(dependencyGraphRef)
        const ambientHandlers = yield* Ref.get(ambientHandlersRef)
        const handlers = ambientHandlers.get(ambientName) ?? []
        if (handlers.length === 0) return

        const handlerNames = handlers.map(h => h.name)
        const sortedNames = topologicalSort(handlerNames, graph)
        const nameToHandler = new Map(handlers.map(h => [h.name, h]))

        for (const name of sortedNames) {
          const handlerItem = nameToHandler.get(name)
          if (handlerItem) {
            yield* handlerItem.handler(value).pipe(
              Effect.catchAllCause((cause) =>
                reporter.report(FrameworkError.ProjectionSignalHandlerError({
                  projectionName: name,
                  signalName: `ambient:${ambientName}`,
                  cause
                }))
              )
            )
            recordObserverRun(name)
            recordRuntimeProjectionChange(name)
          }
        }

        yield* flushSignalQueue
        yield* publishRuntimeObserverChanges
      })
    }
  }))
}
