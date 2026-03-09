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

import { Effect, Context, Layer, Ref, Cause } from 'effect'
import { type BaseEvent, type Timestamped } from './event-bus-core'
import { FrameworkErrorReporter, FrameworkError, type FrameworkErrorReporterService } from './framework-error'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProjectionBusService<E extends BaseEvent> {
  /**
   * Register an event handler for this projection.
   * Handlers run in dependency order based on reads and signal subscriptions.
   */
  register: (
    handler: (event: E) => Effect.Effect<void>,
    eventTypes: readonly E['type'][],
    name: string
  ) => Effect.Effect<void>

  /**
   * Register a signal handler for this projection.
   * Automatically registers a dependency on the signal's source projection.
   */
  registerSignalHandler: (
    signalName: string,
    handler: (value: unknown, sourceState: unknown) => Effect.Effect<void>,
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
   * Get current state of a projection (for read()).
   * Returns the full state for standard projections.
   */
  getProjectionState: (name: string) => unknown

  /**
   * Get fork state of a forked projection.
   * Returns the specific fork's state, or full state if not forked.
   */
  getForkState: (name: string, forkId: string | null) => unknown

  /**
   * Validate no cycles exist in the dependency graph.
   * Should be called after all projections are registered.
   * Throws if a cycle is detected.
   */
  validateNoCycles: () => Effect.Effect<void>
}

// Create a tag for a specific event type E
export const ProjectionBusTag = <E extends BaseEvent>() =>
  Context.GenericTag<ProjectionBusService<E>>('ProjectionBus')

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
      handler: (event: E) => Effect.Effect<void>
    }
    const eventHandlersRef = yield* Ref.make<EventHandlerItem[]>([])

    // Signal handlers: Map<signalName, Array<{name, handler}>>
    type SignalHandlerItem = {
      name: string
      handler: (value: unknown, sourceState: unknown) => Effect.Effect<void>
    }
    const signalHandlersRef = yield* Ref.make<Map<string, SignalHandlerItem[]>>(new Map())

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

    // Cache for sorted handler order (invalidated when handlers registered)
    let cachedEventHandlerOrder: string[] | null = null
    const signalHandlerOrderCache = new Map<string, string[]>()

    // Helper to get dependency graph synchronously
    const getDependencyGraph = () => Effect.runSync(Ref.get(dependencyGraphRef))

    let currentEventTimestamp: number = Date.now()

    return {
      register: (
        handler: (event: E) => Effect.Effect<void>,
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
        handler: (value: unknown, sourceState: unknown) => Effect.Effect<void>,
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

        // Invalidate signal handler cache for this signal
        signalHandlerOrderCache.delete(signalName)
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

      getProjectionState: (name: string) => {
        const entry = stateGetters.get(name)
        if (!entry) {
          throw new Error(`No state getter registered for projection "${name}"`)
        }
        return entry.getter()
      },

      getForkState: (name: string, forkId: string | null) => {
        const entry = stateGetters.get(name)
        if (!entry) {
          throw new Error(`No state getter registered for projection "${name}"`)
        }
        if (!entry.isForked) {
          return entry.getter()
        }
        const state = entry.getter() as { forks: Map<string | null, unknown> }
        return state.forks.get(forkId)
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
        const graph = getDependencyGraph()

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
          }
        }

        // Phase 2: Flush signals iteratively
        let iterations = 0
        const maxIterations = 100 // Safety limit for infinite loops

        while (true) {
          const queue = yield* Ref.getAndSet(signalQueueRef, [])
          if (queue.length === 0) break

          if (iterations++ >= maxIterations) {
            yield* Effect.logWarning(`Signal flush exceeded ${maxIterations} iterations, possible infinite loop`)
            break
          }

          const signalHandlers = yield* Ref.get(signalHandlersRef)

          for (const { signalName, value, sourceState, eventTimestamp } of queue) {
            const timestampedValue = Object.assign({}, value, { timestamp: eventTimestamp })
            const handlers = signalHandlers.get(signalName) ?? []
            if (handlers.length === 0) continue

            // Get or compute sorted order for this signal's handlers
            let sortedNames = signalHandlerOrderCache.get(signalName)
            if (!sortedNames) {
              const handlerNames = handlers.map(h => h.name)
              sortedNames = topologicalSort(handlerNames, graph)
              signalHandlerOrderCache.set(signalName, sortedNames)
            }

            // Build name -> handler map for this signal
            const nameToSignalHandler = new Map(handlers.map(h => [h.name, h]))

            // Run handlers in sorted order
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
              }
            }
          }
        }
      })
    }
  }))
}
