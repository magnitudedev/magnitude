import { Context, Effect, Fiber, Layer, ManagedRuntime } from 'effect'
import type { AddressedError } from '../addressed/errors'
import type { BaseEvent } from '../core/event-bus-core'
import type { FrameworkError } from '../core/framework-error'
import type { Signal } from '../signal/define'
import type { Shape, ExposeConfig } from './make'

/**
 * A projection that has a Tag for state access (standard or forked).
 * This is intentionally structural so the Promise adapter does not need to
 * know projection implementation details.
 */
type ProjectionWithTag =
  | {
    readonly Tag: Context.Tag<unknown, unknown>
    readonly isForked: false
  }
  | {
    readonly Tag: Context.Tag<unknown, unknown>
    readonly isForked: true
  }

/** Extract the value type from a Signal<T> */
type SignalValue<T> = T extends Signal<infer V, infer _SourceState> ? V : never

/**
 * Extract state type from a regular projection's Tag (has .get)
 */
type ProjectionStateFromTag<T> =
  T extends { readonly Tag: Context.Tag<infer _Id, infer Service> }
    ? Service extends { readonly get: Effect.Effect<infer S, infer _E, infer _R> }
      ? S
      : never
    : never

/**
 * Extract fork state type from a forked projection's Tag (has .getFork)
 */
type ForkedProjectionStateFromTag<T> =
  T extends { readonly Tag: Context.Tag<infer _Id, infer Service> }
    ? Service extends { readonly getFork: (forkId: string | null) => Effect.Effect<infer S, infer _E, infer _R> }
      ? S
      : never
    : never

/**
 * Check if a projection is forked (has getFork method)
 */
type IsForkedProjection<T> =
  T extends { readonly Tag: Context.Tag<infer _Id, infer Service> }
    ? Service extends { readonly getFork: (forkId: string | null) => Effect.Effect<infer _S, infer _E, infer _R> }
      ? true
      : false
    : false

/** Client signal subscription API */
type ClientSignals<T extends Record<string, unknown>> = {
  [K in keyof T]: (callback: (value: SignalValue<T[K]>) => void) => () => void
}

/** Client state API for regular projections */
type RegularClientState<T> = {
  get: () => Promise<ProjectionStateFromTag<T>>
  subscribe: (callback: (state: ProjectionStateFromTag<T>) => void) => () => void
}

/** Client state API for forked projections */
type ForkedClientState<T> = {
  getFork: (forkId: string | null) => Promise<ForkedProjectionStateFromTag<T>>
  subscribeFork: (forkId: string | null, callback: (state: ForkedProjectionStateFromTag<T>) => void) => () => void
}

/** Client state subscription API - different interface for regular vs forked */
type ClientState<T extends Record<string, unknown>> = {
  [K in keyof T]: IsForkedProjection<T[K]> extends true
    ? ForkedClientState<T[K]>
    : RegularClientState<T[K]>
}

/** The Promise client interface returned by createClient */
export interface Client<
  TEvent extends BaseEvent,
  TExpose extends ExposeConfig,
  TRuntimeServices = never
> {
  /** Subscribe to exposed signals */
  readonly on: TExpose['signals'] extends Record<string, unknown>
    ? ClientSignals<TExpose['signals']>
    : Record<string, never>

  /** Access exposed state */
  readonly state: TExpose['state'] extends Record<string, unknown>
    ? ClientState<TExpose['state']>
    : Record<string, never>

  /** Send an event to the agent */
  readonly send: (event: TEvent) => Promise<void>

  /** Subscribe to all events flowing through the bus */
  readonly onEvent: (callback: (event: TEvent) => void) => () => void

  /** Interrupt the agent - stops streaming and resets state */
  readonly interrupt: () => Promise<void>

  /**
   * Run an Effect within the agent's managed runtime.
   * Provides access to all internal services (projections, workers, core services).
   */
  readonly runEffect: <A, E>(effect: Effect.Effect<A, E, TRuntimeServices>) => Promise<A>

  /** Subscribe to framework errors (handler failures, sink errors, etc.) */
  readonly onError: (callback: (error: FrameworkError) => void) => () => void

  /** Dispose the client and cleanup resources */
  readonly dispose: () => Promise<void>
}

export async function createManagedClient<
  TEvent extends BaseEvent,
  TExpose extends ExposeConfig,
  TAllServices,
  TExternalReqs
>(options: {
  readonly engineLayer: Layer.Layer<TAllServices | Shape<TEvent, TExpose>, AddressedError, TExternalReqs>
  readonly requirementsLayer?: Layer.Layer<TExternalReqs, never, never>
  readonly expose: TExpose
  readonly getEngine: (
    context: Context.Context<TAllServices | Shape<TEvent, TExpose>>
  ) => Shape<TEvent, TExpose>
}): Promise<Client<TEvent, TExpose, TAllServices>> {
  const ReqLayer = (options.requirementsLayer ?? Layer.empty) as Layer.Layer<TExternalReqs, never, never>
  const FinalLayer = Layer.provideMerge(options.engineLayer, ReqLayer)
  const runtime = ManagedRuntime.make(FinalLayer)

  const rt = await runtime.runtime()
  const engine = options.getEngine(rt.context)

  const fiberPromises = new Map<string, Promise<Fiber.RuntimeFiber<void, never>>>()
  let disposed = false

  const subscribe = (
    setupEffect: Effect.Effect<Fiber.RuntimeFiber<void, never>>,
    key: string
  ): (() => void) => {
    const promise = runtime.runPromise(setupEffect)
    fiberPromises.set(key, promise)
    promise
      .then((fiber) => fiber.addObserver(() => fiberPromises.delete(key)))
      .catch(() => fiberPromises.delete(key))

    return () => {
      const promise = fiberPromises.get(key)
      if (!promise) return

      fiberPromises.delete(key)
      promise
        .then((fiber) => {
          Effect.runPromise(Fiber.interrupt(fiber)).catch(() => {})
        })
        .catch(() => {})
    }
  }

  const onHandlers: Record<string, (callback: (value: unknown) => void) => () => void> = {}
  if (options.expose.signals) {
    for (const name of Object.keys(options.expose.signals)) {
      onHandlers[name] = (callback) =>
        subscribe(engine.subscribeSignal(name, callback), `signal:${name}`)
    }
  }

  const stateHandlers: Record<string, unknown> = {}
  if (options.expose.state) {
    for (const [name, projection] of Object.entries(options.expose.state)) {
      const proj = projection as ProjectionWithTag
      if (proj.isForked) {
        stateHandlers[name] = {
          getFork: (forkId: string | null) =>
            runtime.runPromise(engine.stateGetFork(name, forkId)),
          subscribeFork: (forkId: string | null, callback: (state: unknown) => void) =>
            subscribe(engine.subscribeStateFork(name, forkId, callback), `state:${name}:fork:${forkId ?? 'null'}`)
        }
      } else {
        stateHandlers[name] = {
          get: () => runtime.runPromise(engine.stateGet(name)),
          subscribe: (callback: (state: unknown) => void) =>
            subscribe(engine.subscribeState(name, callback), `state:${name}`)
        }
      }
    }
  }

  return {
    on: onHandlers as Client<TEvent, TExpose, TAllServices>['on'],
    state: stateHandlers as Client<TEvent, TExpose, TAllServices>['state'],
    send: (event) => runtime.runPromise(engine.send(event)),
    onEvent: (callback) => subscribe(engine.subscribeEvent(callback), 'onEvent'),
    onError: (callback) => subscribe(engine.subscribeError(callback), 'onError'),
    runEffect: <A, E>(effect: Effect.Effect<A, E, TAllServices>) =>
      runtime.runPromise(effect),
    interrupt: () => runtime.runPromise(engine.interrupt()),
    dispose: async () => {
      if (disposed) return
      disposed = true
      await runtime.dispose()
    }
  }
}
