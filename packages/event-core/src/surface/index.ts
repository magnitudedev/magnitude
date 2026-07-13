import { Context, Effect, Fiber, Layer, ManagedRuntime, PubSub, Scope, Stream, Schema } from 'effect'
import type { ProjectionInstance } from '../projection/define'
import type { ForkedProjectionInstance } from '../projection/defineForked'
import { Signal as CoreSignal } from '../signal/define'

export type Command<
  TArgs extends readonly unknown[],
  A,
  E,
  R
> = {
  readonly _tag: 'Command'
  readonly run: (...args: TArgs) => Effect.Effect<A, E, R>
}

export type StreamSignal<A, E, R> = {
  readonly _tag: 'Signal'
  readonly kind: 'stream'
  readonly stream: Stream.Stream<A, E, R>
}

export type FnSignal<TArgs extends readonly unknown[], A, E, R> = {
  readonly _tag: 'Signal'
  readonly kind: 'fn'
  readonly stream: (...args: TArgs) => Stream.Stream<A, E, R>
}

export type Signal<
  TArgs extends readonly unknown[],
  A,
  E,
  R
> = StreamSignal<A, E, R> | FnSignal<TArgs, A, E, R>

export type Host<TServices, TSurface, E = never> = {
  readonly layer: Layer.Layer<TServices, E, never>
  readonly surface: TSurface
}

type MissingRequirements<TRequired, TServices> =
  [TRequired] extends [TServices]
    ? []
    : [missing: { readonly __missingSurfaceRequirements: Exclude<TRequired, TServices> }]

export function command<const TArgs extends readonly unknown[], A, E, R>(
  run: (...args: TArgs) => Effect.Effect<A, E, R>
): Command<TArgs, A, E, R> {
  return {
    _tag: 'Command',
    run
  }
}

export function signal<TValue, TSource>(
  signalDefinition: CoreSignal<TValue, TSource>
): StreamSignal<TValue, never, PubSub.PubSub<TValue>>
export function signal<A, E, R>(
  source: Stream.Stream<A, E, R>
): StreamSignal<A, E, R>
export function signal<const TArgs extends readonly unknown[], A, E, R>(
  source: (...args: TArgs) => Stream.Stream<A, E, R>
): FnSignal<TArgs, A, E, R>
export function signal(
  source: CoreSignal<unknown, unknown> | Stream.Stream<unknown, unknown, unknown> | ((...args: readonly unknown[]) => Stream.Stream<unknown, unknown, unknown>)
): Signal<readonly unknown[], unknown, unknown, unknown> {
  if (source instanceof CoreSignal) {
    return {
      _tag: 'Signal',
      kind: 'stream',
      stream: Stream.unwrap(
        Effect.map(source.tag, (pubsub) => Stream.fromPubSub(pubsub))
      )
    }
  }

  if (typeof source === 'function') {
    return {
      _tag: 'Signal',
      kind: 'fn',
      stream: source
    }
  }

  return {
    _tag: 'Signal',
    kind: 'stream',
    stream: source
  }
}

type RegularProjection<TId, TStateSchema extends Schema.Schema.AnyNoContext> = {
  readonly isForked: false
  readonly Tag: Context.Tag<TId, ProjectionInstance<TStateSchema>>
}

type ForkedProjection<TId, TStateSchema extends Schema.Schema.AnyNoContext> = {
  readonly isForked: true
  readonly Tag: Context.Tag<TId, ForkedProjectionInstance<TStateSchema>>
}

export type RegularStateSurface<TId, TStateSchema extends Schema.Schema.AnyNoContext> = {
  readonly get: () => Effect.Effect<Schema.Schema.Type<TStateSchema>, never, TId>
  readonly subscribe: Stream.Stream<Schema.Schema.Type<TStateSchema>, never, TId>
}

export type ForkedStateSurface<TId, TStateSchema extends Schema.Schema.AnyNoContext> = {
  readonly getFork: (forkId: string | null) => Effect.Effect<Schema.Schema.Type<TStateSchema>, never, TId>
  readonly subscribeFork: (forkId: string | null) => Stream.Stream<Schema.Schema.Type<TStateSchema>, never, TId>
}

export function state<TId, TStateSchema extends Schema.Schema.AnyNoContext>(
  projection: RegularProjection<TId, TStateSchema>
): {
  readonly get: Command<readonly [], Schema.Schema.Type<TStateSchema>, never, TId>
  readonly subscribe: StreamSignal<Schema.Schema.Type<TStateSchema>, never, TId>
}
export function state<TId, TStateSchema extends Schema.Schema.AnyNoContext>(
  projection: ForkedProjection<TId, TStateSchema>
): {
  readonly getFork: Command<readonly [forkId: string | null], Schema.Schema.Type<TStateSchema>, never, TId>
  readonly subscribeFork: FnSignal<readonly [forkId: string | null], Schema.Schema.Type<TStateSchema>, never, TId>
}
export function state<TId, TStateSchema extends Schema.Schema.AnyNoContext>(
  projection: RegularProjection<TId, TStateSchema> | ForkedProjection<TId, TStateSchema>
) {
  if (projection.isForked) {
    return {
      getFork: command((forkId: string | null) =>
        Effect.flatMap(projection.Tag, (service) => service.getFork(forkId))
      ),
      subscribeFork: signal((forkId: string | null) =>
        Stream.unwrap(
          Effect.map(projection.Tag, (service) =>
            Stream.concat(
              Stream.fromEffect(service.getFork(forkId)),
              service.state.changes.pipe(
                Stream.mapEffect(() => service.getFork(forkId)),
                Stream.changes
              )
            )
          )
        )
      )
    }
  }

  return {
    get: command(() =>
      Effect.flatMap(projection.Tag, (service) => service.get)
    ),
    subscribe: signal(
      Stream.unwrap(
        Effect.map(projection.Tag, (service) =>
          Stream.concat(
            Stream.fromEffect(service.get),
            service.state.changes
          )
        )
      )
    )
  }
}

type SurfaceRequirements<TValue> =
  TValue extends Command<infer _TArgs, infer _A, infer _E, infer R>
    ? R
    : TValue extends Signal<infer _TArgs, infer _A, infer _E, infer R>
      ? R
      : TValue extends object
        ? { readonly [K in keyof TValue]: SurfaceRequirements<TValue[K]> }[keyof TValue]
        : never

type HostConfig<TServices, E> = {
  readonly layer: Layer.Layer<TServices, E, never>
}

export function host<
  TServices,
  E,
  const TConfig extends HostConfig<TServices, E>
>(
  config: TConfig,
  ..._missing: MissingRequirements<SurfaceRequirements<Omit<TConfig, 'layer'>>, TServices>
): Host<TServices, Omit<TConfig, 'layer'>, E> {
  const { layer, ...surface } = config
  return { layer, surface }
}

type EffectSignalClient<TArgs extends readonly unknown[], A, E> =
  TArgs extends readonly []
    ? Stream.Stream<A, E, never>
    : (...args: TArgs) => Stream.Stream<A, E, never>

type EffectClientValue<TValue> =
  TValue extends Command<infer TArgs, infer A, infer E, infer _R>
    ? (...args: TArgs) => Effect.Effect<A, E, never>
    : TValue extends StreamSignal<infer A, infer E, infer _R>
      ? Stream.Stream<A, E, never>
      : TValue extends FnSignal<infer TArgs, infer A, infer E, infer _R>
        ? (...args: TArgs) => Stream.Stream<A, E, never>
        : TValue extends object
          ? EffectClient<TValue>
          : never

export type EffectClient<TSurface> = {
  readonly [K in keyof TSurface]: EffectClientValue<TSurface[K]>
}

type VanillaSignalClient<TArgs extends readonly unknown[], A> =
  TArgs extends readonly []
    ? (callback: (value: A) => void) => () => void
    : (...args: [...TArgs, callback: (value: A) => void]) => () => void

type VanillaClientValue<TValue> =
  TValue extends Command<infer TArgs, infer A, infer _E, infer _R>
    ? (...args: TArgs) => Promise<A>
    : TValue extends StreamSignal<infer A, infer _E, infer _R>
      ? VanillaSignalClient<readonly [], A>
      : TValue extends FnSignal<infer TArgs, infer A, infer _E, infer _R>
        ? VanillaSignalClient<TArgs, A>
        : TValue extends object
          ? VanillaClientObject<TValue>
          : never

type VanillaClientObject<TSurface> = {
  readonly [K in keyof TSurface]: VanillaClientValue<TSurface[K]>
}

export type VanillaClient<TSurface> = VanillaClientObject<TSurface> & {
  readonly dispose: () => Promise<void>
}

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === 'object' && value !== null

const isCommand = <TServices>(value: unknown): value is Command<readonly unknown[], unknown, unknown, TServices> =>
  isRecord(value) && value._tag === 'Command'

const isSignal = <TServices>(value: unknown): value is Signal<readonly unknown[], unknown, unknown, TServices> =>
  isRecord(value) && value._tag === 'Signal'

const provideEffect = <A, E, TServices>(
  effect: Effect.Effect<A, E, TServices>,
  context: Context.Context<TServices>
): Effect.Effect<A, E, never> =>
  effect.pipe(Effect.provide(context))

const provideStream = <A, E, TServices>(
  stream: Stream.Stream<A, E, TServices>,
  context: Context.Context<TServices>
): Stream.Stream<A, E, never> =>
  stream.pipe(Stream.provideContext(context))

function bindEffectSurface<TServices>(
  value: unknown,
  context: Context.Context<TServices>
): unknown {
  if (isCommand<TServices>(value)) {
    return (...args: unknown[]) =>
      provideEffect(value.run(...args), context)
  }

  if (isSignal<TServices>(value)) {
    if (value.kind === 'fn') {
      return (...args: unknown[]) =>
        provideStream(value.stream(...args), context)
    }

    return provideStream(value.stream, context)
  }

  if (isRecord(value)) {
    const bound: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(value)) {
      bound[key] = bindEffectSurface(child, context)
    }
    return bound
  }

  return value
}

export function effectClient<TServices, TSurface, E>(
  surfaceHost: Host<TServices, TSurface, E>
): Effect.Effect<EffectClient<TSurface>, E, Scope.Scope>
export function effectClient<TServices, TSurface, E>(
  surfaceHost: Host<TServices, TSurface, E>
): Effect.Effect<EffectClient<TSurface>, E, Scope.Scope> {
  return Effect.gen(function* () {
    const context = yield* Layer.build(surfaceHost.layer)
    return bindEffectSurface(surfaceHost.surface, context) as EffectClient<TSurface>
  })
}

type ActiveStream = {
  readonly fiber: Fiber.RuntimeFiber<void, unknown>
  readonly interrupt: () => Promise<void>
  readonly unsubscribe: () => void
}

function bindVanillaSurface<TServices>(
  value: unknown,
  runtime: ManagedRuntime.ManagedRuntime<TServices, unknown>,
  activeStreams: Set<ActiveStream>,
  isDisposed: () => boolean
): unknown {
  const ensureOpen = () => {
    if (isDisposed()) {
      throw new Error('Surface client is disposed')
    }
  }

  const subscribe = (
    stream: Stream.Stream<unknown, unknown, TServices>,
    callback: (value: unknown) => void
  ) => {
    ensureOpen()
    let closed = false
    const fiber = runtime.runFork(
      Stream.runForEach(stream, (value) =>
        Effect.sync(() => callback(value))
      )
    )
    const interrupt = () =>
      Effect.runPromise(Fiber.interrupt(fiber)).then(() => undefined, () => undefined)

    const active: ActiveStream = {
      fiber,
      interrupt,
      unsubscribe: () => {
        if (closed) return
        closed = true
        activeStreams.delete(active)
        void interrupt()
      }
    }

    activeStreams.add(active)
    fiber.addObserver(() => activeStreams.delete(active))
    return active.unsubscribe
  }

  if (isCommand<TServices>(value)) {
    return (...args: unknown[]) => {
      ensureOpen()
      return runtime.runPromise(value.run(...args))
    }
  }

  if (isSignal<TServices>(value)) {
    if (value.kind === 'fn') {
      return (...args: unknown[]) => {
        const callback = args[args.length - 1]
        if (typeof callback !== 'function') {
          throw new Error('Surface signal method requires a callback')
        }
        return subscribe(value.stream(...args.slice(0, -1)), (item) => {
          callback(item)
        })
      }
    }

    return (callback: (value: unknown) => void) =>
      subscribe(value.stream, callback)
  }

  if (isRecord(value)) {
    const bound: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(value)) {
      bound[key] = bindVanillaSurface(child, runtime, activeStreams, isDisposed)
    }
    return bound
  }

  return value
}

export async function vanillaClient<TServices, TSurface, E>(
  surfaceHost: Host<TServices, TSurface, E>
): Promise<VanillaClient<TSurface>> {
  const runtime = ManagedRuntime.make(surfaceHost.layer)
  await runtime.runtime()
  const activeStreams = new Set<ActiveStream>()
  let disposed = false

  const bound = bindVanillaSurface(
    surfaceHost.surface,
    runtime,
    activeStreams,
    () => disposed
  ) as VanillaClientObject<TSurface>

  return {
    ...bound,
    dispose: async () => {
      if (disposed) return
      disposed = true
      const streams = [...activeStreams]
      activeStreams.clear()
      await Promise.allSettled(streams.map((stream) => stream.interrupt()))
      await runtime.dispose()
    }
  }
}
