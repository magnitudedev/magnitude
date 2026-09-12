import type { MenuItemConstructorOptions } from "electron"
import { Context, Effect, Layer, Option, Ref, Schema, ScopedRef, Stream, SubscriptionRef, type Scope } from "effect"
import { FSM } from "@magnitudedev/utils"
import { type TrayRegistration } from "@magnitudedev/sdk/desktop-host"
import type { TrayHostState } from "@magnitudedev/daemon-management/desktop-native"

export class NativeTrayFailed extends Schema.TaggedError<NativeTrayFailed>()("NativeTrayFailed", { message: Schema.String }) {}
export type TrayMenu = readonly MenuItemConstructorOptions[]
export interface NativeTrayHandle { readonly setMenu: (menu: TrayMenu) => Effect.Effect<void, NativeTrayFailed> }
export interface NativeTrayFactory { readonly create: Effect.Effect<NativeTrayHandle, NativeTrayFailed, Scope.Scope> }
export const NativeTrayFactory = Context.GenericTag<NativeTrayFactory>("desktop/NativeTrayFactory")

class Checking extends Schema.TaggedClass<Checking>()("Checking", {}) {}
class Registered extends Schema.TaggedClass<Registered>()("Registered", {}) {}
class Unavailable extends Schema.TaggedClass<Unavailable>()("Unavailable", { message: Schema.String }) {}
class Closed extends Schema.TaggedClass<Closed>()("Closed", {}) {}
const lifecycle = FSM.defineFSM({ Checking, Registered, Unavailable, Closed }, {
  Checking: ["Registered", "Unavailable", "Closed"], Registered: ["Unavailable", "Closed"],
  Unavailable: ["Registered", "Closed"], Closed: [],
} as const)

export interface TrayOwner {
  readonly state: Effect.Effect<TrayRegistration>
  readonly changes: Stream.Stream<TrayRegistration>
  readonly setMenu: (menu: TrayMenu) => Effect.Effect<void>
  readonly observeHost: (host: TrayHostState) => Effect.Effect<void>
}
export const TrayOwner = Context.GenericTag<TrayOwner>("desktop/TrayOwner")

/** Menu updates and host replacement share one resource owner; closing it is terminal. */
export const TrayOwnerLive = Layer.scoped(TrayOwner, Effect.gen(function* () {
  const factory = yield* NativeTrayFactory
  const handle = yield* ScopedRef.fromAcquire(Effect.succeed(Option.none<NativeTrayHandle>()))
  const menu = yield* Ref.make<TrayMenu>([])
  const host = yield* Ref.make(Option.none<TrayHostState>())
  const state = yield* SubscriptionRef.make<TrayRegistration>(new Checking({}))
  const lock = yield* Effect.makeSemaphore(1)
  const unavailable = (message: string) => SubscriptionRef.update(state, current => current._tag === "Closed" ? current
    : current._tag === "Unavailable" ? lifecycle.hold(current, { message }) : lifecycle.transition(current, "Unavailable", { message }))
  const registered = SubscriptionRef.update(state, current => current._tag === "Closed" || current._tag === "Registered" ? current : lifecycle.transition(current, "Registered", {}))
  const replace = Effect.gen(function* () {
    // Close the predecessor first. ScopedRef's ordinary atomic replacement may overlap resources.
    yield* ScopedRef.set(handle, Effect.succeed(Option.none()))
    yield* ScopedRef.set(handle, factory.create.pipe(Effect.tap(value => Ref.get(menu).pipe(Effect.flatMap(value.setMenu))), Effect.map(Option.some)))
    yield* registered
  }).pipe(Effect.catchTag("NativeTrayFailed", error => unavailable(error.message)))
  yield* replace
  yield* Effect.addFinalizer(() => lock.withPermits(1)(SubscriptionRef.update(state, current => current._tag === "Closed" ? current : lifecycle.transition(current, "Closed", {}))))
  return TrayOwner.of({
    state: SubscriptionRef.get(state), changes: state.changes,
    setMenu: next => lock.withPermits(1)(Effect.gen(function* () {
      if ((yield* SubscriptionRef.get(state))._tag === "Closed") return
      yield* Ref.set(menu, next)
      const current = yield* ScopedRef.get(handle)
      if (Option.isSome(current)) yield* current.value.setMenu(next).pipe(Effect.catchTag("NativeTrayFailed", error =>
        ScopedRef.set(handle, Effect.succeed(Option.none())).pipe(Effect.zipRight(unavailable(error.message)))))
    })),
    observeHost: next => lock.withPermits(1)(Effect.gen(function* () {
      if ((yield* SubscriptionRef.get(state))._tag === "Closed") return
      const previous = yield* Ref.getAndSet(host, Option.some(next))
      if (next._tag === "Unavailable") return yield* unavailable(next.message)
      if (Option.isSome(previous) && previous.value._tag === "Available" && previous.value.owner === next.owner) return
      yield* replace
    })),
  })
}))
