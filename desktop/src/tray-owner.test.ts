import { Context, Effect, Exit, Layer, Ref, Scope } from "effect"
import { describe, expect, it } from "vitest"
import { NativeTrayFactory, NativeTrayFailed, TrayOwner, TrayOwnerLive, type TrayMenu } from "./tray-owner"
import { TrayHostOwner } from "../../packages/daemon-management/src/desktop-native/tray-host"

const available = { _tag: "Available" as const, owner: TrayHostOwner.make(":1.1") }
const setup = Effect.gen(function* () {
  const active = yield* Ref.make(0)
  const maximum = yield* Ref.make(0)
  const created = yield* Ref.make(0)
  const fail = yield* Ref.make(false)
  const displayed = yield* Ref.make<TrayMenu>([])
  const factory = Layer.succeed(NativeTrayFactory, { create: Effect.acquireRelease(Effect.gen(function* () {
    if (yield* Ref.get(fail)) return yield* new NativeTrayFailed({ message: "Native creation failed" })
    const count = yield* Ref.updateAndGet(active, count => count + 1)
    yield* Ref.update(maximum, value => Math.max(value, count))
    yield* Ref.update(created, count => count + 1)
    return { setMenu: (menu: TrayMenu) => Ref.set(displayed, menu) }
  }), () => Ref.update(active, count => count - 1)) })
  const scope = yield* Scope.make()
  const context = yield* Layer.buildWithScope(TrayOwnerLive.pipe(Layer.provide(factory)), scope)
  yield* Effect.addFinalizer(exit => Scope.close(scope, exit))
  return { owner: Context.get(context, TrayOwner), active, maximum, created, displayed, fail, scope }
})
const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))

describe("native tray ownership", () => {
  it("restores the latest menu after host loss, without overlapping icons", () => run(Effect.gen(function* () {
    const f = yield* setup
    yield* f.owner.setMenu([{ label: "Stop Model" }])
    yield* f.owner.observeHost(available)
    expect(yield* Ref.get(f.created)).toBe(2)
    yield* f.owner.observeHost({ _tag: "Unavailable", message: "Panel stopped" })
    expect(yield* Ref.get(f.active)).toBe(1)
    yield* f.owner.setMenu([{ label: "No model loaded" }, { label: "Quit Magnitude" }])
    yield* f.owner.observeHost(available)
    expect(yield* Ref.get(f.displayed)).toEqual([{ label: "No model loaded" }, { label: "Quit Magnitude" }])
    expect(yield* Ref.get(f.created)).toBe(3)
    expect(yield* Ref.get(f.maximum)).toBe(1)
    expect((yield* f.owner.state)._tag).toBe("Registered")
    for (let i = 0; i < 10; i++) yield* f.owner.observeHost(available)
    expect(yield* Ref.get(f.created)).toBe(3)
  })))
  it("replaces a changed owner and recovers a failed creation on the next host transition", () => run(Effect.gen(function* () {
    const f = yield* setup
    yield* f.owner.observeHost(available)
    yield* Ref.set(f.fail, true)
    yield* f.owner.observeHost({ _tag: "Available", owner: TrayHostOwner.make(":1.2") })
    expect(yield* f.owner.state).toMatchObject({ _tag: "Unavailable", message: "Native creation failed" })
    expect(yield* Ref.get(f.active)).toBe(0)
    yield* f.owner.setMenu([{ label: "Current menu" }])
    yield* Ref.set(f.fail, false)
    yield* f.owner.observeHost({ _tag: "Unavailable", message: "Host restarting" })
    yield* f.owner.observeHost(available)
    expect(yield* Ref.get(f.displayed)).toEqual([{ label: "Current menu" }])
    expect(yield* Ref.get(f.active)).toBe(1)
  })))
  it("cannot recreate or update the icon after owner scope closes", () => run(Effect.gen(function* () {
    const f = yield* setup
    yield* Scope.close(f.scope, Exit.void)
    expect(yield* Ref.get(f.active)).toBe(0)
    expect((yield* f.owner.state)._tag).toBe("Closed")
    yield* Effect.all([f.owner.observeHost(available), f.owner.setMenu([{ label: "Late update" }])], { concurrency: "unbounded" })
    expect(yield* Ref.get(f.created)).toBe(1)
    expect(yield* Ref.get(f.active)).toBe(0)
  })))
})
