import { createRequire } from "node:module"
import { Context, Effect, ExecutionStrategy, Exit, Layer, Option, Ref, Schema, Scope } from "effect"
import * as FSM from "../fsm/index"
import { WindowsProcessId } from "./windows-process-identity"

export const WindowsPipeName = Schema.String.pipe(Schema.filter(value => value.startsWith("\\\\.\\pipe\\magnitude-") && value.length <= 240 && !value.includes("\0")), Schema.brand("WindowsPipeName"))
export type WindowsPipeName = typeof WindowsPipeName.Type
export class WindowsPipeFailed extends Schema.TaggedError<WindowsPipeFailed>()("WindowsPipeFailed", {
  message: Schema.String,
  win32Code: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
}) {}
const failure = (message: string, error?: unknown) => new WindowsPipeFailed({ message, win32Code: Option.fromNullable(
  typeof error === "object" && error !== null && "win32Code" in error && typeof error.win32Code === "number" && Number.isInteger(error.win32Code)
    ? error.win32Code : undefined,
) })

/** Node-API boundary only. Product code consumes WindowsPrivatePipes effects. */
export interface WindowsPipeBindings {
  readonly createPrivatePipe: (name: string, first: boolean) => object
  readonly acceptPrivatePipe: (pipe: object) => Promise<number>
  readonly readPrivatePipe: (pipe: object) => Promise<Uint8Array>
  readonly writePrivatePipe: (pipe: object, bytes: Buffer) => Promise<number>
  readonly closePrivatePipe: (pipe: object) => Promise<void>
}
class Listening extends Schema.TaggedClass<Listening>()("Listening", {}) {}
class Accepting extends Schema.TaggedClass<Accepting>()("Accepting", {}) {}
class Connected extends Schema.TaggedClass<Connected>()("Connected", { pid: WindowsProcessId }) {}
class Closing extends Schema.TaggedClass<Closing>()("Closing", {}) {}
class Closed extends Schema.TaggedClass<Closed>()("Closed", {}) {}
const lifecycle = FSM.defineFSM({ Listening, Accepting, Connected, Closing, Closed }, {
  Listening: ["Accepting", "Closing"], Accepting: ["Connected", "Closing"], Connected: ["Closing"], Closing: ["Closed"], Closed: [],
} as const)
type State = Listening | Accepting | Connected | Closing | Closed
export interface WindowsPrivatePipe {
  readonly accept: Effect.Effect<WindowsProcessId, WindowsPipeFailed>
  readonly read: Effect.Effect<Uint8Array, WindowsPipeFailed>
  readonly write: (bytes: Uint8Array) => Effect.Effect<void, WindowsPipeFailed>
  readonly close: Effect.Effect<void>
}
export interface WindowsPrivatePipes {
  readonly bind: (name: WindowsPipeName, first: boolean) => Effect.Effect<WindowsPrivatePipe, WindowsPipeFailed, Scope.Scope>
}
export const WindowsPrivatePipes = Context.GenericTag<WindowsPrivatePipes>("@magnitudedev/utils/WindowsPrivatePipes")

export const windowsPrivatePipesLayer = (native: WindowsPipeBindings) => Layer.succeed(WindowsPrivatePipes, {
  bind: (name, first) => Effect.gen(function* () {
    const state = yield* Ref.make<State>(new Listening({}))
    const scope = yield* Scope.fork(yield* Scope.Scope, ExecutionStrategy.sequential)
    const close = Scope.close(scope, Exit.void)
    const pipe = yield* Effect.acquireRelease(
      Effect.try({ try: () => native.createPrivatePipe(name, first), catch: error => failure("Could not create the private Windows pipe.", error) }),
      pipe => Effect.gen(function* () {
        yield* Ref.update(state, value => value._tag === "Closing" || value._tag === "Closed" ? value : lifecycle.transition(value, "Closing", {}))
        yield* Effect.tryPromise({ try: () => native.closePrivatePipe(pipe), catch: error => failure("Could not close the private Windows pipe.", error) }).pipe(Effect.orDie)
        yield* Ref.update(state, value => value._tag === "Closing" ? lifecycle.transition(value, "Closed", {}) : value)
      }),
    ).pipe(Scope.extend(scope), Effect.onError(() => close))
    const connected = Ref.get(state).pipe(Effect.flatMap(value => value._tag === "Connected" ? Effect.void : Effect.fail(failure("The private Windows pipe is not connected."))))
    const readLock = yield* Effect.makeSemaphore(1)
    const writeLock = yield* Effect.makeSemaphore(1)
    const accept = Effect.gen(function* () {
      const claimed = yield* Ref.modify(state, value => value._tag === "Listening"
        ? [true, lifecycle.transition(value, "Accepting", {})] : [false, value])
      if (!claimed) return yield* failure("The private Windows pipe is already accepting or closed.")
      return yield* Effect.tryPromise({ try: () => native.acceptPrivatePipe(pipe), catch: error => failure("Could not accept the Windows pipe connection.", error) }).pipe(
        Effect.flatMap(Schema.decodeUnknown(WindowsProcessId)),
        Effect.mapError(error => error instanceof WindowsPipeFailed ? error : failure("Windows returned an invalid pipe client identity.")),
        Effect.flatMap(pid => Ref.modify(state, value => value._tag === "Accepting"
          ? [true, lifecycle.transition(value, "Connected", { pid })] : [false, value]).pipe(
          Effect.flatMap(published => published ? Effect.succeed(pid) : Effect.fail(failure("The Windows pipe closed during acceptance."))),
        )), Effect.onError(() => close), Effect.onInterrupt(() => close),
      )
    })
    const read = readLock.withPermits(1)(Effect.gen(function* () {
      yield* connected
      const bytes = yield* Effect.tryPromise({ try: () => native.readPrivatePipe(pipe), catch: error => failure("Could not read the Windows pipe.", error) })
      if (!(bytes instanceof Uint8Array) || bytes.byteLength > 65536) return yield* failure("Windows returned an invalid pipe read.")
      yield* connected
      return bytes
    })).pipe(Effect.onError(() => close), Effect.onInterrupt(() => close))
    const write = (bytes: Uint8Array) => writeLock.withPermits(1)(Effect.gen(function* () {
      yield* connected
      const input = Buffer.from(bytes)
      for (let offset = 0; offset < input.length;) {
        const chunk = input.subarray(offset, offset + 65536)
        const count = yield* Effect.tryPromise({ try: () => native.writePrivatePipe(pipe, chunk), catch: error => failure("Could not write the Windows pipe.", error) })
        if (!Number.isInteger(count) || count <= 0 || count > chunk.length) return yield* failure("Windows returned an invalid pipe write count.")
        offset += count
        yield* connected
      }
    })).pipe(Effect.onError(() => close), Effect.onInterrupt(() => close))
    return { accept, read, write, close }
  }),
})

export const nativeWindowsPrivatePipesLayer = (addonPath: string) => Layer.unwrapEffect(Effect.try({
  try: () => {
    const native = createRequire(import.meta.url)(addonPath) as WindowsPipeBindings
    if (![native.createPrivatePipe, native.acceptPrivatePipe, native.readPrivatePipe, native.writePrivatePipe, native.closePrivatePipe].every(value => typeof value === "function"))
      throw new Error("Windows pipe exports are absent")
    return windowsPrivatePipesLayer(native)
  }, catch: error => failure("The installed native host does not provide Windows private pipes.", error),
}))
