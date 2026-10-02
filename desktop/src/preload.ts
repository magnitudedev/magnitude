import { ModelTrayPresentation } from "@magnitudedev/client-common/application/contracts"
import { contextBridge, ipcRenderer } from "electron"
import { RpcClient } from "@effect/rpc"
import { Cause, Context, Effect, Exit, Fiber, Layer, ManagedRuntime, Option, Schema, Stream } from "effect"
import { ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
import { HostError, InferenceHostRpcs, type InferenceHostClient, type DesktopApi } from "./desktop-rpc"
import { makeElectronRpcClientLayer } from "./electron-rpc"
class HostClient extends Context.Tag("InferenceHostClient")<HostClient, InferenceHostClient>() {}
const runtime = ManagedRuntime.make(Layer.scoped(HostClient, RpcClient.make(InferenceHostRpcs)).pipe(Layer.provide(makeElectronRpcClientLayer(ipcRenderer))))
const observe = <A>(select: (client: InferenceHostClient) => Stream.Stream<A, unknown>, value: (value: A) => void, error: (message: string) => void) => {
  const fiber = runtime.runFork(Effect.gen(function* () {
    const client = yield* HostClient
    yield* select(client).pipe(Stream.runForEach(item => Effect.sync(() => value(item))))
  }).pipe(Effect.catchAll(cause => Effect.sync(() => error(String(cause))))))
  return () => { runtime.runFork(Fiber.interrupt(fiber)) }
}
const command = (select: (client: InferenceHostClient) => Effect.Effect<unknown, unknown>) => runtime.runPromiseExit(Effect.gen(function* () { yield* select(yield* HostClient) })).then(Exit.match({
  onSuccess: () => undefined,
  onFailure: cause => {
    const failure = Cause.failureOption(cause)
    // contextBridge preserves ordinary Error messages, not Effect's FiberFailure identity.
    throw new Error(Option.isSome(failure) && Schema.is(HostError)(failure.value)
      ? failure.value.message : "Magnitude could not complete this action. Try again or check Status.")
  },
}))
const query = <A>(select: (client: InferenceHostClient) => Effect.Effect<A, unknown>) => runtime.runPromiseExit(Effect.flatMap(HostClient, select)).then(Exit.match({
  onSuccess: value => value,
  onFailure: cause => {
    const failure = Cause.failureOption(cause)
    throw new Error(Option.isSome(failure) && Schema.is(HostError)(failure.value)
      ? failure.value.message : "Magnitude could not complete this action. Try again or check Status.")
  },
}))
const api: DesktopApi = {
  platform: process.platform,
  observe: (value, error) => observe(client => client.Observe({}), state => value(Schema.encodeSync(ApplicationSnapshot)(state)), error),
  actions: value => observe(client => client.Actions({}), value, message => console.error(message)),
  presentModel: value => command(client => client.PresentModel(Schema.decodeUnknownSync(ModelTrayPresentation)(value))),
  getAppearance: () => runtime.runPromise(Effect.flatMap(HostClient, client => client.GetAppearance({}))),
  setAppearance: preference => command(client => client.SetAppearance({ preference })),
  retry: () => command(client => client.Retry({})),
  resolveQuitFailure: decision => command(client => client.ResolveQuitFailure({ decision })),
}
contextBridge.exposeInMainWorld("__magnitudeDesktop", api)
