import { DesktopConnectRequest, DesktopConnectionsSnapshot } from "@magnitudedev/client-common"
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
const api: DesktopApi = {
  memory: (value, error) => observe(client => client.Memory({}), value, error),
  applicationInfo: () => runtime.runPromise(Effect.flatMap(HostClient, client => client.ApplicationInfo({}))),
  updates: (value, error) => observe(client => client.Updates({}), value, error),
  checkUpdate: () => command(client => client.CheckUpdate({})),
  downloadUpdate: () => command(client => client.DownloadUpdate({})),
  restartUpdate: () => command(client => client.RestartUpdate({})),
  platform: process.platform,
  observe: (value, error) => observe(client => client.Observe({}), state => value(Schema.encodeSync(ApplicationSnapshot)(state)), error),
  actions: value => observe(client => client.Actions({}), value, message => console.error(message)),
  presentModel: value => command(client => client.PresentModel(value)),
  appearance: preference => command(client => client.Appearance({ preference })),
  loginStartup: (value, error) => observe(client => client.LoginStartup({}), value, error),
  setLoginStartup: enabled => command(client => client.SetLoginStartup({ enabled })),
  connections: (value, error) => observe(client => client.Connections({}), rows => value(Schema.encodeSync(DesktopConnectionsSnapshot)(rows)), error),
  connect: input => command(client => client.Connect(Schema.decodeUnknownSync(DesktopConnectRequest)(input))),
  disconnect: harness => command(client => client.Disconnect({ harness })),
  retry: () => command(client => client.Retry({})),
  quit: () => command(client => client.Quit({})),
}
contextBridge.exposeInMainWorld("__magnitudeDesktop", api)
