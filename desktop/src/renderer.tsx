import { Effect, Option, Schema, Stream } from "effect"
import { ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
import { ApplicationHostFailed, HostAction, ModelTrayPresentation, type ApplicationHost } from "@magnitudedev/client-common"
import { renderApplication } from "@magnitudedev/web/run"
import { OpenPathResult, type DesktopApi } from "./desktop-rpc"

/** The preload bridge is the renderer's only Promise boundary; everything past this file is Effect. */
const bridge: DesktopApi = window.__magnitudeDesktop

const failure = (error: unknown) => {
  const decoded = Schema.decodeUnknownEither(Schema.Struct({ message: Schema.String }))(error)
  return new ApplicationHostFailed({ message: decoded._tag === "Right" ? decoded.right.message : "Magnitude could not complete this action. Try again or check Status." })
}
const call = <A,>(action: () => Promise<A>) => Effect.tryPromise({ try: action, catch: failure })
const observe = <A, I>(schema: Schema.Schema<A, I>, subscribe: (value: (encoded: I) => void, error: (message: string) => void) => () => void) =>
  Stream.asyncPush<A, ApplicationHostFailed>(emit => Effect.acquireRelease(
    Effect.sync(() => subscribe(encoded => {
      const value = Schema.decodeUnknownEither(schema)(encoded)
      if (value._tag === "Right") emit.single(value.right)
      else emit.fail(new ApplicationHostFailed({ message: String(value.left) }))
    }, message => emit.fail(new ApplicationHostFailed({ message })))),
    unsubscribe => Effect.sync(unsubscribe),
  ).pipe(Effect.asVoid))

const host: ApplicationHost = {
  window: Option.some({ platform: bridge.platform }),
  appearance: {
    read: call(() => bridge.getAppearance()),
    save: preference => call(() => bridge.setAppearance(preference)),
  },
  foundModelsNotice: {
    readDismissed: call(() => bridge.getFoundModelsNoticeDismissed()),
    dismiss: call(() => bridge.dismissFoundModelsNotice()),
  },
  shell: Option.some({
    actions: observe(HostAction, bridge.actions),
    presentModel: value => call(() => bridge.presentModel(Schema.encodeSync(ModelTrayPresentation)(value))),
    resolveQuitFailure: decision => call(() => bridge.resolveQuitFailure(decision)),
    retryService: call(() => bridge.retry()),
    chooseOpenPath: request => call(() => bridge.chooseOpenPath({ title: request.title, ...Option.match(request.defaultPath, { onNone: () => ({}), onSome: defaultPath => ({ defaultPath }) }) })).pipe(
      Effect.flatMap(Schema.decodeUnknown(OpenPathResult)),
      Effect.mapError(error => error instanceof ApplicationHostFailed ? error : new ApplicationHostFailed({ message: "The file dialog returned an unexpected result." })),
      Effect.map(result => result.path),
    ),
    application: observe(ApplicationSnapshot, bridge.observe),
  }),
}

const firstSnapshot = observe(ApplicationSnapshot, bridge.observe).pipe(
  Stream.runHead,
  Effect.flatMap(Option.match({ onNone: () => Effect.fail(new ApplicationHostFailed({ message: "Magnitude's service state is unavailable." })), onSome: Effect.succeed })),
)

renderApplication({ host, origin: Effect.map(firstSnapshot, snapshot => snapshot.endpoint), navigation: "memory", canSignIn: false, reconnect: "Automatic" })
