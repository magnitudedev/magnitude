import { Effect, Option, Schema, Stream } from "effect"
import { ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
import { ApplicationHostFailed, ModelTrayPresentation, type ApplicationHost } from "@magnitudedev/client-common"
import { renderApplication } from "@magnitudedev/web/run"
import type { DesktopApi } from "./desktop-rpc"

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
  shell: Option.some({
    actions: Stream.asyncPush(emit => Effect.acquireRelease(Effect.sync(() => bridge.actions(action => emit.single(action))), unsubscribe => Effect.sync(unsubscribe)).pipe(Effect.asVoid)),
    presentModel: value => call(() => bridge.presentModel(Schema.encodeSync(ModelTrayPresentation)(value))),
    resolveQuitFailure: decision => call(() => bridge.resolveQuitFailure(decision)),
    retryService: call(() => bridge.retry()),
    application: observe(ApplicationSnapshot, bridge.observe),
  }),
}

const firstSnapshot = observe(ApplicationSnapshot, bridge.observe).pipe(
  Stream.runHead,
  Effect.flatMap(Option.match({ onNone: () => Effect.fail(new ApplicationHostFailed({ message: "Magnitude's service state is unavailable." })), onSome: Effect.succeed })),
)

renderApplication({ host, origin: Effect.map(firstSnapshot, snapshot => snapshot.endpoint), navigation: "memory" })
