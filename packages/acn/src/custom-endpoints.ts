import * as FileSystem from "@effect/platform/FileSystem"
import {
  Context,
  Duration,
  Effect,
  Layer,
  Option,
  Schedule,
  Schema,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  CustomEndpointDeclarationsSchema,
  GlobalStorage,
  MagnitudeConfigSchema,
  MagnitudeStorage,
  type CustomEndpointDeclarations,
} from "@magnitudedev/storage"

export interface CustomEndpointsApi {
  readonly get: Effect.Effect<CustomEndpointDeclarations>
  readonly changes: Stream.Stream<CustomEndpointDeclarations>
}

export class CustomEndpoints extends Context.Tag("CustomEndpoints")<
  CustomEndpoints,
  CustomEndpointsApi
>() {}

const declarationsFromConfig = (
  providers: Option.Option<CustomEndpointDeclarations>,
): CustomEndpointDeclarations => Option.getOrElse(providers, () => ({}))

export const CustomEndpointsLive: Layer.Layer<
  CustomEndpoints,
  never,
  MagnitudeStorage | GlobalStorage | FileSystem.FileSystem
> = Layer.scoped(CustomEndpoints, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  const globalStorage = yield* GlobalStorage
  const fs = yield* FileSystem.FileSystem
  const initialConfig = yield* storage.config.load().pipe(
    Effect.catchAll((error) => Effect.logWarning("Unable to load custom endpoints").pipe(
      Effect.annotateLogs({ error: String(error).slice(0, 1_000) }),
      Effect.as({ providers: Option.none<CustomEndpointDeclarations>() }),
    )),
  )
  const initial = declarationsFromConfig(initialConfig.providers)
  const state = yield* SubscriptionRef.make(initial)
  const equivalent = Schema.equivalence(CustomEndpointDeclarationsSchema)

  const readConfigText = fs.readFileString(globalStorage.paths.configFile).pipe(
    Effect.map(Option.some),
    Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
      ? Effect.succeed(Option.none<string>())
      : Effect.fail(error)),
  )

  const observe = Effect.gen(function* () {
    const decoded = yield* readConfigText.pipe(
      Effect.flatMap(Option.match({
        onNone: () => Effect.succeed({ providers: Option.none<CustomEndpointDeclarations>() }),
        onSome: (text) => Schema.decodeUnknown(
          Schema.parseJson(MagnitudeConfigSchema),
          { onExcessProperty: "preserve" },
        )(text),
      })),
      Effect.either,
    )
    if (decoded._tag === "Left") {
      return
    }
    const declarations = declarationsFromConfig(decoded.right.providers)
    const current = yield* SubscriptionRef.get(state)
    if (!equivalent(current, declarations)) yield* SubscriptionRef.set(state, declarations)
  })

  yield* observe.pipe(
    Effect.repeat(Schedule.spaced(Duration.seconds(1))),
    Effect.catchAllCause((cause) => Effect.logError("Custom endpoint observer stopped").pipe(
      Effect.annotateLogs({ cause: String(cause).slice(0, 1_000) }),
    )),
    Effect.forkScoped,
  )

  return CustomEndpoints.of({
    get: SubscriptionRef.get(state),
    changes: state.changes,
  })
}))
