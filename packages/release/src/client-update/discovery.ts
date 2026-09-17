import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Effect, Option, Schema, Stream } from "effect"
import { admittedChannels, isNewerVersion, newestFirst, releaseChannelOf } from "./release-channels"

export const RELEASE_DIST_TAGS_URL = "https://registry.npmjs.org/-/package/@magnitudedev%2Fcli/dist-tags"
const NPM_RESPONSE_LIMIT = 64 * 1_024
const CHANNEL_TAGS = ["latest", "beta", "alpha"] as const
const NpmDistTagsSchema = Schema.Struct({
  latest: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  beta: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  alpha: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})

export class UpdateDiscoveryFailed extends Schema.TaggedError<UpdateDiscoveryFailed>()(
  "UpdateDiscoveryFailed",
  {
    stage: Schema.Literal("registry", "release"),
    reason: Schema.String,
  },
) {}

/** Select once by release channel; the consumer verifies its own required artifact set. */
export const findReleaseUpdate = <A, E, R>(options: {
  readonly currentVersion: string
  readonly registryUrl: string
  readonly verify: (version: string) => Effect.Effect<A, E, R>
}) => Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const npmPackageUrl = options.registryUrl
  const admitted = admittedChannels(releaseChannelOf(options.currentVersion))
  const fetchDistTags = Effect.gen(function* () {
    const response = yield* http.execute(HttpClientRequest.get(
      npmPackageUrl,
    )).pipe(Effect.mapError((error) => new UpdateDiscoveryFailed({
      stage: "registry",
      reason: String(error),
    })))
    if (response.status < 200 || response.status >= 300) {
      return yield* new UpdateDiscoveryFailed({
        stage: "registry",
        reason: `npm registry returned HTTP ${response.status}`,
      })
    }
    const bytes = yield* response.stream.pipe(
      Stream.runFoldEffect(
        { chunks: [] as Uint8Array[], size: 0 },
        (state, chunk) => {
          const size = state.size + chunk.byteLength
          return size > NPM_RESPONSE_LIMIT
            ? new UpdateDiscoveryFailed({
                stage: "registry",
                reason: "npm registry response exceeds its size bound",
              })
            : Effect.succeed({ chunks: [...state.chunks, chunk], size })
        },
      ),
      Effect.mapError((error) => error instanceof UpdateDiscoveryFailed
        ? error
        : new UpdateDiscoveryFailed({
            stage: "registry",
            reason: String(error),
          })),
    )
    const body = new Uint8Array(bytes.size)
    let offset = 0
    for (const chunk of bytes.chunks) {
      body.set(chunk, offset)
      offset += chunk.byteLength
    }
    return yield* Schema.decodeUnknown(
      Schema.parseJson(NpmDistTagsSchema),
    )(new TextDecoder().decode(body)).pipe(
      Effect.mapError((error) => new UpdateDiscoveryFailed({
        stage: "registry",
        reason: String(error),
      })),
    )
  }).pipe(Effect.timeoutFail({
    duration: "30 seconds",
    onTimeout: () => new UpdateDiscoveryFailed({
      stage: "registry",
      reason: "npm registry request timed out",
    }),
  }))

  const tags = yield* fetchDistTags
  const candidates = CHANNEL_TAGS.flatMap(tag => {
    const version = tags[tag]
    return Option.isSome(version) && admitted.has(releaseChannelOf(version.value)) && isNewerVersion(version.value, options.currentVersion)
      ? [version.value] : []
  })
  for (const version of newestFirst([...new Set(candidates)])) {
    const result = yield* options.verify(version).pipe(
      Effect.map(Option.some),
      Effect.catchAll(error => Effect.logDebug(`Update candidate ${version} is not ready: ${String(error)}`).pipe(Effect.as(Option.none<A>()))),
    )
    if (Option.isSome(result)) return result
  }
  return Option.none<A>()
})
