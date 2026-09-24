import { FileSystem } from "@effect/platform"
import { decodePublisherPublicKey } from "@magnitudedev/release/hosted-update"
import { Effect, Schema, Stream } from "effect"
import { join } from "node:path"

export const ApplicationUpdateConfiguration = Schema.Struct({
  origin: Schema.NonEmptyString,
  keyId: Schema.NonEmptyString,
  publicKey: Schema.NonEmptyString,
  acceptance: Schema.Boolean,
  windowsPublisher: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
}).pipe(Schema.filter(config => !config.acceptance || config.origin !== "https://magnitude.dev"))

/** Trust is an application build input; release offers and runtime environment cannot supply it. */
export const decodeApplicationUpdateConfiguration = (input: unknown) => Schema.decodeUnknown(ApplicationUpdateConfiguration)(input, { onExcessProperty: "error" }).pipe(
  Effect.flatMap(config => decodePublisherPublicKey(config.publicKey).pipe(Effect.map(key => ({
    ...config, trustedPublishers: new Map([[config.keyId, key]]),
  })))),
)

export const readInstalledUpdateConfiguration = (resources: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const bytes = yield* fs.stream(join(resources, "update-configuration.json"), { bytesToRead: 16385 }).pipe(Stream.runCollect)
  const content = Buffer.concat(Array.from(bytes))
  if (content.length > 16384) return yield* new InstalledUpdateConfigurationFailed()
  const text = yield* Effect.try(() => new TextDecoder("utf-8", { fatal: true }).decode(content))
  const input = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(text)
  return yield* decodeApplicationUpdateConfiguration(input)
}).pipe(Effect.mapError(() => new InstalledUpdateConfigurationFailed()))
export class InstalledUpdateConfigurationFailed extends Schema.TaggedError<InstalledUpdateConfigurationFailed>()("InstalledUpdateConfigurationFailed", {}) {
  override get message() { return "The installed application update configuration could not be read. Reinstall Magnitude." }
}
