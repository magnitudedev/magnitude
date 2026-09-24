import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { decodePublisherPublicKey } from "../../src/hosted-update/manifest"

const Configuration = Schema.Struct({
  origin: Schema.String.pipe(Schema.pattern(/^https:\/\/[a-zA-Z0-9.-]+(?::[0-9]+)?$/)),
  appleTeam: Schema.String.pipe(Schema.pattern(/^[A-Z0-9]{10}$/)),
  publicKey: Schema.NonEmptyString,
})

/** Publisher identity is a release input; downloaded offers cannot override it. */
export const renderUnixInstallationScript = (input: typeof Configuration.Type) => Effect.gen(function* () {
  const configuration = yield* Schema.decodeUnknown(Configuration)(input, { onExcessProperty: "error" })
  yield* decodePublisherPublicKey(configuration.publicKey)
  const fs = yield* FileSystem.FileSystem
  const template = yield* fs.readFileString(fileURLToPath(new URL("../../resources/install.sh", import.meta.url)))
  return template.replaceAll("@MAGNITUDE_INSTALL_ORIGIN@", configuration.origin)
    .replaceAll("@MAGNITUDE_APPLE_TEAM@", configuration.appleTeam)
    .replaceAll("@MAGNITUDE_PUBLISHER_KEY@", Buffer.from(configuration.publicKey).toString("base64"))
})

const WindowsConfiguration = Schema.Struct({
  origin: Configuration.fields.origin,
  publisher: Schema.NonEmptyString.pipe(Schema.maxLength(256), Schema.filter(value => !/[\x00-\x1f\x7f]/.test(value))),
})
export const renderWindowsInstallationScript = (input: typeof WindowsConfiguration.Type) => Effect.gen(function* () {
  const configuration = yield* Schema.decodeUnknown(WindowsConfiguration)(input, { onExcessProperty: "error" })
  const fs = yield* FileSystem.FileSystem
  const template = yield* fs.readFileString(fileURLToPath(new URL("../../resources/install.ps1", import.meta.url)))
  return template.replaceAll("@MAGNITUDE_INSTALL_ORIGIN@", configuration.origin)
    .replaceAll("@MAGNITUDE_WINDOWS_PUBLISHER@", configuration.publisher.replaceAll("'", "''"))
})
