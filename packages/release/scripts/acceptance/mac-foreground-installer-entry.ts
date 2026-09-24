import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { startMacForegroundInstallation } from "../../../daemon-management/src/application-update/mac-foreground-installation"
import { MacInstallerRequest } from "../../../daemon-management/src/application-update/mac-installer-command"
import { acquireApplicationMaintenance } from "../../../daemon-management/src/desktop-native/application-owner"
import { acquireUpdateInstallationLease } from "../../../daemon-management/src/desktop-native/update-installation-lease"
import { nativeHostLayer } from "../../../daemon-management/src/desktop-native/index"

const Input = Schema.Struct({ resources: Schema.String, stateDirectory: Schema.String, dataDirectory: Schema.String,
  version: Schema.String, architecture: Schema.Literal("arm64", "x64"), continuation: MacInstallerRequest.fields.continuation })

// Compiled with the real publisher identity. Each fixture is a fresh isolated installation.
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  const input = yield* Schema.decodeUnknown(Schema.parseJson(Input))(process.argv[2], { onExcessProperty: "error" })
  return yield* Effect.gen(function* () {
    yield* acquireApplicationMaintenance(input.stateDirectory)
    yield* acquireUpdateInstallationLease(input.stateDirectory)
    return yield* startMacForegroundInstallation(input)
  }).pipe(Effect.provide(nativeHostLayer(join(input.resources, "desktop-host.node"))))
})).pipe(Effect.provide(BunContext.layer)))
