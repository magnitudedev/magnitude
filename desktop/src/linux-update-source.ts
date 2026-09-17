import { Effect } from "effect"
import { PreparedUpdateStore, startLinuxUpdateHandoff } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"
import { PreparedUpdateInstaller } from "./prepared-update-installation"
import { hostedUpdateSource, type HostedUpdateSourceOptions } from "./hosted-update-source"

export const makeLinuxUpdateSource = (options: HostedUpdateSourceOptions & {
  readonly dataDirectory: string
  readonly stateDirectory: string
}) => Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  return {
    source: hostedUpdateSource(options, (archive, release) => store.prepare(archive, release).pipe(
      Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
    )),
    installer: PreparedUpdateInstaller.of({
      requiresAuthorization: true,
      install: (_archive, release, showWindow) => startLinuxUpdateHandoff({
        dataDirectory: options.dataDirectory, stateDirectory: options.stateDirectory, release, showWindow,
      }).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message }))),
    }),
  }
})
