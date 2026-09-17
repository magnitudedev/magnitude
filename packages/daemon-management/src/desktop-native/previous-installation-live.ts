import { NodeContext } from "@effect/platform-node"
import { Effect, Layer } from "effect"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { PreviousInstallation, makePreviousInstallationUpgrade } from "./previous-installation"
import { makeUnixPreviousInstallation } from "./previous-installation-unix"
import { previousInstallationJournal } from "./previous-installation-journal"
import { NativeLegacyStartupCommands } from "./legacy-startup-command"
import { UnixLegacyProcessTable } from "./legacy-tree"
import { macLegacyStartupLayer, linuxLegacyStartupLayer } from "./legacy-startup"
import { unixPrivateFilePermissions } from "./private-files"

/** Only macOS and Linux had a previous standalone installation. */
export const previousInstallationUpgrade = (options: {
  readonly home: string; readonly dataDirectory: string; readonly stateDirectory: string
}) => Effect.gen(function* () {
  const installation = yield* makeUnixPreviousInstallation(options).pipe(
    Effect.provide([
      (process.platform === "darwin" ? macLegacyStartupLayer(options.home) : linuxLegacyStartupLayer({ home: options.home })).pipe(Layer.provideMerge(NativeLegacyStartupCommands)),
      UnixLegacyProcessTable,
    ]))
  return yield* makePreviousInstallationUpgrade.pipe(Effect.provideService(PreviousInstallation, installation),
    Effect.provide(previousInstallationJournal(options.stateDirectory).pipe(Layer.provide(unixPrivateFilePermissions))))
}).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive), Effect.provide(NodeContext.layer))
