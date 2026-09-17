/** Isolated old-binary acceptance driver; never uses the real user's registration home. */
import { join } from "node:path"
import { mkdir } from "node:fs/promises"
import { NodeContext } from "@effect/platform-node"
import { Effect, Layer, Option } from "effect"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { BunSqliteDriverLayer } from "../src/bun"
import { makePreviousInstallationUpgrade, PreviousInstallation, PreviousInstallationJournal } from "../src/desktop-native/previous-installation"
import { previousInstallationJournal } from "../src/desktop-native/previous-installation-journal"
import { makeUnixPreviousInstallation } from "../src/desktop-native/previous-installation-unix"
import { NativeLegacyStartupCommands } from "../src/desktop-native/legacy-startup-command"
import { macLegacyStartupLayer, linuxLegacyStartupLayer } from "../src/desktop-native/legacy-startup"
import { UnixLegacyProcessTable } from "../src/desktop-native/legacy-tree"
import { unixPrivateFilePermissions } from "../src/desktop-native/private-files"

const home = process.argv[2]
if (!home || !home.includes("magnitude-upgrade-validation")) throw new Error("An isolated upgrade validation home is required")
const dataDirectory = join(home, ".magnitude")
const stateDirectory = join(dataDirectory, "state")
await mkdir(stateDirectory, { recursive: true, mode: 0o700 })
await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const installation = yield* makeUnixPreviousInstallation({ home, dataDirectory })
  if (process.argv.includes("--prepare-only")) {
    const plan = yield* installation.inspect
    if (Option.isNone(plan)) return yield* Effect.dieMessage("Expected an old installation to prepare")
    const journal = yield* PreviousInstallationJournal.pipe(Effect.provide(previousInstallationJournal(stateDirectory).pipe(Layer.provide(unixPrivateFilePermissions))))
    yield* journal.write(plan.value)
    console.log("Saved upgrade checkpoint without retiring installation")
    return
  }
  const upgrade = yield* makePreviousInstallationUpgrade.pipe(Effect.provideService(PreviousInstallation, installation),
    Effect.provide(previousInstallationJournal(stateDirectory).pipe(Layer.provide(unixPrivateFilePermissions))))
  yield* upgrade
  yield* upgrade
  console.log("Previous installation retired; repeated admission succeeded")
})).pipe(
  Effect.provide(process.platform === "darwin" ? macLegacyStartupLayer(home, process.argv.includes("--installed") ? undefined : "dev.magnitude.upgrade-validation") : linuxLegacyStartupLayer({ home, ...(process.argv.includes("--installed") ? {} : { unit: "magnitude-upgrade-validation.service" }) })),
  Effect.provide(NativeLegacyStartupCommands), Effect.provide(UnixLegacyProcessTable), Effect.provide(BunSqliteDriverLayer),
  Effect.provideService(ProcessGroupController, ProcessGroupControllerLive), Effect.provide(NodeContext.layer)))
