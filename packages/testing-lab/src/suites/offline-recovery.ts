import { Effect, Schema } from "effect"
import { ApplicationIdentity } from "../application-identity"
import { DesktopDriver } from "../desktop-driver"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { GenerationExecution } from "../generation-evidence"
import { NetworkFault, NetworkIsolation, NetworkProbeAddress } from "../network-fault"
import { CliTests } from "./cli"

export const OfflineRecoveryEvidence = Schema.Union(
  Schema.TaggedStruct("Before", { generation: GenerationExecution }),
  Schema.TaggedStruct("Isolation", { isolation: NetworkIsolation, address: NetworkProbeAddress }),
  Schema.TaggedStruct("Offline", { generation: GenerationExecution }),
  Schema.TaggedStruct("Restored", { address: NetworkProbeAddress }),
  Schema.TaggedStruct("Owners", { before: ApplicationIdentity, after: ApplicationIdentity }),
)

export const verifyOfflineRecovery = (
  observe: Effect.Effect<typeof GenerationExecution.Type, AssertionFailure | InfrastructureFailure>,
  attest: (generation: typeof GenerationExecution.Type) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>,
  record: (evidence: typeof OfflineRecoveryEvidence.Type) => Effect.Effect<void, InfrastructureFailure>,
  onCleanupError: (message: string) => void,
) => Effect.gen(function* () {
  const network = yield* NetworkFault, desktop = yield* DesktopDriver, cli = yield* CliTests
  const address = yield* network.resolve
  const requireOnline = network.reachable(address).pipe(Effect.flatMap(reachable => reachable ? Effect.void
    : Effect.fail(new InfrastructureFailure({ operation: "offline-control", message: "External network control is not reachable" }))))
  const requireOffline = network.reachable(address).pipe(Effect.flatMap(reachable => reachable
    ? Effect.fail(new AssertionFailure({ message: "External traffic remained reachable during offline generation" })) : Effect.void))
  yield* requireOnline
  const owner = yield* desktop.identity()
  const before = yield* observe
  yield* record({ _tag: "Before", generation: before })
  yield* attest(before)
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const isolation = yield* network.isolate(onCleanupError)
    yield* record({ _tag: "Isolation", isolation, address })
    yield* requireOffline
    yield* cli.reloadModel
    const offline = yield* observe
    yield* record({ _tag: "Offline", generation: offline })
    yield* attest(offline)
    if (offline.native.workerGeneration === before.native.workerGeneration) return yield* new AssertionFailure({ message: "Offline reload reused the previous worker generation" })
    yield* desktop.ready()
    yield* requireOffline
  })).pipe(Effect.exit)
  // Restoration is observable even when the offline operation fails. Preserve the original failure too.
  const restored = yield* requireOnline.pipe(Effect.exit)
  if (restored._tag === "Success") yield* record({ _tag: "Restored", address })
  else onCleanupError("External connectivity could not be verified after removing isolation")
  yield* result
  yield* restored
  const current = yield* desktop.identity()
  yield* record({ _tag: "Owners", before: owner, after: current })
  if (!Schema.equivalence(ApplicationIdentity)(owner, current)) return yield* new AssertionFailure({ message: "Offline recovery replaced the owning application or service" })
})
