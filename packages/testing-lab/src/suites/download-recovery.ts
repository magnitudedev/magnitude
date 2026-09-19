import { Effect, Option, Schema } from "effect"
import { ApplicationIdentity } from "../application-identity"
import { DisposableDesktopUser } from "../desktop-environment"
import { DesktopDriver } from "../desktop-driver"
import { DownloadProgress } from "../download-controls"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { GenerationExecution } from "../generation-evidence"
import { ModelFileReceipt } from "../model-files"
import { NetworkFault, NetworkIsolation, NetworkProbeAddress } from "../network-fault"
import { CliTests } from "./cli"

export const DownloadRecoveryEvidence = Schema.Union(
  Schema.TaggedStruct("Baseline", { files: ModelFileReceipt }),
  Schema.TaggedStruct("Transferring", { progress: DownloadProgress }),
  Schema.TaggedStruct("Isolation", { isolation: NetworkIsolation, address: NetworkProbeAddress }),
  Schema.TaggedStruct("Interrupted", { isolation: NetworkIsolation, address: NetworkProbeAddress }),
  Schema.TaggedStruct("Restored", { address: NetworkProbeAddress }),
  Schema.TaggedStruct("Recovered", { files: ModelFileReceipt, generation: GenerationExecution }),
  Schema.TaggedStruct("Owners", { before: ApplicationIdentity, after: ApplicationIdentity }),
)
export const verifyDownloadRecovery = (
  model: string,
  files: Effect.Effect<typeof ModelFileReceipt.Type, AssertionFailure | InfrastructureFailure>,
  observe: Effect.Effect<typeof GenerationExecution.Type, AssertionFailure | InfrastructureFailure>,
  attest: (generation: typeof GenerationExecution.Type) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>,
  record: (evidence: typeof DownloadRecoveryEvidence.Type) => Effect.Effect<void, InfrastructureFailure>,
  onCleanupError: (message: string) => void,
) => Effect.gen(function* () {
  if (Option.isNone(yield* Effect.serviceOption(DisposableDesktopUser))) return yield* new InfrastructureFailure({ operation: "download-recovery", message: "Model removal/reacquisition requires a qualified disposable guest" })
  const network = yield* NetworkFault, desktop = yield* DesktopDriver, cli = yield* CliTests
  const address = yield* network.resolve
  const online = network.reachable(address).pipe(Effect.flatMap(value => value ? Effect.void : Effect.fail(new InfrastructureFailure({ operation: "download-recovery", message: "External network control is unavailable" }))))
  const offline = network.reachable(address).pipe(Effect.flatMap(value => value ? Effect.fail(new AssertionFailure({ message: "Download fault did not block external traffic" })) : Effect.void))
  yield* online
  // Prove the fault mechanism and its cleanup before removing any installed model.
  yield* Effect.scoped(network.isolate(onCleanupError).pipe(Effect.zipRight(offline)))
  yield* online
  const owner = yield* desktop.identity()
  const baseline = yield* files
  yield* record({ _tag: "Baseline", files: baseline })
  yield* cli.removeModel
  yield* desktop.search(model)
  yield* desktop.downloads.absent(model)
  yield* desktop.downloads.begin(model)
  const progress = yield* desktop.downloads.transferring(model)
  yield* record({ _tag: "Transferring", progress })
  const interrupted = yield* Effect.scoped(Effect.gen(function* () {
    const isolation = yield* network.isolate(onCleanupError)
    yield* record({ _tag: "Isolation", isolation, address })
    yield* offline
    yield* desktop.downloads.failed(model)
    yield* record({ _tag: "Interrupted", isolation, address })
  })).pipe(Effect.exit)
  const restored = yield* online.pipe(Effect.exit)
  if (restored._tag === "Success") yield* record({ _tag: "Restored", address })
  else onCleanupError("External connectivity was not restored after interrupted acquisition")
  yield* interrupted
  yield* restored
  // A user-visible retry creates a new catalog installation occurrence; never retry assertions.
  yield* desktop.downloads.begin(model)
  yield* desktop.downloads.complete(model)
  const recovered = yield* files
  if (!Schema.equivalence(ModelFileReceipt)(baseline, recovered)) return yield* new AssertionFailure({ message: "Recovered model identities differ from the admitted baseline" })
  yield* desktop.load(model)
  const generation = yield* observe
  yield* record({ _tag: "Recovered", files: recovered, generation })
  yield* attest(generation)
  const current = yield* desktop.identity()
  yield* record({ _tag: "Owners", before: owner, after: current })
  if (!Schema.equivalence(ApplicationIdentity)(owner, current)) return yield* new AssertionFailure({ message: "Download recovery replaced the owning application or service" })
})
