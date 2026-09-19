import { Effect, Schema } from "effect"
import { ApplicationIdentity, LabProcessId } from "../application-identity"
import { DesktopDriver } from "../desktop-driver"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { GenerationExecution } from "../generation-evidence"
import { WorkerFault, WorkerFaultReceipt } from "../worker-fault"
import { CliTests } from "./cli"

export const WorkerRecoveryEvidence = Schema.Union(
  Schema.TaggedStruct("Before", { generation: GenerationExecution }),
  Schema.TaggedStruct("Fault", { receipt: WorkerFaultReceipt }),
  Schema.TaggedStruct("After", { generation: GenerationExecution }),
  Schema.TaggedStruct("Owners", { before: ApplicationIdentity, after: ApplicationIdentity }),
)

/** Explicit reload is product policy: a failed resident worker is not automatically reloaded. */
export const verifyWorkerRecovery = (
  profile: string,
  observe: Effect.Effect<typeof GenerationExecution.Type, AssertionFailure | InfrastructureFailure>,
  attest: (generation: typeof GenerationExecution.Type) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>,
  record: (evidence: typeof WorkerRecoveryEvidence.Type) => Effect.Effect<void, InfrastructureFailure>,
) => Effect.gen(function* () {
  const desktop = yield* DesktopDriver, cli = yield* CliTests, fault = yield* WorkerFault
  const owner = yield* desktop.identity()
  const before = yield* observe
  yield* record({ _tag: "Before", generation: before })
  yield* attest(before)
  const receipt = yield* fault.crash({ owner, workerPid: LabProcessId.make(before.native.workerPid), profile })
  yield* record({ _tag: "Fault", receipt })
  yield* cli.failedModel
  yield* desktop.ready()
  if (!Schema.equivalence(ApplicationIdentity)(owner, yield* desktop.identity())) return yield* new AssertionFailure({ message: "Worker fault replaced the owning application or service" })
  yield* cli.loadModel
  const after = yield* observe
  yield* record({ _tag: "After", generation: after })
  yield* attest(after)
  if (before.native.workerGeneration === after.native.workerGeneration) return yield* new AssertionFailure({ message: "Generation reused the faulted native worker generation" })
  yield* fault.verifyParent(receipt)
  const current = yield* desktop.identity()
  yield* record({ _tag: "Owners", before: owner, after: current })
  if (!Schema.equivalence(ApplicationIdentity)(owner, current)) return yield* new AssertionFailure({ message: "Recovery replaced the owning application or service" })
})
