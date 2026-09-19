import { Effect, Schema } from "effect"
import { AssertionFailure, InfrastructureFailure } from "./domain"

export const LabProcessId = Schema.Int.pipe(Schema.positive(), Schema.brand("LabProcessId"))
export const ServiceInstanceId = Schema.NonEmptyString.pipe(Schema.brand("LabServiceInstanceId"))
export const ApplicationIdentity = Schema.Struct({ applicationPid: LabProcessId, servicePid: LabProcessId, serviceInstance: ServiceInstanceId })
export type ApplicationIdentity = typeof ApplicationIdentity.Type
export const ReadyApplicationSnapshot = Schema.Struct({ pid: LabProcessId, endpoint: Schema.String,
  service: Schema.TaggedStruct("Ready", { health: Schema.Struct({ pid: LabProcessId, id: ServiceInstanceId }) }),
})
/** Signal zero observes existence only. Never kill a service to make this assertion pass. */
export const assertServiceExited = (pid: typeof LabProcessId.Type) => Effect.try({ try: () => {
  try { process.kill(pid, 0); return false }
  catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ESRCH") return true
    throw error
  }
}, catch: () => new InfrastructureFailure({ operation: "service-exit", message: "Cannot verify whether the previous service process exited" }) }).pipe(
  Effect.flatMap(exited => exited ? Effect.void : Effect.fail(new AssertionFailure({ message: "Previous owning service process survived application shutdown" }))),
)
