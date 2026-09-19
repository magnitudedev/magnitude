import { Context, Effect } from "effect"
import { InfrastructureFailure } from "./domain"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"

export interface GuestExecutor {
  readonly run: (invocation: typeof WorkerInvocation.Type, root: string) => Effect.Effect<typeof WorkerReply.Type, InfrastructureFailure>
}
export const GuestExecutor = Context.GenericTag<GuestExecutor>("@magnitudedev/testing-lab/GuestExecutor")
