import { Schema } from "effect"
import { Backend, LeaseId, Provider, RunId, TargetId } from "./domain"
import { WorkId } from "./work-identity"

export const RunProgress = Schema.Struct({ runId: RunId, state: Schema.Literal("Queued", "Running", "Cancelling", "Finished"),
  deadline: Schema.DateTimeUtc, stages: Schema.Array(Schema.Struct({ id: WorkId, kind: Schema.Literal("build", "test"),
    targetId: TargetId, backend: Backend, state: Schema.Literal("Queued", "Running", "Finished"), attempts: Schema.Int,
    producer: Schema.optionalWith(WorkId, { as: "Option", exact: true }),
    leases: Schema.Array(Schema.Struct({ id: LeaseId, provider: Provider, machine: Schema.String,
      state: Schema.Literal("Allocating", "Ready", "Releasing", "Released") })),
  })) })
export type RunProgress = typeof RunProgress.Type
export const formatProgress = (progress: RunProgress) => progress.stages.map(stage => {
  const current = stage.leases.at(-1)
  const state = stage.state === "Finished" ? "finished" : current?.state === "Allocating" ? "preparing worker"
    : current?.state === "Releasing" ? "cleaning up" : stage.state.toLowerCase()
  return `${stage.kind === "build" ? "Build" : "Test"} ${stage.targetId} (${stage.backend}): ${state}${stage.attempts ? `, attempt ${stage.attempts}` : ""}`
}).join("\n")
