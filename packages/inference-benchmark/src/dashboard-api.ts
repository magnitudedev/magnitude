import { Schema } from "effect"

export const DashboardExperiment = Schema.Struct({
  id: Schema.String,
  title: Schema.String,
  path: Schema.String,
  profile: Schema.String,
  prepared: Schema.Boolean,
  requestPolicy: Schema.Struct({
    contextTokensPerSequence: Schema.Number,
    parallelSequences: Schema.Number,
    maxOutputTokens: Schema.Number,
    temperature: Schema.Number,
    topP: Schema.Number,
    seed: Schema.Number,
    enableThinking: Schema.Boolean,
  }),
  execution: Schema.Struct({ blocks: Schema.Number, variantOrder: Schema.String }),
  variants: Schema.Array(Schema.Struct({
    id: Schema.String,
    engine: Schema.String,
    artifact: Schema.Struct({
      kind: Schema.String,
      repository: Schema.String,
      revision: Schema.String,
      quantization: Schema.String,
    }),
  })),
})

export const DashboardRun = Schema.Struct({
  runId: Schema.String,
  state: Schema.Literal("running", "completed", "failed"),
  startedAt: Schema.String,
  experimentId: Schema.String,
  directory: Schema.String,
})

export const DashboardRunDetail = Schema.Struct({
  run: DashboardRun,
  manifest: Schema.Unknown,
  result: Schema.NullOr(Schema.Unknown),
  events: Schema.Array(Schema.Unknown),
})

export type DashboardExperiment = typeof DashboardExperiment.Type
export type DashboardRun = typeof DashboardRun.Type
export type DashboardRunDetail = typeof DashboardRunDetail.Type
