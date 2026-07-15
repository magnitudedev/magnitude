import { Schema } from "effect"

export const OnboardingFlowId = Schema.Literal("model_setup")
export type OnboardingFlowId = Schema.Schema.Type<typeof OnboardingFlowId>

export const OnboardingFlowState = Schema.Struct({
  currentVersion: Schema.Number.pipe(Schema.int(), Schema.positive()),
  completedVersion: Schema.NullOr(Schema.Number.pipe(Schema.int(), Schema.positive())),
  completedAt: Schema.NullOr(Schema.String),
  required: Schema.Boolean,
})
export type OnboardingFlowState = Schema.Schema.Type<typeof OnboardingFlowState>

export const OnboardingState = Schema.Struct({
  flows: Schema.Record({ key: OnboardingFlowId, value: OnboardingFlowState }),
})
export type OnboardingState = Schema.Schema.Type<typeof OnboardingState>
