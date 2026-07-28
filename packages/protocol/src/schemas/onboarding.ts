import { Schema } from "effect"

export const OnboardingState = Schema.Struct({
  completed: Schema.Boolean,
})
export type OnboardingState = Schema.Schema.Type<typeof OnboardingState>
