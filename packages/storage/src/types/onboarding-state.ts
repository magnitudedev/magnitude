import { Option, Schema } from 'effect'

const PersistedOnboardingStateSchema = Schema.Struct({
  completed: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
})

const RuntimeOnboardingStateSchema = Schema.Struct({ completed: Schema.Boolean })

export const OnboardingStateSchema = Schema.transform(
  PersistedOnboardingStateSchema,
  RuntimeOnboardingStateSchema,
  {
    strict: true,
    decode: ({ completed }) => ({ completed: Option.getOrElse(completed, () => false) }),
    encode: ({ completed }) => ({ completed: Option.some(completed) }),
  },
)

export type OnboardingState = typeof OnboardingStateSchema.Type

export const EMPTY_ONBOARDING_STATE: OnboardingState = { completed: false }
