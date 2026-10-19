import { Schema } from "effect"

/**
 * Why Magnitude isn't signed in to Hugging Face after an attempt or a session ended: the person
 * declined in the browser, the sign-in wasn't finished in time, it failed, or the saved session
 * could no longer be renewed.
 */
export const HuggingFaceSignedOutReasonSchema = Schema.Literal("Denied", "Expired", "Failed", "SessionExpired")
export type HuggingFaceSignedOutReason = typeof HuggingFaceSignedOutReasonSchema.Type

/** How a sign-in in progress is being completed: the browser on this computer, or a code entered on any device. */
export const HuggingFaceSignInMethodSchema = Schema.Literal("Browser", "Device")
export type HuggingFaceSignInMethod = typeof HuggingFaceSignInMethodSchema.Type

/** The Hugging Face account ACN uses for gated models, private repositories and higher rate limits. It never carries a token. */
export const HuggingFaceAccountSchema = Schema.Union(
  Schema.TaggedStruct("SignedOut", {
    reason: Schema.optionalWith(HuggingFaceSignedOutReasonSchema, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("SigningIn", { method: HuggingFaceSignInMethodSchema }),
  Schema.TaggedStruct("SignedIn", {
    username: Schema.String,
    avatarUrl: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
)
export type HuggingFaceAccount = typeof HuggingFaceAccountSchema.Type

/** One sign-in attempt, so a client can follow its own attempt rather than whatever the account shows. */
export const HuggingFaceSignInAttemptIdSchema = Schema.String.pipe(Schema.brand("HuggingFaceSignInAttemptId"))
export type HuggingFaceSignInAttemptId = typeof HuggingFaceSignInAttemptIdSchema.Type

/**
 * How one attempt ended: still pending, signed in, failed (declined, expired or failed), cancelled,
 * replaced by a newer attempt, or unknown to this ACN (it restarted or forgot older attempts).
 */
export const HuggingFaceSignInOutcomeSchema = Schema.Union(
  Schema.TaggedStruct("Pending", {}),
  Schema.TaggedStruct("SignedIn", { username: Schema.String }),
  Schema.TaggedStruct("Failed", { reason: Schema.Literal("Denied", "Expired", "Failed") }),
  Schema.TaggedStruct("Cancelled", {}),
  Schema.TaggedStruct("Superseded", {}),
  Schema.TaggedStruct("Unknown", {}),
)
export type HuggingFaceSignInOutcome = typeof HuggingFaceSignInOutcomeSchema.Type

/** A browser sign-in to open: ACN receives the redirect on this computer and completes it. */
export const HuggingFaceBrowserSignInSchema = Schema.Struct({
  attemptId: HuggingFaceSignInAttemptIdSchema,
  authorizationUrl: Schema.String,
})
export type HuggingFaceBrowserSignIn = typeof HuggingFaceBrowserSignInSchema.Type

/**
 * A device sign-in: the person enters `userCode` at `verificationUri`, or opens `verificationUriComplete`
 * when Hugging Face returns one with the code filled in, and ACN polls until it completes. Durations are seconds.
 */
export const HuggingFaceDeviceSignInSchema = Schema.Struct({
  attemptId: HuggingFaceSignInAttemptIdSchema,
  verificationUri: Schema.String,
  verificationUriComplete: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  userCode: Schema.String,
  expiresIn: Schema.Int.pipe(Schema.positive()),
  interval: Schema.Int.pipe(Schema.positive()),
})
export type HuggingFaceDeviceSignIn = typeof HuggingFaceDeviceSignInSchema.Type

/** What a gated Hugging Face repository needs before Magnitude can read it. */
export const HuggingFaceAccessSchema = Schema.Literal("SignIn", "AcceptTerms")
export type HuggingFaceAccess = typeof HuggingFaceAccessSchema.Type

/** The page where a repository's terms are accepted. */
export const huggingFaceRepositoryUrl = (repository: string): string => `https://huggingface.co/${repository}`
