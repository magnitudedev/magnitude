import { Data } from "effect"

/** Update setup found this build can't update itself, and says why in a sentence written where it found it. */
export class UpdateSetupRefused extends Data.TaggedError("UpdateSetupRefused")<{ readonly reason: string }> {}

export const UPDATE_SETUP_FAILED = "Application update setup could not be read."

/**
 * Why updates are unavailable after update setup failed: the reason a refusal wrote, or, for anything unexpected (a
 * file or decoding error, whose message is no sentence for people), the general one.
 */
export const updateSetupReason = (error: unknown): string =>
  error instanceof UpdateSetupRefused ? error.reason : UPDATE_SETUP_FAILED
