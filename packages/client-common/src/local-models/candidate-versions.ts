import { Option } from "effect"
import type { CandidateVersion } from "@magnitudedev/sdk"
import { unsupportedReasonMessages } from "./failure-messages"

/** A version may run here unless the engine said it is too large or can't run it. */
const mayRun = (version: CandidateVersion): boolean =>
  Option.match(version.assessment, { onNone: () => true, onSome: ({ _tag }) => _tag === "Fits" })

/**
 * Why no version of a repository runs on this computer, from the engine's verdicts, or nothing when
 * some version may run: every version too large, every version unsupported with its reasons said
 * once, or some of each.
 */
export const noRunnableVersionReason = (versions: readonly CandidateVersion[]): Option.Option<string> => {
  if (versions.length === 0 || versions.some(mayRun)) return Option.none()
  const unsupportedReasons = versions.flatMap(({ assessment }) =>
    Option.isSome(assessment) && assessment.value._tag === "Unsupported" ? assessment.value.reasons : [])
  const tooLarge = versions.some(({ assessment }) => Option.exists(assessment, ({ _tag }) => _tag === "DoesNotFit"))
  const reasons = unsupportedReasonMessages(unsupportedReasons).join(". ")
  if (unsupportedReasons.length === 0) return Option.some("Every version is too large for this machine.")
  return Option.some(tooLarge
    ? `Some versions are too large for this machine, and Magnitude can't run the others: ${reasons}.`
    : `Magnitude can't run any version: ${reasons}.`)
}
