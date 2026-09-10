import * as Command from "@effect/platform/Command"
import { Data, Effect } from "effect"

// Substituted by release compilation, never read from the installed user's environment.
declare const MAGNITUDE_APPLE_TEAM_ID: string
export const APPLE_TEAM_ID = typeof MAGNITUDE_APPLE_TEAM_ID === "undefined" ? "" : MAGNITUDE_APPLE_TEAM_ID

export class AppleSignatureInvalid extends Data.TaggedError("AppleSignatureInvalid")<{
  readonly path: string
  readonly message: string
}> {}

export const appleRequirement = (identifier: string, team = APPLE_TEAM_ID): string => {
  if (!/^[a-zA-Z0-9.-]+$/.test(identifier) || (team !== "" && !/^[A-Z0-9]{10}$/.test(team))) {
    throw new Error("Invalid compiled Apple signing identity")
  }
  const identity = `identifier "${identifier}"`
  return team === "" ? identity : `anchor apple generic and ${identity} and certificate leaf[subject.OU] = "${team}" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists`
}

export const verifyAppleCode = (path: string, identifier: string) =>
  Command.make("/usr/bin/codesign", "--verify", "--strict", "--deep", "-R", `=${appleRequirement(identifier)}`, path).pipe(
    Command.exitCode,
    Effect.filterOrFail((code) => code === 0, () => new AppleSignatureInvalid({ path, message: "Apple code signature does not match Magnitude's publisher" })),
    Effect.timeout("30 seconds"),
    Effect.mapError((error) => error instanceof AppleSignatureInvalid ? error : new AppleSignatureInvalid({ path, message: String(error) })),
    Effect.asVoid,
  )
