import { Effect, Schema } from "effect"
import { WindowsProcessIdentity } from "@magnitudedev/utils/windows-native"
import type { LegacyOwner } from "./legacy-owner"

// Historical PowerShell StartTime.ToUniversalTime().Ticks starts at year 1;
// native FILETIME starts at 1601. Both use 100-nanosecond units.
const fileTimeEpochTicks = 504911232000000000n
const maximumDateTimeTicks = 3155378975999999999n
const LegacyWindowsStart = Schema.String.pipe(
  Schema.pattern(/^windows:[1-9][0-9]{17,18}$/),
  Schema.filter(value => {
    const ticks = BigInt(value.slice("windows:".length))
    return ticks >= fileTimeEpochTicks && ticks <= maximumDateTimeTicks
  }),
  Schema.brand("LegacyWindowsStartIdentity"),
)

export class LegacyWindowsIdentityInvalid extends Schema.TaggedError<LegacyWindowsIdentityInvalid>()(
  "LegacyWindowsIdentityInvalid", { message: Schema.String },
) {}

/** Converts frozen migration data only; this does not observe or authorize a process. */
export const legacyWindowsProcessIdentity = (owner: LegacyOwner) => Effect.gen(function* () {
  const started = yield* Schema.decodeUnknown(LegacyWindowsStart)(owner.processStartIdentity)
  const creationTime = (BigInt(started.slice("windows:".length)) - fileTimeEpochTicks)
    .toString(16).padStart(16, "0")
  return yield* Schema.decodeUnknown(WindowsProcessIdentity)({ pid: owner.pid, creationTime })
}).pipe(Effect.mapError(() => new LegacyWindowsIdentityInvalid({
  message: "Legacy Windows owner has an invalid PID or UTC process-start identity",
})))
