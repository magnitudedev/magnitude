import { createRequire } from "node:module"
import { MachineIdentity, type MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import { Effect, Schema } from "effect"

const FirmwareIdentity = Schema.Struct({ manufacturer: Schema.String, model: Schema.String })
// Firmware can contain placeholders or control characters. Those are not device names.
const normalize = (value: string) => value.replace(/[\x00-\x1f\x7f]/g, " ").trim()
const placeholder = /^(?:unknown|none|not (?:specified|applicable)|default string|system (?:manufacturer|product name)|to be filled by o\.?e\.?m\.?)$/i
export const readMachineIdentity = (load: () => unknown): Effect.Effect<MachineIdentityObservation> => Effect.gen(function* () {
  const binding = yield* Effect.try(load)
  const raw = yield* Effect.try(() => (binding as { machineIdentity: () => unknown }).machineIdentity())
  const fields = yield* Schema.decodeUnknown(FirmwareIdentity)(raw)
  const identity = yield* Schema.decodeUnknown(MachineIdentity)({ manufacturer: normalize(fields.manufacturer), model: normalize(fields.model) })
  if (placeholder.test(identity.manufacturer) || placeholder.test(identity.model)) return { _tag: "Unavailable" as const }
  return { _tag: "Identified" as const, ...identity }
}).pipe(Effect.catchAll(() => Effect.succeed({ _tag: "Unavailable" as const })))

export const nativeMachineIdentity = (addonPath: string) => readMachineIdentity(() => createRequire(import.meta.url)(addonPath))
