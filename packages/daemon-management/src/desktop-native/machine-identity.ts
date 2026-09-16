import { createRequire } from "node:module"
import { MachineIdentity, type MachineIdentityObservation, type MachineFormFactor } from "@magnitudedev/sdk/desktop-host"
import { Effect, Option, Schema } from "effect"

const FirmwareIdentity = Schema.Struct({
  manufacturer: Schema.String, model: Schema.String,
  family: Schema.String, version: Schema.String, chassisType: Schema.Int,
})
// Firmware can contain placeholders or control characters. Those are not device names.
const normalize = (value: string) => value.replace(/[\x00-\x1f\x7f]/g, " ").trim().replace(/\s+/g, " ")
const placeholder = /^(?:unknown|none|not (?:specified|applicable)|default string|system (?:manufacturer|product name|version|family)|to be filled by o\.?e\.?m\.?)$/i
const label = (raw: string) => Option.filter(Option.some(normalize(raw)), value => value.length > 0 && value.length <= 255 && !placeholder.test(value))
/** SMBIOS Type 3 enclosure type; the native reader removes the chassis-lock bit. */
export const machineFormFactor = (type: number): MachineFormFactor => {
  if ([8, 9, 10, 11, 14, 30, 31, 32].includes(type)) return "Portable"
  if ([3, 4, 5, 6, 7, 15, 16].includes(type)) return "Desktop"
  if (type === 13) return "AllInOne"
  if ([35, 36].includes(type)) return "MiniPc"
  if ([17, 23, 28, 29].includes(type)) return "Server"
  return "Unknown"
}
export const readMachineIdentity = (load: () => unknown): Effect.Effect<MachineIdentityObservation> => Effect.gen(function* () {
  const binding = yield* Effect.try(load)
  const raw = yield* Effect.try(() => (binding as { machineIdentity: () => unknown }).machineIdentity())
  const fields = yield* Schema.decodeUnknown(FirmwareIdentity)(raw)
  const formFactor = machineFormFactor(fields.chassisType)
  const manufacturer = label(fields.manufacturer), model = label(fields.model)
  if (Option.isNone(manufacturer) || Option.isNone(model)) return { _tag: "Unavailable" as const, formFactor }
  const identity = yield* Schema.validate(MachineIdentity)({
    manufacturer: manufacturer.value, model: model.value,
    family: label(fields.family), version: label(fields.version), formFactor,
  })
  return { _tag: "Identified" as const, ...identity }
}).pipe(Effect.catchAll(() => Effect.succeed({ _tag: "Unavailable" as const, formFactor: "Unknown" as const })))

export const nativeMachineIdentity = (addonPath: string) => readMachineIdentity(() => createRequire(import.meta.url)(addonPath))
