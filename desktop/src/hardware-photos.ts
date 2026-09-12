import { Option, Schema } from "effect"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import inventory from "../../assets/hardware/inventory.json"

export const HardwarePhoto = Schema.Struct({
  src: Schema.String, subject: Schema.String,
  kind: Schema.Literal("Device", "Component"),
})
export type HardwarePhoto = typeof HardwarePhoto.Type
const HardwarePhotoEntry = Schema.Struct({
  id: Schema.NonEmptyString.pipe(Schema.brand("HardwarePhotoId")), file: Schema.NonEmptyString,
  subject: Schema.NonEmptyString, manufacturer: Schema.NonEmptyString,
  source: Schema.NonEmptyString, imageUrl: Schema.NonEmptyString,
  match: Schema.Union(
    Schema.TaggedStruct("Apple", { models: Schema.NonEmptyArray(Schema.NonEmptyString) }),
    Schema.TaggedStruct("Pc", { manufacturers: Schema.NonEmptyArray(Schema.NonEmptyString), models: Schema.NonEmptyArray(Schema.NonEmptyString) }),
    Schema.TaggedStruct("Gpu", { names: Schema.NonEmptyArray(Schema.NonEmptyString) }),
  ),
})
const images = import.meta.glob<string>("../../assets/hardware/*.{jpg,png}", { eager: true, query: "?url", import: "default" })
export const hardwarePhotoInventory = Schema.decodeUnknownSync(Schema.Array(HardwarePhotoEntry))(inventory).map(entry => {
  const src = images[`../../assets/hardware/${entry.file}`]
  if (!src) throw new Error(`Missing bundled hardware photo: ${entry.file}`)
  const kind: HardwarePhoto["kind"] = entry.match._tag === "Gpu" ? "Component" : "Device"
  return { ...entry, src, kind }
})
export const hardwarePhotos: Readonly<Record<string, HardwarePhoto>> = Object.fromEntries(hardwarePhotoInventory.map(entry => [entry.id, entry]))
const normalize = (value: string) => value.trim().replace(/\s+/g, " ").toLowerCase()
// Preserve GPU variant suffixes: Ti, SUPER, XT/XTX and laptop names are different hardware.
const gpuName = (value: string) => normalize(value).replace(/^(?:nvidia |amd |intel\(r\) |intel )/, "").replace(/^(?:geforce |radeon )/, "")

export const hardwarePresentation = (identity: MachineIdentityObservation | null, accelerators: readonly string[]) => {
  const identified = identity?._tag === "Identified" ? identity : null
  const devicePhoto = Option.fromNullable(identified ? hardwarePhotoInventory.find(({ match }) => {
    if (match._tag === "Gpu") return false
    const manufacturer = normalize(identified.manufacturer)
    const accepted = match._tag === "Apple" ? ["apple", "apple inc."] : match.manufacturers.map(normalize)
    return accepted.includes(manufacturer) && match.models.some(model => normalize(model) === normalize(identified.model))
  }) : undefined)
  const photo = Option.orElse(devicePhoto, () => Option.fromNullable(hardwarePhotoInventory.find(({ match }) =>
    match._tag === "Gpu" && accelerators.some(name => match.names.some(candidate => gpuName(candidate) === gpuName(name))))))
  const name = Option.orElse(Option.map(devicePhoto, value => value.subject), () => identified ? Option.some(`${identified.manufacturer} ${identified.model}`) : Option.none<string>())
  return { name, photo }
}
