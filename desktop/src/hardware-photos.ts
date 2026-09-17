import { Option, Schema } from "effect"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import inventory from "../../assets/hardware/inventory.json"

const PhotoFraming = Schema.Struct({
  sourceWidth: Schema.Positive, sourceHeight: Schema.Positive,
  x: Schema.NonNegative, y: Schema.NonNegative,
  width: Schema.Positive, height: Schema.Positive,
})
export const HardwarePhoto = Schema.Struct({
  src: Schema.String, subject: Schema.String,
  framing: PhotoFraming,
  kind: Schema.Literal("Device", "Component"),
})
export type HardwarePhoto = typeof HardwarePhoto.Type
const DeviceCategory = Schema.Literal("Laptop", "Gaming laptop", "Gaming tablet", "Mobile workstation", "Mini PC", "AI mini PC", "Desktop", "Workstation", "All-in-one", "Server", "Graphics card")
const HardwarePhotoEntry = Schema.Struct({
  id: Schema.NonEmptyString.pipe(Schema.brand("HardwarePhotoId")), file: Schema.NonEmptyString,
  framing: PhotoFraming,
  subject: Schema.NonEmptyString, manufacturer: Schema.NonEmptyString, category: DeviceCategory,
  source: Schema.NonEmptyString, imageUrl: Schema.NonEmptyString,
  processorModels: Schema.optionalWith(Schema.NonEmptyArray(Schema.NonEmptyString), { as: "Option", exact: true }),
  match: Schema.Union(
    Schema.TaggedStruct("Apple", { models: Schema.NonEmptyArray(Schema.NonEmptyString) }),
    Schema.TaggedStruct("Pc", {
      manufacturers: Schema.NonEmptyArray(Schema.NonEmptyString),
      field: Schema.Literal("model", "family", "version"), models: Schema.NonEmptyArray(Schema.NonEmptyString),
    }),
    Schema.TaggedStruct("Gpu", { names: Schema.NonEmptyArray(Schema.NonEmptyString) }),
  ),
})
const images = import.meta.glob<string>("../../assets/hardware/*.{jpg,png,webp}", { eager: true, query: "?url", import: "default" })
export const hardwarePhotoInventory = Schema.decodeUnknownSync(Schema.Array(HardwarePhotoEntry))(inventory).map(entry => {
  const src = images[`../../assets/hardware/${entry.file}`]
  if (!src) throw new Error(`Missing bundled hardware photo: ${entry.file}`)
  const kind: HardwarePhoto["kind"] = entry.match._tag === "Gpu" ? "Component" : "Device"
  return { ...entry, src, kind }
})
export const hardwarePhotos: Readonly<Record<string, HardwarePhoto>> = Object.fromEntries(hardwarePhotoInventory.map(entry => [entry.id, entry]))
export const normalizeHardwareName = (value: string) => value.trim().replace(/\s+/g, " ").toLowerCase()
// Suffixes are identity: never collapse Ti, SUPER, XT/XTX, D, or Laptop GPU.
export const normalizeGpuName = (value: string) => normalizeHardwareName(value)
  .replace(/\((?:r|tm)\)|[®™]/g, "")
  .replace(/\s+/g, " ")
  .replace(/^(?:nvidia |amd |intel )/, "").replace(/^(?:geforce |radeon )/, "")
  .replace(/\s+\(radv [a-z0-9_]+\)$/, "")
  .replace(/ graphics$/, "")
const portableCategories = new Set(["Laptop", "Gaming laptop", "Gaming tablet", "Mobile workstation", "All-in-one"])

/** Only pass accelerators backed by observed dedicated memory as component candidates. */
export const hardwarePresentation = (identity: MachineIdentityObservation | null, dedicatedAccelerators: readonly string[], processor: Option.Option<string>) => {
  const identified = identity?._tag === "Identified" ? identity : null
  const devicePhoto = Option.fromNullable(identified ? hardwarePhotoInventory.find(({ match, processorModels }) => {
    if (match._tag === "Gpu") return false
    if (Option.isSome(processorModels) && (Option.isNone(processor) || !processorModels.value.some(model => normalizeHardwareName(processor.value).split(/[^a-z0-9+-]+/).includes(normalizeHardwareName(model))))) return false
    const manufacturer = normalizeHardwareName(identified.manufacturer)
    const accepted = match._tag === "Apple" ? ["apple", "apple inc."] : match.manufacturers.map(normalizeHardwareName)
    const field = match._tag === "Apple" || match.field === "model" ? Option.some(identified.model) : identified[match.field]
    return accepted.includes(manufacturer) && Option.isSome(field) && match.models.some(model => normalizeHardwareName(model) === normalizeHardwareName(field.value))
  }) : undefined)
  const enclosure = Option.match(devicePhoto, {
    onSome: entry => entry.category,
    onNone: () => ({ Portable: "Laptop / tablet", Desktop: "Desktop", AllInOne: "All-in-one", MiniPc: "Mini PC", Server: "Server", Unknown: "Computer" })[identity?.formFactor ?? "Unknown"],
  })
  const portable = identity?.formFactor === "Portable" || identity?.formFactor === "AllInOne" || portableCategories.has(enclosure)
  const photo = Option.orElse(devicePhoto, () => portable ? Option.none() : Option.fromNullable(hardwarePhotoInventory.find(({ match }) =>
    match._tag === "Gpu" && dedicatedAccelerators.some(name => match.names.some(candidate => normalizeGpuName(candidate) === normalizeGpuName(name))))))
  const name = Option.orElse(Option.map(devicePhoto, value => value.subject), () => identified ? Option.some(`${identified.manufacturer} ${identified.model}`) : Option.none<string>())
  return { name, photo, category: enclosure, deviceId: Option.map(devicePhoto, entry => entry.id) }
}
