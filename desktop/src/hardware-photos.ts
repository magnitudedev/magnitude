import { Option, Schema } from "effect"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import book16 from "../../assets/hardware/macbook-pro-16.jpg"
import book14 from "../../assets/hardware/macbook-pro-14.jpg"
import air13 from "../../assets/hardware/macbook-air-13.jpg"
import studio from "../../assets/hardware/mac-studio.jpg"
import mini from "../../assets/hardware/mac-mini-2024.jpg"
import rtx5090 from "../../assets/hardware/rtx-5090.png"

export const HardwarePhoto = Schema.Struct({
  src: Schema.String, subject: Schema.String, author: Schema.String,
  source: Schema.String, license: Schema.String, licenseUrl: Schema.String,
  kind: Schema.Literal("Device", "Component"),
})
export type HardwarePhoto = typeof HardwarePhoto.Type
const commons = (name: string) => `https://commons.wikimedia.org/wiki/File:${encodeURIComponent(name.replaceAll(" ", "_"))}`
const cc0 = { author: "AzureSaturn", license: "CC0", licenseUrl: "https://creativecommons.org/publicdomain/zero/1.0/", kind: "Device" as const }
export const hardwarePhotos = {
  book16: { ...cc0, src: book16, subject: "16-inch MacBook Pro", source: commons("MacBook Pro (16-inch, M4 Pro, Silver).jpg") },
  book14: { ...cc0, src: book14, subject: "14-inch MacBook Pro", source: commons("MacBook Pro (14-inch, M5, Space Black).jpg") },
  air13: { ...cc0, src: air13, subject: "13-inch MacBook Air", source: commons("MacBook Air (13-inch, M4, Silver).jpg") },
  studio: { src: studio, subject: "Mac Studio", author: "Yasu", source: commons("Mac Studio (2022) front.jpg"), license: "CC BY-SA 3.0", licenseUrl: "https://creativecommons.org/licenses/by-sa/3.0/", kind: "Device" },
  mini: { src: mini, subject: "Mac mini (2024)", author: "Seasider53", source: commons("Mac mini 2024 M4.jpg"), license: "CC BY 4.0", licenseUrl: "https://creativecommons.org/licenses/by/4.0/", kind: "Device" },
  rtx5090: { src: rtx5090, subject: "GeForce RTX 5090 Founders Edition", author: "ZMASLO", source: commons("RTX 5090 - duża wydajność dużym kosztem (2160p 30fps VP9 LQ-96kbit AAC)-00.05.24.568.png"), license: "CC BY 3.0", licenseUrl: "https://creativecommons.org/licenses/by/3.0/", kind: "Component" },
} satisfies Record<string, HardwarePhoto>

// Explicit enclosure matches from Apple's model-identification pages (see assets/hardware/README.md).
// Shared enclosure artwork does not claim the stock finish or internal chip is the user's configuration.
const macFamilies: ReadonlyArray<{ readonly ids: readonly string[]; readonly photo: HardwarePhoto }> = [
  { ids: ["Mac16,5", "Mac16,7", "Mac15,7", "Mac15,9", "Mac15,11", "Mac14,6", "Mac14,10", "MacBookPro18,1", "MacBookPro18,2", "Mac17,6", "Mac17,8"], photo: hardwarePhotos.book16 },
  { ids: ["Mac16,1", "Mac16,6", "Mac16,8", "Mac15,3", "Mac15,6", "Mac15,8", "Mac15,10", "Mac14,5", "Mac14,9", "MacBookPro18,3", "MacBookPro18,4", "Mac17,2", "Mac17,7", "Mac17,9"], photo: hardwarePhotos.book14 },
  { ids: ["Mac16,12", "Mac17,3"], photo: hardwarePhotos.air13 },
  { ids: ["Mac13,1", "Mac13,2", "Mac14,13", "Mac14,14", "Mac15,14", "Mac16,9"], photo: hardwarePhotos.studio },
  { ids: ["Mac16,10", "Mac16,11"], photo: hardwarePhotos.mini },
]

export const hardwarePresentation = (identity: MachineIdentityObservation | null, accelerators: readonly string[]) => {
  const identified = identity?._tag === "Identified" ? identity : null
  const devicePhoto = identified && /^(?:Apple|Apple Inc\.)$/i.test(identified.manufacturer)
    ? Option.fromNullable(macFamilies.find(family => family.ids.includes(identified.model))?.photo) : Option.none<HardwarePhoto>()
  const photo = Option.orElse(devicePhoto, () => accelerators.some(name => /^(?:NVIDIA\s+)?(?:GeForce\s+)?RTX\s+5090(?:\s+Founders Edition)?$/i.test(name.trim()))
    ? Option.some(hardwarePhotos.rtx5090) : Option.none<HardwarePhoto>())
  const name = Option.orElse(Option.map(devicePhoto, value => value.subject), () => identified ? Option.some(`${identified.manufacturer} ${identified.model}`) : Option.none<string>())
  return { name, photo }
}
