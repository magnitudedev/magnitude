import { readFileSync, readdirSync } from "node:fs"
import { createHash } from "node:crypto"
import { Option } from "effect"
import { expect, it } from "vitest"
import { hardwarePhotos, hardwarePresentation, hardwarePhotoInventory } from "./hardware-photos"

const mac = (model: string) => ({ _tag: "Identified" as const, manufacturer: "Apple", model })
it.each([
  ["Mac16,5", "book16"], ["Mac16,8", "book14"], ["Mac16,9", "studio"],
  ["Mac16,11", "mini"], ["Mac16,12", "air13"],
] as const)("matches %s by device identity, not the M4 processor name", (model, photo) => {
  expect(hardwarePresentation(mac(model), ["Apple M4 Max"]).photo).toEqual(Option.some(hardwarePhotos[photo]))
})
it("does not assign the redesigned mini or a guessed Mac image to an unknown device", () => {
  expect(hardwarePresentation(mac("Macmini9,1"), []).photo).toEqual(Option.some(hardwarePhotos["mini-m1"]))
  expect(hardwarePresentation(mac("Mac99,1"), ["Apple M4 Max"]).photo).toEqual(Option.none())
  expect(hardwarePresentation({ _tag: "Identified", manufacturer: "Example", model: "Mac16,5" }, []).photo).toEqual(Option.none())
})
it.each(["NVIDIA GeForce RTX 5090", "GeForce RTX 5090", "RTX 5090 Founders Edition"])("uses a labeled component image for %s", name => {
  expect(hardwarePresentation({ _tag: "Unavailable" }, [name]).photo).toEqual(Option.some(hardwarePhotos.rtx5090))
})
it.each(["RTX 5090 Laptop GPU", "NVIDIA GeForce RTX 5090 D", "RTX 50900", "RTX 590", "RX 590", "RTX 4090 Laptop GPU"])("never substitutes a different GPU photo for %s", name => {
  expect(hardwarePresentation(null, [name]).photo).toEqual(Option.none())
})
it("keeps the enclosure as the primary photo and preserves unknown PC identity", () => {
  expect(hardwarePresentation(mac("Mac16,5"), ["RTX 5090"]).photo).toEqual(Option.some(hardwarePhotos.book16))
  expect(hardwarePresentation({ _tag: "Identified", manufacturer: "Dell", model: "Precision 3680" }, []).name).toEqual(Option.some("Dell Precision 3680"))
})

it.each(hardwarePhotoInventory)("matches the verified identity for $id", entry => {
  const match = entry.match
  if (match._tag === "Gpu") {
    for (const name of match.names) {
      expect(hardwarePresentation(null, [name]).photo).toEqual(Option.some(entry))
      expect(hardwarePresentation(null, [name + " Laptop GPU"]).photo).toEqual(Option.none())
      expect(hardwarePresentation(null, [name + " D"]).photo).toEqual(Option.none())
    }
  } else {
    for (const model of match.models) {
      const manufacturer = match._tag === "Apple" ? "Apple Inc." : match.manufacturers[0]
      expect(hardwarePresentation({ _tag: "Identified", manufacturer, model }, []).photo).toEqual(Option.some(entry))
      expect(hardwarePresentation({ _tag: "Identified", manufacturer: "Unrelated vendor", model }, []).photo).toEqual(Option.none())
    }
  }
})
it("contains distinct professional product photographs and unambiguous matches", () => {
  expect(hardwarePhotoInventory.length).toBeGreaterThanOrEqual(30)
  expect(new Set(hardwarePhotoInventory.map(entry => entry.file)).size).toBe(hardwarePhotoInventory.length)
  expect(new Set(hardwarePhotoInventory.map(entry => entry.id)).size).toBe(hardwarePhotoInventory.length)
  const identities = hardwarePhotoInventory.flatMap(({ match }) => match._tag === "Gpu" ? match.names : match.models)
  expect(new Set(identities).size).toBe(identities.length)
  for (const entry of hardwarePhotoInventory) {
    expect(entry.source).toMatch(/^https:\/\//)
    expect(entry.source).not.toContain("wikimedia")
    expect(entry.imageUrl).toMatch(/^https:\/\//)
    expect(entry.manufacturer).toBeTruthy()
    expect(entry.src).toBeTruthy()
    expect(entry.kind).toBe(entry.match._tag === "Gpu" ? "Component" : "Device")
  }
})

it("bundles every inventory file exactly once without duplicate image bytes", () => {
  const directory = new URL("../../assets/hardware/", import.meta.url)
  const files = readdirSync(directory).filter(file => /\.(jpg|png)$/.test(file)).sort()
  expect(files).toEqual(hardwarePhotoInventory.map(entry => entry.file).sort())
  const hashes = files.map(file => createHash("sha256").update(readFileSync(new URL(file, directory))).digest("hex"))
  expect(new Set(hashes).size).toBe(files.length)
})
