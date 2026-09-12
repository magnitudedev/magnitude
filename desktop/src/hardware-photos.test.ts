import { Option } from "effect"
import { expect, it } from "vitest"
import { hardwarePhotos, hardwarePresentation } from "./hardware-photos"

const mac = (model: string) => ({ _tag: "Identified" as const, manufacturer: "Apple", model })
it.each([
  ["Mac16,5", "book16"], ["Mac16,8", "book14"], ["Mac16,9", "studio"],
  ["Mac16,11", "mini"], ["Mac16,12", "air13"],
] as const)("matches %s by device identity, not the M4 processor name", (model, photo) => {
  expect(hardwarePresentation(mac(model), ["Apple M4 Max"]).photo).toEqual(Option.some(hardwarePhotos[photo]))
})
it("does not assign the redesigned mini or a guessed Mac image to an unknown device", () => {
  expect(hardwarePresentation(mac("Macmini9,1"), []).photo).toEqual(Option.none())
  expect(hardwarePresentation(mac("Mac99,1"), ["Apple M4 Max"]).photo).toEqual(Option.none())
  expect(hardwarePresentation({ _tag: "Identified", manufacturer: "Example", model: "Mac16,5" }, []).photo).toEqual(Option.none())
})
it.each(["NVIDIA GeForce RTX 5090", "GeForce RTX 5090", "RTX 5090 Founders Edition"])("uses a labeled component image for %s", name => {
  expect(hardwarePresentation({ _tag: "Unavailable" }, [name]).photo).toEqual(Option.some(hardwarePhotos.rtx5090))
})
it.each(["RTX 5090 Laptop GPU", "NVIDIA GeForce RTX 5090 D", "RTX 50900", "RTX 590", "RX 590", "RTX 4090"])("never substitutes a different GPU photo for %s", name => {
  expect(hardwarePresentation(null, [name]).photo).toEqual(Option.none())
})
it("keeps the enclosure as the primary photo and preserves unknown PC identity", () => {
  expect(hardwarePresentation(mac("Mac16,5"), ["RTX 5090"]).photo).toEqual(Option.some(hardwarePhotos.book16))
  expect(hardwarePresentation({ _tag: "Identified", manufacturer: "Dell", model: "Precision 3680" }, []).name).toEqual(Option.some("Dell Precision 3680"))
})
